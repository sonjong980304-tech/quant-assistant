"""미국주식 도메인 에이전트(src/agents/domain_us.py) 단위/통합 테스트 (TDD, HA-7).

HA-6(한국주식 도메인 에이전트, src/agents/domain_kr.py)과 대칭 역할이지만, 미국은
재무데이터 출처가 us_financials 테이블 하나뿐이라 HA-2(DART/FnGuide 소스판단)에
해당하는 로직 자체가 없다.

검증 대상:
- 질문을 보고 재무(HA-2 대응 없음, us_financials 기반 신규 헬퍼) / 주가·기술지표
  (HA-4, get_price_snapshot_us)로 위임 라우팅한다 — AC6(미국 부분).
- 회사명/티커 추출: 이미 티커 형태(AAPL 등)인 입력은 그대로 쓰고, 아닌 경우
  us_company 테이블(HA-4가 이미 채운 이름↔티커) 및 llm_fn 폴백으로 해석한다.
- 가까운 계층 재시도: 데이터 에이전트 호출이 실패(예외 또는 ok=False)하면 도메인
  에이전트 레벨에서 1회 더 재시도하고, 그래도 실패하면 예외를 전파하지 않고 실패
  사유를 반환값에 담는다(HA-6과 동일 원칙).
"""
from __future__ import annotations

from src.agents.domain_us import (
    _call_with_retry,
    _classify_intent,
    _extract_ticker_token,
    _filter_common_stock,
    _is_common_stock_by_name_fallback,
    answer_us_question,
    answer_us_screening,
    get_financials_us,
    resolve_ticker_us,
)
from src.db import connect, connect_readonly, init_db


def _seed_us_db(tmp_path) -> str:
    db = tmp_path / "domain_us.db"
    init_db(str(db))
    conn = connect(str(db))
    try:
        conn.execute(
            "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            ("AAPL", "Apple Inc.", "NASDAQ", "Technology", 3.0e12, "2026-07-01"),
        )
        conn.execute(
            "INSERT INTO us_prices(stock_code, date, open, high, low, close, volume) "
            "VALUES (?,?,?,?,?,?,?)",
            ("AAPL", "2026-07-11", 196.5, 199.0, 195.5, 198.2, 4.2e7),
        )
        for statement_type, item_key, value in [
            ("income_stmt", "Total Revenue", 1.0e11),
            ("income_stmt", "Operating Income", 3.0e10),
            ("income_stmt", "Net Income", 2.5e10),
            ("balance_sheet", "Stockholders Equity", 8.0e10),
        ]:
            conn.execute(
                "INSERT INTO us_financials"
                "(stock_code, as_of_date, period_type, statement_type, item_key, item_value, "
                "disclosed_date, source, collected_at) VALUES (?,?,?,?,?,?,?,?,?)",
                ("AAPL", "2026-06-30", "quarterly", statement_type, item_key, value,
                 "2026-07-12", "yfinance", "2026-07-12T00:00:00"),
            )
        # PER=시총/순이익(TTM)이 계산되려면 최근 4개 quarterly Net Income이 필요하다
        # (data_access_us._ttm_net_income, 4개 미만이면 추정하지 않고 None).
        for as_of_date, value in [
            ("2026-03-31", 2.4e10),
            ("2025-12-31", 2.3e10),
            ("2025-09-30", 2.2e10),
        ]:
            conn.execute(
                "INSERT INTO us_financials"
                "(stock_code, as_of_date, period_type, statement_type, item_key, item_value, "
                "disclosed_date, source, collected_at) VALUES (?,?,?,?,?,?,?,?,?)",
                ("AAPL", as_of_date, "quarterly", "income_stmt", "Net Income", value,
                 as_of_date, "yfinance", "2026-07-12T00:00:00"),
            )
        conn.commit()
    finally:
        conn.close()
    return str(db)


# ── _call_with_retry: 가까운 계층 재시도 (단위테스트) ────────────────────────
def test_call_with_retry_returns_success_on_first_try():
    calls = []

    def fn(x):
        calls.append(x)
        return x * 2

    outcome = _call_with_retry(fn, 3)
    assert outcome == {"ok": True, "result": 6, "error": None}
    assert calls == [3]  # 성공하면 재시도하지 않는다


def test_call_with_retry_retries_once_then_succeeds():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("일시 실패")
        return "복구됨"

    outcome = _call_with_retry(fn)
    assert outcome == {"ok": True, "result": "복구됨", "error": None}
    assert len(calls) == 2  # 최초 1회 + 재시도 1회


def test_call_with_retry_always_fails_does_not_propagate_exception():
    """fake로 항상 실패하도록 주입된 상황에서도 예외가 전파되지 않는다(HA-6과 동일 원칙)."""
    calls = []

    def always_fails():
        calls.append(1)
        raise ValueError("영구 실패")

    outcome = _call_with_retry(always_fails)  # 예외를 던지지 않아야 함
    assert outcome["ok"] is False
    assert outcome["result"] is None
    assert "영구 실패" in outcome["error"]
    assert len(calls) == 2  # 최초 1회 + 재시도 1회, 그 이상 재시도하지 않음


def test_call_with_retry_treats_ok_false_dict_as_failure():
    calls = []

    def fn():
        calls.append(1)
        return {"ok": False, "error": "문법 오류"}

    outcome = _call_with_retry(fn)
    assert outcome["ok"] is False
    assert "문법 오류" in outcome["error"]
    assert len(calls) == 2


# ── _classify_intent: 재무/주가/둘다 판단 (LLM 우선, 키워드 폴백) ────────────────
def test_us_classify_intent_keyword_fallback_without_llm():
    """llm_fn 미주입 시 기존 키워드 휴리스틱 그대로(회귀)."""
    assert _classify_intent("AAPL PER 알려줘") == "financial"
    assert _classify_intent("AAPL 주가 알려줘") == "price"
    assert _classify_intent("AAPL 알려줘") == "both"  # 둘 다 없으면 안전 폴백


def test_us_classify_intent_llm_first_overrides_keyword():
    """LLM 우선: 재무 키워드(PER)가 있어도 llm_fn 판단(price)을 먼저 채택한다."""
    calls: list[str] = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return "price"

    assert _classify_intent("AAPL PER 알려줘", llm_fn=fake_llm) == "price"
    assert len(calls) == 1
    assert "AAPL PER 알려줘" in calls[0]


def test_us_classify_intent_falls_back_to_keyword_when_llm_unparseable():
    """llm_fn 응답에서 intent 를 못 뽑으면 키워드 휴리스틱으로 폴백한다."""
    assert _classify_intent("AAPL 주가 알려줘", llm_fn=lambda p: "???") == "price"


def test_answer_us_question_threads_llm_fn_into_intent_classification(tmp_path):
    """answer_us_question 이 llm_fn 을 intent 분류까지 관통시킨다(LLM 우선) — 주가 키워드가
    있어도 LLM 이 financial 로 판단하면 주가 에이전트를 호출하지 않는다."""
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)
    seen: list[str] = []

    def fake_llm(prompt: str) -> str:
        seen.append(prompt)
        return "financial"

    try:
        result = answer_us_question("AAPL 주가 알려줘", conn, llm_fn=fake_llm)
    finally:
        conn.close()
    assert any("AAPL 주가 알려줘" in p for p in seen)  # intent 프롬프트로 LLM 호출됨
    assert result["intent"] == "financial"
    assert result["price"] is None  # LLM 판단(financial)이 '주가' 키워드를 이김


# ── resolve_ticker_us: 티커/회사명 추출 ──────────────────────────────────────
def test_resolve_ticker_us_returns_ticker_when_already_ticker_form(tmp_path):
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)
    try:
        assert resolve_ticker_us("AAPL 주가 알려줘", conn) == "AAPL"
    finally:
        conn.close()


def test_resolve_ticker_us_does_not_mistake_financial_keyword_for_ticker(tmp_path):
    """PER/RSI 같은 대문자 키워드가 티커로 오인되지 않아야 한다(AAPL이 실제 티커)."""
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)
    try:
        assert resolve_ticker_us("PER 알려줘 AAPL", conn) == "AAPL"
    finally:
        conn.close()


def test_resolve_ticker_us_falls_back_to_llm_for_korean_company_name(tmp_path):
    """'애플' 같은 한글 회사명은 규칙기반으로 못 찾으므로 llm_fn 폴백에 위임한다.

    llm_fn에는 원본 질문을 그대로 넘기면 안 된다 — 실측 확인: 원문("앤비디아 주식
    주가")을 role="sql" LLM에 그대로 넘기면 "저는 실시간 시세를 조회할 수 없습니다..."
    같은 장문 챗봇 답변이 오고, 그 안에서 회사명/티커를 못 뽑아내 매번 실패했다
    (son-checker류 재현 아님, 실서버 직접 재현: "앤비디아 주식 주가" → error).
    반드시 "티커만 답하라"는 지시가 포함된 프롬프트로 감싸 호출해야 한다.
    """
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return "AAPL"

    try:
        ticker = resolve_ticker_us("애플 주가 알려줘", conn, llm_fn=fake_llm)
    finally:
        conn.close()
    assert ticker == "AAPL"
    assert len(calls) == 1
    assert "애플 주가 알려줘" in calls[0]
    assert "티커" in calls[0]


def test_resolve_ticker_us_resolves_llm_company_name_guess_via_us_company_table(tmp_path):
    """llm_fn이 티커가 아니라 회사명(예: 'Apple')을 반환해도 us_company로 티커를 찾는다."""
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)

    def fake_llm(question):
        return "Apple"

    try:
        ticker = resolve_ticker_us("애플 주가 알려줘", conn, llm_fn=fake_llm)
    finally:
        conn.close()
    assert ticker == "AAPL"


def test_extract_ticker_token_does_not_mistake_lowercase_company_name_for_ticker():
    """'nvidia'(6글자)는 티커 형식 정규식(대문자 1~6글자)과 우연히 길이가 맞아,
    소문자로 써도 upper() 변환 후 '이미 티커'로 오판되던 버그(재현: 실서버에서
    "현재 nvidia 주식 주가"가 stock_code='NVIDIA'로 조회돼 매 시도 실패).
    사용자가 실제로 대문자 티커를 입력했을 때만(예: AAPL) 채택해야 한다."""
    assert _extract_ticker_token("현재 nvidia 주식 주가") is None


def test_extract_ticker_token_still_accepts_uppercase_ticker_in_question():
    assert _extract_ticker_token("AAPL 주가 알려줘") == "AAPL"


def test_resolve_ticker_us_falls_back_to_llm_for_lowercase_company_name(tmp_path):
    """소문자 회사명(nvidia)은 규칙기반 즉시채택 없이 llm_fn 폴백으로 넘어가야 한다."""
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return "NVDA"

    try:
        ticker = resolve_ticker_us("현재 nvidia 주식 주가", conn, llm_fn=fake_llm)
    finally:
        conn.close()
    assert ticker == "NVDA"
    assert len(calls) == 1
    assert "현재 nvidia 주식 주가" in calls[0]


def test_resolve_ticker_us_extracts_ticker_from_verbose_llm_response(tmp_path):
    """LLM이 프롬프트 지시를 어기고 부연설명을 붙여도(실측 재현: role=sql 모델에 원문을
    그대로 넣었을 때 실제로 나온 응답 "엔비디아(NVIDIA, 티커: **NVDA**) 주가를
    말씀하시는 거라면, 저는 실시간 시세를 직접 조회할 수는 없습니다...") 그 안에 있는
    진짜 티커를 최종적으로 뽑아낼 수 있어야 한다(방어적 파싱 — 프롬프트 개선만 믿지 않음)."""
    db = _seed_us_db(tmp_path)
    seed_conn = connect(db)
    seed_conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        ("NVDA", "NVIDIA Corporation Common Stock", "NASDAQ", "Technology", 3.5e12, "2026-07-01"),
    )
    seed_conn.commit()
    seed_conn.close()
    conn = connect_readonly(db)

    def verbose_llm(prompt):
        return (
            "엔비디아(NVIDIA, 티커: **NVDA**) 주가를 말씀하시는 거라면, "
            "저는 실시간 시세를 직접 조회할 수는 없습니다."
        )

    try:
        ticker = resolve_ticker_us("현재 nvidia 주식 주가", conn, llm_fn=verbose_llm)
    finally:
        conn.close()
    assert ticker == "NVDA"


def test_resolve_ticker_us_prefers_exact_ticker_over_name_substring(tmp_path):
    """LLM이 올바른 티커(META)를 줬는데 그 문자열이 다른 회사명("Aqua Metals")의 부분
    문자열이라 회사명 부분조회가 엉뚱한 종목(AQMS)을 먼저 잡아채던 실측 버그 회귀 방지
    (experiment/us-domain-llm-flexible 비교에서 세 접근법 모두 공통 실패했던 지점).
    stock_code 완전일치(META)를 이름 부분일치(Aqua Metals)보다 먼저 확인해야 한다."""
    db = _seed_us_db(tmp_path)
    seed_conn = connect(db)
    seed_conn.executemany(
        "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        [
            ("AQMS", "Aqua Metals Inc. Common Stock", "NASDAQ", "Materials", 1.0e8, "2026-07-01"),
            ("META", "Meta Platforms Inc. Class A Common Stock", "NASDAQ", "Technology", 1.2e12, "2026-07-01"),
        ],
    )
    seed_conn.commit()
    seed_conn.close()
    conn = connect_readonly(db)

    try:
        ticker = resolve_ticker_us("메타 주가 알려줘", conn, llm_fn=lambda p: "META")
    finally:
        conn.close()
    assert ticker == "META"  # AQMS가 아니라 완전일치 티커 META


def test_resolve_ticker_us_returns_none_when_unresolvable(tmp_path):
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)
    try:
        ticker = resolve_ticker_us("알 수 없는 회사 주가", conn, llm_fn=None)
    finally:
        conn.close()
    assert ticker is None


# ── get_financials_us: 미국 재무 조회 헬퍼 ───────────────────────────────────
def test_get_financials_us_returns_metrics_for_known_ticker(tmp_path):
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)
    try:
        result = get_financials_us(conn, "AAPL", asof="2026-07-14")
    finally:
        conn.close()
    assert result is not None
    assert result["stock_code"] == "AAPL"
    assert result["per"] is not None


def test_get_financials_us_returns_none_for_unknown_ticker(tmp_path):
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)
    try:
        result = get_financials_us(conn, "ZZZZ", asof="2026-07-14")
    finally:
        conn.close()
    assert result is None


def test_get_financials_us_defaults_asof_when_not_given(tmp_path):
    """asof 미지정 시 us_financials에 실제 존재하는 최신 disclosed_date를 기준으로 조회한다."""
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)
    try:
        result = get_financials_us(conn, "AAPL")
    finally:
        conn.close()
    assert result is not None
    assert result["stock_code"] == "AAPL"


# ── answer_us_question: 통합(라우팅 + 종합 반환) ─────────────────────────────
def test_answer_us_question_routes_financial_keyword_to_financial_agent_only(tmp_path):
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)
    price_calls = []

    def fake_price_fn(conn, code, **kwargs):
        price_calls.append(code)
        return []

    try:
        result = answer_us_question(
            "AAPL PER 알려줘", conn, price_fn=fake_price_fn,
        )
    finally:
        conn.close()
    assert result["ok"] is True
    assert result["stock_code"] == "AAPL"
    assert result["financial"] is not None
    assert result["price"] is None
    assert price_calls == []  # 주가 에이전트는 호출되지 않아야 함


def test_answer_us_question_routes_price_keyword_to_price_agent_only(tmp_path):
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)
    financial_calls = []

    def fake_financial_fn(conn, code, **kwargs):
        financial_calls.append(code)
        return {"stock_code": code}

    try:
        result = answer_us_question(
            "AAPL 주가 알려줘", conn, financial_fn=fake_financial_fn,
        )
    finally:
        conn.close()
    assert result["ok"] is True
    assert result["price"] is not None
    assert result["price"][0]["stock_code"] == "AAPL"
    assert result["financial"] is None
    assert financial_calls == []  # 재무 에이전트는 호출되지 않아야 함


def test_answer_us_question_calls_both_agents_when_intent_ambiguous(tmp_path):
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)
    try:
        result = answer_us_question("AAPL 알려줘", conn)
    finally:
        conn.close()
    assert result["ok"] is True
    assert result["financial"] is not None
    assert result["price"] is not None


def test_answer_us_question_uses_real_data_agents_end_to_end(tmp_path):
    """fake 없이 실제 get_financials_us/get_price_snapshot_us(HA-4)가 호출되는
    end-to-end 통합테스트 — AC6(미국 부분)."""
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)
    try:
        result = answer_us_question("AAPL PER랑 주가 알려줘", conn)
    finally:
        conn.close()
    assert result["ok"] is True
    assert result["stock_code"] == "AAPL"
    assert result["financial"]["per"] is not None
    assert result["price"][0]["close"] == 198.2


def test_answer_us_question_returns_error_without_exception_when_ticker_unresolvable(tmp_path):
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)
    try:
        result = answer_us_question("알 수 없는 회사 주가", conn)
    finally:
        conn.close()
    assert result["ok"] is False
    assert result["stock_code"] is None
    assert result["error"]


# ── price_history 첨부(버그A 대칭, KR과 동일 원칙) ────────────────────────────
def test_answer_us_question_attaches_price_history_for_chart_question(tmp_path):
    db = _seed_us_db(tmp_path)

    def fake_history(conn, code, *a, **k):
        return [
            {"stock_code": code, "date": "2025-07-15", "close": 180.0},
            {"stock_code": code, "date": "2026-07-14", "close": 198.0},
        ]

    conn = connect_readonly(db)
    try:
        result = answer_us_question(
            "AAPL 최근 1년 주가 그래프 그려줘", conn, price_history_fn=fake_history
        )
    finally:
        conn.close()

    assert result.get("price_history") is not None
    assert result["price_history"]["count"] == 2
    assert result["price_history"]["start_date"] == "2025-07-15"
    assert result["price_history"]["end_date"] == "2026-07-14"


def test_answer_us_question_no_price_history_for_plain_price_question(tmp_path):
    db = _seed_us_db(tmp_path)
    called: list[int] = []

    def fake_history(conn, code, *a, **k):
        called.append(1)
        return []

    conn = connect_readonly(db)
    try:
        result = answer_us_question(
            "AAPL 주가 알려줘", conn, price_history_fn=fake_history
        )
    finally:
        conn.close()

    assert result.get("price_history") is None
    assert called == []


# ── 가까운 계층 재시도가 answer_us_question 레벨에서도 동작 ──────────────────
def test_answer_us_question_retries_failing_data_agent_and_absorbs_final_failure(tmp_path):
    """데이터 에이전트 호출이 fake로 항상 실패하도록 주입되면(HA-6과 동일 원칙) 1회
    재시도하고, 그래도 실패하면 예외 없이 실패 사유가 결과에 담긴다."""
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)
    calls = []

    def always_failing_financial_fn(conn, code, **kwargs):
        calls.append(code)
        raise RuntimeError("데이터 에이전트 영구 실패")

    try:
        result = answer_us_question(
            "AAPL PER 알려줘", conn, financial_fn=always_failing_financial_fn,
        )
    finally:
        conn.close()
    assert result["ok"] is False
    assert result["financial"] is None
    assert "데이터 에이전트 영구 실패" in result["error"]
    assert len(calls) == 2  # 최초 1회 + 재시도 1회


def test_answer_us_question_retries_and_recovers_on_second_attempt(tmp_path):
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)
    calls = []

    def flaky_financial_fn(conn, code, **kwargs):
        calls.append(code)
        if len(calls) < 2:
            raise RuntimeError("일시 실패")
        return {"stock_code": code, "per": 30.0}

    try:
        result = answer_us_question(
            "AAPL PER 알려줘", conn, financial_fn=flaky_financial_fn,
        )
    finally:
        conn.close()
    assert result["ok"] is True
    assert result["financial"]["per"] == 30.0
    assert len(calls) == 2


# ── 빈 데이터를 성공으로 오보고하지 않는지(회귀) ──────────────────────────────
# 외부 리뷰 지적: 재무/주가 데이터 에이전트가 예외 없이 "데이터 없음"(None/빈 리스트)을
# 돌려주면 _is_failure_result가 실패로 못 잡아(dict+ok=False 형태만 체크) 도메인 에이전트가
# ok=True로 오보고했다. exec_fallback._is_meaningfully_empty(이미 검증된 빈결과 판별 패턴)를
# 재사용해 None/[]/{} 도 실패로 취급하도록 고친다.
def test_answer_us_question_reports_failure_when_financial_data_is_empty(tmp_path):
    """financial_fn이 예외 없이 None(데이터 없음)을 돌려주면 ok=True로 오보고하면 안 된다."""
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)
    calls = []

    def empty_financial_fn(conn, code, **kwargs):
        calls.append(code)
        return None

    try:
        result = answer_us_question(
            "AAPL PER 알려줘", conn, financial_fn=empty_financial_fn,
        )
    finally:
        conn.close()
    assert result["ok"] is False
    assert result["financial"] is None
    assert result["error"]
    assert len(calls) == 2  # 최초 1회 + 재시도 1회(빈 결과도 재시도 대상)


def test_answer_us_question_reports_failure_when_price_data_is_empty(tmp_path):
    """price_fn이 예외 없이 빈 리스트(데이터 없음)를 돌려주면 ok=True로 오보고하면 안 된다."""
    db = _seed_us_db(tmp_path)
    conn = connect_readonly(db)
    calls = []

    def empty_price_fn(conn, code, **kwargs):
        calls.append(code)
        return []

    try:
        result = answer_us_question(
            "AAPL 주가 알려줘", conn, price_fn=empty_price_fn,
        )
    finally:
        conn.close()
    assert result["ok"] is False
    assert result["price"] is None
    assert result["error"]
    assert len(calls) == 2


# ── security_type 필터 (증권종류 — 워런트/ADR 등 제외, HA15 후속(B)) ──────────────────
# 배경: 실서버 curl 재현 — 나스닥 저PER 스크리닝 상위권에 SKHYV(ADR)/RNWWW·BNCWW(Warrant)
# 같은 파생·특수 증권이 섞여 나왔다. us_company.name에 이미 "…Warrant"/"…American
# Depositary Shares" 처럼 판별 정보가 있으므로, us_company.security_type(배치 스크립트가
# LLM으로 미리 분류해 캐싱)을 1차로 쓰고, 아직 분류 안 된(NULL) 종목만 이름 키워드
# 안전망으로 최소한 걸러낸다("판단은 AI, 실행은 캐싱" 원칙, route_question/classify_intent/
# is_screening_question과 동일한 "AI 판단 우선, 실패시 규칙 폴백" 패턴).

def test_is_common_stock_by_name_fallback_true_for_plain_company_names():
    assert _is_common_stock_by_name_fallback("Apple Inc.")
    assert _is_common_stock_by_name_fallback("SK hynix Inc.")


def test_is_common_stock_by_name_fallback_false_for_known_special_security_suffixes():
    assert not _is_common_stock_by_name_fallback("CEA Industries Inc. Warrant")
    assert not _is_common_stock_by_name_fallback("ReNew Energy Global plc Warrant")
    assert not _is_common_stock_by_name_fallback(
        "Cresud S.A.C.I.F. y A. American Depositary Shares"
    )
    assert not _is_common_stock_by_name_fallback("GRAVITY Co. Ltd. American Depository Shares")
    assert not _is_common_stock_by_name_fallback(
        "SK hynix Inc. American Depositary Shares When Issued"
    )
    assert not _is_common_stock_by_name_fallback("Some Corp Preferred Stock")
    assert not _is_common_stock_by_name_fallback("Some Corp Rights")
    assert not _is_common_stock_by_name_fallback("Some Corp Units")


def test_filter_common_stock_keeps_tagged_common_and_drops_tagged_special():
    rows = [
        {"stock_code": "AAPL", "name": "Apple Inc.", "security_type": "common"},
        {"stock_code": "BNCWW", "name": "CEA Industries Inc. Warrant", "security_type": "warrant"},
        {"stock_code": "CRESY", "name": "Cresud ADR", "security_type": "adr"},
        {"stock_code": "PFD", "name": "Some Corp Preferred", "security_type": "preferred"},
    ]
    out = _filter_common_stock(rows)
    assert [r["stock_code"] for r in out] == ["AAPL"]


def test_filter_common_stock_falls_back_to_name_keyword_when_untagged():
    rows = [
        {"stock_code": "AAPL", "name": "Apple Inc.", "security_type": None},  # 미분류 + 정상 이름 → 유지
        {"stock_code": "BNCWW", "name": "CEA Industries Inc. Warrant", "security_type": None},  # 미분류 + 워런트 이름 → 제외
    ]
    out = _filter_common_stock(rows)
    assert [r["stock_code"] for r in out] == ["AAPL"]


def test_filter_common_stock_missing_security_type_key_treated_as_untagged():
    """security_type 키 자체가 없는(구버전 fixture 등) rows 도 안전하게 처리한다(KeyError 없음)."""
    rows = [{"stock_code": "NAS1", "name": "나스닥가", "market": "NASDAQ", "per": 5.0}]
    out = _filter_common_stock(rows)
    assert [r["stock_code"] for r in out] == ["NAS1"]  # 키워드 없는 정상 이름 → 유지(회귀 없음)


def _seed_us_db_multi(tmp_path, companies: list[dict]) -> str:
    """여러 종목을 metrics_at_us 가 실제로 인식할 수 있는 완전한 데이터로 시딩한다.

    companies: [{"code","name","security_type"(optional),"market_cap","net_income"}, ...].
    _seed_us_db(AAPL 단일종목, 기존 다수 테스트가 의존)와 완전히 별개 함수 — 기존 테스트에
    영향 없이 security_type 필터를 실제 DB(get_cross_section→metrics_at_us) 경로로
    end-to-end 검증하기 위한 전용 헬퍼.
    """
    db = tmp_path / "domain_us_multi.db"
    init_db(str(db))
    conn = connect(str(db))
    try:
        for co in companies:
            code = co["code"]
            ni = co.get("net_income", 2.5e9)
            conn.execute(
                "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, "
                "security_type, updated_at) VALUES (?,?,?,?,?,?,?)",
                (code, co["name"], "NASDAQ", "Technology", co.get("market_cap", 3.0e11),
                 co.get("security_type"), "2026-07-01"),
            )
            conn.execute(
                "INSERT INTO us_prices(stock_code, date, open, high, low, close, volume) "
                "VALUES (?,?,?,?,?,?,?)",
                (code, "2026-07-11", 96.5, 99.0, 95.5, 98.2, 4.2e7),
            )
            for statement_type, item_key, value in [
                ("income_stmt", "Total Revenue", 1.0e10),
                ("income_stmt", "Operating Income", 3.0e9),
                ("income_stmt", "Net Income", ni),
                ("balance_sheet", "Stockholders Equity", 8.0e9),
            ]:
                conn.execute(
                    "INSERT INTO us_financials"
                    "(stock_code, as_of_date, period_type, statement_type, item_key, item_value, "
                    "disclosed_date, source, collected_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (code, "2026-06-30", "quarterly", statement_type, item_key, value,
                     "2026-07-12", "yfinance", "2026-07-12T00:00:00"),
                )
            for as_of_date, mult in [("2026-03-31", 0.9), ("2025-12-31", 0.8), ("2025-09-30", 0.7)]:
                conn.execute(
                    "INSERT INTO us_financials"
                    "(stock_code, as_of_date, period_type, statement_type, item_key, item_value, "
                    "disclosed_date, source, collected_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (code, as_of_date, "quarterly", "income_stmt", "Net Income", ni * mult,
                     as_of_date, "yfinance", "2026-07-12T00:00:00"),
                )
        conn.commit()
    finally:
        conn.close()
    return str(db)


def test_answer_us_screening_end_to_end_excludes_warrant_and_adr_via_real_db(tmp_path):
    """실제 metrics_at_us(→get_cross_section) 경로로 워런트/ADR가 스크리닝 후보에서
    제외되고, top_n 이 정상 종목(보통주)만으로 채워짐을 확인한다(fixture 가 아니라 진짜
    DB 경로 — 필터가 top_n 선정 '이전'에 적용됨을 함께 증명: WRNT 는 시총을 비정상적으로
    작게 시딩해 필터가 없으면 PER 최상위(가장 낮음)를 차지했을 것이다)."""
    companies = [
        {"code": "COM1", "name": "Common One Inc.", "security_type": "common",
         "market_cap": 3.0e11, "net_income": 5.0e9},
        {"code": "WRNT", "name": "Warrant Corp Warrant", "security_type": "warrant",
         "market_cap": 1.0e6, "net_income": 5.0e9},  # 시총 왜곡으로 PER 극단적으로 낮아짐(실사용 재현 버그 패턴)
        {"code": "COM2", "name": "Common Two Inc.", "security_type": "common",
         "market_cap": 3.5e11, "net_income": 4.0e9},
        {"code": "ADRX", "name": "Foreign Co American Depositary Shares", "security_type": None,
         "market_cap": 2.0e6, "net_income": 5.0e9},  # 미분류(NULL) + 이름 키워드로 폴백 제외돼야 함
    ]
    db = _seed_us_db_multi(tmp_path, companies)
    conn = connect_readonly(db)
    try:
        result = answer_us_screening(
            "PER 낮은 2종목", conn,
            llm_fn=lambda p: '{"criteria":[{"key":"per","direction":"low"}],"top_n":2}',
            asof="2026-07-14",
        )
    finally:
        conn.close()
    assert result["errors"] == []
    codes = {r["stock_code"] for r in result["result"]}
    assert codes == {"COM1", "COM2"}  # 워런트/ADR 제외, top_n=2 정확히 정상 종목으로 채워짐
    assert len(result["result"]) == 2


def test_answer_us_screening_common_stock_unaffected_regression(tmp_path):
    """일반 보통주만 있는 경우(security_type='common' 또는 미분류+정상 이름) 필터 도입 전과
    동일하게 전부 후보에 남는다(회귀 없음)."""
    companies = [
        {"code": "COM1", "name": "Common One Inc.", "security_type": "common", "net_income": 5.0e9},
        {"code": "COM2", "name": "Common Two Inc.", "security_type": None, "net_income": 4.0e9},
    ]
    db = _seed_us_db_multi(tmp_path, companies)
    conn = connect_readonly(db)
    try:
        result = answer_us_screening(
            "PER 낮은 5종목", conn,
            llm_fn=lambda p: '{"criteria":[{"key":"per","direction":"low"}],"top_n":5}',
            asof="2026-07-14",
        )
    finally:
        conn.close()
    assert result["errors"] == []
    assert {r["stock_code"] for r in result["result"]} == {"COM1", "COM2"}


# ── data_asof: 기간 미지정 시 실제 사용된 데이터 시점 라벨링(KR과 대칭) ──────────────
def test_answer_us_question_unspecified_period_labels_data_asof(tmp_path):
    """기간 미지정 미국 질문(PER+주가)은 실제 재무 기준시점/종가일을 data_asof에 담는다."""
    db = _seed_us_db(tmp_path)  # us_financials as_of_date=2026-06-30, us_prices date=2026-07-11
    conn = connect_readonly(db)
    try:
        result = answer_us_question("AAPL PER랑 주가 알려줘", conn)
    finally:
        conn.close()

    assert result["data_asof"]["financial_quarter"] == "2026-06-30"
    assert result["data_asof"]["price_date"] == "2026-07-11"


def test_answer_us_question_financial_only_data_asof_has_only_quarter(tmp_path):
    """기간 미지정 재무 단독(PER)은 재무 기준시점만 담는다(주가 스냅샷 없어 가격 기준일 없음)."""
    db = _seed_us_db(tmp_path)
    price_calls = []

    def fake_price_fn(conn, code, **kwargs):
        price_calls.append(code)
        return []

    conn = connect_readonly(db)
    try:
        result = answer_us_question("AAPL PER 알려줘", conn, price_fn=fake_price_fn)
    finally:
        conn.close()

    assert price_calls == []  # 재무 단독 질문 → 주가 에이전트 미호출
    assert result["data_asof"]["financial_quarter"] == "2026-06-30"
    assert "price_date" not in result["data_asof"]
