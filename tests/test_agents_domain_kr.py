"""한국주식 도메인 에이전트(src/agents/domain_kr.py) 단위/통합 테스트 (TDD, HA-6).

검증 대상:
- 질문에서 종목명/종목코드를 인식해 stock_code로 변환한다(company 테이블, execute_sql 경유).
- 질문을 보고 재무데이터(HA-2 resolve_metric)/주가·기술지표(HA-3 get_price_snapshot_kr)
  중 무엇이 필요한지 판단해 위임한다.
- "가까운 계층 재시도": 하위 데이터 에이전트 호출이 실패(예외)하면 이 도메인 에이전트가
  즉시 1회 재시도하고, 그래도 실패하면 예외를 전파하지 않고 실패 사유를 반환값에 담는다
  (상위 총괄 에이전트, HA-10까지 예외가 뚫고 올라가지 않게).
"""
from __future__ import annotations

import pytest

from src.agents.domain_kr import (
    _parse_period,
    _strip_retry_feedback,
    answer_kr_question,
    classify_intent,
    find_stock_code,
    find_stock_codes,
    resolve_computed_metric,
)
from src.db import connect, connect_readonly, init_db


def _seed(tmp_path, name: str = "삼성전자", code: str = "005930") -> str:
    """company + metrics(per) + prices 최소 데이터를 시드한 임시 DB 경로를 반환."""
    db = tmp_path / "domain_kr.db"
    init_db(str(db))
    conn = connect(str(db))
    try:
        conn.execute(
            "INSERT INTO company(stock_code, name, market, sector) VALUES(?,?,?,?)",
            (code, name, "KOSPI", "반도체"),
        )
        conn.execute(
            "INSERT INTO metrics(stock_code, quarter, price_date, market_cap, per) "
            "VALUES(?,?,?,?,?)",
            (code, "2025Q1", "2026-07-11", 4.1e14, 12.5),
        )
        conn.execute(
            "INSERT INTO prices(stock_code, date, close, market_cap, open, high, low, volume) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (code, "2026-07-11", 71000.0, 4.1e14, 70000.0, 71500.0, 69800.0, 1.2e7),
        )
        conn.commit()
    finally:
        conn.close()
    return str(db)


def _seed_two_companies(tmp_path) -> str:
    """"삼성전자와 SK하이닉스 종가/PER 알려줘" 같은 다중종목 질문 재현용 2개사 시드."""
    db = tmp_path / "domain_kr_multi.db"
    init_db(str(db))
    conn = connect(str(db))
    try:
        conn.execute(
            "INSERT INTO company(stock_code, name, market, sector) VALUES(?,?,?,?)",
            ("005930", "삼성전자", "KOSPI", "전기·전자"),
        )
        conn.execute(
            "INSERT INTO company(stock_code, name, market, sector) VALUES(?,?,?,?)",
            ("000660", "SK하이닉스", "KOSPI", "반도체"),
        )
        conn.execute(
            "INSERT INTO metrics(stock_code, quarter, price_date, market_cap, per) "
            "VALUES(?,?,?,?,?)",
            ("005930", "2025Q1", "2026-07-15", 4.1e14, 12.5),
        )
        conn.execute(
            "INSERT INTO metrics(stock_code, quarter, price_date, market_cap, per) "
            "VALUES(?,?,?,?,?)",
            ("000660", "2025Q1", "2026-07-15", 1.5e14, 20.0),
        )
        conn.execute(
            "INSERT INTO prices(stock_code, date, close, market_cap, open, high, low, volume) "
            "VALUES(?,?,?,?,?,?,?,?)",
            ("005930", "2026-07-15", 71000.0, 4.1e14, 70000.0, 71500.0, 69800.0, 1.2e7),
        )
        conn.execute(
            "INSERT INTO prices(stock_code, date, close, market_cap, open, high, low, volume) "
            "VALUES(?,?,?,?,?,?,?,?)",
            ("000660", "2026-07-15", 210000.0, 1.5e14, 205000.0, 212000.0, 204000.0, 3.0e6),
        )
        conn.commit()
    finally:
        conn.close()
    return str(db)


def _seed_with_price_history(tmp_path, name: str = "삼성전자", code: str = "005930") -> str:
    """company + financials(유효분기) + 12개월 간격 prices 2건을 시드한 DB 경로를 반환.

    return_12m(metrics_at, tests/test_return_12m.py와 동일 시드 패턴)이 값을 계산하려면
    effective_quarter_at이 유효분기를 찾을 financials 행과, asof·asof 1년전 종가가 모두
    필요하다 — 기존 _seed()는 metrics 테이블만 채워 financials가 없으므로 재사용 불가.
    """
    db = tmp_path / "domain_kr_r12m.db"
    init_db(str(db))
    conn = connect(str(db))
    try:
        conn.execute(
            "INSERT INTO company(stock_code, name, market, sector) VALUES(?,?,?,?)",
            (code, name, "KOSPI", "반도체"),
        )
        conn.execute(
            "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
            "VALUES(?,?,?,?,?)",
            (code, "2025Q1", "2025-05-15", "net_income", 1_000.0),
        )
        conn.execute(
            "INSERT INTO prices(stock_code, date, close, market_cap) VALUES(?,?,?,?)",
            (code, "2025-07-11", 60000.0, 3.5e14),  # 12개월 전 종가
        )
        conn.execute(
            "INSERT INTO prices(stock_code, date, close, market_cap) VALUES(?,?,?,?)",
            (code, "2026-07-11", 72000.0, 4.1e14),  # 기준시점(최신 거래일) 종가
        )
        conn.commit()
    finally:
        conn.close()
    return str(db)


# ── find_stock_code: 종목명/종목코드 인식 ────────────────────────────────────

def test_find_stock_code_resolves_company_name_embedded_in_question(tmp_path):
    db = _seed(tmp_path)
    conn = connect_readonly(db)
    try:
        code = find_stock_code(conn, "삼성전자 PER 알려줘")
    finally:
        conn.close()
    assert code == "005930"


def test_find_stock_code_resolves_explicit_six_digit_code_in_question(tmp_path):
    db = _seed(tmp_path)
    conn = connect_readonly(db)
    try:
        code = find_stock_code(conn, "005930 주가 알려줘")
    finally:
        conn.close()
    assert code == "005930"


def test_find_stock_code_returns_none_when_no_match(tmp_path):
    db = _seed(tmp_path)
    conn = connect_readonly(db)
    try:
        code = find_stock_code(conn, "존재하지않는회사 실적 알려줘")
    finally:
        conn.close()
    assert code is None


def test_find_stock_code_prefers_longer_more_specific_name_match(tmp_path):
    """"SK" 와 "SK하이닉스" 둘 다 질문에 부분 포함될 수 있으면 더 구체적인(긴) 이름 우선."""
    db = tmp_path / "multi.db"
    init_db(str(db))
    conn = connect(str(db))
    try:
        conn.execute(
            "INSERT INTO company(stock_code, name, market, sector) VALUES(?,?,?,?)",
            ("034730", "SK", "KOSPI", "지주"),
        )
        conn.execute(
            "INSERT INTO company(stock_code, name, market, sector) VALUES(?,?,?,?)",
            ("000660", "SK하이닉스", "KOSPI", "반도체"),
        )
        conn.commit()
    finally:
        conn.close()

    ro_conn = connect_readonly(str(db))
    try:
        code = find_stock_code(ro_conn, "SK하이닉스 주가 알려줘")
    finally:
        ro_conn.close()
    assert code == "000660"


def test_find_stock_code_handles_quote_injection_attempt_safely(tmp_path):
    """execute_sql은 파라미터 바인딩이 없어 질문 텍스트를 SQL 문자열에 직접 끼워 넣는다 —
    작은따옴표가 포함된 입력이 들어와도 SQL이 깨지거나 DB가 손상되지 않아야 한다."""
    db = _seed(tmp_path)
    conn = connect_readonly(db)
    try:
        code = find_stock_code(conn, "삼성전자' OR '1'='1")
    finally:
        conn.close()
    assert code == "005930"  # 회사명 매칭은 정상 동작(인젝션으로 전체 테이블 노출 안 됨)

    verify_conn = connect(db)
    try:
        count = verify_conn.execute("SELECT COUNT(*) FROM company").fetchone()[0]
    finally:
        verify_conn.close()
    assert count == 1  # DB 손상 없음


def test_find_stock_code_goes_through_execute_sql(tmp_path, monkeypatch):
    """execute_sql이 실제로 호출되는지 monkeypatch 스파이로 검증한다."""
    import src.agents.domain_kr as mod

    db = _seed(tmp_path)
    calls: list[tuple] = []
    real_execute_sql = mod.execute_sql

    def spy_execute_sql(sql, conn, *args, **kwargs):
        calls.append((sql, conn))
        return real_execute_sql(sql, conn, *args, **kwargs)

    monkeypatch.setattr(mod, "execute_sql", spy_execute_sql)

    conn = connect_readonly(db)
    try:
        code = find_stock_code(conn, "삼성전자 PER 알려줘")
    finally:
        conn.close()

    assert len(calls) == 1
    assert "company" in calls[0][0]
    assert code == "005930"


# ── classify_intent: 재무/주가/둘다 판단 ─────────────────────────────────────

def test_classify_intent_detects_financial_only():
    assert classify_intent("삼성전자 PER 알려줘") == "financial"
    assert classify_intent("삼성전자 매출액 얼마야") == "financial"


def test_classify_intent_detects_price_only():
    assert classify_intent("삼성전자 주가 알려줘") == "price"
    assert classify_intent("삼성전자 이동평균 알려줘") == "price"


def test_classify_intent_detects_both():
    assert classify_intent("삼성전자 PER이랑 주가 같이 알려줘") == "both"


def test_classify_intent_llm_first_used_when_available():
    """LLM 우선: llm_fn 이 주어지면 먼저 LLM 에게 물어보고 그 명확한 판단을 채택한다
    (질문 텍스트는 판단용 프롬프트에 감싸져 전달된다 — route_question 과 동일 패턴)."""
    calls: list[str] = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return "price"

    result = classify_intent("삼성전자 어때?", llm_fn=fake_llm)
    assert len(calls) == 1
    assert "삼성전자 어때?" in calls[0]  # 원 질문이 프롬프트에 포함됨
    assert result == "price"


def test_classify_intent_unclear_without_llm_fn_falls_back_to_both():
    assert classify_intent("삼성전자 어때?") == "both"


def test_classify_intent_llm_first_overrides_keyword_heuristic():
    """LLM 우선순위 전환: 재무 키워드(PER)가 있어도 llm_fn 이 있으면 LLM 판단을 먼저 채택한다
    (기존 '키워드 우선, LLM 최후 폴백'에서 순서가 뒤집힘)."""
    calls: list[str] = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return "both"

    result = classify_intent("삼성전자 PER 알려줘", llm_fn=fake_llm)
    assert len(calls) == 1  # LLM 이 먼저 호출됨(키워드로 단락하지 않음)
    assert result == "both"  # LLM 판단(both)이 키워드(financial)를 이김


def test_classify_intent_falls_back_to_keyword_when_llm_unparseable():
    """llm_fn 응답에서 financial/price/both 를 못 뽑으면 키워드 휴리스틱으로 폴백한다."""
    assert classify_intent("삼성전자 주가 알려줘", llm_fn=lambda p: "???") == "price"


def test_classify_intent_llm_exception_falls_back_to_keyword():
    """llm_fn 이 예외를 던져도 키워드 휴리스틱으로 안전 폴백한다(예외 전파 없음)."""

    def boom(prompt: str) -> str:
        raise RuntimeError("LLM 다운")

    assert classify_intent("삼성전자 주가 알려줘", llm_fn=boom) == "price"


# ── answer_kr_question: 통합 위임 ────────────────────────────────────────────

def test_answer_kr_question_routes_financial_question_to_resolve_metric(tmp_path, monkeypatch):
    """"삼성전자 PER" 같은 질문 → 재무데이터 에이전트(resolve_metric)가 호출됨을 스파이로 확인."""
    import src.agents.domain_kr as mod

    db = _seed(tmp_path)
    calls: list[tuple] = []
    real_resolve_metric = mod.resolve_metric

    def spy_resolve_metric(conn, stock_code, metric, llm_fn=None):
        calls.append((stock_code, metric))
        return real_resolve_metric(conn, stock_code, metric, llm_fn=llm_fn)

    monkeypatch.setattr(mod, "resolve_metric", spy_resolve_metric)

    conn = connect_readonly(db)
    try:
        result = answer_kr_question("삼성전자 PER 알려줘", conn)
    finally:
        conn.close()

    assert calls == [("005930", "per")]
    assert result["stock_code"] == "005930"
    assert result["intent"] == "financial"
    assert result["financial"]["value"] == 12.5
    assert result["financial"]["source"] == "DART"
    assert result["price"] is None
    assert result["errors"] == []


def test_answer_kr_question_routes_price_question_to_price_snapshot(tmp_path, monkeypatch):
    import src.agents.domain_kr as mod

    db = _seed(tmp_path)
    calls: list[tuple] = []
    real_snapshot = mod.get_price_snapshot_kr

    def spy_snapshot(conn, stock_codes, **kwargs):
        calls.append((stock_codes,))
        return real_snapshot(conn, stock_codes, **kwargs)

    monkeypatch.setattr(mod, "get_price_snapshot_kr", spy_snapshot)

    conn = connect_readonly(db)
    try:
        result = answer_kr_question("삼성전자 주가 알려줘", conn)
    finally:
        conn.close()

    assert calls == [("005930",)]
    assert result["stock_code"] == "005930"
    assert result["intent"] == "price"
    assert result["price"][0]["close"] == 71000.0
    assert result["financial"] is None
    assert result["errors"] == []


def test_answer_kr_question_both_intent_calls_both_data_agents(tmp_path, monkeypatch):
    import src.agents.domain_kr as mod

    db = _seed(tmp_path)
    fin_calls: list[tuple] = []
    price_calls: list[tuple] = []
    real_resolve_metric = mod.resolve_metric
    real_snapshot = mod.get_price_snapshot_kr

    def spy_resolve_metric(conn, stock_code, metric, llm_fn=None):
        fin_calls.append((stock_code, metric))
        return real_resolve_metric(conn, stock_code, metric, llm_fn=llm_fn)

    def spy_snapshot(conn, stock_codes, **kwargs):
        price_calls.append((stock_codes,))
        return real_snapshot(conn, stock_codes, **kwargs)

    monkeypatch.setattr(mod, "resolve_metric", spy_resolve_metric)
    monkeypatch.setattr(mod, "get_price_snapshot_kr", spy_snapshot)

    conn = connect_readonly(db)
    try:
        result = answer_kr_question("삼성전자 PER이랑 주가 같이 알려줘", conn)
    finally:
        conn.close()

    assert fin_calls == [("005930", "per")]
    assert price_calls == [("005930",)]
    assert result["intent"] == "both"
    assert result["financial"]["value"] == 12.5
    assert result["price"][0]["close"] == 71000.0


def test_answer_kr_question_unknown_company_reports_error_without_calling_data_agents(
    tmp_path, monkeypatch
):
    import src.agents.domain_kr as mod

    db = _seed(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        mod, "resolve_metric", lambda *a, **k: calls.append("financial") or {}
    )
    monkeypatch.setattr(
        mod, "get_price_snapshot_kr", lambda *a, **k: calls.append("price") or []
    )

    conn = connect_readonly(db)
    try:
        result = answer_kr_question("존재하지않는회사 PER 알려줘", conn)
    finally:
        conn.close()

    assert result["stock_code"] is None
    assert result["errors"]  # 실패 사유가 담김
    assert calls == []  # 하위 데이터 에이전트는 호출되지 않음


# ── "가까운 계층 재시도" ──────────────────────────────────────────────────────

def test_answer_kr_question_retries_once_on_financial_agent_failure_then_succeeds(tmp_path):
    db = _seed(tmp_path)
    attempts: list[int] = []

    def flaky_resolve_metric(conn, stock_code, metric, llm_fn=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise SyntaxError("생성된 코드 문법 오류(시뮬레이션)")
        return {"stock_code": stock_code, "metric": metric, "value": 99.0, "source": "DART"}

    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "삼성전자 PER 알려줘", conn, resolve_metric_fn=flaky_resolve_metric
        )
    finally:
        conn.close()

    assert len(attempts) == 2  # 1회 재시도
    assert result["financial"]["value"] == 99.0
    assert result["errors"] == []


def test_answer_kr_question_retries_once_then_reports_failure_without_raising(tmp_path):
    """항상 실패하도록 주입된 데이터 에이전트 → 1회 재시도 후에도 실패하면 예외 없이
    실패 사유가 반환값에 담긴다(총괄 에이전트까지 예외가 뚫고 올라가지 않음)."""
    db = _seed(tmp_path)
    attempts: list[int] = []

    def always_fails(conn, stock_code, metric, llm_fn=None):
        attempts.append(1)
        raise RuntimeError("항상 실패(시뮬레이션)")

    conn = connect_readonly(db)
    try:
        result = answer_kr_question(  # 예외가 여기서 전파되면 테스트 자체가 실패함
            "삼성전자 PER 알려줘", conn, resolve_metric_fn=always_fails
        )
    finally:
        conn.close()

    assert len(attempts) == 2  # 최초 시도 + 재시도 1회 = 총 2회, 그 이상 재시도하지 않음
    assert result["financial"] is None
    assert any("항상 실패" in e or "RuntimeError" in e for e in result["errors"])


# ── 계산전용 지표(return_12m 등) 단일종목 배선 (HA-6 실사용 버그 복구) ──────────────
# 배경: "삼성전자 직전 12개월 수익률" 같은 단일종목 질문이 classify_intent에서 결정론
# 키워드 매치 없이 LLM 폴백으로 "price"로 잘못 분류되고, price 분기(get_price_snapshot_kr)는
# return_12m을 계산하지 않아 uncertain으로 끝나던 실서버 재현 버그.

def test_classify_intent_detects_return_12m_keyword_as_financial():
    assert classify_intent("삼성전자 직전 12개월 수익률") == "financial"
    assert classify_intent("삼성전자 모멘텀") == "financial"
    assert classify_intent("삼성전자 가격수익률 알려줘") in ("financial", "both")


def test_classify_intent_return_12m_keyword_protected_by_fallback_when_llm_unclear():
    """LLM 우선으로 바뀌었어도, LLM 이 명확한 답을 못 주면(파싱 실패) 키워드 휴리스틱 안전망으로
    폴백한다 — '직전 12개월 수익률' 은 폴백에서 financial 로 보호된다(return_12m 회귀 방지)."""

    def unclear_llm(prompt: str) -> str:
        return "잘 모르겠습니다"  # financial/price/both 토큰 없음 → 파싱 실패 → 키워드 폴백

    result = classify_intent("삼성전자 직전 12개월 수익률", llm_fn=unclear_llm)
    assert result == "financial"


def test_classify_intent_detects_other_computed_only_metric_keywords_as_financial():
    """return_12m 외에 metrics_at()이 계산하는 다른 계산전용 지표(ROA/PSR)도 일관되게 인식."""
    assert classify_intent("삼성전자 ROA 알려줘") == "financial"
    assert classify_intent("삼성전자 PSR 알려줘") == "financial"


def test_classify_intent_still_detects_per_as_financial_unaffected_by_computed_change():
    """회귀 방지: PER처럼 이미 resolve_metric으로 동작하던 재무지표 분류는 그대로."""
    assert classify_intent("삼성전자 PER 알려줘") == "financial"


def test_answer_kr_question_routes_return_12m_question_to_computed_metric(tmp_path):
    db = _seed_with_price_history(tmp_path)
    conn = connect_readonly(db)
    try:
        result = answer_kr_question("삼성전자 직전 12개월 수익률", conn)
    finally:
        conn.close()

    assert result["stock_code"] == "005930"
    assert result["intent"] == "financial"
    assert result["financial"] is not None
    assert result["financial"]["metric"] == "return_12m"
    # (72000-60000)/60000*100 = 20.0
    assert result["financial"]["value"] == pytest.approx(20.0)
    assert result["financial"]["source"] == "computed"
    assert result["price"] is None
    assert result["errors"] == []


def test_answer_kr_question_momentum_keyword_also_resolves_return_12m(tmp_path):
    db = _seed_with_price_history(tmp_path)
    conn = connect_readonly(db)
    try:
        result = answer_kr_question("삼성전자 모멘텀 알려줘", conn)
    finally:
        conn.close()

    assert result["financial"]["metric"] == "return_12m"
    assert result["financial"]["value"] == pytest.approx(20.0)
    assert result["errors"] == []


def test_answer_kr_question_computed_metric_fn_is_injectable(tmp_path):
    """computed_metric_fn이 (stock_code, metric)으로 실제 호출되는지 스파이로 확인."""
    db = _seed_with_price_history(tmp_path)
    calls: list[tuple] = []

    def fake_computed(conn, stock_code, metric, **kwargs):
        calls.append((stock_code, metric))
        return {"stock_code": stock_code, "metric": metric, "value": 42.0, "source": "computed", "period": "x"}

    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "삼성전자 모멘텀 알려줘", conn, computed_metric_fn=fake_computed
        )
    finally:
        conn.close()

    assert calls == [("005930", "return_12m")]
    assert result["financial"]["value"] == 42.0


def test_answer_kr_question_revenue_growth_keyword_prefers_computed_metric_over_shorter_financial_alias(
    tmp_path,
):
    """"매출성장"(계산지표, revenue_growth)이 "매출"(재무지표, revenue)의 상위 문자열이라
    구체적인 계산지표 별칭이 먼저 매치돼야 한다 — computed_metric_fn이 호출되고 resolve_metric_fn
    (financial, revenue)은 호출되지 않아야 한다."""
    db = _seed_with_price_history(tmp_path)
    financial_calls: list[tuple] = []
    computed_calls: list[tuple] = []

    def spy_resolve_metric(conn, stock_code, metric, llm_fn=None):
        financial_calls.append((stock_code, metric))
        return {"stock_code": stock_code, "metric": metric, "value": -1.0, "source": "DART", "period": None}

    def fake_computed(conn, stock_code, metric, **kwargs):
        computed_calls.append((stock_code, metric))
        return {"stock_code": stock_code, "metric": metric, "value": 5.0, "source": "computed", "period": "x"}

    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "삼성전자 매출성장 알려줘",
            conn,
            resolve_metric_fn=spy_resolve_metric,
            computed_metric_fn=fake_computed,
        )
    finally:
        conn.close()

    assert computed_calls == [("005930", "revenue_growth")]
    assert financial_calls == []
    assert result["financial"]["value"] == 5.0


def test_answer_kr_question_per_question_still_routes_to_resolve_metric_unaffected(tmp_path, monkeypatch):
    """회귀 방지: 계산전용 지표 배선 추가 후에도 PER은 여전히 resolve_metric_fn(재무 경로)로 간다."""
    import src.agents.domain_kr as mod

    db = _seed_with_price_history(tmp_path)
    calls: list[tuple] = []
    real_resolve_metric = mod.resolve_metric

    def spy_resolve_metric(conn, stock_code, metric, llm_fn=None):
        calls.append((stock_code, metric))
        return real_resolve_metric(conn, stock_code, metric, llm_fn=llm_fn)

    monkeypatch.setattr(mod, "resolve_metric", spy_resolve_metric)

    conn = connect_readonly(db)
    try:
        answer_kr_question("삼성전자 PER 알려줘", conn)
    finally:
        conn.close()

    assert calls == [("005930", "per")]


def test_answer_kr_question_no_retry_when_price_agent_succeeds_first_try(tmp_path):
    db = _seed(tmp_path)
    attempts: list[int] = []

    def succeeds_first_try(conn, stock_codes, **kwargs):
        attempts.append(1)
        return [{"stock_code": stock_codes, "close": 1000.0}]

    conn = connect_readonly(db)
    try:
        answer_kr_question(
            "삼성전자 주가 알려줘", conn, price_snapshot_fn=succeeds_first_try
        )
    finally:
        conn.close()

    assert len(attempts) == 1  # 성공 시 불필요한 재시도 없음


# ── 기간 파싱(_parse_period) — 버그B: "25년 전체 영업이익"이 최신분기만 반환 ──────────

def test_parse_period_two_digit_year_only_is_annual():
    assert _parse_period("삼성전자 25년 전체 영업이익") == {"kind": "annual", "year": 2025}


def test_parse_period_four_digit_year_only_is_annual():
    assert _parse_period("삼성전자 2025년 영업이익") == {"kind": "annual", "year": 2025}


def test_parse_period_year_and_quarter_is_specific_quarter():
    assert _parse_period("삼성전자 26년 1분기 영업이익") == {"kind": "quarter", "quarter": "2026Q1"}
    assert _parse_period("삼성전자 2025년 3분기 매출") == {"kind": "quarter", "quarter": "2025Q3"}


def test_parse_period_none_when_no_period_mentioned():
    assert _parse_period("삼성전자 영업이익 알려줘") is None
    assert _parse_period("삼성전자 PER 알려줘") is None


def test_parse_period_does_not_false_match_stock_code_or_large_number():
    # 6자리 종목코드/큰 숫자를 연도로 오인하지 않는다.
    assert _parse_period("005930 영업이익") is None
    assert _parse_period("삼성전자 시가총액 200000") is None


def test_parse_period_quarter_only_without_year_is_none():
    # 연도 없는 분기 단독은 현행(최신) 유지 — None.
    assert _parse_period("삼성전자 1분기 영업이익") is None


def test_strip_retry_feedback_removes_appended_feedback():
    original = "삼성전자 25년 전체 영업이익"
    decorated = (
        f"{original}\n\n[이전 시도 실패 피드백] 직전 시도가 다음 이유로 검증에 실패했습니다: "
        "도메인 결과는 2026Q1 분기 영업이익만 제공해 기간이 맞지 않습니다.\n같은 방식을 반복하지 마세요."
    )
    assert _strip_retry_feedback(decorated) == original
    # 피드백 제거 후 파싱하면 피드백의 '2026Q1'/'1분기'에 오염되지 않고 연간으로 남는다.
    assert _parse_period(_strip_retry_feedback(decorated)) == {"kind": "annual", "year": 2025}


def test_strip_retry_feedback_noop_when_no_feedback():
    assert _strip_retry_feedback("삼성전자 영업이익") == "삼성전자 영업이익"


# ── period 배선: answer_kr_question이 파싱한 기간을 resolve_metric_fn에 전달 ──────────

def test_answer_kr_question_threads_annual_period_into_resolve_metric(tmp_path):
    db = _seed(tmp_path)
    seen: dict = {}

    def spy_resolve_metric(conn, stock_code, metric, llm_fn=None, period=None):
        seen["period"] = period
        return {"stock_code": stock_code, "metric": metric,
                "value": 42.0, "source": "DART", "period": "2025 연간"}

    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "삼성전자 25년 전체 영업이익", conn, resolve_metric_fn=spy_resolve_metric
        )
    finally:
        conn.close()

    assert seen["period"] == {"kind": "annual", "year": 2025}
    assert result["financial"]["value"] == 42.0


def test_answer_kr_question_annual_period_ignores_retry_feedback(tmp_path):
    # 재시도로 피드백이 붙어도 원본(연간) 기준으로 기간을 전달한다.
    db = _seed(tmp_path)
    seen: dict = {}

    def spy_resolve_metric(conn, stock_code, metric, llm_fn=None, period=None):
        seen["period"] = period
        return {"stock_code": stock_code, "metric": metric,
                "value": 1.0, "source": "DART", "period": "2025 연간"}

    decorated = (
        "삼성전자 25년 전체 영업이익\n\n[이전 시도 실패 피드백] 직전 시도가 다음 이유로 "
        "검증에 실패했습니다: 결과는 2026Q1 분기 영업이익만 제공해 기간이 맞지 않습니다."
    )
    conn = connect_readonly(db)
    try:
        answer_kr_question(decorated, conn, resolve_metric_fn=spy_resolve_metric)
    finally:
        conn.close()

    assert seen["period"] == {"kind": "annual", "year": 2025}


def test_answer_kr_question_no_period_does_not_pass_period_kwarg(tmp_path):
    # 회귀: 기간 없는 질문은 period 인자 없이 resolve_metric_fn을 호출한다
    # (기존 (conn, stock_code, metric, llm_fn) 시그니처 fake가 깨지지 않게).
    db = _seed(tmp_path)

    def spy_resolve_metric(conn, stock_code, metric, llm_fn=None):  # period 파라미터 없음
        return {"stock_code": stock_code, "metric": metric,
                "value": 12.5, "source": "DART", "period": "2025Q1"}

    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "삼성전자 PER 알려줘", conn, resolve_metric_fn=spy_resolve_metric
        )
    finally:
        conn.close()

    assert result["financial"]["value"] == 12.5


# ── price_history 첨부(버그A): "최근 1년 주가 그래프" 질문에 1년 시계열을 담아 검증 통과 ────

def test_answer_kr_question_attaches_price_history_for_chart_question(tmp_path):
    db = _seed(tmp_path)

    def fake_history(conn, code, *a, **k):
        return [
            {"stock_code": code, "date": "2025-07-15", "close": 60000.0},
            {"stock_code": code, "date": "2026-07-14", "close": 72000.0},
        ]

    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "삼성전자 최근 1년 주가 그래프 그려줘", conn, price_history_fn=fake_history
        )
    finally:
        conn.close()

    assert result.get("price_history") is not None
    assert result["price_history"]["count"] == 2
    assert result["price_history"]["start_date"] == "2025-07-15"
    assert result["price_history"]["end_date"] == "2026-07-14"


def test_answer_kr_question_no_price_history_for_plain_price_question(tmp_path):
    db = _seed(tmp_path)
    called: list[int] = []

    def fake_history(conn, code, *a, **k):
        called.append(1)
        return []

    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "삼성전자 주가 알려줘", conn, price_history_fn=fake_history
        )
    finally:
        conn.close()

    assert result.get("price_history") is None
    assert called == []  # 차트/기간 의도가 없으면 히스토리 조회 자체를 하지 않는다


# ── find_stock_codes(복수): 한 질문에 여러 종목이 함께 언급된 경우 ──────────────────
# 실서버 재현 버그: "삼성전자와 SK하이닉스 종가 알려줘"처럼 두 종목을 한 번에 물으면
# find_stock_code(단수)는 하나만 찾아 검증기가 "대상이 안 맞음"으로 계속 재시도만 하다
# 실패했다(재시도해도 단수 함수는 결정론적으로 같은 종목만 반환하므로 절대 안 고쳐짐).

def test_find_stock_codes_finds_multiple_companies_named_in_question(tmp_path):
    db = _seed_two_companies(tmp_path)
    conn = connect_readonly(db)
    try:
        codes = find_stock_codes(conn, "삼성전자와 SK하이닉스 종가 알려줘")
    finally:
        conn.close()
    assert set(codes) == {"005930", "000660"}


def test_find_stock_codes_returns_single_code_for_single_stock_question(tmp_path):
    db = _seed_two_companies(tmp_path)
    conn = connect_readonly(db)
    try:
        codes = find_stock_codes(conn, "삼성전자 PER 알려줘")
    finally:
        conn.close()
    assert codes == ["005930"]


def test_find_stock_codes_returns_empty_list_when_no_match(tmp_path):
    db = _seed_two_companies(tmp_path)
    conn = connect_readonly(db)
    try:
        codes = find_stock_codes(conn, "존재하지않는회사 실적 알려줘")
    finally:
        conn.close()
    assert codes == []


def test_find_stock_codes_dedupes_overlapping_substring_names(tmp_path):
    """"SK"와 "SK하이닉스" 둘 다 부분 포함 매치돼도 더 구체적인(긴) 이름 하나로만 남는다
    (find_stock_code 단수의 우선순위 규칙과 동일 원칙)."""
    db = tmp_path / "overlap.db"
    init_db(str(db))
    conn = connect(str(db))
    try:
        conn.execute(
            "INSERT INTO company(stock_code, name, market, sector) VALUES(?,?,?,?)",
            ("034730", "SK", "KOSPI", "지주"),
        )
        conn.execute(
            "INSERT INTO company(stock_code, name, market, sector) VALUES(?,?,?,?)",
            ("000660", "SK하이닉스", "KOSPI", "반도체"),
        )
        conn.commit()
    finally:
        conn.close()

    ro_conn = connect_readonly(str(db))
    try:
        codes = find_stock_codes(ro_conn, "SK하이닉스 주가 알려줘")
    finally:
        ro_conn.close()
    assert codes == ["000660"]  # "SK"는 겹쳐서 걸러짐


def test_find_stock_codes_includes_explicit_six_digit_codes(tmp_path):
    db = _seed_two_companies(tmp_path)
    conn = connect_readonly(db)
    try:
        codes = find_stock_codes(conn, "005930 000660 종가 비교")
    finally:
        conn.close()
    assert set(codes) == {"005930", "000660"}


# ── answer_kr_question: 다중종목(named multi-entity) 경로 ───────────────────────

def test_answer_kr_question_multi_entity_price_question_returns_entities_for_each_stock(tmp_path):
    db = _seed_two_companies(tmp_path)
    conn = connect_readonly(db)
    try:
        result = answer_kr_question("삼성전자와 SK하이닉스 종가 알려줘", conn)
    finally:
        conn.close()

    assert result["stock_code"] is None  # 단일 종목으로 특정할 수 없음(다중종목 경로)
    assert set(result["stock_codes"]) == {"005930", "000660"}
    assert len(result["entities"]) == 2
    by_code = {e["stock_code"]: e for e in result["entities"]}
    assert by_code["005930"]["price"][0]["close"] == 71000.0
    assert by_code["000660"]["price"][0]["close"] == 210000.0
    assert result["errors"] == []


def test_answer_kr_question_multi_entity_financial_question_returns_entities_for_each_stock(tmp_path):
    db = _seed_two_companies(tmp_path)
    conn = connect_readonly(db)
    try:
        result = answer_kr_question("삼성전자와 SK하이닉스 PER 비교해줘", conn)
    finally:
        conn.close()

    by_code = {e["stock_code"]: e for e in result["entities"]}
    assert by_code["005930"]["financial"]["value"] == 12.5
    assert by_code["000660"]["financial"]["value"] == 20.0


def test_answer_kr_question_multi_entity_calls_data_agent_once_per_stock_code(tmp_path, monkeypatch):
    import src.agents.domain_kr as mod

    db = _seed_two_companies(tmp_path)
    calls: list[str] = []
    real_snapshot = mod.get_price_snapshot_kr

    def spy_snapshot(conn, stock_codes, **kwargs):
        calls.append(stock_codes)
        return real_snapshot(conn, stock_codes, **kwargs)

    monkeypatch.setattr(mod, "get_price_snapshot_kr", spy_snapshot)

    conn = connect_readonly(db)
    try:
        answer_kr_question("삼성전자와 SK하이닉스 종가 알려줘", conn)
    finally:
        conn.close()

    assert set(calls) == {"005930", "000660"}
    assert len(calls) == 2  # 종목 하나당 정확히 1회


def test_answer_kr_question_single_entity_question_has_no_entities_key(tmp_path):
    """회귀 방지: 종목이 하나뿐이면 기존 단일종목 응답 형태 그대로(entities 키 없음)."""
    db = _seed_two_companies(tmp_path)
    conn = connect_readonly(db)
    try:
        result = answer_kr_question("삼성전자 PER 알려줘", conn)
    finally:
        conn.close()

    assert "entities" not in result
    assert result["stock_code"] == "005930"
    assert result["financial"]["value"] == 12.5


# ── resolve_computed_metric: {metric}_estimated 컴패니언 필드 병기 ──────────────
# "삼성전자 투하자본수익률 알려줘" 류 단일종목 질문은 get_cross_section 행 전체가 아니라
# value 하나만 골라내므로, roc_estimated(감가상각비 근사 여부)를 명시적으로 함께 넘기지
# 않으면 근사치라는 사실이 최종 답변에서 조용히 사라진다.
def test_resolve_computed_metric_surfaces_estimated_companion_field():
    fake_execute_sql = lambda sql, conn: {"ok": True, "rows": [{"d": "2026-07-15"}]}
    fake_cross_section = lambda conn, asof: [
        {"stock_code": "005930", "roc": 13.0, "roc_estimated": True},
    ]
    result = resolve_computed_metric(
        None, "005930", "roc",
        execute_sql_fn=fake_execute_sql, cross_section_fn=fake_cross_section,
    )
    assert result["value"] == 13.0
    assert result["estimated"] is True


def test_resolve_computed_metric_estimated_is_none_without_companion_field():
    # return_12m처럼 {metric}_estimated 컴패니언이 아예 없는 지표는 estimated=None.
    fake_execute_sql = lambda sql, conn: {"ok": True, "rows": [{"d": "2026-07-15"}]}
    fake_cross_section = lambda conn, asof: [
        {"stock_code": "005930", "return_12m": 20.0},
    ]
    result = resolve_computed_metric(
        None, "005930", "return_12m",
        execute_sql_fn=fake_execute_sql, cross_section_fn=fake_cross_section,
    )
    assert result["value"] == 20.0
    assert result["estimated"] is None


# ── 실서버 재현 버그: 마법공식(EY/ROC)/GPA는 스크리닝에서만 인식되고 단일종목
#    질문("삼성전자 투하자본수익률")에서는 _extract_metric이 지표명을 못 찾아 매번
#    "재무 지표를 인식하지 못함"으로 결정론적으로 실패했다(_SCREEN_METRIC_ALIASES에
#    roc/earnings_yield/gp_a 한국어 별칭이 등록 안 됨). ──────────────────────────
def test_answer_kr_question_recognizes_roc_single_stock(tmp_path):
    db = _seed_two_companies(tmp_path)
    conn = connect_readonly(db)

    def fake_cross_section(conn, asof):
        return [{"stock_code": "005930", "roc": 13.0, "roc_estimated": True}]

    try:
        result = answer_kr_question(
            "삼성전자 투하자본수익률 알려줘", conn,
            computed_metric_fn=lambda conn, code, metric, **kw: resolve_computed_metric(
                conn, code, metric, cross_section_fn=fake_cross_section,
                execute_sql_fn=lambda sql, c: {"ok": True, "rows": [{"d": "2026-07-15"}]},
            ),
        )
    finally:
        conn.close()

    assert "재무 지표를 인식하지 못함" not in result["errors"]
    assert result["financial"]["value"] == 13.0
    assert result["financial"]["estimated"] is True


def test_answer_kr_question_recognizes_earnings_yield_single_stock(tmp_path):
    db = _seed_two_companies(tmp_path)
    conn = connect_readonly(db)

    def fake_cross_section(conn, asof):
        return [{"stock_code": "005930", "earnings_yield": 8.5}]

    try:
        result = answer_kr_question(
            "삼성전자 이익수익률 알려줘", conn,
            computed_metric_fn=lambda conn, code, metric, **kw: resolve_computed_metric(
                conn, code, metric, cross_section_fn=fake_cross_section,
                execute_sql_fn=lambda sql, c: {"ok": True, "rows": [{"d": "2026-07-15"}]},
            ),
        )
    finally:
        conn.close()

    assert "재무 지표를 인식하지 못함" not in result["errors"]
    assert result["financial"]["value"] == 8.5


# ── 실서버 재현 버그: "삼성전자 25년 투하자본수익률"처럼 연도가 명시된 계산지표
#    질문이 항상 오늘 날짜(최신 거래일) 기준으로만 계산돼 검증에서 결정론적으로
#    탈락했다(질문 연도와 결과 period가 안 맞음). resolve_metric(기존 DART 재무지표)
#    경로는 이미 period를 반영하는데, computed_metric_fn(roc/ey/gp_a/return_12m) 경로는
#    period를 아예 안 넘기고 있었다. ──────────────────────────────────────────

def test_resolve_computed_metric_uses_explicit_asof_when_given():
    fake_execute_sql = lambda sql, conn: {"ok": True, "rows": [{"d": "2026-07-16"}]}
    calls: list = []

    def fake_cross_section(conn, asof):
        calls.append(asof)
        return [{"stock_code": "005930", "roc": 13.0}]

    result = resolve_computed_metric(
        None, "005930", "roc", asof="2025-12-30",
        execute_sql_fn=fake_execute_sql, cross_section_fn=fake_cross_section,
    )
    assert calls == ["2025-12-30"]  # _default_screening_asof(오늘)이 아니라 명시값 사용
    assert result["period"] == "2025-12-30"
    assert result["value"] == 13.0


def test_resolve_computed_metric_falls_back_to_default_asof_when_not_given():
    """asof 생략(기존 호출부 하위호환) — 기존처럼 최신 거래일로 폴백."""
    fake_execute_sql = lambda sql, conn: {"ok": True, "rows": [{"d": "2026-07-16"}]}
    fake_cross_section = lambda conn, asof: [{"stock_code": "005930", "roc": 13.0}]

    result = resolve_computed_metric(
        None, "005930", "roc",
        execute_sql_fn=fake_execute_sql, cross_section_fn=fake_cross_section,
    )
    assert result["period"] == "2026-07-16"


def test_answer_kr_question_passes_explicit_year_to_computed_metric(tmp_path):
    """'삼성전자 25년 투하자본수익률' — 질문의 2025년이 computed_metric_fn의 asof로 전달돼야
    한다(오늘 날짜로 계산해 질문 연도와 안 맞다고 검증 실패하는 결정론적 버그 재현 방지)."""
    db = _seed_two_companies(tmp_path)
    # _seed_two_companies는 2026-07-15 가격만 시드하므로, "25년" 질문이 실제로 2025년
    # 이하 최신 거래일을 찾을 수 있도록 2025-12-30 가격을 추가로 시드한다.
    seed_conn = connect(db)
    seed_conn.execute(
        "INSERT INTO prices(stock_code, date, close, market_cap) VALUES(?,?,?,?)",
        ("005930", "2025-12-30", 65000.0, 3.9e14),
    )
    seed_conn.commit()
    seed_conn.close()
    conn = connect_readonly(db)
    seen_asof: list = []

    def spy_computed_metric_fn(conn, code, metric, asof=None, **kw):
        seen_asof.append(asof)
        return {"stock_code": code, "metric": metric, "value": 22.65,
                "source": "computed", "period": asof, "estimated": True}

    try:
        result = answer_kr_question(
            "삼성전자 25년 투하자본수익률", conn, computed_metric_fn=spy_computed_metric_fn,
        )
    finally:
        conn.close()

    assert seen_asof == ["2025-12-30"]  # 2025년 말일 이하 최신 거래일로 확정돼야 함
    assert result["financial"]["period"] == "2025-12-30"


def test_answer_kr_question_recognizes_gpa_single_stock(tmp_path):
    db = _seed_two_companies(tmp_path)
    conn = connect_readonly(db)

    def fake_cross_section(conn, asof):
        return [{"stock_code": "005930", "gp_a": 21.0}]

    try:
        result = answer_kr_question(
            "삼성전자 GPA 알려줘", conn,
            computed_metric_fn=lambda conn, code, metric, **kw: resolve_computed_metric(
                conn, code, metric, cross_section_fn=fake_cross_section,
                execute_sql_fn=lambda sql, c: {"ok": True, "rows": [{"d": "2026-07-15"}]},
            ),
        )
    finally:
        conn.close()

    assert "재무 지표를 인식하지 못함" not in result["errors"]
    assert result["financial"]["value"] == 21.0
