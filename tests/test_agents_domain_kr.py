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

from datetime import date

import pytest

from src.agents.domain_kr import (
    _COMPUTED_ONLY_FIELDS,
    _extract_indicators,
    _extract_metric,
    _parse_period,
    _parse_periods,
    _parse_price_target_date,
    _parse_recent_return_months,
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


def test_find_stock_code_does_not_treat_price_amount_as_code(tmp_path):
    """회귀(리뷰 지적 (c)): '삼성전자 200000원 돌파했어?'의 '200000'은 주가 금액이지
    종목코드가 아니다 — 6자리 숫자를 무조건 코드로 취급하면 엉뚱한 종목을 조회한다.
    통화 단위 '원'이 붙은 6자리 숫자는 종목코드 후보에서 제외하고 회사명 매칭으로 넘어간다."""
    db = _seed(tmp_path)  # 삼성전자 = 005930
    conn = connect_readonly(db)
    try:
        for q in (
            "삼성전자 200000원 돌파했어?",
            "삼성전자 100000원 넘었어?",
            "삼성전자 주가 500000원 이상이야?",
            "삼성전자 300000원 이하로 떨어졌어?",
        ):
            assert find_stock_code(conn, q) == "005930", q
    finally:
        conn.close()


def test_find_stock_codes_excludes_price_amount_from_candidates(tmp_path):
    """find_stock_codes도 '…원' 금액을 코드 후보에서 빼야 한다 — 안 그러면 가짜 코드가
    섞여 다중종목 경로로 잘못 분기된다(['200000','005930']로 2개 판정 → 다중종목 오분기)."""
    db = _seed(tmp_path)
    conn = connect_readonly(db)
    try:
        codes = find_stock_codes(conn, "삼성전자 200000원 돌파했어?")
    finally:
        conn.close()
    assert codes == ["005930"]


def test_find_stock_code_still_recognizes_genuine_six_digit_code(tmp_path):
    """회귀: 진짜 종목코드(원이 붙지 않은 6자리)는 그대로 인식한다. 코드와 금액이 섞여 있으면
    금액('…원')은 버리고 코드를 고른다."""
    db = _seed(tmp_path)
    conn = connect_readonly(db)
    try:
        assert find_stock_code(conn, "005930 주가 알려줘") == "005930"
        assert find_stock_code(conn, "005930 200000원 돌파?") == "005930"
    finally:
        conn.close()


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


# ── find_stock_code: 그룹명 생략 구어체 종목명(하이닉스=SK하이닉스) 정합성 ──────────────
# 실서버 재현 버그: "하이닉스 24년 영업이익률"을 물으면 SK하이닉스(000660)가 아니라 우연히
# "하이닉스" 안에 들어있는 무관한 짧은 회사 "이닉스"(452400)가 역방향 LIKE로 걸려 조용히
# 틀린 종목 데이터를 정답처럼 반환했다. "SK/LG/…" 그룹명을 생략하고 부르는 한국어 관행
# ("하이닉스"="SK하이닉스", "이노텍"="LG이노텍")을 결정론적으로 보강한다.

def _seed_group_prefix_companies(tmp_path) -> str:
    """그룹명 생략 구어체 매칭 재현용: SK하이닉스 + 우연히 부분문자열인 무관한 이닉스 +
    LG이노텍 + 삼성전자 + 짧은 지주명 SK 를 한 DB 에 시드한다."""
    db = tmp_path / "group_prefix.db"
    init_db(str(db))
    conn = connect(str(db))
    try:
        rows = [
            ("005930", "삼성전자", "KOSPI", "전기·전자"),
            ("000660", "SK하이닉스", "KOSPI", "반도체"),
            ("452400", "이닉스", "KOSDAQ", "자동차부품"),  # "하이닉스"에 우연히 포함되는 무관한 회사
            ("011070", "LG이노텍", "KOSPI", "전기·전자"),
            ("034730", "SK", "KOSPI", "지주"),
        ]
        for r in rows:
            conn.execute(
                "INSERT INTO company(stock_code, name, market, sector) VALUES(?,?,?,?)", r
            )
        conn.commit()
    finally:
        conn.close()
    return str(db)


def test_find_stock_code_resolves_group_prefix_omitted_hynix(tmp_path):
    """"하이닉스"(SK 생략)는 무관한 이닉스(452400)가 아니라 SK하이닉스(000660)로 resolve돼야 한다."""
    db = _seed_group_prefix_companies(tmp_path)
    conn = connect_readonly(db)
    try:
        code = find_stock_code(conn, "하이닉스 24년 영업이익률")
    finally:
        conn.close()
    assert code == "000660"


def test_find_stock_code_resolves_group_prefix_omitted_inotek(tmp_path):
    """"이노텍"(LG 생략)도 LG이노텍(011070)으로 resolve된다(하이닉스 외 그룹명 생략 사례)."""
    db = _seed_group_prefix_companies(tmp_path)
    conn = connect_readonly(db)
    try:
        code = find_stock_code(conn, "이노텍 실적 알려줘")
    finally:
        conn.close()
    assert code == "011070"


# ── find_stock_code: 불규칙 축약형(중간 글자 생략) 종목명 정합성 ────────────────────────
# 실서버 재현 버그: "현대차 PER 알려줘"를 물으면 "현대차"가 정식 사명 "현대자동차"의 부분
# 문자열이 아니라서(가운데 "자동" 두 글자가 빠짐) 종목을 못 찾았다. 그룹접두어 제거
# (_GROUP_PREFIXES)는 "앞부분만 잘라내는" 알고리즘이라 이런 중간 축약은 다루지 못한다 —
# 별도 명시 별칭 사전(_IRREGULAR_NAME_ALIASES)으로 보강한다. "현대차증권"처럼 우연히
# "현대차"를 포함하는 무관한 회사가 있어도(느슨한 LIKE가 아니라 명시 별칭 매칭이므로)
# 잘못 걸리면 안 된다.

def _seed_irregular_alias_companies(tmp_path) -> str:
    """불규칙 축약형 매칭 재현용: 현대자동차 + 우연히 "현대차"를 포함하는 무관한 현대차증권."""
    db = tmp_path / "irregular_alias.db"
    init_db(str(db))
    conn = connect(str(db))
    try:
        rows = [
            ("005380", "현대자동차", "KOSPI", "운수장비"),
            ("001500", "현대차증권", "KOSPI", "금융"),  # "현대차"를 우연히 포함하는 무관한 회사
            ("000270", "기아", "KOSPI", "운수장비"),
        ]
        for r in rows:
            conn.execute(
                "INSERT INTO company(stock_code, name, market, sector) VALUES(?,?,?,?)", r
            )
        conn.commit()
    finally:
        conn.close()
    return str(db)


def test_find_stock_code_resolves_irregular_abbreviation_hyundai_motor(tmp_path):
    """"현대차"(중간 축약)는 무관한 현대차증권(001500)이 아니라 현대자동차(005380)로 resolve돼야 한다."""
    db = _seed_irregular_alias_companies(tmp_path)
    conn = connect_readonly(db)
    try:
        code = find_stock_code(conn, "현대차 PER 알려줘")
    finally:
        conn.close()
    assert code == "005380"


def test_find_stock_code_resolves_irregular_abbreviation_kia(tmp_path):
    """"기아차"(중간 축약)도 기아(000270)로 resolve된다(현대차 외 불규칙 축약 사례)."""
    db = _seed_irregular_alias_companies(tmp_path)
    conn = connect_readonly(db)
    try:
        code = find_stock_code(conn, "기아차 실적 알려줘")
    finally:
        conn.close()
    assert code == "000270"


def test_find_stock_code_full_official_name_still_resolves_correct_company(tmp_path):
    """회귀: 정식 사명("현대자동차")을 그대로 물으면 여전히 정확히 매칭된다(별칭 추가로 안 깨짐)."""
    db = _seed_irregular_alias_companies(tmp_path)
    conn = connect_readonly(db)
    try:
        code = find_stock_code(conn, "현대자동차 PER 알려줘")
    finally:
        conn.close()
    assert code == "005380"


def test_find_stock_code_keeps_genuine_short_name_when_no_expansion(tmp_path):
    """안전장치: 진짜로 "이닉스"를 물으면(그룹명+이닉스 회사가 실제로 없으므로) 확장하지 않고
    이닉스(452400) 그대로 둔다 — 무조건 확장하는 게 아니라 확장형이 실재할 때만 우선한다."""
    db = _seed_group_prefix_companies(tmp_path)
    conn = connect_readonly(db)
    try:
        code = find_stock_code(conn, "이닉스 주가 알려줘")
    finally:
        conn.close()
    assert code == "452400"


def test_find_stock_code_full_name_regression_with_group_prefix_fix(tmp_path):
    """회귀: 정식명칭(SK하이닉스)·타사 정식명칭(삼성전자)은 이번 보강으로 결과가 안 바뀐다."""
    db = _seed_group_prefix_companies(tmp_path)
    conn = connect_readonly(db)
    try:
        assert find_stock_code(conn, "SK하이닉스 매출 알려줘") == "000660"
        assert find_stock_code(conn, "삼성전자 PER 알려줘") == "005930"
    finally:
        conn.close()


# ── classify_intent: 재무/주가/기술지표 3갈래 판단(여러 개 동시 가능) ─────────────────
# 반환은 {"financial","price","technical"} 의 정렬된 부분집합 튜플 — 하나면 단독 서브에이전트,
# 여러 개면 여러 서브에이전트를 함께 부른다.

def test_classify_intent_detects_financial_only():
    assert classify_intent("삼성전자 PER 알려줘") == ("financial",)
    assert classify_intent("삼성전자 매출액 얼마야") == ("financial",)


def test_classify_intent_detects_price_only():
    assert classify_intent("삼성전자 주가 알려줘") == ("price",)
    assert classify_intent("삼성전자 종가랑 거래량 알려줘") == ("price",)


def test_classify_intent_detects_technical_only():
    """이동평균/RSI/MACD/볼린저 등 기술지표만 물으면 technical 단독으로 분류된다
    (순수 시세 price 와 별개 축)."""
    assert classify_intent("삼성전자 이동평균 알려줘") == ("technical",)
    assert classify_intent("삼성전자 RSI 얼마야") == ("technical",)
    assert classify_intent("삼성전자 MACD 보여줘") == ("technical",)
    assert classify_intent("삼성전자 볼린저밴드 알려줘") == ("technical",)


def test_classify_intent_detects_financial_and_price():
    assert classify_intent("삼성전자 PER이랑 주가 같이 알려줘") == ("financial", "price")


def test_classify_intent_detects_price_and_technical():
    """순수 시세와 기술지표를 함께 물으면 price+technical 두 축이 모두 잡힌다
    (여러 서브에이전트 동시 작동)."""
    assert classify_intent("삼성전자 종가랑 RSI 알려줘") == ("price", "technical")


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
    assert result == ("price",)


def test_classify_intent_llm_can_select_multiple_axes():
    """LLM 이 여러 축을 쉼표로 답하면 모두 담아 여러 서브에이전트를 함께 부를 수 있다."""
    result = classify_intent("삼성전자 분석", llm_fn=lambda p: "financial, price, technical")
    assert result == ("financial", "price", "technical")


def test_classify_intent_unclear_without_llm_fn_falls_back_to_financial_and_price():
    """불명확하면 과다조회가 안전하나, 무거운 기술지표는 명시 신호가 있을 때만 켠다
    (재무+주가로만 폴백, technical 은 제외)."""
    assert classify_intent("삼성전자 어때?") == ("financial", "price")


def test_classify_intent_llm_first_overrides_keyword_heuristic():
    """LLM 우선순위: 재무 키워드(PER)가 있어도 llm_fn 이 있으면 LLM 판단을 먼저 채택한다."""
    calls: list[str] = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return "financial, price"

    result = classify_intent("삼성전자 PER 알려줘", llm_fn=fake_llm)
    assert len(calls) == 1  # LLM 이 먼저 호출됨(키워드로 단락하지 않음)
    assert result == ("financial", "price")  # LLM 판단이 키워드(financial 단독)를 이김


def test_classify_intent_falls_back_to_keyword_when_llm_unparseable():
    """llm_fn 응답에서 축 토큰을 못 뽑으면 키워드 휴리스틱으로 폴백한다."""
    assert classify_intent("삼성전자 주가 알려줘", llm_fn=lambda p: "???") == ("price",)


def test_classify_intent_llm_exception_falls_back_to_keyword():
    """llm_fn 이 예외를 던져도 키워드 휴리스틱으로 안전 폴백한다(예외 전파 없음)."""

    def boom(prompt: str) -> str:
        raise RuntimeError("LLM 다운")

    assert classify_intent("삼성전자 주가 알려줘", llm_fn=boom) == ("price",)


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
    assert result["intent"] == ("financial",)
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
    assert result["intent"] == ("price",)
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
    assert result["intent"] == ("financial", "price")
    assert result["financial"]["value"] == 12.5
    assert result["price"][0]["close"] == 71000.0


def test_answer_kr_question_technical_question_calls_price_snapshot_with_indicators(
    tmp_path, monkeypatch
):
    """기술지표 질문("삼성전자 RSI") → get_price_snapshot_kr 이 indicators 를 채워 호출된다
    (순수 시세 price 와 달리 TA-Lib 지표 계산 경로 발동). intent 는 technical 단독."""
    import src.agents.domain_kr as mod

    db = _seed(tmp_path)
    captured: list[dict] = []

    def spy_snapshot(conn, stock_codes, **kwargs):
        captured.append(kwargs)
        return [{"stock_code": stock_codes, "date": "2026-07-11", "close": 71000.0}]

    monkeypatch.setattr(mod, "get_price_snapshot_kr", spy_snapshot)

    conn = connect_readonly(db)
    try:
        result = answer_kr_question("삼성전자 RSI 알려줘", conn)
    finally:
        conn.close()

    assert result["intent"] == ("technical",)
    assert len(captured) == 1  # 기술지표 축 하나만 발동(단독 호출)
    assert captured[0]["indicators"] == [{"name": "rsi"}]


def test_answer_kr_question_financial_only_does_not_call_price_snapshot(tmp_path, monkeypatch):
    """재무만 필요한 질문("삼성전자 PER") → 주가/기술지표 서브에이전트는 호출되지 않는다
    (단독 호출: 필요한 축만 작동)."""
    import src.agents.domain_kr as mod

    db = _seed(tmp_path)
    price_calls: list = []
    monkeypatch.setattr(
        mod, "get_price_snapshot_kr", lambda *a, **k: price_calls.append(1) or []
    )

    conn = connect_readonly(db)
    try:
        result = answer_kr_question("삼성전자 PER 알려줘", conn)
    finally:
        conn.close()

    assert result["intent"] == ("financial",)
    assert price_calls == []  # 주가 스냅샷 미호출
    assert result["financial"]["value"] == 12.5


def test_answer_kr_question_pure_price_question_passes_no_indicators(tmp_path, monkeypatch):
    """순수 시세 질문("삼성전자 주가")은 indicators 없이 스냅샷만 조회한다(기술지표 미계산)."""
    import src.agents.domain_kr as mod

    db = _seed(tmp_path)
    captured: list[dict] = []

    def spy_snapshot(conn, stock_codes, **kwargs):
        captured.append(kwargs)
        return [{"stock_code": stock_codes, "date": "2026-07-11", "close": 71000.0}]

    monkeypatch.setattr(mod, "get_price_snapshot_kr", spy_snapshot)

    conn = connect_readonly(db)
    try:
        result = answer_kr_question("삼성전자 주가 알려줘", conn)
    finally:
        conn.close()

    assert result["intent"] == ("price",)
    assert len(captured) == 1
    assert captured[0].get("indicators") is None  # 순수 시세 → 지표 미부착


# ── _extract_indicators: 질문 → 기술지표 스펙 리스트 ──────────────────────────────

def test_extract_indicators_maps_keywords_to_specs():
    assert _extract_indicators("삼성전자 RSI 알려줘") == [{"name": "rsi"}]
    assert _extract_indicators("삼성전자 MACD 보여줘") == [{"name": "macd"}]
    assert _extract_indicators("삼성전자 볼린저밴드") == [{"name": "bollinger"}]
    assert _extract_indicators("삼성전자 이동평균") == [{"name": "sma"}]


def test_extract_indicators_multiple_and_default():
    specs = _extract_indicators("삼성전자 RSI 랑 MACD 같이")
    assert {s["name"] for s in specs} == {"rsi", "macd"}
    # 기술지표 신호는 있으나 특정 지표를 못 집으면 대표 지표(sma)로 안전 폴백
    assert _extract_indicators("삼성전자 기술지표 분석") == [{"name": "sma"}]


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
    assert classify_intent("삼성전자 직전 12개월 수익률") == ("financial",)
    assert classify_intent("삼성전자 모멘텀") == ("financial",)
    assert "financial" in classify_intent("삼성전자 가격수익률 알려줘")


def test_classify_intent_return_12m_keyword_protected_by_fallback_when_llm_unclear():
    """LLM 우선으로 바뀌었어도, LLM 이 명확한 답을 못 주면(파싱 실패) 키워드 휴리스틱 안전망으로
    폴백한다 — '직전 12개월 수익률' 은 폴백에서 financial 로 보호된다(return_12m 회귀 방지)."""

    def unclear_llm(prompt: str) -> str:
        return "잘 모르겠습니다"  # financial/price/both 토큰 없음 → 파싱 실패 → 키워드 폴백

    result = classify_intent("삼성전자 직전 12개월 수익률", llm_fn=unclear_llm)
    assert result == ("financial",)


def test_classify_intent_detects_other_computed_only_metric_keywords_as_financial():
    """return_12m 외에 metrics_at()이 계산하는 다른 계산전용 지표(ROA/PSR)도 일관되게 인식."""
    assert classify_intent("삼성전자 ROA 알려줘") == ("financial",)
    assert classify_intent("삼성전자 PSR 알려줘") == ("financial",)


def test_classify_intent_still_detects_per_as_financial_unaffected_by_computed_change():
    """회귀 방지: PER처럼 이미 resolve_metric으로 동작하던 재무지표 분류는 그대로."""
    assert classify_intent("삼성전자 PER 알려줘") == ("financial",)


def test_answer_kr_question_routes_return_12m_question_to_computed_metric(tmp_path):
    db = _seed_with_price_history(tmp_path)
    conn = connect_readonly(db)
    try:
        result = answer_kr_question("삼성전자 직전 12개월 수익률", conn)
    finally:
        conn.close()

    assert result["stock_code"] == "005930"
    assert result["intent"] == ("financial",)
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
    """"매출성장"(revenue_growth)이 "매출"(재무지표, revenue)의 상위 문자열이라 더 구체적인
    별칭이 먼저 매치돼야 한다. revenue_growth는 이번 세션에 EAV 직접 quarter 매치 경로로
    옮겨져(가격 불필요) 이제 computed_metric_fn이 아니라 resolve_metric_fn을 통해 조회된다
    (routing 목적지는 바뀌었지만, "매출"로 오매핑되지 않는다는 핵심 회귀는 그대로 지킨다)."""
    db = _seed_with_price_history(tmp_path)
    financial_calls: list[tuple] = []
    computed_calls: list[tuple] = []

    def spy_resolve_metric(conn, stock_code, metric, period=None, llm_fn=None):
        financial_calls.append((stock_code, metric))
        return {"stock_code": stock_code, "metric": metric, "value": 5.0, "source": "DART", "period": "x"}

    def fake_computed(conn, stock_code, metric, **kwargs):
        computed_calls.append((stock_code, metric))
        return {"stock_code": stock_code, "metric": metric, "value": -1.0, "source": "computed", "period": "x"}

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

    assert financial_calls == [("005930", "revenue_growth")]
    assert computed_calls == []
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


# ── 다중분기 파싱(_parse_periods) — 한 질문에 여러 분기/연도가 함께 언급된 경우 ──────────
# 실서버 재현 버그: "하이닉스 25년과 26년 1분기 영업이익률"처럼 한 질문이 여러 분기를 지목하면
# _parse_period(단일 dict만 반환)가 첫 분기(2025Q1)만 잡고 둘째("26년 1분기")를 통째로 버렸다.
# _parse_periods는 언급된 각 기간을 위치 순서대로 리스트로 반환한다.

def test_parse_periods_shared_quarter_across_two_years():
    # "25년과 26년 1분기": 뒤에 붙은 단일 분기(1분기)를 두 연도가 공유 → 각 연도의 1분기 비교.
    # (분기가 여럿 명시되면 각자 자기 분기를 쓰고, 단 하나만 명시되면 앞의 맨연도가 그 분기를
    #  공유하는 게 한국어의 자연스러운 비교 표현이다.)
    assert _parse_periods("하이닉스 25년과 26년 1분기 영업이익률") == [
        {"kind": "quarter", "quarter": "2025Q1"},
        {"kind": "quarter", "quarter": "2026Q1"},
    ]


def test_parse_periods_each_year_has_own_quarter():
    assert _parse_periods("삼성전자 2025년 1분기와 2026년 1분기 매출") == [
        {"kind": "quarter", "quarter": "2025Q1"},
        {"kind": "quarter", "quarter": "2026Q1"},
    ]


def test_parse_periods_two_quarters_share_one_year():
    assert _parse_periods("삼성전자 2025년 1분기와 2분기 영업이익") == [
        {"kind": "quarter", "quarter": "2025Q1"},
        {"kind": "quarter", "quarter": "2025Q2"},
    ]


def test_parse_periods_single_quarter_matches_parse_period():
    # 단일 기간은 _parse_period와 동일한 원소 하나만 담는다(회귀 없음).
    assert _parse_periods("삼성전자 26년 1분기 영업이익") == [
        {"kind": "quarter", "quarter": "2026Q1"},
    ]


def test_parse_periods_single_annual_matches_parse_period():
    assert _parse_periods("삼성전자 2025년 영업이익") == [
        {"kind": "annual", "year": 2025},
    ]


def test_parse_periods_two_bare_years_are_annual_each():
    # 분기가 전혀 없으면 각 연도를 그 해 연간으로 본다.
    assert _parse_periods("삼성전자 2024년과 2025년 매출") == [
        {"kind": "annual", "year": 2024},
        {"kind": "annual", "year": 2025},
    ]


def test_parse_periods_empty_when_no_year():
    # 연도가 없으면(분기 단독 포함) 빈 리스트 — 기존 최신분기 유지 경로로 흐른다.
    assert _parse_periods("삼성전자 PER 알려줘") == []
    assert _parse_periods("삼성전자 1분기 영업이익") == []


def test_parse_periods_no_share_when_metric_repeated_between_years():
    # 실서버 재현 버그: "25년 영업이익률과 26년 1분기 영업이익률"은 지표명이 각 절마다
    # 반복돼("25년 영업이익률" + "26년 1분기 영업이익률") 두 연도가 서로 다른 기간 타입을
    # 독립적으로 지목한 것이다. 연도 사이가 순수 접속사("과 ")뿐인
    # test_parse_periods_shared_quarter_across_two_years와 달리, 여기는 연도 사이에
    # "영업이익률과"라는 실질 단어가 끼어 있어 공유하면 안 된다 — 앞의 25년은 연간으로
    # 남아야 한다(수정 전에는 [2025Q1, 2026Q1]로 잘못 공유되어 2025Q1 값을 돌려주고 있었다).
    assert _parse_periods("SK하이닉스 25년 영업이익률과 26년 1분기 영업이익률") == [
        {"kind": "annual", "year": 2025},
        {"kind": "quarter", "quarter": "2026Q1"},
    ]


# ── 지표 파싱(_extract_metric) — 회귀: "영업이익률"(비율)이 "영업이익"(금액)으로 오매핑 ─────
# 실서버 재현 버그: "하이닉스 …영업이익률" 질문에서 _extract_metric이 부분문자열 "영업이익"을
# 먼저 매치해 operating_profit(원본 계정 금액)을 뽑았다. operating_margin은 METRIC_SOURCE_MAP에
# 있어 computed-only 별칭 경로(_COMPUTED_KO_ALIASES)에 안 들어가고, _METRIC_KO_ALIASES에도
# "영업이익률" 키가 없어 발생. (net_margin은 우연히 computed-only라 "순이익률"이 먼저 잡혀 정상.)

def test_extract_metric_operating_margin_ratio_not_raw_profit():
    # 비율 지표 "영업이익률"은 operating_margin이어야 한다(원본 계정 operating_profit이 아님).
    assert _extract_metric("삼성전자 영업이익률") == "operating_margin"
    assert _extract_metric("하이닉스 25년과 26년 1분기 영업이익률") == "operating_margin"
    # 회귀: "영업이익"(률 없음, 원본 계정)은 여전히 operating_profit으로 남아야 한다.
    assert _extract_metric("삼성전자 영업이익") == "operating_profit"
    # 회귀: "순이익률"은 computed 경로로 net_margin(기존 정상 동작 유지).
    assert _extract_metric("삼성전자 순이익률") == "net_margin"


# ── 지표 파싱(_extract_metric) — psr/pcr/ev_ebitda가 METRIC_SOURCE_MAP으로 옮겨가며
# computed-only 별칭(_COMPUTED_KO_ALIASES)에서 빠졌는데, _METRIC_KO_ALIASES에 한국어 별칭이
# 새로 등록되지 않으면 한국어 질문이 인식되지 않는다(영어 약어 "PSR"/"PCR"는 METRIC_SOURCE_MAP
# 원본 키 매치로 그대로 인식되지만 "주가매출비율" 같은 한국어 표현은 별도 등록이 필요하다).
# "주가매출비율"은 "매출"을 부분문자열로 포함하므로, "매출"(revenue) 별칭보다 먼저 매치돼야
# 한다(영업이익률/순이익률과 동일한 순서 규율).

def test_extract_metric_psr_pcr_ev_ebitda_korean_aliases():
    assert _extract_metric("삼성전자 주가매출비율 알려줘") == "psr"
    assert _extract_metric("삼성전자 주가현금흐름비율 알려줘") == "pcr"
    assert _extract_metric("삼성전자 EV/EBITDA 알려줘") == "ev_ebitda"
    # 회귀: "매출"(원본 계정)은 여전히 revenue로 남아야 한다.
    assert _extract_metric("삼성전자 매출 알려줘") == "revenue"


def test_answer_kr_question_operating_margin_routes_ratio_metric(tmp_path):
    # end-to-end: "영업이익률" 질문이 resolve_metric_fn에 operating_margin으로 전달돼야 한다
    # (operating_profit 금액이 아니라). 실서버에서 kr 도메인 경로가 영업이익 '금액'만 답하던 원인.
    db = _seed(tmp_path)
    seen: dict = {}

    def spy_resolve_metric(conn, stock_code, metric, llm_fn=None, period=None):
        seen["metric"] = metric
        return {"stock_code": stock_code, "metric": metric,
                "value": 21.09, "source": "DART", "period": "2025Q1"}

    conn = connect_readonly(db)
    try:
        answer_kr_question(
            "삼성전자 25년 1분기 영업이익률", conn, resolve_metric_fn=spy_resolve_metric
        )
    finally:
        conn.close()

    assert seen["metric"] == "operating_margin"


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


# ── 다중분기 배선: answer_kr_question이 여러 분기를 각각 조회해 result["periods"]에 담는다 ──
# 실서버 재현 버그: "하이닉스 25년과 26년 1분기 영업이익률"이 첫 분기(2025Q1)만 조회하고
# 둘째(2026Q1)를 통째로 버렸다. 이제 각 분기를 개별 조회해 기간별로 명확히 구분해 담는다.

def test_answer_kr_question_multi_quarter_returns_value_per_period(tmp_path):
    db = _seed(tmp_path, name="하이닉스", code="000660")
    seen: list = []

    def spy_resolve_metric(conn, stock_code, metric, llm_fn=None, period=None):
        seen.append(period)
        label = period["quarter"] if period and period.get("kind") == "quarter" else None
        return {
            "stock_code": stock_code, "metric": metric,
            "value": 21.0 if label == "2025Q1" else 19.0,
            "source": "DART", "period": label,
        }

    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "하이닉스 25년과 26년 1분기 영업이익률", conn, resolve_metric_fn=spy_resolve_metric
        )
    finally:
        conn.close()

    # 두 분기 모두 개별 조회됐는가(핵심).
    assert seen == [
        {"kind": "quarter", "quarter": "2025Q1"},
        {"kind": "quarter", "quarter": "2026Q1"},
    ]
    periods = result["periods"]
    assert [p["period"] for p in periods] == ["2025Q1", "2026Q1"]
    assert periods[0]["financial"]["value"] == 21.0
    assert periods[1]["financial"]["value"] == 19.0
    # 다중분기는 기간별로 구분해 담으므로 최상위 financial은 쓰지 않는다(다중종목 entities와 동일 관례).
    assert result["financial"] is None


def test_answer_kr_question_single_quarter_unchanged_no_periods_key(tmp_path):
    # 회귀: 단일 분기 질문(압도적 다수)은 기존처럼 financial 하나만, periods 키는 없다.
    db = _seed(tmp_path)

    def spy_resolve_metric(conn, stock_code, metric, llm_fn=None, period=None):
        return {"stock_code": stock_code, "metric": metric,
                "value": 12.5, "source": "DART", "period": "2024Q3"}

    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "삼성전자 2024년 3분기 PER", conn, resolve_metric_fn=spy_resolve_metric
        )
    finally:
        conn.close()

    assert result["financial"]["value"] == 12.5
    assert result.get("periods") is None


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


def test_find_stock_codes_resolves_group_prefix_omitted_single(tmp_path):
    """find_stock_codes(복수)도 그룹명 생략 매칭을 공유한다 — "하이닉스"는 이닉스(452400)가
    아니라 SK하이닉스(000660) 하나로만 나와야 한다(단수/복수 로직 일관성)."""
    db = _seed_group_prefix_companies(tmp_path)
    conn = connect_readonly(db)
    try:
        codes = find_stock_codes(conn, "하이닉스 영업이익률 알려줘")
    finally:
        conn.close()
    assert codes == ["000660"]


def test_find_stock_codes_multi_with_group_prefix_omitted(tmp_path):
    """다중종목에서도 그룹명 생략이 정확히 해석된다 — "삼성전자와 하이닉스"는
    {삼성전자, SK하이닉스}이고 무관한 이닉스(452400)는 섞이지 않는다."""
    db = _seed_group_prefix_companies(tmp_path)
    conn = connect_readonly(db)
    try:
        codes = find_stock_codes(conn, "삼성전자와 하이닉스 비교해줘")
    finally:
        conn.close()
    assert set(codes) == {"005930", "000660"}
    assert "452400" not in codes


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


# ── 실서버 재현 버그: "SK하이닉스 26년 1분기 매출총이익률"이 아직 공시 전인 26년 1분기
#    대신 25년 4분기 값을 아무 경고 없이 그대로 반환했다. gross_margin 등 계산전용 지표는
#    look-ahead 방지를 위해 asof(분기 말일 이하 최신 거래일)까지 공시된 최신 분기를 쓰는데
#    (get_cross_section/effective_quarter_at, 백테스트와 공유하는 안전장치 — 수정 금지),
#    분기 말일과 실제 공시일 사이엔 DART 정기공시 특성상 항상 수십일의 시차가 있어 사용자가
#    지목한 분기 자체가 매번 조용히 그 이전 분기로 대체된다. operating_margin 등
#    resolve_metric(DART financials 직접 quarter 매치) 경로는 이 간접참조를 안 거쳐
#    영향받지 않는다(같은 분기 질문에 정상 응답하는 이유). 안전장치 자체(effective_quarter_at)
#    는 그대로 두고, get_cross_section 각 행에 이미 있는 실제 사용 분기("quarter" 필드)를
#    resolve_computed_metric이 data_quarter로 노출해, 조용한 대체를 정직한 대체로 바꾼다. ──

def test_resolve_computed_metric_surfaces_data_quarter_from_cross_section_row():
    fake_execute_sql = lambda sql, conn: {"ok": True, "rows": [{"d": "2026-03-31"}]}
    fake_cross_section = lambda conn, asof: [
        {"stock_code": "000660", "gross_margin": 68.775, "quarter": "2025Q4"},
    ]
    result = resolve_computed_metric(
        None, "000660", "gross_margin",
        execute_sql_fn=fake_execute_sql, cross_section_fn=fake_cross_section,
    )
    assert result["data_quarter"] == "2025Q4"


def test_answer_kr_question_warns_when_computed_metric_quarter_differs_from_requested(tmp_path):
    """실측 재현(원 버그는 gross_margin이었으나 그건 근본수정으로 이 경로를 더 이상 안 타므로,
    여전히 _COMPUTED_ONLY_FIELDS로 남는 roc로 동일 시나리오를 검증한다): '하이닉스 26년 1분기
    투하자본수익률' 요청 시 아직 공시 전이라 get_cross_section이 2025Q4 데이터를 대신 골랐다면
    (look-ahead 방지 정상 동작), 그 사실을 조용히 숨기지 말고 명시적으로 경고해야 한다."""
    db = _seed_two_companies(tmp_path)
    conn = connect_readonly(db)

    def fake_cross_section(conn, asof):
        return [{"stock_code": "000660", "roc": 68.775, "quarter": "2025Q4"}]

    try:
        result = answer_kr_question(
            "SK하이닉스 26년 1분기 투하자본수익률", conn,
            computed_metric_fn=lambda conn, code, metric, **kw: resolve_computed_metric(
                conn, code, metric, cross_section_fn=fake_cross_section,
                execute_sql_fn=lambda sql, c: {"ok": True, "rows": [{"d": "2026-03-31"}]},
            ),
        )
    finally:
        conn.close()

    assert result["financial"]["value"] == 68.775
    assert result["financial"]["data_quarter"] == "2025Q4"
    assert any("2025Q4" in e and "2026Q1" in e for e in result["errors"]), result["errors"]


def test_answer_kr_question_no_warning_when_computed_metric_quarter_matches_requested(tmp_path):
    """회귀 방지: 요청 분기와 실제 사용 분기가 같으면(정상 케이스) 경고를 붙이지 않는다."""
    db = _seed_two_companies(tmp_path)
    conn = connect_readonly(db)

    def fake_cross_section(conn, asof):
        return [{"stock_code": "000660", "roc": 79.274, "quarter": "2026Q1"}]

    try:
        result = answer_kr_question(
            "SK하이닉스 26년 1분기 투하자본수익률", conn,
            computed_metric_fn=lambda conn, code, metric, **kw: resolve_computed_metric(
                conn, code, metric, cross_section_fn=fake_cross_section,
                execute_sql_fn=lambda sql, c: {"ok": True, "rows": [{"d": "2026-03-31"}]},
            ),
        )
    finally:
        conn.close()

    assert result["financial"]["value"] == 79.274
    assert not any("아직 공시되지" in e for e in result["errors"])


# ── 근본 원인 수정: gross_margin/net_margin/cogs_ratio는 이미 _FLOW_RATIO_ACCOUNTS
#    (src/agents/data_financial.py)로 EAV 두 계정 직접 quarter 매치가 구현/테스트돼
#    있었는데(test_resolve_metric_net_gross_cogs_ratios_fall_back_to_eav), METRIC_SOURCE_MAP
#    에 등록이 안 돼 있어 단일종목 질문 라우팅이 이 경로를 타지 못하고 _COMPUTED_ONLY_FIELDS
#    (asof/look-ahead 경로)로 잘못 빠졌다. operating_margin과 동일하게 등록하면 그 경로가
#    아예 필요 없어진다(경고문이 아니라 애초에 정답이 나옴). ──────────────────────────

def test_gross_margin_net_margin_cogs_ratio_are_not_computed_only():
    assert "gross_margin" not in _COMPUTED_ONLY_FIELDS
    assert "net_margin" not in _COMPUTED_ONLY_FIELDS
    assert "cogs_ratio" not in _COMPUTED_ONLY_FIELDS


def test_answer_kr_question_gross_margin_named_quarter_uses_eav_direct_match(tmp_path):
    """실측 재현 수정 확인: 'SK하이닉스 26년 1분기 매출총이익률'이 이제 asof/cross_section을
    거치지 않고 operating_margin처럼 EAV에서 그 분기를 직접 찾아 정답을 낸다. computed_metric_fn
    (cross_section 경로)이 호출되면 boom을 던지게 해, 정말로 그 경로를 안 타는지도 함께 검증."""
    db = _seed_two_companies(tmp_path)
    conn = connect(db)
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
        "VALUES(?,?,?,?,?)",
        ("000660", "2026Q1", "2026-05-15", "revenue", 52576287000000.0),
    )
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
        "VALUES(?,?,?,?,?)",
        ("000660", "2026Q1", "2026-05-15", "gross_profit", 41679414000000.0),
    )
    conn.commit()
    conn.close()
    conn = connect_readonly(db)

    def _boom(*a, **kw):
        raise AssertionError("gross_margin이 여전히 computed_metric_fn(cross_section) 경로를 탐")

    try:
        result = answer_kr_question(
            "SK하이닉스 26년 1분기 매출총이익률", conn, computed_metric_fn=_boom,
        )
    finally:
        conn.close()

    assert result["financial"]["value"] == pytest.approx(79.274, abs=0.01)
    assert result["financial"]["source"] == "DART"
    assert result["errors"] == []


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
    """"GPA"(대문자)가 gp_a로 인식되고, 이제 gp_a는 가격이 필요 없는 EAV 직접 quarter 매치
    경로(_RATIO_TTM_ACCOUNTS)를 타므로 computed_metric_fn 없이도 resolve_metric_fn(기본
    resolve_metric)만으로 답이 나와야 한다. period 미지정 질문이라 '데이터가 있는 가장
    최근 분기'로 자동 산정되는지도 함께 검증한다."""
    db = _seed_two_companies(tmp_path)
    conn = connect(db)
    for q in ("2025Q2", "2025Q3", "2025Q4", "2026Q1"):  # gp_a 분자는 TTM(최근 4분기 합)
        conn.execute(
            "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
            "VALUES(?,?,?,?,?)",
            ("005930", q, "2026-05-15", "gross_profit", 2.0e12),
        )
    conn.execute(  # 분모(총자산)는 스냅샷이라 해당 분기 하나만
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
        "VALUES(?,?,?,?,?)",
        ("005930", "2026Q1", "2026-05-15", "total_assets", 40.0e12),
    )
    conn.commit()
    conn.close()
    conn = connect_readonly(db)

    def _boom(*a, **kw):
        raise AssertionError("gp_a가 여전히 computed_metric_fn(cross_section) 경로를 탐")

    try:
        result = answer_kr_question("삼성전자 GPA 알려줘", conn, computed_metric_fn=_boom)
    finally:
        conn.close()

    assert "재무 지표를 인식하지 못함" not in result["errors"]
    assert result["financial"]["value"] == 8.0e12 / 40.0e12 * 100.0
    assert result["financial"]["period"] == "2026Q1"


# ── _parse_recent_return_months: "최근 N개월/N년 수익률" → 개월수 ─────────────────
# 실서버 재현 버그: "삼성전자 최근 3개월 수익률"이 12가 아닌 임의 개월수라 인식 실패 →
# free_exec LLM 코드생성으로 폴백해 8년 전 데이터로 틀린 답을 냄. 이 파서가 개월수를 뽑아
# 결정론 함수(price_return_over_months)로 라우팅되게 한다. "직전 12개월 수익률"/"모멘텀"
# (기존 _COMPUTED 경로)은 여기서 매치하지 않아 기존 경로가 그대로 유지된다(회귀).

def test_parse_recent_return_months_recent_three_months():
    assert _parse_recent_return_months("삼성전자 최근 3개월 수익률") == 3


def test_parse_recent_return_months_recent_twelve_months():
    assert _parse_recent_return_months("SK하이닉스 최근 12개월 수익률") == 12


def test_parse_recent_return_months_one_year_is_twelve():
    assert _parse_recent_return_months("1년 수익률 알려줘") == 12


def test_parse_recent_return_months_two_years_span_is_twenty_four():
    assert _parse_recent_return_months("2년간 수익률") == 24


def test_parse_recent_return_months_none_for_unrelated_question():
    assert _parse_recent_return_months("삼성전자 PER 알려줘") is None


def test_parse_recent_return_months_none_for_legacy_return_12m_phrasing():
    # 회귀: "직전 12개월 수익률"/"모멘텀"은 기존 _COMPUTED 경로가 처리해야 하므로 여기선 None.
    assert _parse_recent_return_months("삼성전자 직전 12개월 수익률") is None
    assert _parse_recent_return_months("삼성전자 모멘텀 알려줘") is None


def test_parse_recent_return_months_none_for_calendar_year_phrasing():
    # "2024년 수익률"/"25년 수익률"은 캘린더 연도(기간)이지 "N년 전"이 아니다 → None(오탐 방지).
    assert _parse_recent_return_months("삼성전자 2024년 수익률") is None
    assert _parse_recent_return_months("삼성전자 25년 수익률") is None


def test_answer_kr_question_routes_recent_months_return_to_price_return_fn(tmp_path):
    """'삼성전자 최근 3개월 수익률' → 주입한 price_return_fn이 (code, asof, months)로 호출되고
    그 결과가 result['financial']에 그대로 담긴다. intent는 financial, price는 None."""
    db = _seed(tmp_path)  # prices 최신 거래일 = 2026-07-11
    calls: list[tuple] = []

    def fake_price_return(conn, code, asof, months):
        calls.append((code, asof, months))
        return {"stock_code": code, "months": months, "return_pct": 12.34,
                "start_date": "2026-04-11", "end_date": asof}

    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "삼성전자 최근 3개월 수익률", conn, price_return_fn=fake_price_return
        )
    finally:
        conn.close()

    assert calls == [("005930", "2026-07-11", 3)]
    assert result["intent"] == ("financial",)
    assert result["financial"]["return_pct"] == pytest.approx(12.34)
    assert result["financial"]["months"] == 3
    assert result["price"] is None
    assert result["errors"] == []


def test_answer_kr_question_recent_months_failure_reported_without_raising(tmp_path):
    """price_return_fn이 계속 실패해도 예외가 전파되지 않고 errors에 사유가 담긴다(가까운 계층 재시도)."""
    db = _seed(tmp_path)

    def boom(conn, code, asof, months):
        raise RuntimeError("boom")

    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "삼성전자 최근 6개월 수익률", conn, price_return_fn=boom
        )
    finally:
        conn.close()

    assert result["intent"] == ("financial",)
    assert result["financial"] is None
    assert any("boom" in e for e in result["errors"])


def test_answer_kr_question_multi_entity_recent_months_return_routes_to_price_return_fn(tmp_path):
    """회귀(리뷰 지적 (a)): 다중종목 '최근 N개월 수익률'도 단일종목과 동일하게 결정론
    price_return_fn으로 종목별 계산되어야 한다. 예전엔 다중종목 경로에 이 처리가 빠져
    _extract_metric이 임의 개월수를 못 잡아 종목마다 '재무 지표를 인식하지 못함'만 남고
    결정론 계산이 통째로 빠졌다(LLM 자유 코드생성 폴백으로 새는 실패 패턴의 원인)."""
    db = _seed_two_companies(tmp_path)  # prices 최신 거래일 = 2026-07-15
    calls: list[tuple] = []

    def fake_price_return(conn, code, asof, months):
        calls.append((code, asof, months))
        return {"stock_code": code, "months": months, "return_pct": 20.0}

    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "삼성전자와 SK하이닉스 최근 6개월 수익률", conn, price_return_fn=fake_price_return
        )
    finally:
        conn.close()

    assert {c[0] for c in calls} == {"005930", "000660"}
    assert all(c[2] == 6 for c in calls)          # 6개월
    assert all(c[1] == "2026-07-15" for c in calls)  # asof = 최신 거래일
    assert result["intent"] == ("financial",)
    by_code = {e["stock_code"]: e for e in result["entities"]}
    assert by_code["005930"]["financial"]["return_pct"] == pytest.approx(20.0)
    assert by_code["000660"]["financial"]["months"] == 6
    assert all(e["errors"] == [] for e in result["entities"])


# ── data_asof: 기간 미지정 시 실제 사용된 데이터 시점 라벨링 ─────────────────────────
# 질문에 연도/분기가 없으면 시스템이 자동으로 데이터 시점을 정한다(주가 기반 지표=종가
# 기준일, 재무제표 기반 지표=기준분기). 사용자가 "언제 기준 데이터냐"를 검증할 수 있도록,
# 이미 resolve된 값만 재사용해(effective_* 재호출 없이) 결과 dict에 data_asof로 순수 추가한다.
def test_answer_kr_question_unspecified_period_price_metric_labels_price_date(tmp_path):
    """기간 미지정 주가 기반 지표(PER)는 실제 사용된 종가 기준일을 data_asof.price_date에 담는다."""
    db = _seed(tmp_path)  # metrics: quarter=2025Q1, price_date=2026-07-11, per=12.5
    conn = connect_readonly(db)
    try:
        result = answer_kr_question("삼성전자 PER 알려줘", conn)
    finally:
        conn.close()

    assert result["data_asof"]["price_date"] == "2026-07-11"
    # PER은 분기 재무(EPS)도 함께 쓰므로 재무 기준분기도 담긴다(둘 다 실제 사용된 시점).
    assert result["data_asof"]["financial_quarter"] == "2025Q1"


def test_answer_kr_question_unspecified_period_financial_metric_labels_quarter(tmp_path):
    """기간 미지정 순수 재무지표(영업이익률)는 실제 사용된 기준분기만 담는다(가격 기준일 없음)."""
    db = _seed(tmp_path)
    conn = connect(db)
    try:
        conn.execute(
            "UPDATE metrics SET operating_margin=? WHERE stock_code=? AND quarter=?",
            (15.5, "005930", "2025Q1"),
        )
        conn.commit()
    finally:
        conn.close()

    conn = connect_readonly(db)
    try:
        result = answer_kr_question("삼성전자 영업이익률 알려줘", conn)
    finally:
        conn.close()

    assert result["data_asof"]["financial_quarter"] == "2025Q1"
    assert "price_date" not in result["data_asof"]  # 순수 재무지표엔 가격 기준일 없음


def test_answer_kr_question_unspecified_period_price_snapshot_labels_price_date(tmp_path):
    """기간 미지정 주가 조회는 실제 스냅샷 거래일을 data_asof.price_date에 담는다."""
    db = _seed(tmp_path)  # prices: date=2026-07-11
    conn = connect_readonly(db)
    try:
        result = answer_kr_question("삼성전자 주가 알려줘", conn)
    finally:
        conn.close()

    assert result["data_asof"]["price_date"] == "2026-07-11"


def test_answer_kr_question_explicit_period_preserves_shape_and_omits_data_asof(tmp_path):
    """회귀 방지: 기간 명시 질문은 기존 키/값을 그대로 유지하고 자동 시점 라벨을 붙이지 않는다."""
    db = _seed(tmp_path)  # metrics quarter=2025Q1
    conn = connect_readonly(db)
    try:
        result = answer_kr_question("삼성전자 2025년 1분기 PER 알려줘", conn)
    finally:
        conn.close()

    assert result["stock_code"] == "005930"
    assert result["intent"] == ("financial",)
    assert result["financial"]["value"] == 12.5
    assert result["financial"]["period"] == "2025Q1"
    assert "data_asof" not in result  # 기간을 명시했으므로 중복 라벨 생략


def test_answer_kr_question_multi_entity_unspecified_period_labels_data_asof(tmp_path):
    """다중종목+기간미지정도 실제 사용된 시점을 최상위 data_asof에 담는다."""
    db = _seed_two_companies(tmp_path)  # metrics: price_date=2026-07-15, quarter=2025Q1
    conn = connect_readonly(db)
    try:
        result = answer_kr_question("삼성전자와 SK하이닉스 PER 비교해줘", conn)
    finally:
        conn.close()

    assert result["data_asof"]["price_date"] == "2026-07-15"
    assert result["data_asof"]["financial_quarter"] == "2025Q1"


# ── 문제 A/B/C: 백테스트 옵션 전체 지표를 질의응답 경로에서도 과거 시점에 원본 계산 ────────
# 배경: metrics 사전계산 테이블은 최신 한 분기 스냅샷만 유지해, 과거 시점을 지목한
# per/pbr/roe/operating_margin/debt_ratio/market_cap 질문은 그 분기 metrics 행이 없어
# resolve_metric이 None으로 빠졌다. 원본 재무제표+주가(metrics_at)로는 그 시점 값을
# 재현할 수 있으므로, resolve_metric이 None이고 기간이 명시됐으면 computed 폴백을 덧댄다.
# psr/roa/roc 등 계산전용 지표는 다중분기 경로도 기간별로 개별 계산하도록 배선한다.

def test_answer_kr_question_metrics_col_past_period_falls_back_to_computed(tmp_path):
    """문제 A(단일종목·단일기간): 과거 시점 PER이 metrics 부재로 None이면 계산 폴백이 값을 낸다."""
    db = _seed(tmp_path, name="SK하이닉스", code="000660")  # metrics: 2025Q1만
    # 2024년 말일 이하 거래일이 실재해야 asof가 확정된다(2024-12-30 시드).
    seed_conn = connect(db)
    seed_conn.execute(
        "INSERT INTO prices(stock_code, date, close, market_cap) VALUES(?,?,?,?)",
        ("000660", "2024-12-30", 130000.0, 9.0e13),
    )
    seed_conn.commit()
    seed_conn.close()

    seen: list = []

    def spy_computed_metric_fn(conn, code, metric, asof=None, **kw):
        seen.append((code, metric, asof))
        return {"stock_code": code, "metric": metric, "value": 8.8,
                "source": "computed", "period": asof, "estimated": None}

    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "SK하이닉스 2024년 PER", conn, computed_metric_fn=spy_computed_metric_fn,
        )
    finally:
        conn.close()

    # resolve_metric None → 2024년 말일 이하 최신 거래일(2024-12-30) asof로 계산 폴백 발동.
    assert seen == [("000660", "per", "2024-12-30")]
    assert result["financial"]["value"] == 8.8
    assert result["financial"]["source"] == "computed"
    # resolve_metric과 동일한 키 집합으로 정규화(누락 키 None).
    assert result["financial"]["dart_value"] is None
    assert result["financial"]["fnguide_value"] is None
    assert result["financial"]["price_date"] is None


def test_answer_kr_question_metrics_col_present_period_does_not_fall_back(tmp_path):
    """우선순위 회귀: metrics 캐시에 그 분기 값이 있으면 계산 폴백은 발동하지 않고 캐시값을 쓴다."""
    db = _seed(tmp_path)  # metrics: 2025Q1, per=12.5

    def boom_computed_metric_fn(conn, code, metric, asof=None, **kw):
        raise AssertionError("사전계산값이 있는데 계산 폴백이 발동하면 안 됨")

    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "삼성전자 2025년 1분기 PER", conn, computed_metric_fn=boom_computed_metric_fn,
        )
    finally:
        conn.close()

    assert result["financial"]["value"] == 12.5
    assert result["financial"]["period"] == "2025Q1"


def test_answer_kr_question_computed_only_multi_period_returns_value_per_period(tmp_path):
    """문제 B/C(단일종목·다중분기): 계산전용 지표(PSR)를 다중분기로 물으면 각 기간을 개별 계산해
    periods에 담는다(기존엔 계산전용 지표가 다중분기 분기를 못 타고 단일 asof로만 계산됐다)."""
    db = _seed(tmp_path, name="SK하이닉스", code="000660")
    seed_conn = connect(db)
    for d in ("2024-12-30", "2025-12-30"):
        seed_conn.execute(
            "INSERT INTO prices(stock_code, date, close, market_cap) VALUES(?,?,?,?)",
            ("000660", d, 130000.0, 9.0e13),
        )
    seed_conn.commit()
    seed_conn.close()

    def spy_computed_metric_fn(conn, code, metric, asof=None, **kw):
        val = {"2024-12-30": 1.1, "2025-12-30": 1.5}.get(asof)
        return {"stock_code": code, "metric": metric, "value": val,
                "source": "computed", "period": asof, "estimated": None}

    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "SK하이닉스 2024년과 2025년 PSR", conn, computed_metric_fn=spy_computed_metric_fn,
        )
    finally:
        conn.close()

    assert result["financial"] is None
    periods = result["periods"]
    assert [p["period"] for p in periods] == ["2024 연간", "2025 연간"]
    assert periods[0]["financial"]["value"] == 1.1
    assert periods[1]["financial"]["value"] == 1.5


def test_answer_kr_question_metrics_col_multi_period_falls_back_per_period(tmp_path):
    """문제 A+B(단일종목·다중분기): per가 두 과거 분기 모두 metrics 부재로 None이면 각 기간을
    개별 asof로 계산 폴백한다."""
    db = _seed(tmp_path, name="SK하이닉스", code="000660")  # metrics: 2025Q1만
    seed_conn = connect(db)
    for d in ("2024-12-30", "2025-12-30"):
        seed_conn.execute(
            "INSERT INTO prices(stock_code, date, close, market_cap) VALUES(?,?,?,?)",
            ("000660", d, 130000.0, 9.0e13),
        )
    seed_conn.commit()
    seed_conn.close()

    def spy_computed_metric_fn(conn, code, metric, asof=None, **kw):
        val = {"2024-12-30": 6.6, "2025-12-30": 9.9}.get(asof)
        return {"stock_code": code, "metric": metric, "value": val,
                "source": "computed", "period": asof, "estimated": None}

    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "SK하이닉스 2024년과 2025년 PER", conn, computed_metric_fn=spy_computed_metric_fn,
        )
    finally:
        conn.close()

    periods = result["periods"]
    assert [p["period"] for p in periods] == ["2024 연간", "2025 연간"]
    assert periods[0]["financial"]["value"] == 6.6
    assert periods[1]["financial"]["value"] == 9.9


def test_answer_kr_question_multi_entity_metrics_col_past_period_falls_back(tmp_path):
    """문제 A(다중종목·단일기간): 두 종목의 과거 시점 PER이 metrics 부재로 None이면 각각 계산 폴백."""
    db = _seed_two_companies(tmp_path)  # metrics: 2025Q1
    seed_conn = connect(db)
    for code in ("005930", "000660"):
        seed_conn.execute(
            "INSERT INTO prices(stock_code, date, close, market_cap) VALUES(?,?,?,?)",
            (code, "2024-12-30", 60000.0, 3.0e14),
        )
    seed_conn.commit()
    seed_conn.close()

    def spy_computed_metric_fn(conn, code, metric, asof=None, **kw):
        return {"stock_code": code, "metric": metric,
                "value": 9.9 if code == "005930" else 7.7,
                "source": "computed", "period": asof, "estimated": None}

    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "삼성전자와 SK하이닉스 2024년 PER 비교", conn,
            computed_metric_fn=spy_computed_metric_fn,
        )
    finally:
        conn.close()

    by_code = {e["stock_code"]: e for e in result["entities"]}
    assert by_code["005930"]["financial"]["value"] == 9.9
    assert by_code["000660"]["financial"]["value"] == 7.7


# ── 특정일자(연도 없는 월-일) 시세 조회 — 실서버 재현 버그 ─────────────────────────
# "하이닉스 6월 18일 주가정보"(연도 미명시)가 데이터가 있는 가장 오래된 연도(2015-06-18)를
# 골라버리던 버그. domain_backtest의 "종료연도 미명시 시 오늘이 속한 연도 기본값" 패턴과 동일
# 원칙: 연도가 없는 월-일은 오늘이 속한 연도를 우선 쓰고, 그 시점 이하 가장 가까운 과거
# 거래일 종가로 폴백한다(오래된 과거 연도를 임의로 고르지 않는다).
def test_parse_price_target_date_no_year_uses_current_year():
    assert _parse_price_target_date(
        "하이닉스 6월 18일 주가정보 알려줘", today=date(2026, 7, 20)
    ) == "2026-06-18"


def test_parse_price_target_date_explicit_four_digit_year():
    assert _parse_price_target_date(
        "삼성전자 2025년 6월 18일 종가", today=date(2026, 7, 20)
    ) == "2025-06-18"


def test_parse_price_target_date_explicit_two_digit_year():
    assert _parse_price_target_date(
        "삼성전자 25년 6월 18일 종가", today=date(2026, 7, 20)
    ) == "2025-06-18"


def test_parse_price_target_date_none_when_no_month_day():
    assert _parse_price_target_date("삼성전자 주가 알려줘", today=date(2026, 7, 20)) is None


def test_parse_price_target_date_ignores_relative_months():
    # "최근 3개월"은 특정 월-일이 아니다(개월 표현을 월-일로 오탐하지 않음).
    assert _parse_price_target_date(
        "삼성전자 최근 3개월 수익률", today=date(2026, 7, 20)
    ) is None


def test_parse_price_target_date_invalid_calendar_date_is_none():
    # 존재하지 않는 날짜(2월 30일)는 None(달력 검증).
    assert _parse_price_target_date(
        "삼성전자 2월 30일 종가", today=date(2026, 7, 20)
    ) is None


def _seed_price_dates(tmp_path, code: str, name: str, rows: list[tuple[str, float]]) -> str:
    """단일종목의 (date, close) 여러 건을 시드한 임시 DB 경로를 반환(특정일자 조회 테스트용)."""
    db = tmp_path / "price_dates.db"
    init_db(str(db))
    conn = connect(str(db))
    try:
        conn.execute(
            "INSERT INTO company(stock_code, name, market, sector) VALUES(?,?,?,?)",
            (code, name, "KOSPI", "반도체"),
        )
        for d, close in rows:
            conn.execute(
                "INSERT INTO prices(stock_code, date, close, market_cap, open, high, low, volume) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (code, d, close, 1.5e14, close, close, close, 1.0e6),
            )
        conn.commit()
    finally:
        conn.close()
    return str(db)


def test_answer_kr_question_month_day_without_year_uses_current_year_not_oldest(tmp_path):
    """'하이닉스 6월 18일 주가정보'(연도 미명시) → 2015(가장 오래된 연도)가 아니라 오늘이
    속한 연도(2026) 6월 18일 종가를 반환한다. 정형 경로에서 1차 시도로 성공해야 한다."""
    db = _seed_price_dates(
        tmp_path, "000660", "SK하이닉스",
        [("2015-06-18", 44900.0), ("2026-06-18", 210000.0), ("2026-07-15", 230000.0)],
    )
    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "하이닉스 6월 18일 주가정보 알려줘", conn, today=date(2026, 7, 20),
        )
    finally:
        conn.close()

    assert result["stock_code"] == "000660"
    assert result["intent"] == ("price",)
    assert result["price"][0]["date"] == "2026-06-18"  # 2015-06-18이 아님
    assert result["price"][0]["close"] == 210000.0
    assert result["data_asof"]["price_date"] == "2026-06-18"
    assert result["errors"] == []


def test_answer_kr_question_month_day_falls_back_to_nearest_past_trading_day(tmp_path):
    """월-일이 휴장일/미래여서 그날 데이터가 없으면 그 시점 이하 가장 가까운 과거 거래일로 폴백한다."""
    db = _seed_price_dates(
        tmp_path, "000660", "SK하이닉스",
        [("2026-06-17", 200000.0), ("2026-06-19", 205000.0)],  # 6/18 거래 없음
    )
    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "하이닉스 6월 18일 종가", conn, today=date(2026, 7, 20),
        )
    finally:
        conn.close()

    assert result["price"][0]["date"] == "2026-06-17"  # 6/18 없음 → 직전 거래일


def test_answer_kr_question_month_day_explicit_year_uses_that_year(tmp_path):
    """'2025년 6월 18일'처럼 연도가 명시되면 그 해의 6월 18일을 쓴다(연간 말일 폴백 아님)."""
    db = _seed_price_dates(
        tmp_path, "000660", "SK하이닉스",
        [("2025-06-18", 180000.0), ("2025-12-30", 195000.0), ("2026-06-18", 210000.0)],
    )
    conn = connect_readonly(db)
    try:
        result = answer_kr_question(
            "SK하이닉스 2025년 6월 18일 종가", conn, today=date(2026, 7, 20),
        )
    finally:
        conn.close()

    assert result["price"][0]["date"] == "2025-06-18"  # 2025년 말일(12-30)이 아님
    assert result["price"][0]["close"] == 180000.0
