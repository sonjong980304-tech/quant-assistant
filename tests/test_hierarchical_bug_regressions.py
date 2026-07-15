"""HA-14 (AC19) — 원버그 3건이 신규 계층형 아키텍처에서 재현되지 않음을 못박는 회귀/특성화 테스트.

이 세 버그는 레거시 6단계 파이프라인 시절에 발견됐다. 계층형 재설계 이후에도 안전한지
(또는 구조적으로 재현 불가능한지)를 신규 구조 경유 경로로 검증한다. 기존 단위테스트
(tests/test_llm.py, tests/test_selection.py 등)를 대체하지 않고, "신규 구조에서도 안전"임을
보이는 추가 확인 테스트다.

버그 1. LLM temperature 폴백 비결정성 (src/llm.py._openai)
    max_tokens→max_completion_tokens 재시도에서 temperature까지 빼면 결정론이 깨진다.
    신규 아키텍처의 모든 도메인/총괄 에이전트는 llm_fn: Callable[[str],str] 규약으로 LLM을
    부르며 web/app.py._build_llm_fn 이 role="sql"(temperature 기본 0.0)로 감싼다. 그 규약
    경로에서도 재시도가 temperature=0.0 을 유지함을 확인한다.

버그 2. PER 계산 시 CASE WHEN 누락으로 인한 잘못된 순위 (레거시 LLM 원시 SQL)
    신규 구조는 PER 등 비율지표를 LLM이 매번 SQL로 재계산하지 않고, 사전계산된 컬럼
    (metrics 테이블)을 resolve_metric 이 그대로 읽는다. llm_fn 을 일부러 "틀린 CASE WHEN
    SQL"을 뱉도록 넣어도 값이 영향받지 않음을 보여, 이 버그 클래스가 구조적으로 재현
    불가능함을 특성화한다.

버그 3. 백테스트 criteria 필드 환각 (존재하지 않는 필드명)
    src/backtest/selection.py._validate_criteria_keys 가 존재하지 않는 필드면 ValueError를
    던진다. 신규 백테스트 도메인 에이전트(answer_backtest_question)가 이 검증 경로를 실제로
    타서, 조용히 빈 결과가 아니라 명시적 오류로 이어짐을 단위/통합으로 확인한다.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents.data_financial import resolve_metric
from src.agents.domain_backtest import answer_backtest_question
from src.agents.supervisor import answer_with_verification, dispatch_domains
from src.db import connect, init_db
from src.llm import LLMClient
from tests.conftest import seed_kr_companies


# ===========================================================================
# 버그 1 — temperature 폴백이 신규 구조 llm_fn 규약 경유에서도 유지된다
# ===========================================================================
class _FakeCompletions:
    """1차 호출(max_tokens)은 실패, 재시도(max_completion_tokens)는 성공하는 OpenAI 더블."""

    def __init__(self, calls: list[dict]):
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise Exception(
                "Unsupported parameter: 'max_tokens' is not supported with this "
                "model. Use 'max_completion_tokens' instead."
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


class _FakeClient:
    def __init__(self, calls: list[dict]):
        self.chat = SimpleNamespace(completions=_FakeCompletions(calls))


def _fake_cfg():
    return SimpleNamespace(has_openai_key=True, openai_api_key="fake-key", openai_base_url=None)


def test_new_structure_llm_fn_preserves_temperature_on_max_tokens_fallback(monkeypatch):
    """도메인/총괄 에이전트의 llm_fn 규약(web/app.py._build_llm_fn 과 동일한 role='sql' 래핑)
    을 통해 호출해도, max_tokens 폴백 재시도에서 temperature=0.0 이 유지된다.
    """
    calls: list[dict] = []
    monkeypatch.setattr("openai.OpenAI", lambda **kw: _FakeClient(calls))

    client = LLMClient(cfg=_fake_cfg(), model="gpt-5.4-mini")
    # 신규 계층형 아키텍처의 llm_fn 계약(Callable[[str], str]) — _build_llm_fn 과 동일한 형태.
    llm_fn = lambda prompt: (client.complete(prompt, role="sql").text or "")  # noqa: E731

    out = llm_fn("삼성전자 PER 알려줘")

    assert out == "ok"
    assert len(calls) == 2  # 1차(max_tokens) 실패 + 재시도
    retry = calls[1]
    assert retry.get("temperature") == 0.0  # 재시도에서도 결정론(temperature=0) 유지
    assert retry.get("max_completion_tokens") == 1024  # complete() 기본 max_tokens
    assert "max_tokens" not in retry


# ===========================================================================
# 버그 2 — PER 등 비율지표는 사전계산 컬럼을 읽는다(LLM 원시 SQL 재계산 아님)
# ===========================================================================
_EVIL_CASE_WHEN_SQL = (
    "SELECT SUM(CASE WHEN account_key='net_income' THEN amount END) AS ni FROM financials"
)


def _seed_per(db_path: str, code: str = "005930", per: float = 8.0) -> None:
    conn = connect(db_path)
    try:
        seed_kr_companies(conn, [code])
        conn.execute(
            "INSERT INTO metrics(stock_code, quarter, price_date, market_cap, per, pbr, roe, "
            "operating_margin, debt_ratio) VALUES(?,?,?,?,?,?,?,?,?)",
            (code, "2025Q1", "2026-06-22", 1e12, per, 1.0, 10.0, 12.0, 80.0),
        )
        conn.commit()
    finally:
        conn.close()


def test_resolve_metric_per_reads_precomputed_column_ignoring_evil_llm_sql(tmp_path):
    """resolve_metric(conn, code, 'per') 는 metrics 사전계산 컬럼을 그대로 반환하며,
    llm_fn 이 '틀린 CASE WHEN 원시 SQL'을 뱉어도 결과가 영향받지 않는다(구조적 재현 불가).
    """
    db = str(tmp_path / "per.db")
    init_db(db)
    _seed_per(db, per=8.0)

    llm_calls: list[str] = []

    def evil_llm(prompt: str) -> str:
        llm_calls.append(prompt)
        return _EVIL_CASE_WHEN_SQL

    conn = connect(db)
    try:
        res = resolve_metric(conn, "005930", "per", llm_fn=evil_llm)
    finally:
        conn.close()

    assert res["value"] == 8.0       # 사전계산 metrics.per — evil SQL의 영향 전혀 없음
    assert res["source"] == "DART"
    # 'per'는 규칙표(METRIC_SOURCE_MAP)에 있어 소스판단조차 LLM을 부르지 않는다 →
    # 값 계산은 물론 라우팅에서도 LLM 원시 SQL이 개입할 여지가 없다.
    assert llm_calls == []


def test_kr_domain_per_question_never_generates_raw_ratio_sql(tmp_path):
    """한국주식 도메인 경로(answer_kr_question)에서 PER 질문도 사전계산 값을 읽는다 —
    evil llm_fn 을 넣어도 PER 값이 뒤집히지 않는다(도메인 계층까지 관통 확인)."""
    from src.agents.domain_kr import answer_kr_question
    from src.db import connect_readonly

    db = str(tmp_path / "kr_per.db")
    init_db(db)
    # 종목명으로 찾을 수 있게 회사명을 넣고 per 사전계산값을 시드한다(쓰기 연결).
    seed = connect(db)
    try:
        seed.execute(
            "INSERT INTO company(stock_code, name, market, sector) VALUES(?,?,?,?)",
            ("005930", "삼성전자", "KOSPI", "전기·전자"),
        )
        seed.execute(
            "INSERT INTO metrics(stock_code, quarter, price_date, market_cap, per, pbr, roe, "
            "operating_margin, debt_ratio) VALUES(?,?,?,?,?,?,?,?,?)",
            ("005930", "2025Q1", "2026-06-22", 1e12, 8.0, 1.0, 10.0, 12.0, 80.0),
        )
        seed.commit()
    finally:
        seed.close()

    # 실서비스(/api/query)와 동일하게 읽기전용 연결(check_same_thread=False)로 도메인 실행.
    # find_stock_code 가 execute_sql(워커 스레드)을 경유하므로 읽기전용 연결이 필요하다.
    conn = connect_readonly(db)
    try:
        out = answer_kr_question("삼성전자 PER 알려줘", conn, llm_fn=lambda p: _EVIL_CASE_WHEN_SQL)
    finally:
        conn.close()

    assert out["stock_code"] == "005930"
    assert out["financial"]["metric"] == "per"
    assert out["financial"]["value"] == 8.0       # 사전계산 값 그대로
    assert out["financial"]["source"] == "DART"


# ===========================================================================
# 버그 3 — 백테스트 criteria 필드 환각 시 조용한 빈 결과가 아니라 명시적 오류
# ===========================================================================
def _hallucinated_combine_steps() -> list[dict]:
    """combine 단계에 존재하지 않는 필드(forward_12m_return)를 criteria 로 넣은 파이프라인.

    rows 는 리터럴로 주입(DB 불필요) — select_stocks._validate_criteria_keys 가 rows[0]의
    실제 키(stock_code/per/roe)와 대조해 환각 필드를 즉시 걸러낸다.
    """
    return [
        {
            "op": "combine",
            "params": {
                "rows": [
                    {"stock_code": "000001", "per": 10.0, "roe": 5.0},
                    {"stock_code": "000002", "per": 12.0, "roe": 8.0},
                ],
                "criteria": [{"key": "forward_12m_return", "direction": "high", "weight": 1.0}],
            },
            "out": "picked",
        }
    ]


def test_backtest_hallucinated_criteria_raises_explicit_error():
    """신규 백테스트 도메인 에이전트가 환각 필드에 대해 ValueError 를 던진다(조용한 빈 결과 아님)."""
    with pytest.raises(ValueError, match="존재하지 않는 필드"):
        answer_backtest_question("저per 전략 백테스트", _hallucinated_combine_steps(), conn=None)


def test_backtest_hallucinated_criteria_surfaces_in_supervisor_dispatch():
    """총괄 dispatch_domains 계층에서 그 오류가 domain_results['backtest']['error']로 명시적으로
    노출된다 — 조용히 빈 결과({})로 흡수되지 않는다."""
    results = dispatch_domains(
        ["backtest"], "저per 전략 백테스트", conn=None, llm_fn=None,
        steps=_hallucinated_combine_steps(),
    )
    assert "backtest" in results
    assert "error" in results["backtest"]
    assert "존재하지 않는 필드" in results["backtest"]["error"]


def test_backtest_hallucinated_criteria_makes_supervisor_uncertain():
    """총괄 오케스트레이션(answer_with_verification)은 이 오류를 '검증 실패→불확실'로 처리하고,
    원본 오류를 domain_results 에 그대로 보존한다(빈 성공 결론을 조용히 내지 않는다)."""
    out = answer_with_verification(
        "저per 전략 백테스트", conn=None, llm_fn=None,
        steps=_hallucinated_combine_steps(),
    )
    assert out["uncertain"] is True
    assert out["routes"] == ["backtest"]
    assert "존재하지 않는 필드" in out["domain_results"]["backtest"]["error"]
