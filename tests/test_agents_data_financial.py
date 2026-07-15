"""재무데이터 에이전트(src/agents/data_financial.py) 단위 테스트.

이 에이전트는 질문에 나온 지표별로 값을 어디서 볼지(DART vs FnGuide) 판단한다.
- DART: 표준 재무제표 항목(financials 테이블) + DART 계산 지표(metrics 테이블).
- FnGuide: FnGuide 전용 지표(consensus_target_price/목표주가, 투자의견 등, fnguide_metrics 테이블).
매핑표에 없는 지표는 주입된 llm_fn 판단 경로로 위임한다.
DART·FnGuide 둘 다 값이 있으면 DART를 우선 채택하고, 반환 dict엔 항상 'source'가 담긴다.
"""
from __future__ import annotations

from src.agents.data_financial import (
    METRIC_SOURCE_MAP,
    classify_source,
    resolve_metric,
)
from src.db import connect, init_db
from tests.conftest import seed_kr_companies


# ---------- 규칙기반 라우팅 (매핑표에 있는 지표) ----------

def test_classify_source_maps_standard_financials_to_dart():
    assert classify_source("revenue") == "DART"
    assert classify_source("operating_profit") == "DART"
    assert classify_source("net_income") == "DART"
    assert classify_source("total_assets") == "DART"
    assert classify_source("total_liabilities") == "DART"
    assert classify_source("total_equity") == "DART"


def test_classify_source_maps_dart_computed_ratios_to_dart():
    assert classify_source("per") == "DART"
    assert classify_source("pbr") == "DART"
    assert classify_source("roe") == "DART"


def test_classify_source_maps_fnguide_only_metrics_to_fnguide():
    assert classify_source("target_price") == "FnGuide"
    assert classify_source("consensus_target_price") == "FnGuide"
    assert classify_source("analyst_opinion") == "FnGuide"
    assert classify_source("consensus_opinion_score") == "FnGuide"


def test_classify_source_is_case_and_whitespace_insensitive():
    assert classify_source("  PER  ") == "DART"
    assert classify_source("Target_Price") == "FnGuide"


def test_metric_source_map_is_exposed_for_callers():
    # HA-6/HA-7이 라우팅표를 직접 조회할 수 있어야 한다.
    assert METRIC_SOURCE_MAP["revenue"] == "DART"
    assert METRIC_SOURCE_MAP["target_price"] == "FnGuide"


# ---------- LLM 위임 (매핑표에 없는 지표) ----------

def test_classify_source_unknown_metric_delegates_to_llm():
    calls: list[str] = []

    def fake_llm(metric: str) -> str:
        calls.append(metric)
        return "FnGuide"

    result = classify_source("some_new_exotic_metric", llm_fn=fake_llm)
    assert calls == ["some_new_exotic_metric"]  # 규칙표에 없으니 LLM 경로로 감
    assert result == "FnGuide"


def test_classify_source_known_metric_does_not_call_llm():
    calls: list[str] = []

    def fake_llm(metric: str) -> str:
        calls.append(metric)
        return "FnGuide"

    result = classify_source("per", llm_fn=fake_llm)
    assert calls == []  # 규칙표에 있으니 LLM을 부르면 안 됨
    assert result == "DART"


def test_classify_source_normalizes_llm_return_value():
    # llm_fn이 소문자/여백 섞어 반환해도 정규화돼야 한다.
    assert classify_source("weird_metric", llm_fn=lambda m: "dart") == "DART"
    assert classify_source("weird_metric", llm_fn=lambda m: " fnguide ") == "FnGuide"


# ---------- 값 조회 + DART 우선 (rule #3) ----------

def _seed_two_sources(db_path: str, code: str = "005930") -> None:
    conn = connect(db_path)
    try:
        seed_kr_companies(conn, [code])
        # DART financials: 매출 1000
        conn.execute(
            "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, account_name, amount) "
            "VALUES(?,?,?,?,?,?)",
            (code, "2025Q1", "2025-05-15", "revenue", "매출액", 1000.0),
        )
        # FnGuide fnguide_metrics: 같은 지표 revenue 이지만 값이 다름(2000)
        conn.execute(
            "INSERT INTO fnguide_metrics(stock_code, as_of_date, metric_key, metric_value, source, collected_at) "
            "VALUES(?,?,?,?,?,?)",
            (code, "2025-03-31", "revenue", 2000.0, "fnguide", "2025-07-14T00:00:00"),
        )
        # FnGuide 전용: 컨센서스 목표주가
        conn.execute(
            "INSERT INTO fnguide_metrics(stock_code, as_of_date, metric_key, metric_value, source, collected_at) "
            "VALUES(?,?,?,?,?,?)",
            (code, "2025-07-11", "consensus_target_price", 75000.0, "fnguide", "2025-07-14T00:00:00"),
        )
        conn.commit()
    finally:
        conn.close()


def test_resolve_metric_prefers_dart_when_both_sources_have_value(tmp_path):
    db = str(tmp_path / "both.db")
    init_db(db)
    _seed_two_sources(db)

    conn = connect(db)
    try:
        res = resolve_metric(conn, "005930", "revenue")
    finally:
        conn.close()

    assert res["value"] == 1000.0  # DART 값 채택(FnGuide 2000이 아님)
    assert res["source"] == "DART"


def test_resolve_metric_returns_fnguide_only_metric(tmp_path):
    db = str(tmp_path / "fg.db")
    init_db(db)
    _seed_two_sources(db)

    conn = connect(db)
    try:
        res = resolve_metric(conn, "005930", "target_price")
    finally:
        conn.close()

    assert res["value"] == 75000.0
    assert res["source"] == "FnGuide"


def test_resolve_metric_always_has_source_on_rule_path(tmp_path):
    db = str(tmp_path / "rule.db")
    init_db(db)
    _seed_two_sources(db)

    conn = connect(db)
    try:
        res = resolve_metric(conn, "005930", "revenue")
    finally:
        conn.close()

    assert "source" in res
    assert res["source"] in ("DART", "FnGuide")


# ---------- 기간(period) 라벨 — 실사용 재현 버그: 값이 어느 분기/시점 것인지 안
# 돌려주면 총괄 에이전트(verify_answer)가 "26년 1분기"처럼 특정 기간을 지목한 질문의
# 검증을 절대 통과시키지 못해 매번 uncertain(3회 재시도 실패)으로 빠진다 ----------

def test_resolve_metric_includes_dart_quarter_when_dart_value_used(tmp_path):
    db = str(tmp_path / "period_dart.db")
    init_db(db)
    _seed_two_sources(db)

    conn = connect(db)
    try:
        res = resolve_metric(conn, "005930", "revenue")
    finally:
        conn.close()

    assert res["source"] == "DART"
    assert res["period"] == "2025Q1"  # _seed_two_sources가 심은 financials.quarter


def test_resolve_metric_includes_fnguide_as_of_date_when_fnguide_value_used(tmp_path):
    db = str(tmp_path / "period_fnguide.db")
    init_db(db)
    _seed_two_sources(db)

    conn = connect(db)
    try:
        res = resolve_metric(conn, "005930", "target_price")
    finally:
        conn.close()

    assert res["source"] == "FnGuide"
    assert res["period"] == "2025-07-11"  # _seed_two_sources가 심은 fnguide_metrics.as_of_date


def test_resolve_metric_period_is_none_when_no_value_found(tmp_path):
    db = str(tmp_path / "period_none.db")
    init_db(db)
    _seed_two_sources(db)

    conn = connect(db)
    try:
        res = resolve_metric(conn, "005930", "totally_unknown_metric", llm_fn=lambda m: "FnGuide")
    finally:
        conn.close()

    assert res["value"] is None
    assert res["period"] is None


def test_resolve_metric_always_has_source_on_llm_path(tmp_path):
    # 매핑표에 없고 DB에도 값이 없는 지표 → LLM 판단 소스를 source에 담아 반환.
    db = str(tmp_path / "llm.db")
    init_db(db)
    _seed_two_sources(db)

    conn = connect(db)
    try:
        res = resolve_metric(
            conn, "005930", "totally_unknown_metric", llm_fn=lambda m: "FnGuide"
        )
    finally:
        conn.close()

    assert "source" in res
    assert res["source"] == "FnGuide"
    assert res["value"] is None


# ---------- 기간 지정(period) 조회 — 버그B: "25년 전체 영업이익"이 최신분기만 반환 ----------
# resolve_metric이 period 인자를 받으면 그 기간(연간=4분기 합 / 특정분기)으로 조회한다.
# period 미지정이면 기존과 동일하게 최신 분기 1건을 반환한다(회귀 금지).

def _seed_annual_flow(db_path: str, code: str = "005930") -> None:
    """2025 4개 분기 + 2026Q1 operating_profit(원화 절대값, 분기별 흐름값)을 심는다."""
    conn = connect(db_path)
    try:
        seed_kr_companies(conn, [code])
        rows = [
            ("2025Q1", 6.0e12), ("2025Q2", 4.0e12),
            ("2025Q3", 12.0e12), ("2025Q4", 20.0e12),
            ("2026Q1", 57.0e12),
        ]
        for quarter, amount in rows:
            conn.execute(
                "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
                "VALUES(?,?,?,?,?)",
                (code, quarter, "2025-05-15", "operating_profit", amount),
            )
        conn.commit()
    finally:
        conn.close()


def test_resolve_metric_annual_period_sums_four_quarters(tmp_path):
    db = str(tmp_path / "annual.db")
    init_db(db)
    _seed_annual_flow(db)

    conn = connect(db)
    try:
        res = resolve_metric(
            conn, "005930", "operating_profit",
            period={"kind": "annual", "year": 2025},
        )
    finally:
        conn.close()

    assert res["value"] == 6.0e12 + 4.0e12 + 12.0e12 + 20.0e12  # 2025 4분기 합
    assert res["source"] == "DART"
    assert "2025" in str(res["period"])
    assert "연간" in str(res["period"])


def test_resolve_metric_specific_quarter_period_returns_that_quarter(tmp_path):
    db = str(tmp_path / "q.db")
    init_db(db)
    _seed_annual_flow(db)

    conn = connect(db)
    try:
        res = resolve_metric(
            conn, "005930", "operating_profit",
            period={"kind": "quarter", "quarter": "2025Q3"},
        )
    finally:
        conn.close()

    assert res["value"] == 12.0e12
    assert res["period"] == "2025Q3"


def test_resolve_metric_no_period_still_returns_latest_quarter(tmp_path):
    # 회귀: period 미지정이면 기존처럼 가장 최신 분기(2026Q1)만 반환한다.
    db = str(tmp_path / "latest.db")
    init_db(db)
    _seed_annual_flow(db)

    conn = connect(db)
    try:
        res = resolve_metric(conn, "005930", "operating_profit")
    finally:
        conn.close()

    assert res["value"] == 57.0e12
    assert res["period"] == "2026Q1"


def test_resolve_metric_annual_period_none_when_a_quarter_missing(tmp_path):
    # SoT: 4개 분기 중 하나라도 없으면 추정하지 않고 None(연간값 제시 불가).
    db = str(tmp_path / "partial.db")
    init_db(db)
    conn = connect(db)
    try:
        seed_kr_companies(conn, ["005930"])
        for quarter in ("2024Q1", "2024Q2", "2024Q3"):  # 2024Q4 누락
            conn.execute(
                "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
                "VALUES(?,?,?,?,?)",
                ("005930", quarter, "2024-05-15", "operating_profit", 5.0e12),
            )
        conn.commit()
    finally:
        conn.close()

    conn = connect(db)
    try:
        res = resolve_metric(
            conn, "005930", "operating_profit",
            period={"kind": "annual", "year": 2024},
        )
    finally:
        conn.close()

    assert res["value"] is None
