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


def test_resolve_metric_exposes_both_sources_when_both_have_value(tmp_path):
    # 둘 다 있을 때 DART를 대표값(value/source)으로 쓰되, FnGuide 값도 dart_value/
    # fnguide_value로 함께 노출해야 한다("DART는 얼마, FnGuide는 얼마" 병기).
    db = str(tmp_path / "both2.db")
    init_db(db)
    _seed_two_sources(db)

    conn = connect(db)
    try:
        res = resolve_metric(conn, "005930", "revenue")
    finally:
        conn.close()

    assert res["dart_value"] == 1000.0
    assert res["dart_period"] == "2025Q1"
    assert res["fnguide_value"] == 2000.0
    assert res["fnguide_period"] == "2025-03-31"


def test_resolve_metric_other_source_is_none_when_only_one_has_value(tmp_path):
    # FnGuide 전용 지표(target_price)는 DART 쪽 값이 없으니 dart_value는 None이어야 한다.
    db = str(tmp_path / "onlyone.db")
    init_db(db)
    _seed_two_sources(db)

    conn = connect(db)
    try:
        res = resolve_metric(conn, "005930", "target_price")
    finally:
        conn.close()

    assert res["dart_value"] is None
    assert res["dart_period"] is None
    assert res["fnguide_value"] == 75000.0
    assert res["fnguide_period"] == "2025-07-11"


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


# ---------- 즉석 계산 폴백 — 버그: operating_margin 같은 단순 흐름비율이 metrics
# 사전계산 스냅샷에 없는 과거 분기에서 null로 나온다(실서버 재현: SK하이닉스 24/25년
# 영업이익률 둘 다 null. 원본 EAV엔 operating_profit/revenue가 정상 존재하는데
# metrics 테이블엔 최신 한 분기 스냅샷만 있어 과거 분기를 못 덮음). 분자·분모가 모두
# _SUMMABLE_FLOW_ACCOUNTS인 단순 비율(operating_margin/net_margin/gross_margin/cogs_ratio)은
# metrics 조회 실패 시 financials EAV 두 계정으로 즉석 계산해야 한다.
# 우선순위: metrics 테이블에 그 분기 값이 있으면 그걸 최우선, 없을 때만 폴백. ----------

def _seed_eav(conn, code, quarter, account_key, amount):
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
        "VALUES(?,?,?,?,?)",
        (code, quarter, "2025-05-15", account_key, amount),
    )


def test_resolve_metric_operating_margin_quarter_falls_back_to_eav(tmp_path):
    # metrics 행이 없어도 그 분기 operating_profit/revenue(EAV)로 즉석 계산한다.
    db = str(tmp_path / "om_q.db")
    init_db(db)
    conn = connect(db)
    try:
        seed_kr_companies(conn, ["000660"])
        _seed_eav(conn, "000660", "2024Q4", "operating_profit", 8.0e12)
        _seed_eav(conn, "000660", "2024Q4", "revenue", 20.0e12)
        conn.commit()
        res = resolve_metric(
            conn, "000660", "operating_margin",
            period={"kind": "quarter", "quarter": "2024Q4"},
        )
    finally:
        conn.close()

    assert res["value"] == 8.0e12 / 20.0e12 * 100.0  # 40.0(%)
    assert res["source"] == "DART"
    assert res["period"] == "2024Q4"


def test_resolve_metric_operating_margin_annual_uses_summed_ratio(tmp_path):
    # 연간 비율 = 분자 4분기 합 ÷ 분모 4분기 합(분기별 비율 평균이 아님).
    db = str(tmp_path / "om_a.db")
    init_db(db)
    conn = connect(db)
    try:
        seed_kr_companies(conn, ["000660"])
        for q, op, rev in [
            ("2024Q1", 1.0e12, 5.0e12), ("2024Q2", 2.0e12, 5.0e12),
            ("2024Q3", 3.0e12, 5.0e12), ("2024Q4", 4.0e12, 5.0e12),
        ]:
            _seed_eav(conn, "000660", q, "operating_profit", op)
            _seed_eav(conn, "000660", q, "revenue", rev)
        conn.commit()
        res = resolve_metric(
            conn, "000660", "operating_margin",
            period={"kind": "annual", "year": 2024},
        )
    finally:
        conn.close()

    assert res["value"] == (1.0e12 + 2.0e12 + 3.0e12 + 4.0e12) / (5.0e12 * 4) * 100.0  # 50.0
    assert res["source"] == "DART"
    assert "2024" in str(res["period"]) and "연간" in str(res["period"])


def test_resolve_metric_operating_margin_annual_none_when_quarter_missing(tmp_path):
    # SoT(추정 금지): 4개 분기 중 하나라도 없으면 연간 비율을 계산하지 않는다.
    db = str(tmp_path / "om_partial.db")
    init_db(db)
    conn = connect(db)
    try:
        seed_kr_companies(conn, ["000660"])
        for q in ("2024Q1", "2024Q2", "2024Q3"):  # 2024Q4 누락
            _seed_eav(conn, "000660", q, "operating_profit", 2.0e12)
            _seed_eav(conn, "000660", q, "revenue", 5.0e12)
        conn.commit()
        res = resolve_metric(
            conn, "000660", "operating_margin",
            period={"kind": "annual", "year": 2024},
        )
    finally:
        conn.close()

    assert res["value"] is None


def test_resolve_metric_operating_margin_quarter_none_when_denominator_missing(tmp_path):
    # 분모(매출)가 없으면 0/None 나눗셈 금지 → None.
    db = str(tmp_path / "om_norev.db")
    init_db(db)
    conn = connect(db)
    try:
        seed_kr_companies(conn, ["000660"])
        _seed_eav(conn, "000660", "2024Q4", "operating_profit", 8.0e12)  # revenue 없음
        conn.commit()
        res = resolve_metric(
            conn, "000660", "operating_margin",
            period={"kind": "quarter", "quarter": "2024Q4"},
        )
    finally:
        conn.close()

    assert res["value"] is None


def test_resolve_metric_operating_margin_prefers_metrics_row_over_eav_fallback(tmp_path):
    # metrics 테이블에 그 분기 값이 있으면 즉석계산이 덮어쓰면 안 된다(사전계산값 최우선).
    db = str(tmp_path / "om_pref.db")
    init_db(db)
    conn = connect(db)
    try:
        seed_kr_companies(conn, ["000660"])
        _seed_eav(conn, "000660", "2026Q1", "operating_profit", 8.0e12)  # EAV로는 40%
        _seed_eav(conn, "000660", "2026Q1", "revenue", 20.0e12)
        conn.execute(  # metrics 사전계산값 33.3 — 이게 우선돼야 한다
            "INSERT INTO metrics(stock_code, quarter, price_date, operating_margin) VALUES(?,?,?,?)",
            ("000660", "2026Q1", "2026-04-01", 33.3),
        )
        conn.commit()
        res = resolve_metric(
            conn, "000660", "operating_margin",
            period={"kind": "quarter", "quarter": "2026Q1"},
        )
    finally:
        conn.close()

    assert res["value"] == 33.3  # 즉석계산 40%가 아니라 metrics 사전계산값


def test_resolve_metric_operating_margin_no_period_uses_metrics_snapshot(tmp_path):
    # 회귀: 기간 미지정이면 기존처럼 metrics 최신 스냅샷을 쓴다(폴백은 과거 분기 조회 전용).
    db = str(tmp_path / "om_none.db")
    init_db(db)
    conn = connect(db)
    try:
        seed_kr_companies(conn, ["000660"])
        conn.execute(
            "INSERT INTO metrics(stock_code, quarter, price_date, operating_margin) VALUES(?,?,?,?)",
            ("000660", "2026Q1", "2026-04-01", 27.5),
        )
        conn.commit()
        res = resolve_metric(conn, "000660", "operating_margin")
    finally:
        conn.close()

    assert res["value"] == 27.5
    assert res["period"] == "2026Q1"


def test_resolve_metric_per_not_affected_by_flow_ratio_fallback(tmp_path):
    # 회귀: PER은 이번 폴백 대상이 아니다 — metrics 행이 없으면 EAV가 있어도 여전히 None.
    db = str(tmp_path / "per.db")
    init_db(db)
    conn = connect(db)
    try:
        seed_kr_companies(conn, ["000660"])
        _seed_eav(conn, "000660", "2024Q4", "operating_profit", 8.0e12)
        _seed_eav(conn, "000660", "2024Q4", "revenue", 20.0e12)
        conn.commit()
        res = resolve_metric(
            conn, "000660", "per",
            period={"kind": "quarter", "quarter": "2024Q4"},
        )
    finally:
        conn.close()

    assert res["value"] is None


def test_resolve_metric_net_gross_cogs_ratios_fall_back_to_eav(tmp_path):
    # net_margin/gross_margin/cogs_ratio도 동일하게 EAV 두 계정으로 즉석 계산한다.
    db = str(tmp_path / "ratios.db")
    init_db(db)
    conn = connect(db)
    try:
        seed_kr_companies(conn, ["000660"])
        _seed_eav(conn, "000660", "2024Q4", "revenue", 100.0)
        _seed_eav(conn, "000660", "2024Q4", "net_income", 12.0)
        _seed_eav(conn, "000660", "2024Q4", "gross_profit", 30.0)
        _seed_eav(conn, "000660", "2024Q4", "cost_of_sales", 70.0)
        conn.commit()
        q = {"kind": "quarter", "quarter": "2024Q4"}
        nm = resolve_metric(conn, "000660", "net_margin", period=q)
        gm = resolve_metric(conn, "000660", "gross_margin", period=q)
        cr = resolve_metric(conn, "000660", "cogs_ratio", period=q)
    finally:
        conn.close()

    assert nm["value"] == 12.0 / 100.0 * 100.0
    assert gm["value"] == 30.0 / 100.0 * 100.0
    assert cr["value"] == 70.0 / 100.0 * 100.0
    assert nm["source"] == gm["source"] == cr["source"] == "DART"


# ── 스크리닝 UI에 노출된 나머지 지표(ROA/GP_A/이자보상배율/유동비율/성장률 3종)도 가격이
#    필요 없는 순수 재무비율이라 operating_margin과 동일하게 EAV 직접 quarter 매치로 옮겼다
#    (metrics_at의 TTM/스냅샷 계산식과 동일 정의를 EAV에서 재현 — _RATIO_TTM_ACCOUNTS/
#    _YOY_GROWTH_ACCOUNTS). ──────────────────────────────────────────────────────

def test_resolve_metric_roa_gp_a_use_ttm_numerator_over_snapshot_assets(tmp_path):
    db = str(tmp_path / "roa.db")
    init_db(db)
    conn = connect(db)
    try:
        seed_kr_companies(conn, ["000660"])
        for q, ni, gp in [
            ("2025Q2", 1.0e12, 2.0e12), ("2025Q3", 1.0e12, 2.0e12),
            ("2025Q4", 1.0e12, 2.0e12), ("2026Q1", 1.0e12, 2.0e12),
        ]:
            _seed_eav(conn, "000660", q, "net_income", ni)
            _seed_eav(conn, "000660", q, "gross_profit", gp)
        _seed_eav(conn, "000660", "2026Q1", "total_assets", 40.0e12)  # 스냅샷(그 분기만)
        conn.commit()
        q = {"kind": "quarter", "quarter": "2026Q1"}
        roa = resolve_metric(conn, "000660", "roa", period=q)
        gpa = resolve_metric(conn, "000660", "gp_a", period=q)
    finally:
        conn.close()

    assert roa["value"] == 4.0e12 / 40.0e12 * 100.0  # ni TTM(4x1.0e12)÷총자산
    assert gpa["value"] == 8.0e12 / 40.0e12 * 100.0  # gp TTM(4x2.0e12)÷총자산
    assert roa["source"] == gpa["source"] == "DART"
    assert roa["period"] == gpa["period"] == "2026Q1"


def test_resolve_metric_roa_none_when_any_trailing_quarter_missing(tmp_path):
    # SoT: TTM 4분기 중 하나라도 없으면 억지 추정하지 않고 None.
    db = str(tmp_path / "roa_partial.db")
    init_db(db)
    conn = connect(db)
    try:
        seed_kr_companies(conn, ["000660"])
        for q in ("2025Q3", "2025Q4", "2026Q1"):  # 2025Q2 누락
            _seed_eav(conn, "000660", q, "net_income", 1.0e12)
        _seed_eav(conn, "000660", "2026Q1", "total_assets", 40.0e12)
        conn.commit()
        res = resolve_metric(
            conn, "000660", "roa", period={"kind": "quarter", "quarter": "2026Q1"},
        )
    finally:
        conn.close()

    assert res["value"] is None


def test_resolve_metric_interest_coverage_and_current_ratio(tmp_path):
    db = str(tmp_path / "stability.db")
    init_db(db)
    conn = connect(db)
    try:
        seed_kr_companies(conn, ["000660"])
        for q in ("2025Q2", "2025Q3", "2025Q4", "2026Q1"):
            _seed_eav(conn, "000660", q, "operating_profit", 3.0e12)
            _seed_eav(conn, "000660", q, "interest_expense", 0.5e12)
        _seed_eav(conn, "000660", "2026Q1", "current_assets", 20.0e12)
        _seed_eav(conn, "000660", "2026Q1", "current_liabilities", 10.0e12)
        conn.commit()
        q = {"kind": "quarter", "quarter": "2026Q1"}
        ic = resolve_metric(conn, "000660", "interest_coverage", period=q)
        cr = resolve_metric(conn, "000660", "current_ratio", period=q)
    finally:
        conn.close()

    assert ic["value"] == (3.0e12 * 4) / (0.5e12 * 4)  # TTM÷TTM, 배율(퍼센트 아님)
    assert cr["value"] == 20.0e12 / 10.0e12 * 100.0  # 스냅샷÷스냅샷(%)


def test_resolve_metric_revenue_op_ni_growth_yoy(tmp_path):
    db = str(tmp_path / "growth.db")
    init_db(db)
    conn = connect(db)
    try:
        seed_kr_companies(conn, ["000660"])
        _seed_eav(conn, "000660", "2025Q1", "revenue", 100.0)
        _seed_eav(conn, "000660", "2026Q1", "revenue", 150.0)
        _seed_eav(conn, "000660", "2025Q1", "operating_profit", 10.0)
        _seed_eav(conn, "000660", "2026Q1", "operating_profit", 20.0)
        _seed_eav(conn, "000660", "2025Q1", "net_income", 5.0)
        _seed_eav(conn, "000660", "2026Q1", "net_income", 4.0)
        conn.commit()
        q = {"kind": "quarter", "quarter": "2026Q1"}
        rg = resolve_metric(conn, "000660", "revenue_growth", period=q)
        og = resolve_metric(conn, "000660", "op_growth", period=q)
        ng = resolve_metric(conn, "000660", "ni_growth", period=q)
    finally:
        conn.close()

    assert rg["value"] == (150.0 - 100.0) / 100.0 * 100.0  # 50.0
    assert og["value"] == (20.0 - 10.0) / 10.0 * 100.0  # 100.0
    assert ng["value"] == (4.0 - 5.0) / 5.0 * 100.0  # -20.0(역성장도 유효)
    assert rg["period"] == og["period"] == ng["period"] == "2026Q1"


def test_resolve_metric_growth_none_when_prior_year_quarter_missing(tmp_path):
    db = str(tmp_path / "growth_missing.db")
    init_db(db)
    conn = connect(db)
    try:
        seed_kr_companies(conn, ["000660"])
        _seed_eav(conn, "000660", "2026Q1", "revenue", 150.0)  # 2025Q1 없음
        conn.commit()
        res = resolve_metric(
            conn, "000660", "revenue_growth", period={"kind": "quarter", "quarter": "2026Q1"},
        )
    finally:
        conn.close()

    assert res["value"] is None


# ---------- psr/pcr/ev_ebitda/peg — PER/PBR과 동일한 direct-WHERE(metrics 테이블) 조회 ----------
# metrics 테이블은 ingest 시점에 psr/pcr/ev_ebitda/peg를 이미 per/pbr과 나란히 계산해 저장한다
# (src/ingest/metrics.py). 그런데도 METRIC_SOURCE_MAP에 등록이 안 돼 있어 _COMPUTED_ONLY_FIELDS
# (asof 기반 cross-section 경로, look-ahead/분기 혼동 위험)로 잘못 분류돼 있었다. per/pbr과
# 완전히 동일한 등록(_METRICS_TABLE_COLS/_PRICE_BASED_METRICS/METRIC_SOURCE_MAP)만으로
# _fetch_dart의 기존 범용 로직이 그대로 적용되는지 확인한다(새 계산 로직 불필요).

def test_resolve_metric_psr_pcr_ev_ebitda_peg_route_to_dart_metrics_table(tmp_path):
    db = str(tmp_path / "valuation_ratios.db")
    init_db(db)
    conn = connect(db)
    try:
        seed_kr_companies(conn, ["000660"])
        conn.execute(
            "INSERT INTO metrics(stock_code, quarter, price_date, psr, pcr, ev_ebitda, peg) "
            "VALUES(?,?,?,?,?,?,?)",
            ("000660", "2026Q1", "2026-04-01", 1.5, 8.2, 6.7, 0.9),
        )
        conn.commit()
        psr = resolve_metric(conn, "000660", "psr", period={"kind": "quarter", "quarter": "2026Q1"})
        pcr = resolve_metric(conn, "000660", "pcr", period={"kind": "quarter", "quarter": "2026Q1"})
        ev = resolve_metric(conn, "000660", "ev_ebitda", period={"kind": "quarter", "quarter": "2026Q1"})
        peg = resolve_metric(conn, "000660", "peg", period={"kind": "quarter", "quarter": "2026Q1"})
    finally:
        conn.close()

    assert (psr["value"], psr["source"], psr["period"]) == (1.5, "DART", "2026Q1")
    assert (pcr["value"], pcr["source"], pcr["period"]) == (8.2, "DART", "2026Q1")
    assert (ev["value"], ev["source"], ev["period"]) == (6.7, "DART", "2026Q1")
    assert (peg["value"], peg["source"], peg["period"]) == (0.9, "DART", "2026Q1")
    # per/pbr처럼 주가 기반 지표라 그 값 계산에 쓰인 종가 기준일(price_date)도 함께 노출된다.
    assert psr["price_date"] == pcr["price_date"] == ev["price_date"] == peg["price_date"] == "2026-04-01"


def test_resolve_metric_psr_no_period_uses_metrics_snapshot(tmp_path):
    # 회귀: per/pbr과 동일하게 기간 미지정이면 metrics 최신 스냅샷(quarter DESC)을 쓴다.
    db = str(tmp_path / "psr_latest.db")
    init_db(db)
    conn = connect(db)
    try:
        seed_kr_companies(conn, ["000660"])
        conn.execute(
            "INSERT INTO metrics(stock_code, quarter, price_date, psr) VALUES(?,?,?,?)",
            ("000660", "2026Q1", "2026-04-01", 2.2),
        )
        conn.commit()
        res = resolve_metric(conn, "000660", "psr")
    finally:
        conn.close()

    assert res["value"] == 2.2
    assert res["period"] == "2026Q1"


def test_resolve_metric_psr_annual_uses_q4_snapshot(tmp_path):
    # per/pbr과 동일하게 annual 요청은 흐름값 합산이 아니라 연말(Q4) 스냅샷을 쓴다.
    db = str(tmp_path / "psr_annual.db")
    init_db(db)
    conn = connect(db)
    try:
        seed_kr_companies(conn, ["000660"])
        conn.execute(
            "INSERT INTO metrics(stock_code, quarter, price_date, psr) VALUES(?,?,?,?)",
            ("000660", "2025Q4", "2025-12-30", 1.8),
        )
        conn.commit()
        res = resolve_metric(conn, "000660", "psr", period={"kind": "annual", "year": 2025})
    finally:
        conn.close()

    assert res["value"] == 1.8
    assert res["period"] == "2025Q4"
