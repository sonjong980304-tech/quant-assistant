"""프리미티브 v1(6종) 단위 테스트 (TDD).

.omc/specs/brainstorming-sql-python-primitive-pipeline.md 참고.
모두 순수 함수 — 네트워크/DB는 인자(conn/주입 fn)로만. 파마프렌치 테스트의 DI 관례를 따른다.
get_cross_section/zscore/neutralize/combine은 기존 metrics_at()/select_stocks()를 래핑하고,
regress/optimize_weights는 신규(단, optimize_weights는 Riskfolio-Lib solve만 감싼다).
"""
from __future__ import annotations

import math

import pytest

from src.backtest.primitives import (
    combine,
    compute_ic_primitive,
    get_cross_section,
    neutralize,
    optimize_weights,
    regress,
    run_backtest_primitive,
    winsorize,
    zscore,
)


# --------------------------------------------------------------------------
# get_cross_section — metrics_at() 래핑
# --------------------------------------------------------------------------
def _fake_rows():
    return [
        {"stock_code": "000001", "name": "가", "sector": "화학", "market": "KOSPI",
         "quarter": "2025Q1", "per": 8.0, "roe": 12.0},
        {"stock_code": "000002", "name": "나", "sector": "화학", "market": "KOSPI",
         "quarter": "2025Q1", "per": 15.0, "roe": 8.0},
        {"stock_code": "000003", "name": "다", "sector": "금융", "market": "KOSPI",
         "quarter": "2025Q1", "per": 5.0, "roe": 20.0},
    ]


def test_get_cross_section_wraps_metrics_at_and_passes_conn_asof():
    calls = []

    def fake_metrics(conn, asof):
        calls.append((conn, asof))
        return _fake_rows()

    rows = get_cross_section("CONN", "2025-12-31", metrics_fn=fake_metrics)
    assert calls == [("CONN", "2025-12-31")]
    assert len(rows) == 3


def test_get_cross_section_field_projection_keeps_identifiers():
    rows = get_cross_section("CONN", "2025-12-31", fields=["per"],
                             metrics_fn=lambda c, a: _fake_rows())
    # 요청 필드 + 식별자만 남는다
    assert set(rows[0].keys()) == {"stock_code", "name", "sector", "market", "quarter", "per"}
    assert "roe" not in rows[0]


# --------------------------------------------------------------------------
# zscore — select_stocks(단일 기준 zscore) 래핑
# --------------------------------------------------------------------------
def test_zscore_high_direction_orders_best_first_and_annotates_score():
    ranked = zscore(_fake_rows(), "roe", direction="high")
    # roe 높은 순: 다(20) > 가(12) > 나(8)
    assert [r["name"] for r in ranked] == ["다", "가", "나"]
    assert all("_score" in r for r in ranked)
    # 최우수(다)의 점수가 최하위(나)보다 작다(select_stocks: 낮을수록 우수)
    assert ranked[0]["_score"] < ranked[-1]["_score"]


def test_zscore_low_direction_orders_smallest_first():
    ranked = zscore(_fake_rows(), "per", direction="low")
    assert [r["name"] for r in ranked] == ["다", "가", "나"]  # per 낮은 순 5<8<15


# --------------------------------------------------------------------------
# neutralize — 그룹(섹터) 내 평균 제거
# --------------------------------------------------------------------------
def test_neutralize_demeans_within_sector():
    rows = [
        {"stock_code": "1", "sector": "화학", "roe": 10.0},
        {"stock_code": "2", "sector": "화학", "roe": 20.0},
        {"stock_code": "3", "sector": "금융", "roe": 30.0},
    ]
    out = neutralize(rows, "roe", by="sector")
    by_code = {r["stock_code"]: r for r in out}
    # 화학 평균 15 → -5, +5 ; 금융 평균 30 → 0
    assert by_code["1"]["roe_neutral"] == pytest.approx(-5.0)
    assert by_code["2"]["roe_neutral"] == pytest.approx(5.0)
    assert by_code["3"]["roe_neutral"] == pytest.approx(0.0)


def test_neutralize_leaves_none_for_missing_field():
    rows = [{"stock_code": "1", "sector": "화학", "roe": None},
            {"stock_code": "2", "sector": "화학", "roe": 10.0}]
    out = neutralize(rows, "roe", by="sector")
    assert {r["stock_code"]: r["roe_neutral"] for r in out}["1"] is None


# --------------------------------------------------------------------------
# winsorize — IQR(사분위범위) 기반 이상치 완화(신규, 순위변환·z-score 외 3번째 방식)
# --------------------------------------------------------------------------
def test_winsorize_clips_high_outlier_to_upper_iqr_bound():
    rows = [
        {"stock_code": "1", "roe": 10.0}, {"stock_code": "2", "roe": 11.0},
        {"stock_code": "3", "roe": 12.0}, {"stock_code": "4", "roe": 13.0},
        {"stock_code": "5", "roe": 100.0},  # 극단치
    ]
    out = winsorize(rows, "roe")
    by_code = {r["stock_code"]: r["roe_winsorized"] for r in out}
    # Q1=11, Q3=13, IQR=2, upper=13+1.5*2=16 → 100은 16으로 눌림
    assert by_code["5"] == pytest.approx(16.0)
    assert by_code["1"] == pytest.approx(10.0)  # 경계 안쪽 값은 그대로


def test_winsorize_clips_low_outlier_to_lower_iqr_bound():
    rows = [
        {"stock_code": "1", "roe": 1.0},  # 극단치
        {"stock_code": "2", "roe": 10.0}, {"stock_code": "3", "roe": 11.0},
        {"stock_code": "4", "roe": 12.0}, {"stock_code": "5", "roe": 13.0},
    ]
    out = winsorize(rows, "roe")
    by_code = {r["stock_code"]: r["roe_winsorized"] for r in out}
    # Q1=10, Q3=12, IQR=2, lower=10-1.5*2=7 → 1은 7로 눌림
    assert by_code["1"] == pytest.approx(7.0)
    assert by_code["5"] == pytest.approx(13.0)


def test_winsorize_preserves_original_field_untouched():
    rows = [{"stock_code": "1", "roe": 10.0}, {"stock_code": "2", "roe": 11.0},
            {"stock_code": "3", "roe": 12.0}, {"stock_code": "4", "roe": 100.0}]
    out = winsorize(rows, "roe")
    by_code = {r["stock_code"]: r["roe"] for r in out}
    assert by_code["4"] == 100.0  # 원본 field는 그대로, 눌린 값은 별도 필드에만


def test_winsorize_leaves_none_as_none_and_excludes_from_quartiles():
    rows = [{"stock_code": "1", "roe": None}, {"stock_code": "2", "roe": 10.0},
            {"stock_code": "3", "roe": 11.0}, {"stock_code": "4", "roe": 12.0},
            {"stock_code": "5", "roe": 13.0}]
    out = winsorize(rows, "roe")
    by_code = {r["stock_code"]: r["roe_winsorized"] for r in out}
    assert by_code["1"] is None
    assert by_code["2"] == pytest.approx(10.0)  # None 제외하고 사분위 계산되어 경계 안쪽 유지


def test_winsorize_passes_through_when_too_few_samples_for_quartiles():
    rows = [{"stock_code": "1", "roe": 5.0}, {"stock_code": "2", "roe": 1000.0}]
    out = winsorize(rows, "roe")
    by_code = {r["stock_code"]: r["roe_winsorized"] for r in out}
    assert by_code["1"] == 5.0
    assert by_code["2"] == 1000.0  # 표본 4개 미만이면 경계를 정하지 않고 원본 그대로


# --------------------------------------------------------------------------
# combine — select_stocks(멀티팩터 가중조합) 래핑
# --------------------------------------------------------------------------
def test_combine_wraps_select_stocks_multifactor():
    criteria = [{"key": "per", "direction": "low", "weight": 0.5},
                {"key": "roe", "direction": "high", "weight": 0.5}]
    picked = combine(_fake_rows(), criteria, method="zscore", n=2)
    assert len(picked) == 2
    assert picked[0]["name"] == "다"  # per 최저 + roe 최고 → 종합 1위
    assert all("_score" in r for r in picked)


# --------------------------------------------------------------------------
# regress — 신규 순수 OLS (K-ratio용 기울기/표준오차)
# --------------------------------------------------------------------------
def test_regress_matches_manual_ols():
    import numpy as np

    y = [1.0, 2.1, 2.9, 4.2, 5.0, 5.8]
    res = regress(y)  # x 미지정 → 0..n-1
    x = np.arange(len(y), dtype=float)
    slope_np, intercept_np = np.polyfit(x, y, 1)
    assert res["slope"] == pytest.approx(slope_np, rel=1e-6)
    assert res["intercept"] == pytest.approx(intercept_np, rel=1e-6)
    assert res["n"] == 6
    assert res["se_slope"] > 0
    assert 0.0 <= res["r_squared"] <= 1.0
    # k_ratio = slope / se_slope
    assert res["k_ratio"] == pytest.approx(res["slope"] / res["se_slope"], rel=1e-9)


def test_regress_accepts_explicit_x():
    res = regress([2.0, 4.0, 6.0, 8.0], x=[1.0, 2.0, 3.0, 4.0])
    assert res["slope"] == pytest.approx(2.0, rel=1e-9)
    assert res["intercept"] == pytest.approx(0.0, abs=1e-9)


def test_regress_too_few_points_raises():
    with pytest.raises(ValueError):
        regress([1.0, 2.0])


def test_regress_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        regress([1.0, 2.0, 3.0], x=[1.0, 2.0])


# --------------------------------------------------------------------------
# optimize_weights — Riskfolio-Lib solve만 감싼 래퍼 (3종 메서드만)
# --------------------------------------------------------------------------
def test_optimize_weights_rejects_unknown_method():
    with pytest.raises(ValueError):
        optimize_weights({"A": [0.1, 0.2]}, method="kelly")


def test_optimize_weights_maps_solve_output_to_asset_dict():
    captured = {}

    def fake_solve(matrix, method, rf):
        captured["matrix"] = matrix
        captured["method"] = method
        return [0.6, 0.4]

    w = optimize_weights({"AAA": [0.01, 0.02, -0.01], "BBB": [0.0, 0.01, 0.02]},
                         method="max_sharpe", solve_fn=fake_solve)
    assert w == {"AAA": 0.6, "BBB": 0.4}
    assert captured["method"] == "max_sharpe"
    # dict 입력이 (행=기간, 열=자산) 행렬로 전치됐는지
    assert captured["matrix"] == [[0.01, 0.0], [0.02, 0.01], [-0.01, 0.02]]


def test_optimize_weights_list_input_uses_assets_names():
    def fake_solve(matrix, method, rf):
        return [0.5, 0.5]

    w = optimize_weights([[0.01, 0.0], [0.02, 0.01]], method="min_variance",
                         assets=["X", "Y"], solve_fn=fake_solve)
    assert w == {"X": 0.5, "Y": 0.5}


def test_optimize_weights_all_three_methods_accepted():
    for m in ("max_sharpe", "min_variance", "risk_parity"):
        w = optimize_weights({"A": [0.01, 0.02], "B": [0.0, 0.01]},
                             method=m, solve_fn=lambda mat, meth, rf: [0.5, 0.5])
        assert math.isclose(sum(w.values()), 1.0)


# --------------------------------------------------------------------------
# 7. run_backtest_primitive — engine.run_backtest()를 파이프라인에서 호출 가능하게 래핑
# --------------------------------------------------------------------------
def test_run_backtest_primitive_passes_weights_through_to_engine():
    captured = {}

    def fake_backtest_fn(dates, metrics_fn, price_fn, params, benchmark_fn=None, weights=None):
        captured["dates"] = dates
        captured["weights"] = weights
        captured["benchmark_fn"] = benchmark_fn
        return {"dates": dates, "navs": [1.0, 1.1], "benchmark": None, "performance": {"cagr": 0.1}}

    result = run_backtest_primitive(
        "CONN", start_year=2026, end_year=2026, weights={"AAA": 0.6, "BBB": 0.4},
        dates_fn=lambda sy, ey, freq: ["2026-01-31", "2026-02-28"],
        max_date_fn=lambda conn: "2026-12-31",
        callbacks_fn=lambda conn: (lambda d: [], lambda d, c: None),
        backtest_fn=fake_backtest_fn,
    )
    assert captured["weights"] == {"AAA": 0.6, "BBB": 0.4}
    assert captured["dates"] == ["2026-01-31", "2026-02-28"]
    assert result["navs"] == [1.0, 1.1]


def test_run_backtest_primitive_criteria_mode_when_no_weights():
    captured = {}

    def fake_backtest_fn(dates, metrics_fn, price_fn, params, benchmark_fn=None, weights=None):
        captured["params"] = params
        captured["weights"] = weights
        return {"dates": dates, "navs": [1.0], "benchmark": None, "performance": {}}

    run_backtest_primitive(
        "CONN", start_year=2026, end_year=2026, criteria=[{"field": "per", "direction": "low"}],
        n=10, dates_fn=lambda sy, ey, freq: ["2026-01-31", "2026-02-28"],
        max_date_fn=lambda conn: "2026-12-31",
        callbacks_fn=lambda conn: (lambda d: [], lambda d, c: None),
        backtest_fn=fake_backtest_fn,
    )
    assert captured["weights"] is None
    assert captured["params"]["criteria"] == [{"field": "per", "direction": "low"}]
    assert captured["params"]["n"] == 10


def test_run_backtest_primitive_filters_dates_beyond_max_price_date():
    captured = {}

    def fake_backtest_fn(dates, metrics_fn, price_fn, params, benchmark_fn=None, weights=None):
        captured["dates"] = dates
        return {"dates": dates, "navs": [1.0], "benchmark": None, "performance": {}}

    run_backtest_primitive(
        "CONN", start_year=2026, end_year=2026, weights={"AAA": 1.0},
        dates_fn=lambda sy, ey, freq: ["2026-01-31", "2026-02-28", "2026-12-31"],
        max_date_fn=lambda conn: "2026-02-28",  # 마지막 리밸런싱일은 주가 범위 밖 → 제외
        callbacks_fn=lambda conn: (lambda d: [], lambda d, c: None),
        backtest_fn=fake_backtest_fn,
    )
    assert captured["dates"] == ["2026-01-31", "2026-02-28"]


def test_run_backtest_primitive_raises_when_too_many_dates_before_expensive_calls():
    """window 상한(4000) 초과면 callbacks_fn/benchmark_fn_factory(무거운 연산) 호출 전에 거부한다.

    architect 재검수 지적: start_year/end_year는 정수라 pipeline_exec._check_size가
    이 window를 못 봄(_sizeof(int)=None) — 프리미티브 자체가 연산 시작 '전'에 상한을 강제해야 함.
    """
    expensive_called = {"callbacks": False, "benchmark": False}

    def spy_callbacks(conn):
        expensive_called["callbacks"] = True
        return (lambda d: [], lambda d, c: None)

    def spy_benchmark(dates, mfn, pfn):
        expensive_called["benchmark"] = True
        return None

    with pytest.raises(ValueError):
        run_backtest_primitive(
            "CONN", start_year=1900, end_year=9999, weights={"AAA": 1.0},
            dates_fn=lambda sy, ey, freq: [f"2026-{i:04d}-01" for i in range(4001)],  # 상한 초과
            max_date_fn=lambda conn: None,
            callbacks_fn=spy_callbacks,
            benchmark_fn_factory=spy_benchmark,
            backtest_fn=lambda *a, **kw: {},
        )
    assert expensive_called == {"callbacks": False, "benchmark": False}  # 무거운 연산 전에 거부


def test_run_backtest_primitive_rejects_start_year_before_1990():
    with pytest.raises(ValueError):
        run_backtest_primitive(
            "CONN", start_year=1900, end_year=2026, weights={"AAA": 1.0},
            dates_fn=lambda sy, ey, freq: ["2026-01-31", "2026-02-28"],
            max_date_fn=lambda conn: "2026-12-31",
            callbacks_fn=lambda conn: (lambda d: [], lambda d, c: None),
            backtest_fn=lambda *a, **kw: {},
        )


def test_run_backtest_primitive_rejects_huge_year_span_before_calling_dates_fn():
    """end_year가 병적으로 크면 dates_fn(리스트 생성) 자체를 호출하기 전에 거부한다(LOW, architect 3차 권고)."""
    dates_fn_called = {"n": 0}

    def counting_dates_fn(sy, ey, freq):
        dates_fn_called["n"] += 1
        return []

    with pytest.raises(ValueError):
        run_backtest_primitive(
            "CONN", start_year=2026, end_year=10**9, weights={"AAA": 1.0},
            dates_fn=counting_dates_fn,
            max_date_fn=lambda conn: "2026-12-31",
            callbacks_fn=lambda conn: (lambda d: [], lambda d, c: None),
            backtest_fn=lambda *a, **kw: {},
        )
    assert dates_fn_called["n"] == 0


def test_run_backtest_primitive_rejects_unknown_rebalance_frequency():
    with pytest.raises(ValueError):
        run_backtest_primitive(
            "CONN", start_year=2026, end_year=2026, weights={"AAA": 1.0}, rebalance="daily",
            dates_fn=lambda sy, ey, freq: ["2026-01-31", "2026-02-28"],
            max_date_fn=lambda conn: "2026-12-31",
            callbacks_fn=lambda conn: (lambda d: [], lambda d, c: None),
            backtest_fn=lambda *a, **kw: {},
        )


def test_run_backtest_primitive_raises_when_too_few_dates():
    with pytest.raises(ValueError):
        run_backtest_primitive(
            "CONN", start_year=2026, end_year=2026, weights={"AAA": 1.0},
            dates_fn=lambda sy, ey, freq: ["2026-01-31"],  # 1개뿐 — 시뮬레이션 불가
            max_date_fn=lambda conn: "2026-12-31",
            callbacks_fn=lambda conn: (lambda d: [], lambda d, c: None),
            backtest_fn=lambda *a, **kw: {},
        )


def test_run_backtest_primitive_skips_benchmark_when_disabled():
    captured = {}

    def fake_backtest_fn(dates, metrics_fn, price_fn, params, benchmark_fn=None, weights=None):
        captured["benchmark_fn"] = benchmark_fn
        return {"dates": dates, "navs": [1.0], "benchmark": None, "performance": {}}

    run_backtest_primitive(
        "CONN", start_year=2026, end_year=2026, weights={"AAA": 1.0}, with_benchmark=False,
        dates_fn=lambda sy, ey, freq: ["2026-01-31", "2026-02-28"],
        max_date_fn=lambda conn: "2026-12-31",
        callbacks_fn=lambda conn: (lambda d: [], lambda d, c: None),
        benchmark_fn_factory=lambda dates, mfn, pfn: (_ for _ in ()).throw(
            AssertionError("with_benchmark=False면 호출되면 안 됨")),
        backtest_fn=fake_backtest_fn,
    )
    assert captured["benchmark_fn"] is None


# --------------------------------------------------------------------------
# 7b. run_backtest_primitive market="US" — 미국 콜백 + S&P500·유니버스 이중 벤치마크
# --------------------------------------------------------------------------
def test_run_backtest_primitive_us_uses_us_callbacks_and_dual_benchmark():
    used = {"callbacks": False, "sp500": False, "universe": False}

    def spy_callbacks(conn):
        used["callbacks"] = True
        return (lambda d: [], lambda d, c: None)

    def spy_sp500(dates, fetch_fn=None):
        used["sp500"] = True
        return lambda d: 1.0  # 평탄한 S&P500 레벨

    def spy_universe(dates, mfn, pfn):
        used["universe"] = True
        return lambda d: 1.0  # 평탄한 유니버스 레벨

    def fake_backtest_fn(dates, metrics_fn, price_fn, params, benchmark_fn=None, weights=None):
        # 엔진이 S&P500(메인) 벤치마크로 성과를 계산한 것처럼 흉내
        return {"dates": dates, "navs": [1.0, 1.2], "benchmark": [1.0, 1.0],
                "performance": {"cagr": 20.0, "benchmark_return": 0.0, "excess_return": 20.0, "beta": 0.0},
                "holdings": [{"date": dates[0], "codes": []}]}

    result = run_backtest_primitive(
        "CONN", start_year=2026, end_year=2026, market="US", weights={"AAPL": 1.0},
        dates_fn=lambda sy, ey, freq: ["2026-01-31", "2026-02-28"],
        max_date_fn=lambda conn: "2026-12-31",
        callbacks_fn=spy_callbacks,
        benchmark_fn_factory=spy_universe,
        sp500_fn_factory=spy_sp500,
        backtest_fn=fake_backtest_fn,
    )
    assert used == {"callbacks": True, "sp500": True, "universe": True}  # US 콜백+두 벤치마크 모두 사용
    # 두 벤치마크 레벨 시계열이 모두 결과에 포함
    assert "benchmark_sp500" in result and "benchmark_universe" in result
    # 성과에 S&P500(메인)과 유니버스(보조) 지표가 모두 존재
    perf = result["performance"]
    assert "benchmark_return" in perf   # S&P500(메인)
    assert "universe_return" in perf    # 동일가중 유니버스(보조)


def test_run_backtest_primitive_kr_default_has_no_us_benchmark_keys():
    """market 기본값(KR)이면 미국 이중벤치마크 키가 절대 생기지 않는다(하위호환·회귀 방지)."""
    def fake_backtest_fn(dates, metrics_fn, price_fn, params, benchmark_fn=None, weights=None):
        return {"dates": dates, "navs": [1.0, 1.1], "benchmark": None,
                "performance": {"cagr": 10.0}, "holdings": []}

    result = run_backtest_primitive(
        "CONN", start_year=2026, end_year=2026, weights={"AAAA": 1.0},
        dates_fn=lambda sy, ey, freq: ["2026-01-31", "2026-02-28"],
        max_date_fn=lambda conn: "2026-12-31",
        callbacks_fn=lambda conn: (lambda d: [], lambda d, c: None),
        backtest_fn=fake_backtest_fn,
    )
    assert "benchmark_sp500" not in result
    assert "benchmark_universe" not in result
    assert "universe_return" not in result["performance"]


# --------------------------------------------------------------------------
# 8. compute_ic_primitive — 팩터값 순위 vs 다음구간 수익률 순위의 Spearman IC
# --------------------------------------------------------------------------
def _fake_callbacks(rows_by_date: dict, prices_by_date_code: dict):
    """metrics_fn(asof)->rows, price_fn(asof, code)->price 형태의 테스트용 콜백 쌍."""
    def metrics_fn(asof):
        return rows_by_date.get(asof, [])

    def price_fn(asof, code):
        return prices_by_date_code.get((asof, code))

    return lambda conn: (metrics_fn, price_fn)


def test_compute_ic_primitive_perfect_positive_correlation_gives_ic_near_one():
    rows_by_date = {
        "2026-01-31": [
            {"stock_code": "A", "score": 1},
            {"stock_code": "B", "score": 2},
            {"stock_code": "C", "score": 3},
        ],
    }
    prices = {
        ("2026-01-31", "A"): 100, ("2026-02-28", "A"): 100,   # 0%   (최저 score → 최저 수익률)
        ("2026-01-31", "B"): 100, ("2026-02-28", "B"): 110,   # 10%
        ("2026-01-31", "C"): 100, ("2026-02-28", "C"): 120,   # 20%  (최고 score → 최고 수익률)
    }
    result = compute_ic_primitive(
        "CONN", start_year=2026, end_year=2026, field="score",
        dates_fn=lambda sy, ey, freq: ["2026-01-31", "2026-02-28"],
        max_date_fn=lambda conn: "2026-12-31",
        callbacks_fn=_fake_callbacks(rows_by_date, prices),
    )
    assert result["n"] == 1
    assert math.isclose(result["ic_series"][0], 1.0, abs_tol=1e-9)
    assert math.isclose(result["mean_ic"], 1.0, abs_tol=1e-9)


def test_compute_ic_primitive_perfect_negative_correlation_gives_ic_near_minus_one():
    rows_by_date = {
        "2026-01-31": [
            {"stock_code": "A", "score": 1},
            {"stock_code": "B", "score": 2},
            {"stock_code": "C", "score": 3},
        ],
    }
    prices = {
        ("2026-01-31", "A"): 100, ("2026-02-28", "A"): 120,   # 20%  (최저 score → 최고 수익률)
        ("2026-01-31", "B"): 100, ("2026-02-28", "B"): 110,   # 10%
        ("2026-01-31", "C"): 100, ("2026-02-28", "C"): 100,   # 0%   (최고 score → 최저 수익률)
    }
    result = compute_ic_primitive(
        "CONN", start_year=2026, end_year=2026, field="score",
        dates_fn=lambda sy, ey, freq: ["2026-01-31", "2026-02-28"],
        max_date_fn=lambda conn: "2026-12-31",
        callbacks_fn=_fake_callbacks(rows_by_date, prices),
    )
    assert math.isclose(result["ic_series"][0], -1.0, abs_tol=1e-9)


def test_compute_ic_primitive_returns_summary_stats_over_multiple_periods():
    rows_by_date = {
        "2026-01-31": [{"stock_code": c, "score": i} for i, c in enumerate(["A", "B", "C"], start=1)],
        "2026-02-28": [{"stock_code": c, "score": i} for i, c in enumerate(["A", "B", "C"], start=1)],
    }
    prices = {
        ("2026-01-31", "A"): 100, ("2026-02-28", "A"): 100,
        ("2026-01-31", "B"): 100, ("2026-02-28", "B"): 110,
        ("2026-01-31", "C"): 100, ("2026-02-28", "C"): 120,
        ("2026-03-31", "A"): 120, ("2026-03-31", "B"): 108, ("2026-03-31", "C"): 96,  # 이번엔 역상관
    }
    result = compute_ic_primitive(
        "CONN", start_year=2026, end_year=2026, field="score",
        dates_fn=lambda sy, ey, freq: ["2026-01-31", "2026-02-28", "2026-03-31"],
        max_date_fn=lambda conn: "2026-12-31",
        callbacks_fn=_fake_callbacks(rows_by_date, prices),
    )
    assert result["n"] == 2
    assert len(result["dates"]) == 2
    assert "mean_ic" in result and "ic_std" in result and "ir" in result and "hit_rate" in result


def test_compute_ic_primitive_skips_periods_with_fewer_than_three_samples():
    rows_by_date = {
        "2026-01-31": [
            {"stock_code": "A", "score": 1},
            {"stock_code": "B", "score": 2},
        ],  # 2개뿐 — 상관계수 계산에 부족해 스킵되어야 함
    }
    prices = {
        ("2026-01-31", "A"): 100, ("2026-02-28", "A"): 100,
        ("2026-01-31", "B"): 100, ("2026-02-28", "B"): 110,
    }
    with pytest.raises(ValueError):
        compute_ic_primitive(
            "CONN", start_year=2026, end_year=2026, field="score",
            dates_fn=lambda sy, ey, freq: ["2026-01-31", "2026-02-28"],
            max_date_fn=lambda conn: "2026-12-31",
            callbacks_fn=_fake_callbacks(rows_by_date, prices),
        )


def test_compute_ic_primitive_rejects_start_year_before_1990():
    with pytest.raises(ValueError):
        compute_ic_primitive(
            "CONN", start_year=1989, end_year=2026, field="score",
            dates_fn=lambda sy, ey, freq: ["2026-01-31", "2026-02-28"],
            max_date_fn=lambda conn: "2026-12-31",
            callbacks_fn=_fake_callbacks({}, {}),
        )


def test_compute_ic_primitive_rejects_huge_year_span_before_calling_dates_fn():
    dates_fn_called = {"n": 0}

    def counting_dates_fn(sy, ey, freq):
        dates_fn_called["n"] += 1
        return []

    with pytest.raises(ValueError):
        compute_ic_primitive(
            "CONN", start_year=2000, end_year=999999, field="score",
            dates_fn=counting_dates_fn,
            max_date_fn=lambda conn: "2026-12-31",
            callbacks_fn=_fake_callbacks({}, {}),
        )
    assert dates_fn_called["n"] == 0


def test_compute_ic_primitive_filters_dates_beyond_max_price_date():
    rows_by_date = {
        "2026-01-31": [{"stock_code": c, "score": i} for i, c in enumerate(["A", "B", "C"], start=1)],
    }
    prices = {
        ("2026-01-31", "A"): 100, ("2026-02-28", "A"): 100,
        ("2026-01-31", "B"): 100, ("2026-02-28", "B"): 110,
        ("2026-01-31", "C"): 100, ("2026-02-28", "C"): 120,
    }
    result = compute_ic_primitive(
        "CONN", start_year=2026, end_year=2026, field="score",
        dates_fn=lambda sy, ey, freq: ["2026-01-31", "2026-02-28", "2026-12-31"],
        max_date_fn=lambda conn: "2026-02-28",  # 마지막 리밸런싱일은 주가 범위 밖 → 제외
        callbacks_fn=_fake_callbacks(rows_by_date, prices),
    )
    assert result["dates"] == ["2026-01-31"]


def test_compute_ic_primitive_rejects_unknown_rebalance_frequency():
    with pytest.raises(ValueError):
        compute_ic_primitive(
            "CONN", start_year=2026, end_year=2026, field="score", rebalance="daily",
            dates_fn=lambda sy, ey, freq: ["2026-01-31", "2026-02-28"],
            max_date_fn=lambda conn: "2026-12-31",
            callbacks_fn=_fake_callbacks({}, {}),
        )
