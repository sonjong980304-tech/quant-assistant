"""QVM(Quality-Value-Momentum) 멀티팩터 스크리너/백테스트 단위·통합 테스트 (TDD).

사용자 확정 QVM 파이프라인(브리핑)을 검증한다:
  invert(가치 역수) → winsorize_pct(1%/99%) → 섹터 z-score(<5표본 전체폴백)
  → 카테고리 합성(등가중, 결측 제외) → 결측필터(raw 7개 중 3개 이상 결측 제외)
  → 2차 z-score → 최종점수(z_Q+z_V+z_M)/3.

모두 순수 함수 — DB/엔진은 인자(conn/주입 fn)로만. 기존 프리미티브 테스트의 DI 관례를 따른다.
"""
from __future__ import annotations

import math

import pytest

from src.backtest.primitives import (
    composite_score,
    compute_qvm_scores,
    drop_missing_factors,
    get_cross_section_qvm,
    invert_field,
    run_qvm_backtest,
    sector_zscore_with_fallback,
    winsorize_pct,
)


# ==========================================================================
# 작업 1a: invert_field — 가치 팩터 역수 변환(E/P=1/PER 등)
# ==========================================================================
def test_invert_field_normal_reciprocal():
    rows = [{"per": 8.0}, {"per": 20.0}]
    out = invert_field(rows, "per", "ep")
    assert out[0]["ep"] == pytest.approx(1 / 8.0)
    assert out[1]["ep"] == pytest.approx(1 / 20.0)
    # 원본 필드 보존
    assert out[0]["per"] == 8.0


def test_invert_field_non_positive_becomes_none():
    rows = [{"per": 0.0}, {"per": -5.0}]
    out = invert_field(rows, "per", "ep")
    assert out[0]["ep"] is None
    assert out[1]["ep"] is None


def test_invert_field_none_stays_none():
    rows = [{"per": None}, {}]  # 명시적 None + 필드 없음
    out = invert_field(rows, "per", "ep")
    assert out[0]["ep"] is None
    assert out[1]["ep"] is None


# ==========================================================================
# 작업 1b: winsorize_pct — 1%/99% 분위 기반 극단치 클리핑(기존 IQR winsorize와 별개)
# ==========================================================================
def test_winsorize_pct_clips_extremes_to_boundary():
    # 0..100 균등 + 극단치. 1%/99% 분위 밖은 경계로 눌린다.
    vals = list(range(0, 101))  # 0..100
    rows = [{"x": float(v)} for v in vals]
    out = winsorize_pct(rows, "x", lower_pct=0.01, upper_pct=0.99)
    lo = min(r["x_winsorized"] for r in out)
    hi = max(r["x_winsorized"] for r in out)
    # numpy.percentile([0..100],1)=1.0, 99=99.0 → 0은 1.0, 100은 99.0으로 눌림
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(99.0)


def test_winsorize_pct_keeps_none():
    rows = [{"x": 5.0}, {"x": None}, {"x": 10.0}]
    out = winsorize_pct(rows, "x")
    assert out[1]["x_winsorized"] is None


def test_winsorize_pct_inside_range_unchanged():
    vals = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
    rows = [{"x": v} for v in vals]
    out = winsorize_pct(rows, "x", lower_pct=0.01, upper_pct=0.99)
    mid = next(r for r in out if r["x"] == 15.0)
    assert mid["x_winsorized"] == pytest.approx(15.0)


# ==========================================================================
# 작업 1c: sector_zscore_with_fallback — 섹터 표본<5면 전체 유니버스 z-score 폴백
# ==========================================================================
def _sector_rows():
    # 섹터 '화학' 6종목(>=5), 섹터 '금융' 2종목(<5 → 폴백)
    rows = []
    for i, v in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
        rows.append({"stock_code": f"C{i}", "sector": "화학", "x": v})
    for i, v in enumerate([100.0, 200.0]):
        rows.append({"stock_code": f"F{i}", "sector": "금융", "x": v})
    return rows


def test_sector_zscore_uses_sector_stats_when_enough_samples():
    rows = _sector_rows()
    out = sector_zscore_with_fallback(rows, "x", min_sector_n=5)
    chem = [r for r in out if r["sector"] == "화학"]
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    mean = sum(xs) / len(xs)
    std = math.sqrt(sum((v - mean) ** 2 for v in xs) / len(xs))
    got = next(r for r in chem if r["x"] == 1.0)
    assert got["x_zscore"] == pytest.approx((1.0 - mean) / std)


def test_sector_zscore_falls_back_to_universe_when_too_few():
    rows = _sector_rows()
    out = sector_zscore_with_fallback(rows, "x", min_sector_n=5)
    fin = [r for r in out if r["sector"] == "금융"]
    all_x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0, 200.0]
    mean = sum(all_x) / len(all_x)
    std = math.sqrt(sum((v - mean) ** 2 for v in all_x) / len(all_x))
    got = next(r for r in fin if r["x"] == 100.0)
    # 섹터(2표본) 대신 전체 유니버스 통계로 계산돼야 한다
    assert got["x_zscore"] == pytest.approx((100.0 - mean) / std)


def test_sector_zscore_zero_std_guarded():
    rows = [{"sector": "A", "x": 5.0} for _ in range(6)]
    out = sector_zscore_with_fallback(rows, "x", min_sector_n=5)
    assert all(r["x_zscore"] == 0.0 for r in out)


def test_sector_zscore_none_value_stays_none():
    rows = _sector_rows() + [{"stock_code": "Z", "sector": "화학", "x": None}]
    out = sector_zscore_with_fallback(rows, "x", min_sector_n=5)
    z = next(r for r in out if r["stock_code"] == "Z")
    assert z["x_zscore"] is None


# ==========================================================================
# 작업 1d: composite_score — z-score 필드 가중평균(결측 제외 재정규화)
# ==========================================================================
def test_composite_score_equal_weight_all_present():
    rows = [{"a": 1.0, "b": 2.0, "c": 3.0}]
    out = composite_score(rows, ["a", "b", "c"], "comp")
    assert out[0]["comp"] == pytest.approx(2.0)


def test_composite_score_missing_factor_renormalizes():
    rows = [{"a": 1.0, "b": None, "c": 3.0}]
    out = composite_score(rows, ["a", "b", "c"], "comp")
    # b 결측 → a,c만 등가중 평균
    assert out[0]["comp"] == pytest.approx(2.0)


def test_composite_score_all_missing_is_none():
    rows = [{"a": None, "b": None}]
    out = composite_score(rows, ["a", "b"], "comp")
    assert out[0]["comp"] is None


def test_composite_score_custom_weights():
    rows = [{"a": 1.0, "b": 3.0}]
    out = composite_score(rows, ["a", "b"], "comp", weights=[3.0, 1.0])
    # (1*3 + 3*1)/(3+1) = 6/4 = 1.5
    assert out[0]["comp"] == pytest.approx(1.5)


def test_composite_score_custom_weights_with_missing_renormalizes():
    rows = [{"a": None, "b": 3.0}]
    out = composite_score(rows, ["a", "b"], "comp", weights=[3.0, 1.0])
    # a 결측 → b만 → 3.0
    assert out[0]["comp"] == pytest.approx(3.0)


# ==========================================================================
# 작업 1e: drop_missing_factors — 결측 개수가 max_missing 초과 시 행 제거
# ==========================================================================
def test_drop_missing_factors_boundary_keeps_exactly_max():
    fields = ["a", "b", "c", "d"]
    # 정확히 2개 결측(max_missing=2) → 유지
    rows = [{"a": 1, "b": 2, "c": None, "d": None}]
    out = drop_missing_factors(rows, fields, max_missing=2)
    assert len(out) == 1


def test_drop_missing_factors_over_boundary_drops():
    fields = ["a", "b", "c", "d"]
    # 3개 결측(max_missing=2 초과) → 제거
    rows = [{"a": 1, "b": None, "c": None, "d": None}]
    out = drop_missing_factors(rows, fields, max_missing=2)
    assert out == []


def test_drop_missing_factors_mixed():
    fields = ["a", "b", "c"]
    rows = [
        {"stock_code": "keep", "a": 1, "b": 2, "c": None},   # 1 결측 유지
        {"stock_code": "drop", "a": None, "b": None, "c": None},  # 3 결측 제거(>1)
    ]
    out = drop_missing_factors(rows, fields, max_missing=1)
    codes = [r["stock_code"] for r in out]
    assert codes == ["keep"]


# ==========================================================================
# 작업 2: compute_qvm_scores — 전체 QVM 파이프라인 조립
# ==========================================================================
def _qvm_universe():
    """섹터 '화학' 다수 + 팩터 변동이 있는 유니버스. 7개 raw 팩터 모두 존재."""
    rows = []
    base = [
        # (roe, gp_a, cfo_ratio, per, pbr, psr, momentum)
        (5.0, 10.0, 3.0, 20.0, 2.0, 1.5, -5.0),
        (8.0, 12.0, 4.0, 15.0, 1.8, 1.2, 2.0),
        (12.0, 18.0, 6.0, 10.0, 1.2, 0.9, 10.0),
        (15.0, 22.0, 8.0, 8.0, 1.0, 0.7, 20.0),
        (20.0, 28.0, 10.0, 6.0, 0.8, 0.5, 35.0),
        (3.0, 8.0, 2.0, 25.0, 3.0, 2.0, -10.0),
    ]
    for i, (roe, gpa, cfo, per, pbr, psr, mom) in enumerate(base):
        rows.append({
            "stock_code": f"S{i}", "name": f"종목{i}", "sector": "화학",
            "roe": roe, "gp_a": gpa, "cfo_ratio": cfo,
            "per": per, "pbr": pbr, "psr": psr, "momentum_12_1": mom,
        })
    return rows


def test_compute_qvm_all_rows_get_score_and_intermediates():
    rows = _qvm_universe()
    out = compute_qvm_scores(rows)
    assert len(out) == len(rows)
    for r in out:
        # 최종 점수 + 2차 표준화 카테고리 점수 + 중간값 존재(투명성)
        assert r["qvm_score"] is not None
        assert "quality_z" in r and "value_z" in r and "momentum_z" in r
        assert "roe_winsorized" in r
        assert "roe_winsorized_zscore" in r
        assert "ep" in r  # invert 결과
    # 고품질·저평가·고모멘텀 종목(S4)이 최상위 점수여야 한다
    best = max(out, key=lambda r: r["qvm_score"])
    assert best["stock_code"] == "S4"


def test_compute_qvm_drops_row_with_3_missing_raw_factors():
    rows = _qvm_universe()
    # 가치 3팩터(per/pbr/psr) 전부 결측 → ep/bp/sp 3개 결측 → rule6(3개 이상) 제외
    rows.append({
        "stock_code": "MISS3", "name": "결측3", "sector": "화학",
        "roe": 10.0, "gp_a": 10.0, "cfo_ratio": 5.0,
        "per": None, "pbr": None, "psr": None, "momentum_12_1": 5.0,
    })
    out = compute_qvm_scores(rows)
    assert "MISS3" not in {r["stock_code"] for r in out}


def test_compute_qvm_keeps_row_with_2_missing_raw_factors():
    rows = _qvm_universe()
    # 2개만 결측(per/pbr) → 유지
    rows.append({
        "stock_code": "MISS2", "name": "결측2", "sector": "화학",
        "roe": 10.0, "gp_a": 10.0, "cfo_ratio": 5.0,
        "per": None, "pbr": None, "psr": 1.0, "momentum_12_1": 5.0,
    })
    out = compute_qvm_scores(rows)
    assert "MISS2" in {r["stock_code"] for r in out}


def test_compute_qvm_custom_category_weights_changes_score():
    rows = _qvm_universe()
    eq = compute_qvm_scores(rows, category_weights=(1 / 3, 1 / 3, 1 / 3))
    mom_only = compute_qvm_scores(rows, category_weights=(0.0, 0.0, 1.0))
    eq_map = {r["stock_code"]: r["qvm_score"] for r in eq}
    mom_map = {r["stock_code"]: r["qvm_score"] for r in mom_only}
    # 모멘텀 전용 가중이면 점수 = momentum_z 와 같아야 한다(카테고리 재정규화)
    for r in mom_only:
        assert r["qvm_score"] == pytest.approx(r["momentum_z"])
    # 등가중과 모멘텀전용은 서로 달라야 한다(가중치가 실제로 반영됨)
    assert any(
        abs(eq_map[c] - mom_map[c]) > 1e-9 for c in eq_map if c in mom_map
    )


def test_compute_qvm_category_weights_as_dict_does_not_crash():
    """실사용 재현 버그: LLM이 파이프라인 JSON의 category_weights를 리스트가 아니라
    {"quality":1.0,"value":1.0,"momentum":1.0} 같은 딕셔너리로 생성했다. list(dict)는
    값이 아니라 키(문자열)를 반환하므로, 그 문자열과 z-score를 곱하려다
    "can't multiply sequence by non-int of type 'float'"로 크래시했다(실서버 재현).
    딕셔너리를 quality/value/momentum 순서의 숫자 리스트로 정규화해 등가중과 동일한
    결과가 나와야 한다."""
    rows = _qvm_universe()
    from_dict = compute_qvm_scores(
        rows, category_weights={"quality": 1.0, "value": 1.0, "momentum": 1.0}
    )
    from_tuple = compute_qvm_scores(rows, category_weights=(1 / 3, 1 / 3, 1 / 3))
    dict_map = {r["stock_code"]: r["qvm_score"] for r in from_dict}
    tuple_map = {r["stock_code"]: r["qvm_score"] for r in from_tuple}
    assert dict_map.keys() == tuple_map.keys()
    for code in dict_map:
        assert dict_map[code] == pytest.approx(tuple_map[code])


def test_compute_qvm_category_weights_dict_uppercase_keys():
    """대소문자 무관 — LLM이 "Quality"/"Value"/"Momentum"처럼 대문자로 시작해도 인식한다."""
    rows = _qvm_universe()
    out = compute_qvm_scores(
        rows, category_weights={"Quality": 0.0, "Value": 0.0, "Momentum": 1.0}
    )
    for r in out:
        assert r["qvm_score"] == pytest.approx(r["momentum_z"])


def test_compute_qvm_category_weights_dict_missing_key_raises_clear_error():
    """quality/value/momentum 키가 빠지면 크래시 대신 명확한 ValueError를 낸다."""
    rows = _qvm_universe()
    with pytest.raises(ValueError, match="quality/value/momentum"):
        compute_qvm_scores(rows, category_weights={"quality": 1.0, "value": 1.0})


def test_compute_qvm_sector_fallback_in_full_pipeline():
    rows = _qvm_universe()
    # 소수 섹터(금융 2종목) 추가 → 섹터 z-score가 전체 유니버스 폴백으로 계산돼도 점수 산출
    rows.append({
        "stock_code": "FIN1", "name": "금융1", "sector": "금융",
        "roe": 9.0, "gp_a": 11.0, "cfo_ratio": 4.5,
        "per": 12.0, "pbr": 1.1, "psr": 1.0, "momentum_12_1": 8.0,
    })
    rows.append({
        "stock_code": "FIN2", "name": "금융2", "sector": "금융",
        "roe": 7.0, "gp_a": 9.0, "cfo_ratio": 3.5,
        "per": 14.0, "pbr": 1.3, "psr": 1.1, "momentum_12_1": 3.0,
    })
    out = compute_qvm_scores(rows, min_sector_n=5)
    fin = {r["stock_code"] for r in out} & {"FIN1", "FIN2"}
    assert fin == {"FIN1", "FIN2"}  # 폴백으로 정상 산출·유지


def test_compute_qvm_momentum_missing_category_none_but_scored():
    rows = _qvm_universe()
    # momentum만 결측(1개) → momentum_z None, 하지만 Q/V로 최종점수 산출(재정규화)
    rows.append({
        "stock_code": "NOMOM", "name": "무모멘텀", "sector": "화학",
        "roe": 11.0, "gp_a": 13.0, "cfo_ratio": 5.0,
        "per": 11.0, "pbr": 1.1, "psr": 1.0, "momentum_12_1": None,
    })
    out = compute_qvm_scores(rows)
    nm = next(r for r in out if r["stock_code"] == "NOMOM")
    assert nm["momentum_z"] is None
    assert nm["qvm_score"] is not None  # Q/V만으로 산출


# ==========================================================================
# 작업 2 배선: get_cross_section_qvm — 크로스섹션 + 배치 모멘텀 병합
# ==========================================================================
def test_get_cross_section_qvm_merges_momentum():
    def fake_metrics(conn, asof):
        return [
            {"stock_code": "A", "name": "가", "sector": "화학", "market": "KOSPI",
             "quarter": "2025Q4", "per": 10.0},
            {"stock_code": "B", "name": "나", "sector": "화학", "market": "KOSPI",
             "quarter": "2025Q4", "per": 12.0},
        ]

    def fake_momentum(conn, codes, asof):
        assert set(codes) == {"A", "B"}
        return {"A": 15.0, "B": -3.0}

    out = get_cross_section_qvm(
        "CONN", "2025-12-31", metrics_fn=fake_metrics, momentum_fn=fake_momentum,
    )
    got = {r["stock_code"]: r["momentum_12_1"] for r in out}
    assert got == {"A": 15.0, "B": -3.0}


# ==========================================================================
# 작업 5: run_qvm_backtest — 기존 엔진에 qvm_score 단일 기준으로 배선
# ==========================================================================
def test_run_qvm_backtest_wires_qvm_score_criteria_and_schema():
    captured = {}

    def fake_backtest_fn(dates, metrics_fn, price_fn, params, benchmark_fn=None, weights=None):
        captured["params"] = params
        captured["rows"] = metrics_fn(dates[0])
        captured["price"] = price_fn(dates[0], "S0")
        return {
            "dates": dates, "navs": [1.0, 1.1, 1.2], "benchmark": None,
            "performance": {"cagr": 0.1, "mdd": -0.05, "sharpe": 1.2}, "holdings": [],
        }

    def fake_callbacks_fn(conn):
        return (lambda t: [], lambda t, c: 100.0)

    def fake_cross_section_fn(conn, asof, markets=None):
        return _qvm_universe()

    def fake_dates_fn(sy, ey, reb):
        return ["2024-03-31", "2024-06-30", "2024-09-30"]

    def fake_max_date_fn(conn):
        return "2026-12-31"

    res = run_qvm_backtest(
        "CONN", 2024, 2024, n=20, rebalance="quarterly", with_benchmark=False,
        callbacks_fn=fake_callbacks_fn, cross_section_fn=fake_cross_section_fn,
        dates_fn=fake_dates_fn, max_date_fn=fake_max_date_fn, backtest_fn=fake_backtest_fn,
    )
    # 단일 기준 qvm_score(high)로 엔진 호출
    assert captured["params"]["criteria"] == [
        {"key": "qvm_score", "direction": "high", "weight": 1.0}
    ]
    # metrics_fn이 qvm_score를 얹은 rows를 돌려준다
    assert all("qvm_score" in r for r in captured["rows"])
    assert captured["price"] == 100.0
    # 반환 스키마가 기존 백테스트와 동일
    assert set(res) >= {"dates", "navs", "benchmark", "performance", "holdings"}


def test_run_qvm_backtest_forwards_custom_factor_fields():
    captured = {}

    def fake_backtest_fn(dates, metrics_fn, price_fn, params, benchmark_fn=None, weights=None):
        captured["rows"] = metrics_fn(dates[0])
        return {"dates": dates, "navs": [1.0], "benchmark": None,
                "performance": {}, "holdings": []}

    def fake_callbacks_fn(conn):
        return (lambda t: [], lambda t, c: 1.0)

    def fake_cross_section_fn(conn, asof, markets=None):
        return _qvm_universe()

    run_qvm_backtest(
        "CONN", 2024, 2024, quality_fields=["roe", "gp_a"], with_benchmark=False,
        callbacks_fn=fake_callbacks_fn, cross_section_fn=fake_cross_section_fn,
        dates_fn=lambda s, e, r: ["2024-03-31", "2024-06-30"],
        max_date_fn=lambda c: "2026-12-31", backtest_fn=fake_backtest_fn,
    )
    # cfo_ratio를 뺀 quality_fields로도 정상 산출(qvm_score 존재)
    assert all("qvm_score" in r for r in captured["rows"])


# ==========================================================================
# 작업 4: LLM 파이프라인 프롬프트에 QVM 프리미티브 노출 확인
# ==========================================================================
def test_pipeline_prompt_lists_qvm_primitives():
    from src.agents.domain_backtest import _PIPELINE_PROMPT

    # 프롬프트가 today/question으로 정상 포맷돼야 하고(리터럴 {}의 {{/}} 이스케이프),
    # 신규 QVM 프리미티브 이름이 모두 노출돼야 한다.
    formatted = _PIPELINE_PROMPT.format(today="2026-07-15", question="퀄리티 밸류 모멘텀 상위 20종목")
    for name in (
        "get_cross_section_qvm", "compute_qvm_scores", "run_qvm_backtest",
        "invert_field", "winsorize_pct", "sector_zscore_with_fallback",
        "composite_score", "drop_missing_factors", "qvm_score",
    ):
        assert name in formatted


def test_pipeline_prompt_generates_qvm_pipeline_via_llm():
    """LLM이 프롬프트를 보고 QVM 스크리닝 파이프라인을 생성하면 그대로 파싱된다(배선 확인)."""
    import json

    from src.agents.domain_backtest import generate_backtest_steps

    def fake_llm(prompt: str) -> str:
        assert "get_cross_section_qvm" in prompt  # 프롬프트에 근거 존재
        return json.dumps({"pipeline": [
            {"op": "get_cross_section_qvm", "params": {"asof": "2026-07-15"}, "out": "xs"},
            {"op": "compute_qvm_scores", "params": {"rows": {"$ref": "xs"}}, "out": "scored"},
            {"op": "combine", "params": {"rows": {"$ref": "scored"},
             "criteria": [{"key": "qvm_score", "direction": "high", "weight": 1.0}],
             "method": "zscore", "n": 20}, "out": "picked"},
        ]})

    steps = generate_backtest_steps("퀄리티 밸류 모멘텀 상위 20종목", fake_llm, today="2026-07-15")
    assert steps[0]["op"] == "get_cross_section_qvm"
    assert steps[1]["op"] == "compute_qvm_scores"
    assert steps[2]["params"]["criteria"][0]["key"] == "qvm_score"
