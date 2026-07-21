"""파이프라인 실행기 + 안전장치 단위 테스트 (TDD).

.omc/specs/brainstorming-sql-python-primitive-pipeline.md 참고.
핵심 안전조건(같은 머신에 실거래봇 상주 → 반드시 지킬 것):
- 연산 디스패치는 고정 dict(getattr 등 문자열 동적 조회 금지)
- 단계 최대 20 / 종목·window 최대 4000 / 타임아웃 최대 120초, 초과 시 명시적 에러
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.backtest.pipeline_exec import (
    MAX_SIZE,
    MAX_STEPS,
    MAX_TIMEOUT,
    PRIMITIVE_OPS,
    run_pipeline,
)


# --------------------------------------------------------------------------
# 고정 dict 디스패치 (getattr 부재)
# --------------------------------------------------------------------------
def test_dispatch_table_is_fixed_dict_with_all_primitives():
    assert set(PRIMITIVE_OPS.keys()) == {
        "get_cross_section", "zscore", "neutralize", "winsorize", "combine", "regress",
        "correlation", "quantile_bucket_means", "histogram_buckets", "remove_outliers",
        "scatter_data", "optimize_weights", "run_backtest", "run_signal_backtest",
        "compute_ic", "compute_technical_indicator", "search_strategy", "search_signal_strategy",
        # QVM 멀티팩터(신규)
        "invert_field", "winsorize_pct", "sector_zscore_with_fallback", "composite_score",
        "drop_missing_factors", "compute_qvm_scores", "get_cross_section_qvm", "run_qvm_backtest",
    }
    # 값이 실제 호출 가능한 함수여야 한다
    assert all(callable(v) for v in PRIMITIVE_OPS.values())


def test_qvm_primitives_registered_in_primitive_ops():
    """QVM 프리미티브가 고정 dict 디스패치에 등록돼 파이프라인에서 호출 가능해야 한다."""
    for op in (
        "invert_field", "winsorize_pct", "sector_zscore_with_fallback", "composite_score",
        "drop_missing_factors", "compute_qvm_scores", "get_cross_section_qvm", "run_qvm_backtest",
    ):
        assert op in PRIMITIVE_OPS
        assert callable(PRIMITIVE_OPS[op])


def test_qvm_conn_needing_ops_registered():
    from src.backtest.pipeline_exec import _NEEDS_CONN

    # DB 연결이 필요한 QVM 연산만 자동주입 대상
    assert "get_cross_section_qvm" in _NEEDS_CONN
    assert "run_qvm_backtest" in _NEEDS_CONN


def test_qvm_pure_ops_not_conn_needing():
    from src.backtest.pipeline_exec import _NEEDS_CONN

    # rows(순수 데이터)만 받는 QVM 저수준 연산은 conn 자동주입 대상이 아니다
    for op in (
        "invert_field", "winsorize_pct", "sector_zscore_with_fallback",
        "composite_score", "drop_missing_factors", "compute_qvm_scores",
    ):
        assert op not in _NEEDS_CONN


def test_histogram_buckets_registered_in_primitive_ops():
    """히스토그램 프리미티브가 고정 dict 디스패치에 등록돼 파이프라인에서 호출 가능해야 한다."""
    assert "histogram_buckets" in PRIMITIVE_OPS
    assert callable(PRIMITIVE_OPS["histogram_buckets"])


def test_winsorize_op_is_not_registered_as_conn_needing():
    from src.backtest.pipeline_exec import _NEEDS_CONN

    # winsorize는 rows(순수 데이터)만 받는 함수라 conn 자동주입 대상이 아니다
    assert "winsorize" not in _NEEDS_CONN


def test_run_signal_backtest_op_is_registered_as_conn_needing():
    from src.backtest.pipeline_exec import _NEEDS_CONN

    assert "run_signal_backtest" in _NEEDS_CONN


def test_run_backtest_op_is_registered_as_conn_needing():
    from src.backtest.pipeline_exec import _NEEDS_CONN

    assert "run_backtest" in _NEEDS_CONN


def test_compute_ic_op_is_registered_as_conn_needing():
    from src.backtest.pipeline_exec import _NEEDS_CONN

    assert "compute_ic" in _NEEDS_CONN


def test_compute_technical_indicator_op_is_registered_as_conn_needing():
    from src.backtest.pipeline_exec import _NEEDS_CONN

    assert "compute_technical_indicator" in _NEEDS_CONN


def test_search_strategy_op_is_registered_as_conn_needing():
    from src.backtest.pipeline_exec import _NEEDS_CONN

    assert "search_strategy" in _NEEDS_CONN


# --------------------------------------------------------------------------
# 정적 검사: 신규 프리미티브에 eval/exec 없음 (AC18, 같은 머신에 실거래봇 상주)
# --------------------------------------------------------------------------
def test_new_primitives_source_has_no_eval_or_exec():
    src = Path(__file__).resolve().parent.parent / "src" / "backtest" / "primitives.py"
    text = src.read_text(encoding="utf-8")
    assert "eval(" not in text
    assert "exec(" not in text


def test_executor_source_has_no_dynamic_getattr_dispatch():
    """정적 검사: 실행기 소스에 getattr( 문자열 동적 조회가 없어야 한다(임의 코드 실행 방지)."""
    src = Path(__file__).resolve().parent.parent / "src" / "backtest" / "pipeline_exec.py"
    assert "getattr(" not in src.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 파이프라인 조립/체이닝 ($ref, out)
# --------------------------------------------------------------------------
def test_run_pipeline_chains_via_ref_and_out():
    ops = {"double": lambda x: [v * 2 for v in x], "sum_all": lambda x: sum(x)}
    steps = [
        {"op": "double", "params": {"x": [1, 2, 3]}, "out": "d"},
        {"op": "sum_all", "params": {"x": {"$ref": "d"}}, "out": "total"},
    ]
    assert run_pipeline(steps, ops=ops) == 12  # (1+2+3)*2


def test_run_pipeline_unknown_ref_raises():
    ops = {"noop": lambda x: x}
    steps = [{"op": "noop", "params": {"x": {"$ref": "missing"}}}]
    with pytest.raises(ValueError):
        run_pipeline(steps, ops=ops)


def test_run_pipeline_unknown_op_raises():
    with pytest.raises(ValueError):
        run_pipeline([{"op": "does_not_exist", "params": {}}], ops={"noop": lambda: None})


# --------------------------------------------------------------------------
# 안전장치 상한
# --------------------------------------------------------------------------
def test_too_many_steps_rejected():
    ops = {"noop": lambda **k: None}
    steps = [{"op": "noop", "params": {}} for _ in range(MAX_STEPS + 1)]
    with pytest.raises(ValueError):
        run_pipeline(steps, ops=ops)


def test_caller_cannot_raise_step_cap_above_hard_limit():
    ops = {"noop": lambda **k: None}
    steps = [{"op": "noop", "params": {}} for _ in range(MAX_STEPS + 1)]
    # max_steps를 하드 상한보다 크게 줘도 하드 상한이 이긴다 → 거부
    with pytest.raises(ValueError):
        run_pipeline(steps, ops=ops, max_steps=10_000)


def test_oversized_input_rejected():
    ops = {"noop": lambda x: x}
    steps = [{"op": "noop", "params": {"x": list(range(MAX_SIZE + 1))}}]
    with pytest.raises(ValueError):
        run_pipeline(steps, ops=ops)


def test_oversized_output_rejected():
    ops = {"explode": lambda: list(range(MAX_SIZE + 1))}
    steps = [{"op": "explode", "params": {}}]
    with pytest.raises(ValueError):
        run_pipeline(steps, ops=ops)


def test_timeout_rejected():
    ops = {"slow": lambda: time.sleep(0.5)}
    steps = [{"op": "slow", "params": {}}]
    with pytest.raises(TimeoutError):
        run_pipeline(steps, ops=ops, timeout_s=0.05)


def test_caller_cannot_raise_timeout_above_hard_limit():
    assert MAX_TIMEOUT == 120  # 스펙 고정값(2분) — 완화 금지


# --------------------------------------------------------------------------
# 실제 프리미티브 통합 (DB 없이 실행 가능한 조합)
# --------------------------------------------------------------------------
def test_run_pipeline_with_real_regress_primitive():
    steps = [{"op": "regress", "params": {"y": [1.0, 2.1, 2.9, 4.2, 5.0]}, "out": "reg"}]
    res = run_pipeline(steps)
    assert "slope" in res and "se_slope" in res and "k_ratio" in res
    assert res["slope"] == pytest.approx(1.0, abs=0.2)


# --------------------------------------------------------------------------
# 다중 산출물 파이프라인 (실서버 재현 버그: correlation+quantile_bucket_means+
# scatter_data처럼 서로 다른 3개 산출물을 만드는 파이프라인이 마지막 단계
# 결과 하나만 남기고 나머지를 조용히 버려서, "상관계수를 구해줘"라고 했는데
# 상관계수가 최종 응답에서 사라지는 문제가 있었다. $ref로 뒤 단계에 소비되지
# "않는"(=leaf) out이 2개 이상이면 {out이름: 값} dict로 전부 보존해야 한다.
# leaf가 1개뿐인 기존 모든 파이프라인(run_backtest 등)은 그 값 그대로 반환해
# 완전히 하위호환(회귀 없음).
# --------------------------------------------------------------------------
def test_run_pipeline_returns_dict_of_all_leaf_outputs_when_multiple():
    ops = {
        "source": lambda: [1, 2, 3, 4],
        "sum_all": lambda x: sum(x),
        "max_all": lambda x: max(x),
    }
    steps = [
        {"op": "source", "params": {}, "out": "xs"},
        {"op": "sum_all", "params": {"x": {"$ref": "xs"}}, "out": "total"},
        {"op": "max_all", "params": {"x": {"$ref": "xs"}}, "out": "peak"},
    ]
    res = run_pipeline(steps, ops=ops)
    # xs는 total/peak 둘 다에 $ref로 소비되므로 leaf가 아니다 → 최종 dict에 안 실림.
    assert res == {"total": 10, "peak": 4}


def test_run_pipeline_single_leaf_still_returns_bare_value_not_dict():
    """leaf가 1개면(기존 모든 단일목적 파이프라인) 예전처럼 dict로 안 감싸고 값 그대로."""
    ops = {"double": lambda x: [v * 2 for v in x], "sum_all": lambda x: sum(x)}
    steps = [
        {"op": "double", "params": {"x": [1, 2, 3]}, "out": "d"},
        {"op": "sum_all", "params": {"x": {"$ref": "d"}}, "out": "total"},
    ]
    assert run_pipeline(steps, ops=ops) == 12  # dict가 아니라 스칼라 그대로(회귀 없음)


def test_run_pipeline_single_leaf_survives_trailing_unnamed_step():
    """실서버 재현 버그: leaf(예: run_backtest의 out="bt")가 파이프라인 중간에 있고, 그
    뒤에 "out"이 없는 트레일링 스텝이 이어지면(LLM이 "이게 최종 답이니 이름 필요없다"고
    오판해 마지막 스텝에 out을 안 붙이는 경우), _execute_steps가 실제 leaf(state[leaf])가
    아니라 '마지막으로 실행된 스텝의 반환값'을 그대로 돌려주는 버그가 있었다 —
    run_backtest의 holdings가 사라지고 트레일링 스텝의 엉뚱한 결과가 그 자리를 대신해,
    auditor.post_audit이 그 엉뚱한 값(모양이 다른 dict)을 받아 하드체크가 조용히
    스킵되거나 예상치 못한 모양으로 크래시할 수 있었다. leaf가 1개면 outs_in_order 상
    위치와 무관하게 그 leaf 값이 최종 결과여야 한다."""
    ops = {
        "leaf_op": lambda: {"holdings": [{"date": "d1", "codes": ["005930"]}]},
        "unrelated_op": lambda rows: {"correlation": 0.5},
    }
    steps = [
        {"op": "leaf_op", "params": {}, "out": "bt"},
        {"op": "unrelated_op", "params": {"rows": [1, 2, 3]}},  # out 없음 — 트레일링 스텝
    ]
    res = run_pipeline(steps, ops=ops)
    assert res == {"holdings": [{"date": "d1", "codes": ["005930"]}]}


def test_run_pipeline_neutralize_by_null_then_histogram_buckets_end_to_end():
    """domain_backtest.py 프롬프트에 추가한 "전체 시장 z-score 히스토그램" 예시 파이프라인이
    실제 실행기(PRIMITIVE_OPS, mock 없음)를 통해 끝까지 동작하는지 확인한다. JSON의 null이
    파이썬 None으로 그대로 전달돼 neutralize(by=None)이 섹터 구분 없이 전체를 한 그룹으로
    z-score화하고, 그 결과(per_neutral)를 histogram_buckets가 그대로 받아 구간을 나눈다."""
    rows = [
        {"stock_code": "1", "sector": "화학", "per": 8.0},
        {"stock_code": "2", "sector": "금융", "per": 12.0},
        {"stock_code": "3", "sector": "화학", "per": 10.0},
        {"stock_code": "4", "sector": "금융", "per": 14.0},
    ]
    steps = [
        {"op": "neutralize", "params": {"rows": rows, "field": "per", "by": None, "method": "zscore"}, "out": "xs_z"},
        {"op": "histogram_buckets", "params": {"rows": {"$ref": "xs_z"}, "field": "per_neutral", "num_buckets": 4}, "out": "hist"},
    ]
    res = run_pipeline(steps)
    assert res["field"] == "per_neutral"
    assert res["n"] == 4
    assert sum(res["counts"]) == 4


def test_run_pipeline_correlation_and_quantile_bucket_means_both_survive():
    """실서버 재현 버그의 실제 패턴: 같은 rows에서 correlation과 quantile_bucket_means를
    각각 별도 out으로 뽑는 파이프라인 — 예전엔 quantile_bucket_means(마지막 단계)만 남고
    correlation은 사라졌다."""
    rows = [
        {"pbr": 0.5, "gp_a": 10.0}, {"pbr": 1.0, "gp_a": 20.0},
        {"pbr": 1.5, "gp_a": 25.0}, {"pbr": 2.0, "gp_a": 35.0},
        {"pbr": 2.5, "gp_a": 40.0},
    ]
    steps = [
        {"op": "correlation", "params": {"rows": rows, "field_x": "pbr", "field_y": "gp_a"}, "out": "corr"},
        {"op": "quantile_bucket_means", "params": {
            "rows": rows, "bucket_field": "pbr", "value_field": "gp_a", "n": 5,
        }, "out": "buckets"},
    ]
    res = run_pipeline(steps)
    assert set(res.keys()) == {"corr", "buckets"}
    assert res["corr"]["correlation"] == pytest.approx(1.0, abs=0.01)
    assert len(res["buckets"]) == 5


def test_run_pipeline_chains_winsorize_into_combine():
    """LLM이 실제로 만들 법한 조립: winsorize로 극단치를 누른 뒤 그 필드로 combine."""
    rows = [
        {"stock_code": "1", "roe": 10.0, "per": 8.0},
        {"stock_code": "2", "roe": 11.0, "per": 9.0},
        {"stock_code": "3", "roe": 12.0, "per": 10.0},
        {"stock_code": "4", "roe": 13.0, "per": 11.0},
        {"stock_code": "5", "roe": 500.0, "per": 12.0},  # roe 극단치
    ]
    steps = [
        {"op": "winsorize", "params": {"rows": rows, "field": "roe"}, "out": "w"},
        {"op": "combine", "params": {
            "rows": {"$ref": "w"},
            "criteria": [{"key": "roe_winsorized", "direction": "high", "weight": 1.0}],
            "method": "zscore", "n": 5,
        }, "out": "picked"},
    ]
    res = run_pipeline(steps)
    assert len(res) == 5
    assert res[0]["stock_code"] == "5"  # 눌린 뒤에도 여전히 roe 최상위는 5번
    assert res[0]["roe"] == 500.0  # 원본 roe는 보존
    assert res[0]["roe_winsorized"] < 500.0  # 계산엔 눌린 값이 쓰였다
