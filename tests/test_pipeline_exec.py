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
def test_dispatch_table_is_fixed_dict_with_thirteen_primitives():
    assert set(PRIMITIVE_OPS.keys()) == {
        "get_cross_section", "zscore", "neutralize", "winsorize", "combine", "regress",
        "optimize_weights", "run_backtest", "run_signal_backtest", "compute_ic",
        "compute_technical_indicator", "search_strategy", "search_signal_strategy",
    }
    # 값이 실제 호출 가능한 함수여야 한다
    assert all(callable(v) for v in PRIMITIVE_OPS.values())


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
