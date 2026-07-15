"""선언적 프리미티브 파이프라인 실행기 (결정론적, eval/exec 없음).

.omc/wiki/dart-text2sql-wiki-sql-python.md의 "안전 유지 조건"을 코드로 강제한다.

안전 원칙(같은 macOS 머신에 실거래 퀀트봇 quant_trader가 상주 → 반드시 지킬 것):
- 연산 디스패치는 **고정 dict PRIMITIVE_OPS**. 문자열 기반 동적 속성 조회(리플렉션)는
  임의 코드 실행과 동급의 구멍이라 절대 쓰지 않는다.
- 상한: 단계 최대 20 / 종목·window 최대 4000 / 타임아웃 최대 120초. 초과 시 명시적 에러로
  거부한다(무제한 허용 금지). 호출자가 파라미터로 이 하드 상한을 넘겨도 하드 상한이 이긴다.
- 프리미티브 내부 SQL은 get_cross_section→metrics_at의 바인딩 쿼리 + 읽기전용 연결
  (connect_readonly)만 사용한다(새 SQL 인젝션 경로를 만들지 않는다).

파이프라인 JSON 형식(LLM이 생성, 파이썬 코드 아님):
    {"pipeline": [
       {"op": "get_cross_section", "params": {"asof": "2025-12-31"}, "out": "xs"},
       {"op": "combine", "params": {"rows": {"$ref": "xs"}, "criteria": [...]}, "out": "picked"}
    ]}
각 step의 params 값이 {"$ref": "이름"}이면 앞 step이 out으로 저장한 결과를 주입한다.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

from .primitives import (
    combine,
    compute_ic_primitive,
    compute_technical_indicator,
    get_cross_section,
    neutralize,
    optimize_weights,
    regress,
    run_backtest_primitive,
    search_strategy,
    zscore,
)
from .signal_engine import run_signal_backtest, search_signal_strategy

# 스펙 고정 하드 상한 — 완화 금지(사용자가 "제한 없이"에서 되돌린 값)
MAX_STEPS = 20
MAX_SIZE = 4000
MAX_TIMEOUT = 120

# 고정 dict 디스패치 — getattr 등 문자열 동적 조회 금지(임의 코드 실행 방지)
PRIMITIVE_OPS = {
    "get_cross_section": get_cross_section,
    "zscore": zscore,
    "neutralize": neutralize,
    "combine": combine,
    "regress": regress,
    "optimize_weights": optimize_weights,
    "run_backtest": run_backtest_primitive,
    "run_signal_backtest": run_signal_backtest,
    "compute_ic": compute_ic_primitive,
    "compute_technical_indicator": compute_technical_indicator,
    "search_strategy": search_strategy,
    "search_signal_strategy": search_signal_strategy,
}

# DB 연결(conn)을 실행기가 주입해줘야 하는 연산(LLM이 conn을 지정하지 않도록)
_NEEDS_CONN = {
    "get_cross_section", "run_backtest", "run_signal_backtest", "compute_ic",
    "compute_technical_indicator", "search_strategy", "search_signal_strategy",
}


def _sizeof(value) -> int | None:
    """크기 상한 검사용 원소 개수. list/tuple은 길이, dict는 (키 수, 값 시계열 최대 길이) 중 큰 값."""
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, dict):
        lens = [len(v) for v in value.values() if isinstance(v, (list, tuple))]
        return max([len(value)] + lens)
    return None


def _check_size(values, max_size: int) -> None:
    for v in values:
        n = _sizeof(v)
        if n is not None and n > max_size:
            raise ValueError(f"입력/출력 크기 {n} 개 > 상한 {max_size} 개(종목·window 상한 초과)")


def _resolve_params(params: dict, state: dict) -> dict:
    """params의 {"$ref": name}을 state에 저장된 앞 단계 결과로 치환한다."""
    resolved = {}
    for key, val in params.items():
        if isinstance(val, dict) and set(val.keys()) == {"$ref"}:
            ref = val["$ref"]
            if ref not in state:
                raise ValueError(f"참조 '{ref}'가 아직 정의되지 않았습니다")
            resolved[key] = state[ref]
        else:
            resolved[key] = val
    return resolved


def _execute_steps(steps: list[dict], conn, ops: dict, max_size: int):
    state: dict = {}
    result = None
    for step in steps:
        op = step.get("op")
        if op not in ops:  # 고정 dict 조회 (getattr 아님)
            raise ValueError(f"알 수 없는 연산: {op}")
        fn = ops[op]
        kwargs = _resolve_params(step.get("params", {}), state)
        if op in _NEEDS_CONN:
            if conn is None:
                raise ValueError(f"{op} 연산에는 DB 연결(conn)이 필요합니다")
            kwargs["conn"] = conn
        _check_size(kwargs.values(), max_size)  # 입력 크기 상한
        result = fn(**kwargs)
        _check_size([result], max_size)          # 출력 크기 상한
        out = step.get("out")
        if out:
            state[out] = result
    return result


def run_pipeline(
    steps: list[dict],
    conn=None,
    ops: dict | None = None,
    max_steps: int = MAX_STEPS,
    max_size: int = MAX_SIZE,
    timeout_s: float = MAX_TIMEOUT,
):
    """프리미티브 파이프라인을 결정론적으로 실행하고 마지막 단계 결과를 반환한다.

    ops를 주입하면 그 dict로 디스패치한다(테스트용). 기본은 고정 PRIMITIVE_OPS.
    호출자가 max_steps/max_size/timeout_s에 하드 상한보다 큰 값을 줘도 하드 상한이 이긴다.
    """
    if not isinstance(steps, list):
        raise ValueError("pipeline steps는 list여야 합니다")
    ops = ops if ops is not None else PRIMITIVE_OPS

    # 호출자 파라미터를 하드 상한으로 클램프(무제한 허용 금지)
    eff_steps = min(max_steps, MAX_STEPS)
    eff_size = min(max_size, MAX_SIZE)
    eff_timeout = min(timeout_s, MAX_TIMEOUT)

    if len(steps) > eff_steps:
        raise ValueError(f"파이프라인 단계 수 {len(steps)}개 > 상한 {eff_steps}개")

    # 타임아웃 강제: 워커 스레드에서 실행하고 상한 내 완료를 요구한다(백스톱).
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_execute_steps, steps, conn, ops, eff_size)
        try:
            return future.result(timeout=eff_timeout)
        except FuturesTimeout as exc:
            raise TimeoutError(f"파이프라인 실행 타임아웃({eff_timeout}s 초과)") from exc
