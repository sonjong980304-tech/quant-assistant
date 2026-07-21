"""factcheck eval 공통 비교 유틸 — 허용오차 비교 / 정확 일치 비교.

.omc/specs/brainstorming-factcheck-eval.md AC1 참고. 재무제표(±1% 허용오차) 등
"실제값 대조" 컴포넌트들이 공유하는 순수 함수만 담는다.
"""
from __future__ import annotations

_EPSILON = 1e-9


def within_pct_tolerance(expected: float, actual: float, pct: float) -> bool:
    """|actual-expected| / |expected| <= pct 이면 True (경계값 포함).

    expected == 0 이면 나눗셈이 불가하므로 별도 정의: actual도 0이면 True,
    그 외에는 False.
    """
    if expected == 0:
        return actual == 0
    return abs(actual - expected) / abs(expected) <= pct


def exact_match(expected, actual) -> bool:
    """숫자면 epsilon(1e-9) 이내 근접 비교, 그 외(문자열 등)는 == 비교."""
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(actual - expected) <= _EPSILON
    return expected == actual
