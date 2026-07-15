"""항목 판정 (AC3, AC4).

- exit_zero : exit code 0이면 통과, 그 외 실패.
- threshold : stdout의 JSON에서 extract(dot-path)로 값을 뽑아 비교식과 대조.
              비교 연산자: >=, <=, >, <, == (예: ">= 35.0").
"""
from __future__ import annotations

import json
import operator
from dataclasses import dataclass

from .schema import PASS_EXIT_ZERO, PASS_THRESHOLD, Item

# 비교 연산자 (긴 것 먼저 매칭: >=, <= 가 >, < 보다 우선)
_OPS = [
    (">=", operator.ge),
    ("<=", operator.le),
    ("==", operator.eq),
    (">", operator.gt),
    ("<", operator.lt),
]


class CheckError(ValueError):
    """판정 중 발생한 오류(추출 실패/threshold 형식 오류 등)."""


@dataclass
class CheckResult:
    passed: bool
    expected: str      # 기대 조건 문자열(리포트용)
    actual: str        # 실제 관측값 문자열(리포트용)
    detail: str = ""   # 추가 설명(실패 사유 등)


def check_exit_zero(exit_code: int) -> CheckResult:
    """exit code 0 → 통과."""
    ok = exit_code == 0
    return CheckResult(
        passed=ok,
        expected="exit_code == 0",
        actual=f"exit_code == {exit_code}",
        detail="" if ok else f"비정상 종료(exit {exit_code})",
    )


def extract_metric(stdout: str, dot_path: str):
    """stdout의 JSON에서 dot-path로 값을 추출한다.

    stdout 전체가 JSON이 아니어도, 마지막으로 파싱 가능한 JSON 오브젝트 라인을
    찾아 사용한다(부가 로그 라인 방어).
    """
    data = _parse_json_from_stdout(stdout)
    if data is None:
        raise CheckError("stdout에서 JSON을 찾지 못함")
    cur = data
    for key in dot_path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise CheckError(f"extract 경로 '{dot_path}'에서 '{key}' 없음")
        cur = cur[key]
    return cur


def _parse_json_from_stdout(stdout: str):
    text = (stdout or "").strip()
    if not text:
        return None
    # 1) 전체가 JSON인 경우
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    # 2) 뒤에서부터 JSON 오브젝트로 파싱되는 라인 탐색
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            return json.loads(line)
        except (ValueError, TypeError):
            continue
    return None


def parse_threshold(expr: str) -> tuple[str, object, float]:
    """'>= 35.0' → (op_str, op_func, value). 형식 오류면 CheckError."""
    s = (expr or "").strip()
    for op_str, op_func in _OPS:
        if s.startswith(op_str):
            rest = s[len(op_str):].strip()
            try:
                value = float(rest)
            except ValueError as exc:
                raise CheckError(f"threshold 값 파싱 실패: '{expr}'") from exc
            return op_str, op_func, value
    raise CheckError(f"threshold 연산자 인식 실패: '{expr}' (>=,<=,>,<,== 중 하나 필요)")


def check_threshold(stdout: str, extract: str, threshold_expr: str) -> CheckResult:
    """stdout JSON에서 extract 값을 뽑아 threshold와 비교."""
    op_str, op_func, target = parse_threshold(threshold_expr)
    try:
        raw = extract_metric(stdout, extract)
    except CheckError as exc:
        return CheckResult(
            passed=False,
            expected=f"{extract} {op_str} {target}",
            actual="(추출 실패)",
            detail=str(exc),
        )
    try:
        actual_val = float(raw)
    except (TypeError, ValueError):
        return CheckResult(
            passed=False,
            expected=f"{extract} {op_str} {target}",
            actual=f"{extract} = {raw!r}",
            detail=f"수치 변환 불가: {raw!r}",
        )
    ok = bool(op_func(actual_val, target))
    return CheckResult(
        passed=ok,
        expected=f"{extract} {op_str} {target}",
        actual=f"{extract} = {actual_val}",
        detail="" if ok else f"임계 미달({actual_val} {op_str} {target} 불만족)",
    )


def check_item(item: Item, exit_code: int, stdout: str) -> CheckResult:
    """항목의 pass 규칙에 따라 판정."""
    if item.pass_rule == PASS_EXIT_ZERO:
        return check_exit_zero(exit_code)
    if item.pass_rule == PASS_THRESHOLD:
        # threshold 항목도 우선 exit code가 0이어야 한다(명령 자체가 성공해야 값이 유효).
        if exit_code != 0:
            return CheckResult(
                passed=False,
                expected=f"{item.extract} {item.threshold} (exit 0 후)",
                actual=f"exit_code == {exit_code}",
                detail=f"command 비정상 종료(exit {exit_code}) — 지표 판정 불가",
            )
        return check_threshold(stdout, item.extract, item.threshold)
    # schema에서 걸러지지만 방어적으로
    return CheckResult(
        passed=False,
        expected="(알 수 없는 pass 규칙)",
        actual=item.pass_rule,
        detail=f"지원하지 않는 pass 규칙: {item.pass_rule}",
    )
