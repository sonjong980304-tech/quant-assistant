"""done.yaml 파싱·검증 (AC1).

필수 필드(project, axes, 각 항목의 name·command·pass)가 누락되면 명확한
에러(DoneConfigError)를 낸다. threshold 규칙 항목은 extract·threshold도 요구한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# 판정 규칙 종류
PASS_EXIT_ZERO = "exit_zero"
PASS_THRESHOLD = "threshold"
_VALID_PASS = (PASS_EXIT_ZERO, PASS_THRESHOLD)


class DoneConfigError(ValueError):
    """done.yaml 스키마/필수필드 오류."""


@dataclass
class Item:
    """축에 속한 단일 판정 항목."""

    axis: str
    name: str
    command: str
    pass_rule: str                       # "exit_zero" | "threshold"
    metric: str | None = None
    extract: str | None = None           # threshold일 때 stdout JSON의 dot-path
    threshold: str | None = None         # threshold일 때 비교식(예: ">= 35.0")


@dataclass
class DoneConfig:
    project: str
    env: dict = field(default_factory=dict)
    axes: dict[str, list[Item]] = field(default_factory=dict)  # 축명 → 항목들(순서 유지)
    items: list[Item] = field(default_factory=list)            # 전체 항목(평탄화)
    source_path: str | None = None


def load_done_config(path: str | Path) -> DoneConfig:
    """파일에서 done.yaml을 읽어 파싱·검증한다."""
    p = Path(path)
    if not p.exists():
        raise DoneConfigError(f"done.yaml 파일 없음: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return parse_done_config(data, source_path=str(p))


def parse_done_config(data, source_path: str | None = None) -> DoneConfig:
    """dict를 검증하여 DoneConfig로 변환한다. 문제가 있으면 DoneConfigError."""
    if not isinstance(data, dict):
        raise DoneConfigError("done.yaml 최상위는 매핑(dict)이어야 함")

    project = data.get("project")
    if not project or not isinstance(project, str):
        raise DoneConfigError("필수 필드 누락: 'project'(문자열)")

    axes_raw = data.get("axes")
    if not axes_raw:
        raise DoneConfigError("필수 필드 누락: 'axes'(비어있지 않은 매핑)")
    if not isinstance(axes_raw, dict):
        raise DoneConfigError("'axes'는 축명→항목목록 매핑이어야 함")

    env = data.get("env") or {}
    if not isinstance(env, dict):
        raise DoneConfigError("'env'는 매핑(dict)이어야 함")

    axes: dict[str, list[Item]] = {}
    items: list[Item] = []
    for axis_name, raw_items in axes_raw.items():
        if not isinstance(raw_items, list) or not raw_items:
            raise DoneConfigError(f"축 '{axis_name}'는 비어있지 않은 항목 리스트여야 함")
        parsed: list[Item] = []
        for idx, raw in enumerate(raw_items):
            parsed.append(_parse_item(axis_name, idx, raw))
        axes[axis_name] = parsed
        items.extend(parsed)

    return DoneConfig(
        project=project,
        env=env,
        axes=axes,
        items=items,
        source_path=source_path,
    )


def _parse_item(axis: str, idx: int, raw) -> Item:
    where = f"축 '{axis}' 항목[{idx}]"
    if not isinstance(raw, dict):
        raise DoneConfigError(f"{where}: 매핑(dict)이어야 함")

    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise DoneConfigError(f"{where}: 필수 필드 'name' 누락")

    command = raw.get("command")
    if not command or not isinstance(command, str):
        raise DoneConfigError(f"{where}('{name}'): 필수 필드 'command' 누락")

    pass_rule = raw.get("pass")
    if not pass_rule:
        raise DoneConfigError(f"{where}('{name}'): 필수 필드 'pass' 누락")
    if pass_rule not in _VALID_PASS:
        raise DoneConfigError(
            f"{where}('{name}'): 알 수 없는 pass 규칙 '{pass_rule}' "
            f"(허용: {', '.join(_VALID_PASS)})"
        )

    extract = raw.get("extract")
    threshold = raw.get("threshold")
    if pass_rule == PASS_THRESHOLD:
        if not extract or not isinstance(extract, str):
            raise DoneConfigError(
                f"{where}('{name}'): pass=threshold는 'extract'(dot-path 문자열) 필요"
            )
        if not threshold or not isinstance(threshold, str):
            raise DoneConfigError(
                f"{where}('{name}'): pass=threshold는 'threshold'(비교식 문자열) 필요"
            )

    return Item(
        axis=axis,
        name=name,
        command=command,
        pass_rule=pass_rule,
        metric=raw.get("metric"),
        extract=extract,
        threshold=threshold,
    )
