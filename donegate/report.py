"""verdict 집계·리포트·JSONL 로깅 (AC5, AC6, AC8, AC9).

- all-pass 집계: 모든 항목 통과 → DONE, 하나라도 실패 → NOT_DONE.
- 실패 리포트: 어느 축·항목이 왜(기대 vs 실제) 실패했는지 표시.
- 재현 절차: done.yaml 경로 + 각 command를 리포트에 포함(① 증거).
- 각 실행을 JSONL 한 줄로 로깅: outcome·항목별 latency·cost(오프라인=0).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DONE = "DONE"
NOT_DONE = "NOT_DONE"


@dataclass
class ItemOutcome:
    axis: str
    name: str
    command: str
    pass_rule: str
    passed: bool
    expected: str
    actual: str
    latency_s: float
    cost: float = 0.0        # 오프라인/픽스처 모드는 LLM 미사용 → 0
    detail: str = ""


@dataclass
class GateReport:
    verdict: str
    outcomes: list[ItemOutcome]
    config_path: str
    project: str = ""
    total_latency_s: float = 0.0
    cost_total: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def build_report(project: str, config_path: str, outcomes: list[ItemOutcome]) -> GateReport:
    """항목 결과를 all-pass 규칙으로 집계해 GateReport 생성."""
    verdict = DONE if all(o.passed for o in outcomes) and outcomes else NOT_DONE
    total_latency = round(sum(o.latency_s for o in outcomes), 3)
    cost_total = round(sum(o.cost for o in outcomes), 6)
    return GateReport(
        verdict=verdict,
        outcomes=outcomes,
        config_path=str(config_path),
        project=project,
        total_latency_s=total_latency,
        cost_total=cost_total,
    )


def render_report(report: GateReport) -> str:
    """사람용 리포트 텍스트. 실패 상세(AC6) + 재현 절차(AC9) 포함."""
    lines: list[str] = []
    lines.append("================ 완성도 게이트 리포트 ================")
    lines.append(f"프로젝트 : {report.project}")
    lines.append(f"판정     : {report.verdict}")
    lines.append(f"총 소요  : {report.total_latency_s}s   비용: {report.cost_total}")
    lines.append("")

    # 축별 항목 결과
    axis_order: list[str] = []
    for o in report.outcomes:
        if o.axis not in axis_order:
            axis_order.append(o.axis)
    for axis in axis_order:
        lines.append(f"[{axis}]")
        for o in report.outcomes:
            if o.axis != axis:
                continue
            mark = "PASS" if o.passed else "FAIL"
            lines.append(f"  - [{mark}] {o.name}  ({o.latency_s}s)")
            if not o.passed:
                lines.append(f"          기대: {o.expected}")
                lines.append(f"          실제: {o.actual}")
                if o.detail:
                    lines.append(f"          사유: {o.detail}")
        lines.append("")

    # 실패 요약(AC6)
    failed = [o for o in report.outcomes if not o.passed]
    if failed:
        lines.append("---- 실패 항목 요약 ----")
        for o in failed:
            lines.append(f"  · 축[{o.axis}] {o.name}: 기대 {o.expected} / 실제 {o.actual}")
        lines.append("")

    # 재현 절차(AC9)
    lines.append("---- 재현 절차 (증거) ----")
    lines.append(f"  done.yaml: {report.config_path}")
    for o in report.outcomes:
        lines.append(f"  $ {o.command}")
    lines.append("====================================================")
    return "\n".join(lines)


def log_jsonl(report: GateReport, log_path: str | Path) -> None:
    """실행 1건을 JSONL 한 줄로 누적 기록(AC8).

    스키마: {timestamp, verdict, project, config_path, total_latency_s, cost_total,
             items:[{axis,name,command,passed,latency_s,cost}]}
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": report.timestamp,
        "verdict": report.verdict,
        "project": report.project,
        "config_path": report.config_path,
        "total_latency_s": report.total_latency_s,
        "cost_total": report.cost_total,
        "items": [
            {
                "axis": o.axis,
                "name": o.name,
                "command": o.command,
                "passed": o.passed,
                "latency_s": o.latency_s,
                "cost": o.cost,
            }
            for o in report.outcomes
        ],
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
