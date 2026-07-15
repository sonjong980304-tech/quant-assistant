"""진입점 (AC12).

    python -m donegate.run [done.yaml경로]

done.yaml을 읽어 3축을 command 경계로 실행·판정하고 verdict를 출력한다.
종료코드: DONE=0, NOT_DONE=1. (스키마/설정 오류는 2)
"""
from __future__ import annotations

import sys
from pathlib import Path

from .check import check_item
from .executor import run_command
from .report import ItemOutcome, build_report, log_jsonl, render_report
from .schema import DoneConfigError, load_done_config

# donegate/ 의 부모 = 레포 루트. command는 이 디렉토리에서 실행한다
# (cli.py / tests/ 상대경로가 맞도록).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "done.yaml"
DEFAULT_LOG = REPO_ROOT / ".omc" / "logs" / "donegate.jsonl"


def run_gate(
    config_path: str | Path = DEFAULT_CONFIG,
    cwd: str | Path = REPO_ROOT,
    log_path: str | Path | None = DEFAULT_LOG,
    timeout: float | None = None,
):
    """게이트를 실행하고 GateReport를 반환한다(종료는 호출부에서)."""
    cfg = load_done_config(config_path)
    outcomes: list[ItemOutcome] = []
    for item in cfg.items:
        result = run_command(item.command, cwd=str(cwd), timeout=timeout)
        check = check_item(item, result.exit_code, result.stdout)
        outcomes.append(
            ItemOutcome(
                axis=item.axis,
                name=item.name,
                command=item.command,
                pass_rule=item.pass_rule,
                passed=check.passed,
                expected=check.expected,
                actual=check.actual,
                latency_s=result.latency_s,
                cost=0.0,  # 오프라인/픽스처 게이트 — LLM 미사용이므로 비용 0
                detail=check.detail,
            )
        )
    report = build_report(cfg.project, str(config_path), outcomes)
    if log_path is not None:
        log_jsonl(report, log_path)
    return report


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = argv[0] if argv else DEFAULT_CONFIG
    try:
        report = run_gate(config_path=config_path)
    except DoneConfigError as exc:
        print(f"[donegate] done.yaml 오류: {exc}", file=sys.stderr)
        return 2
    print(render_report(report))
    return 0 if report.verdict == "DONE" else 1


if __name__ == "__main__":
    sys.exit(main())
