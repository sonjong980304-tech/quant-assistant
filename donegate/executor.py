"""command 실행 (AC2).

done.yaml의 각 항목 command를 subprocess로 실행하고 exit code·stdout·stderr·
경과시간(latency)을 캡처한다. dart 코드를 import하지 않고 오직 command 경계로만
상호작용한다(AC11).
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    latency_s: float
    timed_out: bool = False


def run_command(
    command: str,
    cwd: str | None = None,
    timeout: float | None = None,
) -> CommandResult:
    """command를 셸로 실행하고 결과를 캡처한다.

    timeout 초과 시 프로세스를 종료하고 timed_out=True, exit_code=-1로 반환한다.
    """
    start = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        latency = time.monotonic() - start
        return CommandResult(
            command=command,
            exit_code=-1,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + f"\n[donegate] timeout after {timeout}s",
            latency_s=round(latency, 3),
            timed_out=True,
        )
    latency = time.monotonic() - start
    return CommandResult(
        command=command,
        exit_code=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        latency_s=round(latency, 3),
    )
