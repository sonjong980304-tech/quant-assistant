"""donegate 러너 자체 검증 (AC1~AC12).

각 테스트가 어떤 수용기준을 검증하는지 docstring에 표기한다.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from donegate.check import (
    check_exit_zero,
    check_threshold,
    extract_metric,
    parse_threshold,
)
from donegate.executor import run_command
from donegate.report import (
    DONE,
    NOT_DONE,
    ItemOutcome,
    build_report,
    log_jsonl,
    render_report,
)
from donegate.run import run_gate
from donegate.schema import DoneConfigError, load_done_config, parse_done_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DONEGATE_DIR = REPO_ROOT / "donegate"


def _write_yaml(tmp_path: Path, data: dict, name: str = "done.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# AC1 — 스키마 파싱 / 필수필드 누락 에러
# ---------------------------------------------------------------------------
def test_ac1_schema_parses_valid():
    data = {
        "project": "x",
        "axes": {"기능": [{"name": "a", "command": "echo ok", "pass": "exit_zero"}]},
    }
    cfg = parse_done_config(data)
    assert cfg.project == "x"
    assert len(cfg.items) == 1
    assert cfg.items[0].pass_rule == "exit_zero"
    assert cfg.items[0].axis == "기능"


def test_ac1_real_done_yaml_is_valid():
    """레포 루트의 실제 done.yaml이 스키마를 만족한다."""
    cfg = load_done_config(REPO_ROOT / "done.yaml")
    assert cfg.project == "dart-text2sql-wiki"
    assert set(cfg.axes) == {"기능", "안정성", "성능"}
    assert len(cfg.items) == 3


@pytest.mark.parametrize(
    "data",
    [
        {"axes": {"기능": [{"name": "a", "command": "c", "pass": "exit_zero"}]}},  # project 누락
        {"project": "x"},  # axes 누락
        {"project": "x", "axes": {}},  # axes 빈값
        {"project": "x", "axes": {"기능": [{"command": "c", "pass": "exit_zero"}]}},  # name 누락
        {"project": "x", "axes": {"기능": [{"name": "a", "pass": "exit_zero"}]}},  # command 누락
        {"project": "x", "axes": {"기능": [{"name": "a", "command": "c"}]}},  # pass 누락
        {"project": "x", "axes": {"기능": [{"name": "a", "command": "c", "pass": "wat"}]}},  # 잘못된 pass
        # threshold인데 extract/threshold 누락
        {"project": "x", "axes": {"성능": [{"name": "a", "command": "c", "pass": "threshold"}]}},
    ],
)
def test_ac1_missing_fields_raise(data):
    with pytest.raises(DoneConfigError):
        parse_done_config(data)


def test_ac1_missing_file_raises():
    with pytest.raises(DoneConfigError):
        load_done_config("/tmp/definitely_missing_donefile_zzz.yaml")


# ---------------------------------------------------------------------------
# AC2 — command 실행 / exit·stdout 캡처
# ---------------------------------------------------------------------------
def test_ac2_executor_captures_exit_and_stdout():
    r = run_command("echo hello")
    assert r.exit_code == 0
    assert "hello" in r.stdout
    assert r.latency_s >= 0

    r2 = run_command("exit 3")
    assert r2.exit_code == 3


# ---------------------------------------------------------------------------
# AC3 — exit_zero 판정
# ---------------------------------------------------------------------------
def test_ac3_check_exit_zero():
    assert check_exit_zero(0).passed is True
    assert check_exit_zero(1).passed is False
    assert check_exit_zero(2).passed is False


# ---------------------------------------------------------------------------
# AC4 — threshold 비교 / dot-path 추출
# ---------------------------------------------------------------------------
def test_ac4_extract_dot_path():
    so = json.dumps({"execution_accuracy": {"ex_pct": 0.74}})
    assert extract_metric(so, "execution_accuracy.ex_pct") == 0.74


def test_ac4_threshold_pass_and_fail():
    so_hi = json.dumps({"execution_accuracy": {"ex_pct": 0.74}})
    so_lo = json.dumps({"execution_accuracy": {"ex_pct": 0.60}})
    assert check_threshold(so_hi, "execution_accuracy.ex_pct", ">= 0.70").passed is True
    assert check_threshold(so_lo, "execution_accuracy.ex_pct", ">= 0.70").passed is False


def test_ac4_threshold_operators():
    assert parse_threshold(">= 35.0")[2] == 35.0
    assert check_threshold('{"v": 5}', "v", "> 3").passed is True
    assert check_threshold('{"v": 5}', "v", "< 3").passed is False
    assert check_threshold('{"v": 3}', "v", "== 3").passed is True


def test_ac4_missing_path_fails_gracefully():
    res = check_threshold('{"a": 1}', "b.c", ">= 0.5")
    assert res.passed is False
    assert "(추출 실패)" in res.actual


# ---------------------------------------------------------------------------
# AC5 — all-pass verdict (전부통과→DONE, 한 항목 실패→NOT_DONE)
# ---------------------------------------------------------------------------
def test_ac5_verdict_all_pass(tmp_path):
    data = {
        "project": "fixture-pass",
        "axes": {
            "기능": [{"name": "echo통과", "command": "echo ok", "pass": "exit_zero"}],
            "성능": [
                {
                    "name": "지표통과",
                    "command": "echo '{\"score\": 0.9}'",
                    "pass": "threshold",
                    "extract": "score",
                    "threshold": ">= 0.5",
                }
            ],
        },
    }
    cfg_path = _write_yaml(tmp_path, data)
    report = run_gate(config_path=cfg_path, log_path=tmp_path / "log.jsonl")
    assert report.verdict == DONE
    assert all(o.passed for o in report.outcomes)


def test_ac5_verdict_one_fail(tmp_path):
    data = {
        "project": "fixture-fail",
        "axes": {
            "기능": [{"name": "실패항목", "command": "exit 1", "pass": "exit_zero"}],
            "성능": [
                {
                    "name": "지표통과",
                    "command": "echo '{\"score\": 0.9}'",
                    "pass": "threshold",
                    "extract": "score",
                    "threshold": ">= 0.5",
                }
            ],
        },
    }
    cfg_path = _write_yaml(tmp_path, data)
    report = run_gate(config_path=cfg_path, log_path=tmp_path / "log.jsonl")
    assert report.verdict == NOT_DONE


# ---------------------------------------------------------------------------
# AC6 — 실패 리포트에 축·항목·기대·실제 포함
# ---------------------------------------------------------------------------
def test_ac6_failure_report_details(tmp_path):
    data = {
        "project": "ff",
        "axes": {"기능": [{"name": "실패항목", "command": "exit 1", "pass": "exit_zero"}]},
    }
    cfg_path = _write_yaml(tmp_path, data)
    report = run_gate(config_path=cfg_path, log_path=tmp_path / "log.jsonl")
    text = render_report(report)
    assert "실패항목" in text          # 항목
    assert "기능" in text              # 축
    assert "exit_code == 0" in text    # 기대
    assert "exit_code == 1" in text    # 실제


# ---------------------------------------------------------------------------
# AC7 — 결정론 (오프라인 eval 2회 동일 verdict/값)
# ---------------------------------------------------------------------------
def test_ac7_offline_eval_deterministic(seeded_db):
    from src.eval.runner import run_evaluation

    r1 = run_evaluation(db_path=seeded_db, offline=True, limit=10)
    r2 = run_evaluation(db_path=seeded_db, offline=True, limit=10)
    assert r1["execution_accuracy"]["ex_pct"] == r2["execution_accuracy"]["ex_pct"]
    assert r1["sql_source"] == r2["sql_source"]


# ---------------------------------------------------------------------------
# AC8 — JSONL 로깅 스키마 (outcome·latency·cost)
# ---------------------------------------------------------------------------
def test_ac8_jsonl_logging_schema(tmp_path):
    outcomes = [
        ItemOutcome(
            axis="기능", name="a", command="echo ok", pass_rule="exit_zero",
            passed=True, expected="exit_code == 0", actual="exit_code == 0",
            latency_s=0.01, cost=0.0,
        )
    ]
    report = build_report("proj", "done.yaml", outcomes)
    logp = tmp_path / "g.jsonl"
    log_jsonl(report, logp)

    lines = logp.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["verdict"] in (DONE, NOT_DONE)
    assert "timestamp" in rec
    assert isinstance(rec["items"], list) and rec["items"]
    item = rec["items"][0]
    assert isinstance(item["latency_s"], (int, float))
    assert item["cost"] == 0.0  # 오프라인 → 비용 0

    # 누적(append) 확인
    log_jsonl(report, logp)
    assert len(logp.read_text(encoding="utf-8").strip().splitlines()) == 2


# ---------------------------------------------------------------------------
# AC9 — 재현 절차(done.yaml 경로 + 각 command)
# ---------------------------------------------------------------------------
def test_ac9_report_includes_reproduction(tmp_path):
    data = {
        "project": "rp",
        "axes": {"기능": [{"name": "a", "command": "echo ok", "pass": "exit_zero"}]},
    }
    cfg_path = _write_yaml(tmp_path, data)
    report = run_gate(config_path=cfg_path, log_path=tmp_path / "log.jsonl")
    text = render_report(report)
    assert str(cfg_path) in text   # done.yaml 경로
    assert "echo ok" in text       # 각 command
    assert "재현 절차" in text


# ---------------------------------------------------------------------------
# AC11 — import 격리 정적 테스트 (donegate가 src를 import하지 않음)
# ---------------------------------------------------------------------------
_SRC_IMPORT = re.compile(r"^\s*(from|import)\s+src(\.|\s|$)")


def test_ac11_donegate_does_not_import_src():
    offenders: list[tuple[str, str]] = []
    py_files = sorted(DONEGATE_DIR.glob("*.py"))
    assert py_files, "donegate 소스 파일을 찾지 못함"
    for py in py_files:
        source = py.read_text(encoding="utf-8")
        # 1) AST 기반 검사 (권위)
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                for n in node.names:
                    if n.name == "src" or n.name.startswith("src."):
                        offenders.append((py.name, n.name))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "src" or mod.startswith("src."):
                    offenders.append((py.name, mod))
        # 2) 텍스트 스캔 (회귀 방지 이중 확인)
        for line in source.splitlines():
            if _SRC_IMPORT.match(line):
                offenders.append((py.name, line.strip()))
    assert not offenders, f"donegate가 src를 import함(분리 불변식 위반): {offenders}"


# ---------------------------------------------------------------------------
# AC12 — CLI 종료코드 (python -m donegate.run: DONE=0 / NOT_DONE=1)
# ---------------------------------------------------------------------------
def test_ac12_cli_exit_code_done(tmp_path):
    data = {
        "project": "cli-pass",
        "axes": {"기능": [{"name": "a", "command": "echo ok", "pass": "exit_zero"}]},
    }
    cfg_path = _write_yaml(tmp_path, data)
    proc = subprocess.run(
        [sys.executable, "-m", "donegate.run", str(cfg_path)],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "DONE" in proc.stdout


def test_ac12_cli_exit_code_not_done(tmp_path):
    data = {
        "project": "cli-fail",
        "axes": {"기능": [{"name": "실패", "command": "exit 1", "pass": "exit_zero"}]},
    }
    cfg_path = _write_yaml(tmp_path, data)
    proc = subprocess.run(
        [sys.executable, "-m", "donegate.run", str(cfg_path)],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "NOT_DONE" in proc.stdout
