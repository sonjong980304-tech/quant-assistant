"""SEC 배치 자동화(launchd plist + 실행 스크립트) 검증 (AC5).

.omc/specs/brainstorming-sec-edgar-us-financials-backfill.md AC5: 주간 갱신은 기존
com.darttext.usfinancials.plist 와 동일한 주기(매주 토요일)로 등록돼야 한다 — plist
파일 내용으로 확인한다. 스크립트가 import 되는지도 스모크 테스트한다.
"""
from __future__ import annotations

import importlib.util
import plistlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PLIST = _ROOT / "scripts" / "launchd" / "com.darttext.usfinancialssec.plist"


def test_sec_plist_scheduled_weekly_saturday():
    data = plistlib.loads(_PLIST.read_bytes())
    assert data["Label"] == "com.darttext.usfinancialssec"
    # Weekday 6 = 토요일(기존 us_financials plist 와 동일 주기, AC5)
    assert data["StartCalendarInterval"]["Weekday"] == 6


def test_sec_plist_runs_weekly_update_script():
    data = plistlib.loads(_PLIST.read_bytes())
    args = data["ProgramArguments"]
    # 주간 갱신 스크립트(run_us_financials_sec.py)를 실행해야 한다(백필 스크립트가 아님).
    assert any(a.endswith("run_us_financials_sec.py") for a in args)


def _can_import(path: Path) -> bool:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return True


def test_weekly_update_script_imports():
    assert _can_import(_ROOT / "scripts" / "run_us_financials_sec.py")


def test_backfill_script_imports():
    assert _can_import(_ROOT / "scripts" / "backfill_us_financials_sec.py")
