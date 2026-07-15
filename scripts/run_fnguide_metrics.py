"""FnGuide 재무 하이라이트+컨센서스 목표주가 크롤러 실행 스크립트 (launchd용).

company 전종목의 재무 하이라이트+컨센서스 목표주가를 FnGuide에서 받아
fnguide_metrics에 upsert한다. 실패 종목은 건너뛰고 Slack 알림만 보낸다.

실행: python3 scripts/run_fnguide_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.fnguide_metrics import ingest_fnguide_metrics

if __name__ == "__main__":
    result = ingest_fnguide_metrics()
    print(result)
