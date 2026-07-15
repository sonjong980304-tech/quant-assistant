"""US 재무제표(yfinance) 크롤러 실행 스크립트 (launchd용).

us_company 전종목의 손익/재무상태/현금흐름(quarterly+annual)을 yfinance에서
받아 us_financials에 upsert한다. 실패 종목은 건너뛰고 Slack 알림만 보낸다.

실행: python3 scripts/run_us_financials.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.us_financials import ingest_us_financials

if __name__ == "__main__":
    result = ingest_us_financials()
    print(result)
