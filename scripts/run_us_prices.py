"""US 주가(yfinance) 수집 실행 스크립트 (launchd용).

us_company 전종목의 OHLCV를 yfinance에서 받아 us_prices에 upsert한다. 신규
종목은 10년 백필, 기존 종목은 최근 구간만 증분 수집한다(ingest_us_prices
내부 로직). 실패 종목은 건너뛰고 Slack 알림만 보낸다.

실행: python3 scripts/run_us_prices.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.us_prices import ingest_us_prices

if __name__ == "__main__":
    result = ingest_us_prices()
    print(result)
