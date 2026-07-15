"""네이버 fchart 일별 수정주가 크롤러 실행 스크립트 (launchd용).

company 전종목의 OHLCV를 네이버 fchart에서 받아 prices에 upsert한다.
실패 종목은 건너뛰고 Slack 알림만 보낸다(pykrx 자동 폴백 없음).

실행: python3 scripts/run_naver_prices.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.naver_prices import ingest_naver_prices

if __name__ == "__main__":
    result = ingest_naver_prices()
    print(result)
