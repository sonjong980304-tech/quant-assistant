"""네이버 fchart 일별 수정주가 크롤러 실행 스크립트 (launchd용).

company 전종목의 OHLCV를 네이버 fchart에서 받아 prices에 upsert한다. "KOSPI"·"KOSDAQ"(코스피·
코스닥 지수, backfill_kospi_index.py/backfill_kosdaq_index.py로 최초 적재)도 함께 갱신해
백테스트 벤치마크(코스피·코스닥 각각 별도 비교선)가 매일 최신 상태를 유지하게 한다 —
ingest_naver_prices(codes=None)의 하위호환 기본 동작(company 테이블 전체 순회)은 그대로 두고,
이 스크립트에서만 명시적으로 목록에 추가한다.
실패 종목은 건너뛰고 Slack 알림만 보낸다(pykrx 자동 폴백 없음).

실행: python3 scripts/run_naver_prices.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import connect
from src.ingest.naver_prices import ingest_naver_prices

if __name__ == "__main__":
    conn = connect()
    try:
        codes = [r["stock_code"] for r in conn.execute("SELECT stock_code FROM company").fetchall()]
    finally:
        conn.close()
    result = ingest_naver_prices(codes=codes + ["KOSPI", "KOSDAQ"])
    print(result)
