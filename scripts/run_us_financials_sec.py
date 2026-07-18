"""SEC companyfacts 주간 갱신 크롤러 실행 스크립트 (launchd용, AC5).

cik 가 매핑된 us_company 전종목의 companyfacts 를 종목별 SEC API 로 받아, 최신 분기
(최신 제출분)만 us_financials_sec 에 upsert 한다. 초당 10건 이하 + User-Agent 헤더는
src/ingest/us_financials_sec.py 가 강제한다. 실패 종목은 격리하고 Slack 알림만 보낸다.

기존 run_us_financials.py(yfinance 주간 갱신)와 동일한 명명·구조 관례. 초기 대량 백필은
이 스크립트가 아니라 scripts/backfill_us_financials_sec.py(사용자 감독 1회성)로 수행한다.

실행: python3 scripts/run_us_financials_sec.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.us_financials_sec import ingest_companyfacts_api

if __name__ == "__main__":
    # latest_only=True: 매주 최신 분기만 갱신(15년치 전체 재적재 안 함).
    result = ingest_companyfacts_api(latest_only=True)
    print(result)
