"""US 티커/섹터 유니버스(NASDAQ 스크리너 API) 수집 실행 스크립트 (launchd용).

NASDAQ 공식 스크리너 API(api.nasdaq.com/api/screener/stocks)를 거래소별(NASDAQ/
NYSE/AMEX)로 호출해 종목을 모아 us_company에 upsert한다. 인증 불필요, 거래소당
1회 호출로 전량(NASDAQ 4,110+NYSE 2,718+AMEX 295=7,123건, 2026-07-12 실측) 수신.
거래소 실패는 격리하고 나머지는 계속 진행하며 Slack 알림을 보낸다.

2026-07-12 소스 전환: 원래 investing.com Stock Screener를 Playwright로 순회했으나,
게스트+무료로그인 모두 2페이지 이상이 InvestingPro 유료게이트로 막힘을 확정해
전환했다(us_universe.py 모듈 docstring 참고).

실행: python3 scripts/run_us_universe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.us_universe import ingest_us_universe

if __name__ == "__main__":
    result = ingest_us_universe()
    print(result)
