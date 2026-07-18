"""FMP 미국 상장폐지 목록 수집 실행 스크립트 (launchd 주간 갱신 + 초기 백필 겸용).

.omc/specs/brainstorming-us-delisting-survivorship.md 참고.

FMP delisted-companies 는 증분 API 가 없어 매번 전체 목록을 반환한다 → us_delisting 에
멱등 upsert(UNIQUE(stock_code, listing_date, delisting_date))하므로 이 스크립트 하나로
- 초기 역사적 전체 백필(AC7, 사용자가 최초 1회 실행·리포트 확인)과
- 매주 신규 상장폐지만 반영하는 갱신(AC8, launchd 주간)을 모두 처리한다.
두 경우 모두 소량 페이지 호출(전체 상폐목록 ≈ 수십 페이지)이라 하루 250회 한도를 크게
남긴다(AC10). API 키는 .env 의 FMP_API_KEY 를 os.getenv 로만 읽는다(값 미노출, AC9).

기존 run_us_financials_sec.py(SEC 주간 갱신)와 동일한 명명·구조 관례.
실행: python3 scripts/run_us_delisting.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.us_delisting import ingest_us_delisting

if __name__ == "__main__":
    result = ingest_us_delisting()
    # AC7 리포트: 수집된 상장폐지 구간 수·가장 오래된 상장폐지일·재사용 티커 수·API 호출 수.
    # (API 키 값은 절대 출력하지 않는다 — result 에 키가 포함되지 않음, AC9.)
    print(
        f"[us_delisting] episodes_upserted={result['episodes_upserted']} "
        f"tickers={result['tickers']} reused={len(result['reused_tickers'])} "
        f"oldest_delisting_date={result['oldest_delisting_date']} "
        f"api_calls={result['api_calls']}"
    )
