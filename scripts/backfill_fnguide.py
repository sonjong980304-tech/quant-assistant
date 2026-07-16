"""FnGuide 전체 종목 백필 실행 스크립트 (gicode 'A' 접두사 버그 수정 후 1회성 실행용).

company 테이블의 모든 KR 상장주에 대해 재무 하이라이트+컨센서스 목표주가를
fnguide_metrics에 채운다. ThrottledFetcher 기본 지연(2초/요청)을 그대로 사용해
FnGuide 서버에 부담을 주지 않는다.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

from src.ingest.fnguide_metrics import ingest_fnguide_metrics  # noqa: E402


def main() -> None:
    start = time.monotonic()
    result = ingest_fnguide_metrics(db_path="data/market.db")
    elapsed = time.monotonic() - start
    print(f"완료: {result['succeeded']}/{result['tickers']} 성공, "
          f"실패 {len(result['failed'])}개, 총 {result['metric_rows']}행, "
          f"{elapsed:.0f}초 소요")
    if result["failed"]:
        print("실패 종목:", result["failed"])


if __name__ == "__main__":
    main()
