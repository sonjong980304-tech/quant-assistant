"""SEC companyfacts 초기 대량 백필 (사용자 감독 1회성, AC4/AC1/AC2/AC10/AC11).

하이브리드 접근의 "초기 백필" 경로: SEC 전체회사 스냅샷 companyfacts.zip(~1.39GB)을
받아 추적 대상 종목(us_company)만 걸러 새 테이블 us_financials_sec 에 전체 히스토리를
적재한다. 자동 백그라운드 스케줄이 아니라 사용자가 직접 실행하며 진행상황을 지켜보는
1회성 작업이다(스펙 Constraints/AC4 — 주간 갱신은 run_us_financials_sec.py 담당).

단계:
  1) (선택) CIK 매핑: us_company 전종목에 SEC company_tickers.json 로 CIK 부여(AC1),
     매핑률<95% 면 실패 종목 목록 경고(AC2).
  2) companyfacts.zip 을 --zip-path 로 지정(없고 --download 지정 시 SEC 에서 내려받음).
  3) 추적 대상만 걸러 us_financials_sec 에 적재.
  4) quarterly 커버리지(AC10)·디스크 용량(AC11) 리포트 출력.

실행 예:
  python3 scripts/backfill_us_financials_sec.py --map-ciks --download
  python3 scripts/backfill_us_financials_sec.py --zip-path /path/to/companyfacts.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.data_access_us_sec import ttm_coverage
from src.db import connect
from src.ingest.cik_mapping import backfill_ciks
from src.ingest.us_financials_sec import (
    SEC_USER_AGENT,
    backfill_from_zip,
    quarterly_coverage,
    table_disk_bytes,
)

# SEC 전체회사 companyfacts 대량 스냅샷(확인된 실제 파일, ~1.39GB).
_COMPANYFACTS_ZIP_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"


def _download_zip(dest: str) -> str:
    """companyfacts.zip 을 SEC 에서 스트리밍 다운로드한다(User-Agent 필수). 경로 반환."""
    import requests

    print(f"companyfacts.zip 다운로드 시작 → {dest} (~1.39GB, 시간이 걸립니다)", flush=True)
    with requests.get(
        _COMPANYFACTS_ZIP_URL, headers={"User-Agent": SEC_USER_AGENT}, stream=True, timeout=120
    ) as resp:
        resp.raise_for_status()
        total = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1MB
                f.write(chunk)
                total += len(chunk)
        print(f"다운로드 완료: {total:,} bytes", flush=True)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="SEC companyfacts 초기 백필(사용자 감독 1회성)")
    parser.add_argument("--zip-path", default=None, help="로컬 companyfacts.zip 경로")
    parser.add_argument("--download", action="store_true", help="zip 을 SEC 에서 내려받아 사용")
    parser.add_argument("--map-ciks", action="store_true", help="백필 전에 CIK 매핑을 먼저 수행")
    parser.add_argument("--db-path", default=None, help="대상 DB 경로(기본: CONFIG.db_path)")
    args = parser.parse_args()

    if args.map_ciks:
        print("[1/4] CIK 매핑 시작...", flush=True)
        rep = backfill_ciks(db_path=args.db_path)
        print(f"  대상 {rep['total']} / 매핑 {rep['matched']} "
              f"({rep['matched_rate']:.1%}) / 실패 {len(rep['unmatched'])}")
        if rep["warning"]:
            print(f"  ⚠️ 매핑률 {rep['matched_rate']:.1%} < {rep['threshold']:.0%} — "
                  f"매핑 실패 종목(상위 30): {rep['unmatched'][:30]}"
                  + (" ..." if len(rep["unmatched"]) > 30 else ""))

    zip_path = args.zip_path
    if args.download:
        default_dest = str(Path(__file__).resolve().parent.parent / "data" / "companyfacts.zip")
        zip_path = _download_zip(args.zip_path or default_dest)
    if not zip_path:
        parser.error("--zip-path 또는 --download 중 하나는 필요합니다")

    print("[2/4] companyfacts.zip 적재 시작...", flush=True)
    rep = backfill_from_zip(db_path=args.db_path, zip_path=zip_path)
    print(f"  추적 대상 {rep['targets']} / 적재 {rep['tickers_loaded']}종목 "
          f"/ 팩트 {rep['facts_loaded']:,}개 / zip 누락 {len(rep['missing_in_zip'])}종목")

    conn = connect(args.db_path)
    try:
        print("[3/4] quarterly 커버리지(AC10)...", flush=True)
        cov = quarterly_coverage(conn)
        print(f"  종목당 최소 1개 quarterly 팩트 보유: "
              f"{cov['with_quarterly']}/{cov['total_tracked']} ({cov['coverage_rate']:.1%})")
        # TTM 계산가능 커버리지(Q4 역산까지 성공한 종목 — quarterly 존재율이 못 잡는 진짜 지표)
        ttm = ttm_coverage(conn)
        print(f"  TTM(4분기, Q4 역산 포함) 계산가능: "
              f"{ttm['ttm_computable']}/{ttm['total_tracked']} ({ttm['ttm_coverage_rate']:.1%})")

        print("[4/4] 디스크 용량(AC11)...", flush=True)
        size = table_disk_bytes(conn, "us_financials_sec")
        print(f"  us_financials_sec 실제 사용량: {size:,} bytes ({size / 1e9:.2f} GB) "
              f"— 사전 추정치(6GB 안팎)와 비교")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
