"""일자별 상장주식수 백필 (pykrx get_market_cap_by_date → daily_shares).

company 테이블의 전종목을 순회하며, 종목별로 pykrx 를 1회 호출해 그 종목의 일자별
상장주식수를 daily_shares 에 적재한다(액면분할·무상증자 시 결산일 지연 없이 정확한
시점별 주식수를 확보 → backfill_marketcap 이 이를 최우선 소스로 시총을 재계산).

fromdate: 종목의 prices 최소 date(없으면 프로젝트 시작 START_DATE). todate: 오늘.
- 종목 단위 try/except 로 실패 격리(한 종목 실패가 전체를 막지 않음).
- 진행률 로그(PROGRESS_EVERY 종목마다).
- idempotent: (code,date) 는 INSERT OR REPLACE 라 재실행 안전. 기본은 이미 적재된 종목도
  다시 조회(최신 구간 갱신). --skip-covered 로 이미 오늘까지 커버된 종목은 건너뛴다.

⚠️ 3,900여 전종목 전체 실행은 오래 걸린다(종목당 ~6초 → 수 시간). 소규모 검증은
   --codes 로 특정 종목만 돌린다: python3 scripts/backfill_shares_daily.py --codes 134380,003350,298000

실행: python3 scripts/backfill_shares_daily.py            # 전종목(오래 걸림)
      python3 scripts/backfill_shares_daily.py --dry-run  # 대상/시작일만 확인(적재 없음)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG  # noqa: F401 — .env 로드(pykrx KRX 로그인 자격) 보장
from src.db import connect, init_db
from src.ingest.shares_daily import fetch_daily_shares, upsert_daily_shares

START_DATE = "20150101"   # prices 가 없는 종목의 폴백 시작일(프로젝트 데이터 시작)
PROGRESS_EVERY = 100      # 이 종목 수마다 진행률 로그
_CHECKPOINT_EVERY = 200   # 이 종목 수마다 WAL 체크포인트(TRUNCATE)로 -wal 파일 크기 관리


def _company_codes(conn) -> list[str]:
    return [r[0] for r in conn.execute("SELECT stock_code FROM company ORDER BY stock_code")]


def _fromdate_for(conn, code: str) -> str:
    """증분 갱신 시작일(YYYYMMDD).

    daily_shares 에 이미 데이터가 있으면 그 마지막 날짜 다음날부터만(매일 예약 실행 시
    전체 재수집을 피해 몇 초 안에 끝나게 하기 위함). 없으면(최초 백필) 기존대로 종목의
    prices 최소 date, 그마저도 없으면 START_DATE.
    """
    row = conn.execute("SELECT MAX(date) AS d FROM daily_shares WHERE stock_code=?", (code,)).fetchone()
    if row and row["d"]:
        last = date.fromisoformat(row["d"])
        return (last + timedelta(days=1)).strftime("%Y%m%d")
    row = conn.execute("SELECT MIN(date) AS d FROM prices WHERE stock_code=?", (code,)).fetchone()
    if row and row["d"]:
        return str(row["d"]).replace("-", "")
    return START_DATE


def _is_covered(conn, code: str, todate_iso: str) -> bool:
    """이미 이 종목의 daily_shares 가 todate 까지 커버돼 있으면 True(--skip-covered 용)."""
    row = conn.execute("SELECT MAX(date) AS d FROM daily_shares WHERE stock_code=?", (code,)).fetchone()
    return bool(row and row["d"] and row["d"] >= todate_iso)


def backfill_shares_daily(
    db_path: str | None = None,
    *,
    codes: list[str] | None = None,
    dry_run: bool = False,
    skip_covered: bool = False,
    on: date | None = None,
) -> dict:
    """전종목(또는 codes)의 일자별 상장주식수를 daily_shares 에 적재한다."""
    on = on or date.today()
    todate = on.strftime("%Y%m%d")
    todate_iso = on.isoformat()
    init_db(db_path)
    conn = connect(db_path)
    try:
        targets = codes if codes else _company_codes(conn)
        total = len(targets)

        ok = empty = failed = skipped = rows_total = 0
        first_elapsed: float | None = None
        started = time.time()

        for idx, code in enumerate(targets, 1):
            if skip_covered and _is_covered(conn, code, todate_iso):
                skipped += 1
                continue
            frm = _fromdate_for(conn, code)
            if frm > todate:  # 증분분 없음(이미 오늘까지 커버) → pykrx 호출 자체를 생략
                skipped += 1
                continue
            if dry_run:
                print(f"[dry-run] {code} fromdate={frm} todate={todate}", flush=True)
                continue
            try:
                t0 = time.time()
                rows = fetch_daily_shares(code, frm, todate)
                n = upsert_daily_shares(conn, code, rows)
                conn.commit()
                if first_elapsed is None:
                    first_elapsed = time.time() - t0
                    print(f"[timing] 1종목({code}) 조회+적재 {first_elapsed:.2f}s "
                          f"({frm}~{todate}, {len(rows)}행)", flush=True)
                if n:
                    ok += 1
                    rows_total += n
                else:
                    empty += 1
            except Exception as exc:  # noqa: BLE001 — 종목 단위 실패 격리
                failed += 1
                print(f"[error] {code}: {exc}", flush=True)

            if not dry_run and (idx % _CHECKPOINT_EVERY == 0):
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # -wal 파일 축소(디스크 보호)
            if idx % PROGRESS_EVERY == 0 or idx == total:
                elapsed = time.time() - started
                rate = idx / elapsed if elapsed else 0
                eta = (total - idx) / rate if rate else 0
                print(f"[{idx}/{total}] ok={ok} empty={empty} failed={failed} skipped={skipped} "
                      f"rows={rows_total:,} 경과={elapsed:.0f}s ETA={eta:.0f}s", flush=True)

        if not dry_run:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        report = {
            "mode": "dry-run" if dry_run else "applied",
            "codes_total": total,
            "ok": ok,
            "empty": empty,
            "failed": failed,
            "skipped": skipped,
            "rows_upserted": rows_total,
            "first_stock_elapsed_sec": round(first_elapsed, 2) if first_elapsed else None,
            "elapsed_sec": round(time.time() - started, 1),
        }
        print(f"일자별 상장주식수 백필 {report['mode']} 완료: {report}", flush=True)
        return report
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="일자별 상장주식수(daily_shares) 백필")
    ap.add_argument("--codes", default=None, help="쉼표구분 종목코드만 대상(예: 134380,003350,298000)")
    ap.add_argument("--dry-run", action="store_true", help="적재 없이 대상/시작일만 출력")
    ap.add_argument("--skip-covered", action="store_true", help="이미 오늘까지 커버된 종목은 스킵")
    ap.add_argument("--db", default=None, help="DB 경로(기본: CONFIG.db_path)")
    args = ap.parse_args()
    code_list = [c.strip() for c in args.codes.split(",") if c.strip()] if args.codes else None
    backfill_shares_daily(
        db_path=args.db, codes=code_list, dry_run=args.dry_run, skip_covered=args.skip_covered
    )
