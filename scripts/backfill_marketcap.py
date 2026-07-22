"""과거 시가총액 재계산 (prices.market_cap = 종가 × 그 시점 유효 상장주식수).

⚠️ 반드시 "분기별 상장주식수 수집(backfill_full / update_financials)" 후에 실행할 것.
   financials.shares_outstanding 이 충분히 쌓인 뒤 실행해야 의미 있는 결과가 나온다.

이 스크립트는 DART/pykrx 를 호출하지 않는다(이미 적재된 financials.shares_outstanding 만 사용)
→ DART 일일 한도와 무관하다.

동작
----
- prices 의 각 (stock_code, date) 행에 대해 그 시점 유효 상장주식수로 market_cap 을 갱신한다.
- "그 시점 유효 주식수": 그 주가일(date)에 **실제로 공시돼 있던**(disclosed_date <= date)
  shares_outstanding 중 가장 최근 값(look-ahead 방지). staleness·×1000 단위오류 이상치 가드
  포함 — 판정 로직은 src/ingest/metrics._effective_shares_outstanding 를 그대로 재사용해
  ingest(krx)·backtest·이 마이그레이션이 완전히 동일한 규약을 쓴다.
- 시점별 값이 없으면(financials 미적재 종목/기간, stale, 이상치) 그 행은 건드리지 않는다
  (기존 값 보존 — "지금보다 나쁘게 만들지 않는다"). 새 값이 기존과 같으면 UPDATE 를 건너뛴다.
- UPDATE 만 한다(새 테이블/컬럼 없음) → data/market.db 파일이 커지지 않는다.
  종목 단위로 커밋하고 주기적으로 WAL 체크포인트(TRUNCATE)해 -wal 파일이 디스크를 잠식하지
  않게 한다(38GB DB·여유 디스크 부족 환경 안전장치). idempotent: 다시 돌려도 안전.

실행
----
  python3 scripts/backfill_marketcap.py --dry-run   # 쓰기 없이 영향 규모/예상 시간만 보고
  python3 scripts/backfill_marketcap.py             # 실제 재계산(UPDATE)
"""
from __future__ import annotations

import argparse
import bisect
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import connect, init_db
from src.ingest.metrics import _SHARES_ANOMALY_RATIO, _effective_shares_outstanding

_CHECKPOINT_EVERY = 200  # 이 종목 수마다 WAL 체크포인트(TRUNCATE)로 -wal 파일 크기를 잡는다


def _shares_series(conn) -> dict:
    """종목별 상장주식수 시계열: {stock_code: [(disclosed_date'YYYY-MM-DD', shares), ...]} (공시일 오름차순).

    disclosed_date 결측/빈문자·비양수 행은 시점을 알 수 없어 제외(look-ahead 방지).
    (fix_split_discontinuities.py 가 이 함수를 재사용하므로 시그니처를 유지한다 — 전체 dict 반환.)
    """
    rows = conn.execute(
        """SELECT stock_code, disclosed_date, amount
             FROM financials
            WHERE account_key = 'shares_outstanding' AND amount IS NOT NULL AND amount > 0
              AND disclosed_date IS NOT NULL AND disclosed_date != ''
            ORDER BY stock_code, disclosed_date"""
    ).fetchall()
    series: dict = {}
    for r in rows:
        series.setdefault(r["stock_code"], []).append((str(r["disclosed_date"]), r["amount"]))
    for code in series:
        series[code].sort(key=lambda t: t[0])  # 'YYYY-MM-DD' 문자열 정렬 = 시간순
    return series


def _effective_shares(series: list, on_compact: str):
    """series(공시일 오름차순)에서 on_compact 시점까지 **공시된** 최신 주식수(가드 없음, 하위호환).

    fix_split_discontinuities.py 가 import 하는 순수 look-ahead 헬퍼(staleness/이상치 가드 없음).
    이 마이그레이션 본체(backfill_marketcap)는 가드가 포함된 _effective_shares_outstanding 을 쓴다.
    """
    best = None
    for disc, shares in series:
        if disc <= on_compact:
            best = shares  # 오름차순이라 마지막으로 통과한 값이 "그 시점까지 공시된 최신"
        else:
            break
    return best


def _daily_shares_for(conn, code: str) -> tuple[list, list]:
    """daily_shares 에서 code 의 (date 오름차순 리스트, shares 리스트). 비양수/NULL 제외.

    pykrx 일자별 정식 상장주식수 — backfill_marketcap 의 최우선 시총 소스다. date 는
    'YYYY-MM-DD' 라 문자열 정렬=시간순이므로 bisect 로 date<=asof 최신값을 O(log n) 조회한다.
    """
    rows = conn.execute(
        "SELECT date, shares_outstanding FROM daily_shares "
        "WHERE stock_code=? AND shares_outstanding IS NOT NULL AND shares_outstanding>0 "
        "ORDER BY date",
        (code,),
    ).fetchall()
    dates = [r["date"] for r in rows]
    vals = [r["shares_outstanding"] for r in rows]
    return dates, vals


def _daily_shares_at(dates: list, vals: list, asof: str):
    """date<=asof 중 최신 상장주식수(없으면 None). 가드 없음 — 그 값 자체가 그 날짜의 실제 값.

    daily_shares 는 KRX 일자별 정식 데이터라 disclosed_date/staleness/이상치 가드가 필요 없다
    (financials 기반 _effective_shares_outstanding 과 달리 단순 date<=asof 최신값만 고른다).
    """
    i = bisect.bisect_right(dates, asof)  # asof 초과 첫 인덱스
    return vals[i - 1] if i else None


def backfill_marketcap(
    db_path: str | None = None,
    *,
    dry_run: bool = False,
    progress_every: int = 500,
    codes: list[str] | None = None,
) -> dict:
    """prices.market_cap 을 종가 × (그 시점 유효 상장주식수)로 재계산한다.

    상장주식수 소스 우선순위: daily_shares(pykrx 일자별 정식값, 결산일 지연 없음) > financials.
    shares_outstanding(분기 공시, look-ahead/staleness/이상치 가드 포함) 폴백.
    dry_run=True 면 UPDATE 하지 않고 바뀔 행/종목 수만 집계해 보고한다(사전 규모 파악용).
    codes 지정 시 그 종목들만 대상으로 한다(daily_shares 검증 등 소규모 실행용).
    """
    init_db(db_path)
    conn = connect(db_path)
    try:
        series_map = _shares_series(conn)  # {code: [(disc, shares)...]} — 82k행, 메모리 무해
        if codes:
            qmarks = ",".join("?" * len(codes))
            code_list = [r["stock_code"] for r in conn.execute(
                f"SELECT DISTINCT stock_code FROM prices WHERE close IS NOT NULL "
                f"AND stock_code IN ({qmarks}) ORDER BY stock_code",
                codes,
            ).fetchall()]
        else:
            code_list = [r["stock_code"] for r in conn.execute(
                "SELECT DISTINCT stock_code FROM prices WHERE close IS NOT NULL ORDER BY stock_code"
            ).fetchall()]
        total_codes = len(code_list)

        price_rows = changed = unchanged = no_shares = codes_changed = spike_nulled = 0
        started = time.time()

        for idx, code in enumerate(code_list, 1):
            series = series_map.get(code)
            # 종목 median(이상치 판정 기준). _effective_shares_outstanding 과 동일 정의.
            amounts = sorted(a for _, a in series) if series else []
            median = amounts[len(amounts) // 2] if amounts else 0
            # 최우선 소스: daily_shares(pykrx 일자별 정식 상장주식수, 결산일 지연 없음).
            d_dates, d_vals = _daily_shares_for(conn, code)
            rows = conn.execute(
                "SELECT date, close, market_cap FROM prices WHERE stock_code=? AND close IS NOT NULL",
                (code,),
            ).fetchall()
            code_changed = 0
            for r in rows:
                price_rows += 1
                # prices.date·disclosed_date 모두 'YYYY-MM-DD' → 변환 없이 사전식=시간순 비교
                # daily_shares 우선(그 날짜 실제 값), 없는 구간만 financials 폴백(공시일 기반 가드값).
                shares = _daily_shares_at(d_dates, d_vals, r["date"]) if d_dates else None
                if not shares:
                    shares = _effective_shares_outstanding(series, r["date"]) if series else None
                if not shares:
                    no_shares += 1
                    # 확정 스파이크 오염 정리: 가드는 None 이지만 무가드 최신값(raw)이 이상치이고,
                    # 저장된 cap 이 정확히 '종가×그 스파이크값'이면 구버전 무가드 backfill 이 써둔
                    # ×1000 단위오류 오염이다(실측 1,343행/14종목). staleness(raw 정상)는 건드리지
                    # 않고 이 경우만 NULL 로 정리한다(1000x 틀린 값보다 결측이 낫다 — 랭킹 오염 제거).
                    if series and r["market_cap"] is not None and median > 0:
                        raw = _effective_shares(series, r["date"])
                        if (raw and (raw > median * _SHARES_ANOMALY_RATIO or raw < median / _SHARES_ANOMALY_RATIO)
                                and abs(r["market_cap"] - round(r["close"] * raw)) < 1):
                            spike_nulled += 1
                            code_changed += 1
                            if not dry_run:
                                conn.execute(
                                    "UPDATE prices SET market_cap=NULL WHERE stock_code=? AND date=?",
                                    (code, r["date"]),
                                )
                    continue  # 시점별 값 없음 → (오염 아니면) 기존 market_cap 보존
                new_cap = round(r["close"] * shares)
                if new_cap == r["market_cap"]:
                    unchanged += 1
                    continue
                changed += 1
                code_changed += 1
                if not dry_run:
                    conn.execute(
                        "UPDATE prices SET market_cap=? WHERE stock_code=? AND date=?",
                        (new_cap, code, r["date"]),
                    )
            if code_changed:
                codes_changed += 1

            if not dry_run and (idx % _CHECKPOINT_EVERY == 0):
                conn.commit()
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # -wal 파일 축소(디스크 보호)
            if idx % progress_every == 0 or idx == total_codes:
                elapsed = time.time() - started
                rate = idx / elapsed if elapsed else 0
                eta = (total_codes - idx) / rate if rate else 0
                print(f"[{idx}/{total_codes}] changed={changed:,} rows={price_rows:,} "
                      f"경과={elapsed:.0f}s ETA={eta:.0f}s", flush=True)

        if not dry_run:
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        report = {
            "mode": "dry-run" if dry_run else "applied",
            "codes_total": total_codes,
            "codes_changed": codes_changed,
            "price_rows_scanned": price_rows,
            "market_cap_changed": changed,
            "market_cap_unchanged": unchanged,
            "no_time_varying_shares": no_shares,
            "spike_pollution_nulled": spike_nulled,
            "elapsed_sec": round(time.time() - started, 1),
        }
        print(f"시가총액 재계산 {report['mode']} 완료: {report}", flush=True)
        return report
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="prices.market_cap 시점별 상장주식수로 재계산")
    ap.add_argument("--dry-run", action="store_true", help="쓰기 없이 영향 규모/예상 시간만 보고")
    ap.add_argument("--codes", default=None, help="쉼표구분 종목코드만 대상(예: 134380,003350,298000)")
    ap.add_argument("--db", default=None, help="DB 경로(기본: CONFIG.db_path)")
    args = ap.parse_args()
    code_list = [c.strip() for c in args.codes.split(",") if c.strip()] if args.codes else None
    print(backfill_marketcap(db_path=args.db, dry_run=args.dry_run, codes=code_list))
