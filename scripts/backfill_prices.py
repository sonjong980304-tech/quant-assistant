"""과거 주가 백필 (종목별 루프, 10년치 1회 호출).

⚠️ DART 재무 백필(backfill_full.py)이 완료된 후에 실행할 것.
   재무 백필이 launchd로 진행 중인 동안에는 절대 실행하지 말 것
   (pykrx/DART 한도 및 진행 보호). 이 스크립트는 작성만 해두고 대기한다.

배경
----
pykrx 전종목 일괄 API(get_market_ohlcv market="ALL", get_market_ticker_list,
get_market_cap_by_date)는 전부 막혀 있다. 유일하게 작동하는
get_market_ohlcv_by_date(fromdate, todate, ticker)(개별 종목 기간 시계열)만
사용한다. 따라서 "종목별 루프"로 수집한다(날짜별 전종목 금지).

동작
----
- company 테이블의 모든 stock_code를 순회.
- 각 종목: get_market_ohlcv_by_date("20150101", 오늘, code)로 10년치 1회 호출.
- CONFIG.price_history_freq=="monthly"면 각 월의 마지막 거래일 종가만, "daily"면
  전체 일별을 prices에 UPSERT(INSERT OR REPLACE; UNIQUE(stock_code,date)로 중복 방지).
- market_cap은 pykrx 시총 API가 막혀 알 수 없으므로 NULL로 둔다(close만 저장).
  상장주식수가 추후 채워지면 별도로 계산한다.
- resume: "이미 prices에 데이터가 있는 종목은 스킵"(재무 백필과 동일 패턴).
  체크포인트(ingest_meta.price_backfill_checkpoint)에 마지막 완료 종목을 기록한다.
- 종목별 try/except로 실패를 격리하고, 실패한 종목 목록을 모아 1회 재시도한다.

실행: python3 scripts/backfill_prices.py   ← 재무 백필 완료 후에만!
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.db import connect, init_db, set_meta

START_DATE = "20150101"  # 10년치 시작
SLEEP_SEC = 0.5          # 종목 간 딜레이(API 보호)


def _company_codes(conn) -> list[str]:
    """company 테이블의 모든 종목코드."""
    return [r[0] for r in conn.execute("SELECT stock_code FROM company ORDER BY stock_code")]


def _codes_with_prices(conn) -> set:
    """이미 prices에 데이터가 있는 종목코드 집합 (resume용 스킵 대상)."""
    return {r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM prices")}


def _ingest_one(stock, conn, code: str, to: str) -> int:
    """종목 1개 10년치 수집 → prices UPSERT. 적재 행수 반환."""
    df = stock.get_market_ohlcv_by_date(START_DATE, to, code)
    if df is None or len(df) == 0 or "종가" not in df.columns:
        return 0

    if CONFIG.price_history_freq == "monthly":
        # 월별 마지막 거래일만: 'YYYY-MM' 그룹의 마지막 인덱스를 고른다.
        last_idx: dict = {}
        for dt in df.index:
            last_idx[dt.strftime("%Y-%m")] = dt
        rows = [(dt, df.loc[dt]) for dt in last_idx.values()]
    else:
        rows = list(df.iterrows())

    n = 0
    for dt, row in rows:
        close = float(row["종가"])
        if close <= 0:
            continue
        d = dt.strftime("%Y-%m-%d")
        # market_cap은 시총 API 차단으로 알 수 없어 NULL이지만, 이미 다른 경로
        # (backfill_marketcap 등)로 채워진 값이 있으면 지우지 않는다. close도
        # 네이버(수정주가, AC3 source of truth)가 이미 채웠으면 보존하고, 없을 때만
        # pykrx 원본값으로 채운다(액면분할 미조정 값이 수정주가를 덮어쓰지 않도록).
        conn.execute(
            "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,NULL) "
            "ON CONFLICT(stock_code, date) DO UPDATE SET close=COALESCE(close, excluded.close)",
            (code, d, close),
        )
        n += 1
    conn.commit()
    return n


def backfill_prices(db_path: str | None = None, on: date | None = None) -> dict:
    """company 전 종목 10년치 과거 주가를 종목별 루프로 백필."""
    from pykrx import stock  # 지연 import (네트워크 의존)

    on = on or date.today()
    to = on.strftime("%Y%m%d")
    init_db(db_path)
    conn = connect(db_path)
    try:
        all_codes = _company_codes(conn)
        done = _codes_with_prices(conn)  # 이미 데이터 있는 종목 스킵
        targets = [c for c in all_codes if c not in done]

        total = len(all_codes)
        ok = skipped = rows_total = 0
        failed: list[str] = []
        print(
            f"[price] {json.dumps({'event': 'start', 'freq': CONFIG.price_history_freq, 'total': total, 'targets': len(targets), 'already': len(done)}, ensure_ascii=False)}",
            flush=True,
        )

        for code in targets:
            try:
                n = _ingest_one(stock, conn, code, to)
                if n == 0:
                    # 데이터 없음(상장 전/비거래)도 실패로 보지 않고 스킵 처리.
                    skipped += 1
                    print(f"[price] {json.dumps({'code': code, 'status': 'empty'}, ensure_ascii=False)}", flush=True)
                else:
                    ok += 1
                    rows_total += n
                    set_meta(conn, "price_backfill_checkpoint", code)
                    print(f"[price] {json.dumps({'code': code, 'status': 'ok', 'rows': n}, ensure_ascii=False)}", flush=True)
            except Exception as exc:  # noqa: BLE001 — 종목 실패 격리
                failed.append(code)
                print(f"[price] {json.dumps({'code': code, 'status': 'error', 'error': str(exc)}, ensure_ascii=False)}", flush=True)
            time.sleep(SLEEP_SEC)

        # 실패 종목 1회 재시도
        retried_ok = 0
        still_failed: list[str] = []
        if failed:
            print(f"[price] {json.dumps({'event': 'retry', 'count': len(failed)}, ensure_ascii=False)}", flush=True)
            for code in failed:
                try:
                    n = _ingest_one(stock, conn, code, to)
                    if n > 0:
                        retried_ok += 1
                        rows_total += n
                        set_meta(conn, "price_backfill_checkpoint", code)
                        print(f"[price] {json.dumps({'code': code, 'status': 'retry_ok', 'rows': n}, ensure_ascii=False)}", flush=True)
                    else:
                        still_failed.append(code)
                except Exception as exc:  # noqa: BLE001
                    still_failed.append(code)
                    print(f"[price] {json.dumps({'code': code, 'status': 'retry_error', 'error': str(exc)}, ensure_ascii=False)}", flush=True)
                time.sleep(SLEEP_SEC)

        report = {
            "total_companies": total,
            "already_had_prices": len(done),
            "ok": ok,
            "retried_ok": retried_ok,
            "empty_or_skipped": skipped,
            "failed": still_failed,
            "price_rows": rows_total,
            "freq": CONFIG.price_history_freq,
            "range": f"{START_DATE}~{to}",
        }
        print(f"[price] {json.dumps({'event': 'done', **report}, ensure_ascii=False)}", flush=True)
        return report
    finally:
        conn.close()


if __name__ == "__main__":
    # ⚠️ DART 재무 백필 완료 후에만 실행할 것.
    print(backfill_prices())
