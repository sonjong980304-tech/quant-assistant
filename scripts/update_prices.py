"""일별 주가 증분 갱신 (장마감 후 1회).

prices의 마지막 적재일 다음날 ~ 오늘 범위에서 빠진 거래일 종가를 개별 종목으로
수집·적재한다(INSERT OR REPLACE). 주식수는 매일 수집하지 않고(분기별로 financials에
이미 있음), 주가만 넣은 뒤 실행 끝에 backfill_marketcap이 그 분기 주식수로 시총을 채운다.

Mac 꺼짐/잠자기로 누락된 날짜는 다음 실행 때 갭(마지막 적재일~오늘)을 다시
훑어 메꾼다. 주말/공휴일은 pykrx에 데이터가 없어 자연히 건너뛴다.

실행: python3 scripts/update_prices.py
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import connect, init_db, set_meta


def _codes(conn) -> list:
    """갱신 대상 종목코드: company 테이블(DART corpCode 확정분)을 1순위로 사용.

    pykrx 전종목 목록 API(get_market_ticker_list)는 막혀 있으므로 종목 목록은
    company 테이블에서 가져온다. 비어 있으면 universe→COMPANIES 순으로 폴백.
    """
    rows = [r[0] for r in conn.execute("SELECT stock_code FROM company ORDER BY stock_code")]
    if rows:
        return rows
    try:
        from src.ingest.universe import get_universe

        return [row[0] for row in get_universe()]
    except Exception:
        from src.ingest.companies import COMPANIES

        return [row[0] for row in COMPANIES]


# 첫 실행 등으로 prices가 비어 있을 때 시작점 (너무 과거로 가지 않도록 최근 14일)
DEFAULT_LOOKBACK_DAYS = 14


def _last_price_date(conn) -> date | None:
    row = conn.execute("SELECT MAX(date) AS d FROM prices").fetchone()
    if row and row["d"]:
        return datetime.strptime(row["d"], "%Y-%m-%d").date()
    return None


def update_prices(db_path: str | None = None, on: date | None = None) -> dict:
    """마지막 적재일 다음날~오늘 갭 구간의 거래일 종가를 증분 적재."""
    from pykrx import stock  # 지연 import (네트워크 의존)

    on = on or date.today()
    init_db(db_path)
    conn = connect(db_path)
    try:
        last = _last_price_date(conn)
        start = (last + timedelta(days=1)) if last else (on - timedelta(days=DEFAULT_LOOKBACK_DAYS))
        if start > on:
            print(f"갱신 불필요: 마지막 적재일 {last} >= 오늘 {on}")
            return {"price_rows": 0, "dates": [], "start": start.isoformat(), "end": on.isoformat()}

        frm = start.strftime("%Y%m%d")
        to = on.strftime("%Y%m%d")
        # 상장주식수는 분기별로 financials에 이미 적재돼 있으므로 매일 실시간 수집하지 않는다
        # (매일 전종목 DART 재조회는 한도 낭비이고, corpCode 실패 시 전체가 막힌다).
        # 여기서는 주가만 넣고(market_cap=NULL), 실행 후 backfill_marketcap이 분기 주식수로 시총을 채운다.
        dates: set = set()
        n = 0
        for code in _codes(conn):
            try:
                df = stock.get_market_ohlcv_by_date(frm, to, code)
            except Exception:
                continue
            if df is None or len(df) == 0 or "종가" not in df.columns:
                continue
            for dt, row in df.iterrows():
                close = float(row["종가"])
                if close <= 0:
                    continue
                d = dt.strftime("%Y-%m-%d")
                # INSERT OR REPLACE는 네이버 크롤러가 채운 OHLCV와 기존 market_cap을
                # 지운다 — ON CONFLICT DO UPDATE로 close만 갱신한다. market_cap은
                # 이 함수 실행 직후 backfill_marketcap()이 항상 재계산하므로(멱등)
                # 여기서 NULL로 미리 지울 필요가 없다.
                # close는 네이버(수정주가, AC3 source of truth)가 이미 채웠으면 보존하고,
                # 없을 때만 pykrx 원본값으로 채운다 — 이 스크립트는 launchd로 하루 3회
                # 자동 실행되는데, 무조건 덮어쓰면 액면분할/병합 전후 가격 불연속이
                # 매일 재발생해 return_12m 등 장기수익률 계산이 전부 오염된다(실측 확인).
                conn.execute(
                    """INSERT INTO prices(stock_code, date, close, market_cap)
                       VALUES (?,?,?,NULL)
                       ON CONFLICT(stock_code, date) DO UPDATE SET close=COALESCE(close, excluded.close)""",
                    (code, d, close),
                )
                dates.add(d)
                n += 1
        conn.commit()

        latest = max(dates) if dates else (last.isoformat() if last else None)
        if latest:
            set_meta(conn, "latest_price_date", latest)

        result = {
            "price_rows": n,
            "dates": sorted(dates),
            "latest_price_date": latest,
            "range": f"{frm}~{to}",
        }
        print(f"적재 거래일 {len(dates)}일 ({n}행), 최신 종가일 {latest}")
        return result
    finally:
        conn.close()


if __name__ == "__main__":
    res = update_prices()
    print(res)
    # 새 주가가 적재됐다면, 실시간 주식수 수집 성공 여부와 무관하게 이미 적재된
    # 상장주식수(financials)로 시총을 채운다. → 시총이 매일 자동으로 최신화된다.
    # (새 주가가 없으면 재계산할 것도 없으므로 스킵)
    if res.get("price_rows"):
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from backfill_marketcap import backfill_marketcap

            backfill_marketcap()
        except Exception as exc:  # noqa: BLE001 — 시총 백필 실패는 주가 갱신을 무효화하지 않는다
            print(f"시총 백필 스킵: {exc}")
