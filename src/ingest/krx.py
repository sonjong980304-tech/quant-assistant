"""pykrx 주가/시가총액 적재 (종가 스냅샷).

장중 실시간이 아니라 종가 스냅샷(일 1회) 기준. 마감 후 그날 종가로
prices를 갱신하고, metrics(PER/PBR/시총)를 재계산한다.

주의: KRX 전종목 일괄 API(get_market_cap_by_ticker)는 차단되어, 개별 종목
일자별 조회(get_market_ohlcv_by_date)로 우회한다. 시가총액은 KRX 시총 API도
막혀 있어 '종가 × 상장주식수'로 계산한다(상장주식수는 DART 사업보고서).
"""
from __future__ import annotations

from datetime import date, timedelta

from ..config import CONFIG
from ..db import connect, init_db, set_meta
from .companies import COMPANIES
from .metrics import compute_metrics


def _latest_close(stock, ticker: str, on: date, lookback: int = 30):
    """on 기준 직전 lookback일 범위에서 가장 최근 거래일 종가."""
    frm = (on - timedelta(days=lookback)).strftime("%Y%m%d")
    to = on.strftime("%Y%m%d")
    try:
        df = stock.get_market_ohlcv_by_date(frm, to, ticker)
    except Exception:
        return None, None
    if df is None or len(df) == 0 or "종가" not in df.columns:
        return None, None
    last = df.iloc[-1]
    return float(last["종가"]), df.index[-1].strftime("%Y-%m-%d")


def _collect_shares(on: date) -> dict:
    """DART 사업보고서에서 종목별 상장주식수 (최근 연도 우선)."""
    if not CONFIG.has_dart_key:
        return {}
    from .dart import fetch_shares, get_corp_codes

    corp_map = get_corp_codes(CONFIG.dart_api_key)
    shares: dict = {}
    for code, *_ in COMPANIES:
        corp = corp_map.get(code)
        if not corp:
            continue
        for yr in (on.year - 1, on.year - 2, on.year):
            s = fetch_shares(CONFIG.dart_api_key, corp, yr)
            if s:
                shares[code] = s
                break
    return shares


def ingest_price_history(db_path: str | None = None, years: int = 3, on: date | None = None) -> dict:
    """백테스트용 월별 종가 시계열 적재 (개별 종목 + 종가×상장주식수 시총).

    pykrx 개별 종목 일자별 조회(freq='m')로 월말 종가를 받는다.
    """
    from pykrx import stock

    on = on or date.today()
    frm = (on - timedelta(days=365 * years + 31)).strftime("%Y%m%d")
    to = on.strftime("%Y%m%d")
    init_db(db_path)
    conn = connect(db_path)
    try:
        shares_map = _collect_shares(on)
        n = 0
        done = 0
        for code, *_ in COMPANIES:
            try:
                df = stock.get_market_ohlcv_by_date(frm, to, code, freq="m")
            except Exception:
                continue
            if df is None or len(df) == 0 or "종가" not in df.columns:
                continue
            shares = shares_map.get(code)
            for dt, row in df.iterrows():
                close = float(row["종가"])
                if close <= 0:
                    continue
                d = dt.strftime("%Y-%m-%d")
                cap = round(close * shares) if shares else None
                # INSERT OR REPLACE는 행 전체를 갈아끼워 네이버 크롤러(naver_prices.py)가
                # 채운 open/high/low/volume을 NULL로 지운다 — ON CONFLICT DO UPDATE로
                # market_cap만 갱신한다. close는 네이버(수정주가, AC3 source of truth)가
                # 이미 채웠으면 보존하고, 없을 때만 pykrx 원본값으로 채운다(액면분할
                # 미조정 pykrx 값이 수정주가를 덮어써 불연속을 만드는 것을 방지).
                conn.execute(
                    "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,?) "
                    "ON CONFLICT(stock_code, date) DO UPDATE SET "
                    "close=COALESCE(close, excluded.close), market_cap=excluded.market_cap",
                    (code, d, close, cap),
                )
                n += 1
            done += 1
        conn.commit()
        return {"price_rows": n, "tickers": done, "shares_collected": len(shares_map),
                "range": f"{frm}~{to}"}
    finally:
        conn.close()


def ingest_prices(db_path: str | None = None, on: date | None = None, recompute: bool = True) -> dict:
    """최근 거래일의 종가 + (종가×상장주식수) 시가총액 스냅샷을 적재."""
    from pykrx import stock  # 지연 import (네트워크 의존)

    on = on or date.today()
    init_db(db_path)
    conn = connect(db_path)
    try:
        shares_map = _collect_shares(on)

        price_date = None
        n = 0
        no_shares = []
        for code, *_ in COMPANIES:
            close, d = _latest_close(stock, code, on)
            if close is None:
                continue
            price_date = max(price_date or d, d)
            shares = shares_map.get(code)
            cap = round(close * shares) if shares else None
            if not shares:
                no_shares.append(code)
            # INSERT OR REPLACE는 네이버 크롤러가 채운 OHLCV를 지운다 — ON CONFLICT
            # DO UPDATE로 market_cap만 갱신한다. close는 네이버(수정주가, AC3 source of
            # truth)가 이미 채웠으면 보존하고, 없을 때만 pykrx 원본값으로 채운다.
            conn.execute(
                """INSERT INTO prices(stock_code, date, close, market_cap)
                   VALUES (?,?,?,?)
                   ON CONFLICT(stock_code, date) DO UPDATE SET
                   close=COALESCE(close, excluded.close), market_cap=excluded.market_cap""",
                (code, d, close, cap),
            )
            n += 1
        if price_date:
            set_meta(conn, "latest_price_date", price_date)
        conn.commit()

        n_metrics = compute_metrics(conn, on) if recompute else 0
        return {
            "prices_rows": n,
            "price_date": price_date,
            "shares_collected": len(shares_map),
            "no_shares": no_shares,
            "metrics": n_metrics,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    print(ingest_prices())
