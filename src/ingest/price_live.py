"""오늘(최신) 주가 실시간 1건 호출 + 당일 캐시.

"오늘/현재 PER" 같이 최신 주가가 필요한 질의에서 사용한다.
흐름: DB(prices)에 오늘 날짜 주가가 있으면 그대로 사용 → 없으면 pykrx로 해당
종목의 최근 거래일 종가 1건을 실시간 호출 → prices에 당일 행으로 저장(같은 날
재호출 방지) → 사용.

pykrx 전종목 일괄 API는 막혀 있어 개별 종목 조회만 쓴다. 시가총액은 시총 API가
막혀 알 수 없으므로 NULL로 둔다(상장주식수가 있으면 추후 별도 계산).

질문이 "오늘/현재/실시간"을 명시하지 않으면 호출하지 않는다(불필요한 네트워크 회피).
"""
from __future__ import annotations

import re
from datetime import date, timedelta

# "오늘/현재/지금/실시간" 등 최신 주가가 필요함을 시사하는 키워드.
_LIVE_HINT = re.compile(r"오늘|현재|지금|실시간|today|now|current", re.IGNORECASE)


def needs_live_price(question: str, route: str) -> bool:
    """질문이 '오늘/현재' 류이고 주가가 필요한 route면 True."""
    if route not in ("price", "both"):
        return False
    return bool(_LIVE_HINT.search(question or ""))


def _has_today(conn, code: str, today: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM prices WHERE stock_code=? AND date=? LIMIT 1", (code, today)
    ).fetchone()
    return row is not None


def _fetch_latest_close(stock, code: str, on: date, lookback: int = 10):
    """on 기준 최근 lookback일 범위에서 가장 최근 거래일 종가 1건."""
    frm = (on - timedelta(days=lookback)).strftime("%Y%m%d")
    to = on.strftime("%Y%m%d")
    try:
        df = stock.get_market_ohlcv_by_date(frm, to, code)
    except Exception:  # noqa: BLE001 — 네트워크/차단은 격리
        return None
    if df is None or len(df) == 0 or "종가" not in df.columns:
        return None
    last = df.iloc[-1]
    close = float(last["종가"])
    return close if close > 0 else None


def ensure_live_prices(conn, codes, on: date | None = None) -> dict:
    """codes 각각에 대해 오늘 종가를 DB에 보장(없으면 실시간 1건 호출 후 저장).

    반환: {"fetched": 신규 호출 건수, "cached": 이미 있던 건수, "today": 'YYYY-MM-DD'}
    """
    on = on or date.today()
    today = on.strftime("%Y-%m-%d")
    fetched = cached = 0
    stock = None
    for code in codes:
        if _has_today(conn, code, today):
            cached += 1
            continue
        if stock is None:
            from pykrx import stock as _stock  # 지연 import (네트워크 의존)

            stock = _stock
        close = _fetch_latest_close(stock, code, on)
        if close is None:
            continue
        # 당일 행으로 저장 → 같은 날 재호출 방지. market_cap은 시총 API 차단으로 NULL이지만,
        # 이미 다른 경로(backfill_marketcap 등)로 채워진 값이 있으면 지우지 않는다.
        # close도 네이버(수정주가, AC3 source of truth)가 이미 채웠으면 보존하고,
        # 없을 때만 pykrx 원본값으로 채운다(경쟁상황에서 수정주가를 덮어쓰지 않도록).
        conn.execute(
            "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,NULL) "
            "ON CONFLICT(stock_code, date) DO UPDATE SET close=COALESCE(close, excluded.close)",
            (code, today, close),
        )
        fetched += 1
    if fetched:
        conn.commit()
    return {"fetched": fetched, "cached": cached, "today": today}


