"""US 주가 — yfinance 히스토리 매퍼 + 오케스트레이터.

`Ticker(symbol).history(...)`가 반환하는 pandas DataFrame(DatetimeIndex,
Open/High/Low/Close/Volume 컬럼)을 `us_prices` 스키마 행으로 변환한다.
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from ..db import connect, init_db
from .notify import send_slack_alert
from .robust import log_ingest

_BACKFILL_PERIOD = "10y"
_INCREMENTAL_PERIOD = "5d"


def normalize_price_history(stock_code: str, history_df: pd.DataFrame) -> list[dict]:
    """yfinance 히스토리 DataFrame을 us_prices 행 리스트로 변환. Close 결측 행은 제외."""
    rows: list[dict] = []
    for ts, row in history_df.iterrows():
        close = row.get("Close")
        if pd.isna(close):
            continue
        rows.append({
            "stock_code": stock_code,
            "date": ts.strftime("%Y-%m-%d"),
            "open": None if pd.isna(row.get("Open")) else float(row["Open"]),
            "high": None if pd.isna(row.get("High")) else float(row["High"]),
            "low": None if pd.isna(row.get("Low")) else float(row["Low"]),
            "close": float(close),
            "volume": None if pd.isna(row.get("Volume")) else float(row["Volume"]),
        })
    return rows


def _fetch_yf_history(stock_code: str, period: str) -> pd.DataFrame:
    import yfinance as yf  # 지연 import — pykrx.py의 기존 패턴과 동일

    # auto_adjust=True 명시 — 배당·액면분할 반영 수정주가로 일관되게 받는다(라이브러리 기본값에 암묵 의존 금지).
    return yf.Ticker(stock_code).history(period=period, auto_adjust=True)


def ingest_us_prices(
    db_path: str | None = None,
    fetch_history: Optional[Callable[[str, str], pd.DataFrame]] = None,
    commit_every: int = 1,
) -> dict:
    """us_company 전종목의 OHLCV를 yfinance에서 받아 us_prices에 upsert.

    기존 데이터가 없는 신규 종목은 10년 백필, 있는 종목은 최근 구간만 증분
    수집한다(스펙 Round 3). 실패 종목은 건너뛰고 Slack 알림만 보낸다.

    commit_every개 종목마다 주기적으로 commit한다. 기본값 1(종목마다 즉시 commit) —
    commit_every가 크면 쓰기 트랜잭션이 오래 열려있어 동시에 뜬 웹 서버의 다른 쓰기가
    `database is locked`로 실패한다(us_financials.py에서 실제로 발생·확인됨). 종목마다
    commit하면 (1) 프로세스가 죽어도 진행이 안 날아가고, (2) 잠금 구간이 수 밀리초로
    줄어 동시 접근과 공존 가능하다.
    """
    fetch_history = fetch_history or _fetch_yf_history
    init_db(db_path)
    conn = connect(db_path)
    try:
        codes = [r["stock_code"] for r in conn.execute("SELECT stock_code FROM us_company").fetchall()]
        succeeded = 0
        failed: list[str] = []
        price_rows = 0
        for code in codes:
            has_history = conn.execute(
                "SELECT 1 FROM us_prices WHERE stock_code = ? LIMIT 1", (code,)
            ).fetchone()
            period = _INCREMENTAL_PERIOD if has_history else _BACKFILL_PERIOD
            try:
                history_df = fetch_history(code, period)
                rows = normalize_price_history(code, history_df)
                if not rows:
                    raise ValueError("빈 응답(필드 누락)")
            except Exception as exc:  # noqa: BLE001 — 종목별 실패를 격리해 다음 종목으로 계속
                failed.append(code)
                log_ingest({"source": "us_prices", "stock_code": code, "status": "fail", "error": str(exc)})
                send_slack_alert(f"[us_prices] {code} 수집 실패: {exc}")
                continue
            for row in rows:
                conn.execute(
                    "INSERT OR REPLACE INTO us_prices(stock_code, date, open, high, low, close, volume) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (code, row["date"], row["open"], row["high"], row["low"], row["close"], row["volume"]),
                )
                price_rows += 1
            succeeded += 1
            if succeeded % commit_every == 0:
                conn.commit()
        conn.commit()
        return {"tickers": len(codes), "succeeded": succeeded, "failed": failed, "price_rows": price_rows}
    finally:
        conn.close()
