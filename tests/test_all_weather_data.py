"""올웨더 데이터 수집 테스트 (AC1/AC2/AC6/AC7).

4종목(QQQ/삼성전자/TLT/ACE KRX금현물) 가격 패널을 만든다. 삼성전자는 기존 prices 테이블을
재사용하고 나머지 3종목은 yfinance(주입 가능)로 수집한다. 무위험이자율은 ^IRX를 시점별 과거값으로
조회한다. 실제 yfinance 호출은 주입한 fake로 대체해 네트워크 없이 검증한다(기존 DI 관례).
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from src.allweather.data import (
    IRX_TICKER,
    SAMSUNG_CODE,
    SAMSUNG_TICKER,
    TICKERS,
    build_price_panel,
    fetch_irx_series,
    risk_free_rate_at,
)
from src.db import init_db


def _db_with_samsung(tmp_path) -> str:
    db = str(tmp_path / "aw.db")
    init_db(db)
    conn = sqlite3.connect(db)
    dates = pd.date_range("2016-01-04", periods=30, freq="B")
    for i, d in enumerate(dates):
        conn.execute(
            "INSERT INTO prices(stock_code, date, close, market_cap) VALUES(?,?,?,?)",
            (SAMSUNG_CODE, d.strftime("%Y-%m-%d"), 50000.0 + i * 100, 3e14),
        )
    conn.commit()
    conn.close()
    return db


def test_all_four_tickers_present():
    # AC1: 4종목 모두 조회 대상에 포함.
    assert set(TICKERS) == {"QQQ", SAMSUNG_TICKER, "TLT", "411060.KS"}


def test_samsung_from_db_others_from_yfinance(tmp_path):
    # AC2: 삼성전자는 기존 prices 조회, 나머지 3종목만 yfinance(주입 fetch)로 수집.
    db = _db_with_samsung(tmp_path)
    called: list[str] = []

    def fake_fetch(ticker: str) -> pd.Series:
        called.append(ticker)
        dates = pd.date_range("2016-01-04", periods=30, freq="B")
        return pd.Series([100.0 + i for i in range(30)], index=dates)

    conn = sqlite3.connect(db)
    try:
        panel = build_price_panel(conn, fetch_fn=fake_fetch)
    finally:
        conn.close()

    assert set(called) == {"QQQ", "TLT", "411060.KS"}
    assert SAMSUNG_TICKER not in called
    assert set(panel.columns) == set(TICKERS)
    # 삼성 컬럼은 DB(50000대)에서 온 값이어야 한다.
    assert panel[SAMSUNG_TICKER].iloc[0] == 50000.0


def test_irx_ticker_constant_is_caret_irx():
    # AC6: 무위험이자율은 미국 3개월 국채(yfinance 티커 ^IRX).
    assert IRX_TICKER == "^IRX"


def test_fetch_irx_series_uses_irx_ticker():
    seen = {}

    def fake_dl(ticker: str) -> pd.Series:
        seen["ticker"] = ticker
        dates = pd.date_range("2016-01-04", periods=5, freq="B")
        return pd.Series([5.0, 5.1, 5.2, 5.3, 5.4], index=dates)

    fetch_irx_series(fetch_fn=fake_dl)
    assert seen["ticker"] == "^IRX"


def test_build_price_panel_handles_tz_aware_yfinance_index(tmp_path):
    # 실측 버그 재현: 실제 yfinance는 tz-aware DatetimeIndex를 반환한다(예: QQQ/TLT는
    # America/New_York). 삼성전자(DB, tz-naive)와 섞이면 pd.DataFrame(dict) 생성 시점에
    # "Cannot join tz-naive with tz-aware DatetimeIndex"로 죽는다 — 실제 배치 실행에서 재현됨.
    db = _db_with_samsung(tmp_path)

    def fake_fetch_tz_aware(ticker: str) -> pd.Series:
        dates = pd.date_range("2016-01-04", periods=30, freq="B", tz="America/New_York")
        return pd.Series([100.0 + i for i in range(30)], index=dates)

    conn = sqlite3.connect(db)
    try:
        panel = build_price_panel(conn, fetch_fn=fake_fetch_tz_aware)
    finally:
        conn.close()

    assert getattr(panel.index, "tz", None) is None
    assert set(panel.columns) == set(TICKERS)
    assert len(panel) > 0


def test_risk_free_rate_at_returns_point_in_time_value():
    # AC7: 리밸런싱 시점마다 그 시점의 과거 ^IRX 값을 쓴다(단일 현재값 고정 아님).
    dates = pd.to_datetime(["2016-01-04", "2017-06-01", "2018-01-02"])
    s = pd.Series([0.5, 1.5, 2.5], index=dates)  # percent
    r1 = risk_free_rate_at(s, pd.Timestamp("2017-01-01"))  # 가장 최근 <= 시점 = 0.5%
    r2 = risk_free_rate_at(s, pd.Timestamp("2018-01-02"))  # 2.5%
    assert abs(r1 - 0.005) < 1e-9
    assert abs(r2 - 0.025) < 1e-9
    assert r1 != r2
