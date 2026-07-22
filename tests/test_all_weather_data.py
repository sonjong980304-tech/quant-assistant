"""올웨더 데이터 수집 테스트 (AC1/AC2/AC6/AC7 + 20년 이력 확보를 위한 재설계).

4종목(QQQ/삼성전자/TLT/ACE KRX금현물) 모두 yfinance로 수집한다(삼성전자도 DB 대신 yfinance —
2000년부터 제공돼 기존 DB(2014~)보다 훨씬 길다). ACE KRX금현물(411060.KS)은 2021년 말에야
상장돼 그 이전 데이터가 없으므로, GLD(달러 표시 금ETF)×USDKRW=X(환율) 합성값으로 과거를
채우고 실제 411060.KS 데이터가 있는 구간은 항상 진짜 시세를 우선한다(스플라이스).
무위험이자율은 ^IRX를 시점별 과거값으로 조회한다. 실제 yfinance 호출은 주입한 fake로
대체해 네트워크 없이 검증한다(기존 DI 관례).
"""
from __future__ import annotations

import pandas as pd

from src.allweather.data import (
    FX_TICKER,
    GOLD_ETF_TICKER,
    IRX_TICKER,
    KRX_GOLD_TICKER,
    SAMSUNG_TICKER,
    TICKERS,
    build_price_panel,
    build_synthetic_krx_gold_series,
    fetch_irx_series,
    risk_free_rate_at,
)


def test_all_six_tickers_present():
    # AC1 + 2026-07 MDD 제약 도입: 6종목(기존 4 + IEF/TIP) 모두 조회 대상에 포함.
    assert set(TICKERS) == {"QQQ", SAMSUNG_TICKER, "TLT", KRX_GOLD_TICKER, "IEF", "TIP"}


def test_samsung_sourced_from_yfinance_not_db():
    # 재설계: 삼성전자도 이제 DB가 아니라 yfinance로 조회한다(2000년~ 확보 목적).
    called: list[str] = []

    def fake_fetch(ticker: str) -> pd.Series:
        called.append(ticker)
        dates = pd.date_range("2010-01-04", periods=1500, freq="B")
        return pd.Series([100.0 + i for i in range(len(dates))], index=dates)

    panel = build_price_panel(fetch_fn=fake_fetch)

    assert SAMSUNG_TICKER in called
    assert set(panel.columns) == set(TICKERS)
    assert len(panel) > 0


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


def test_build_price_panel_handles_tz_aware_yfinance_index():
    # 실측 버그 재현: 실제 yfinance는 tz-aware DatetimeIndex를 반환한다(예: QQQ/TLT는
    # America/New_York). 서로 다른 티커를 섞으면 pd.DataFrame(dict) 생성 시점에
    # "Cannot join tz-naive with tz-aware DatetimeIndex"로 죽는다 — 실제 배치 실행에서 재현됨.
    def fake_fetch_tz_aware(ticker: str) -> pd.Series:
        dates = pd.date_range("2010-01-04", periods=1500, freq="B", tz="America/New_York")
        return pd.Series([100.0 + i for i in range(len(dates))], index=dates)

    panel = build_price_panel(fetch_fn=fake_fetch_tz_aware)

    assert getattr(panel.index, "tz", None) is None
    assert set(panel.columns) == set(TICKERS)
    assert len(panel) > 0


def test_gold_series_uses_synthetic_before_krx_listing_and_real_after():
    # 411060.KS 상장 이전은 GLD×환율 합성값, 이후는 실제 시세를 쓴다(20년 이력 확보).
    listing_date = pd.Timestamp("2021-06-01")

    def fake_fetch(ticker: str) -> pd.Series:
        if ticker == GOLD_ETF_TICKER:
            dates = pd.date_range("2018-01-01", periods=1000, freq="B")
            return pd.Series([100.0] * len(dates), index=dates)  # 달러 금가격 100 고정
        if ticker == FX_TICKER:
            dates = pd.date_range("2018-01-01", periods=1000, freq="B")
            return pd.Series([1000.0] * len(dates), index=dates)  # 환율 1000원 고정
        if ticker == KRX_GOLD_TICKER:
            dates = pd.date_range(listing_date, periods=100, freq="B")
            return pd.Series([999999.0] * len(dates), index=dates)  # 실제값 구분용 특이값
        raise AssertionError(f"unexpected ticker: {ticker}")

    combined = build_synthetic_krx_gold_series(fetch_fn=fake_fetch)

    before = combined[combined.index < listing_date]
    after = combined[combined.index >= listing_date]

    assert len(before) > 0
    assert len(after) > 0
    assert (before == 100.0 * 1000.0).all()  # 합성값 = GLD(달러) × 환율
    assert (after == 999999.0).all()  # 실제 411060.KS 값을 그대로 스플라이스


def test_gold_series_falls_back_to_synthetic_when_no_real_data():
    # 411060.KS fetch가 빈 결과를 반환해도(예: 네트워크 이슈) 합성값만으로라도 채운다.
    def fake_fetch(ticker: str) -> pd.Series:
        dates = pd.date_range("2018-01-01", periods=500, freq="B")
        if ticker == GOLD_ETF_TICKER:
            return pd.Series([100.0] * len(dates), index=dates)
        if ticker == FX_TICKER:
            return pd.Series([1000.0] * len(dates), index=dates)
        if ticker == KRX_GOLD_TICKER:
            return pd.Series(dtype="float64")
        raise AssertionError(f"unexpected ticker: {ticker}")

    combined = build_synthetic_krx_gold_series(fetch_fn=fake_fetch)

    assert len(combined) > 0
    assert (combined == 100.0 * 1000.0).all()


def test_risk_free_rate_at_returns_point_in_time_value():
    # AC7: 리밸런싱 시점마다 그 시점의 과거 ^IRX 값을 쓴다(단일 현재값 고정 아님).
    dates = pd.to_datetime(["2016-01-04", "2017-06-01", "2018-01-02"])
    s = pd.Series([0.5, 1.5, 2.5], index=dates)  # percent
    r1 = risk_free_rate_at(s, pd.Timestamp("2017-01-01"))  # 가장 최근 <= 시점 = 0.5%
    r2 = risk_free_rate_at(s, pd.Timestamp("2018-01-02"))  # 2.5%
    assert abs(r1 - 0.005) < 1e-9
    assert abs(r2 - 0.025) < 1e-9
    assert r1 != r2
