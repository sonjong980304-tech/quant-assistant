"""올웨더 4종목 가격 패널 + 무위험이자율(^IRX) 조회.

.omc/specs/brainstorming-all-weather-portfolio.md AC1/AC2/AC6/AC7 참고.

- 삼성전자(005930)는 기존 prices 테이블(2014~ 데이터)을 그대로 재사용한다(AC2).
- 나머지 3종목(QQQ/TLT/ACE KRX금현물 411060.KS)은 yfinance로 신규 수집한다(AC2).
- 무위험이자율은 미국 3개월 국채(^IRX)를 리밸런싱 시점별 과거값으로 조회한다(AC6/AC7).

실제 yfinance 호출은 지연 import + 주입 가능한 fetch_fn 으로 분리한다(us_prices.py 등 기존 DI 관례).
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

# 4종목 티커(quant_trader SAFE_TICKERS와 동일) → 표시명.
SAMSUNG_TICKER = "005930.KS"
SAMSUNG_CODE = "005930"  # 기존 prices 테이블의 stock_code
IRX_TICKER = "^IRX"  # 미국 3개월 국채 금리(퍼센트 표기)

TICKERS: dict[str, str] = {
    "QQQ": "QQQ (나스닥 ETF)",
    SAMSUNG_TICKER: "삼성전자",
    "TLT": "TLT (미국 장기채)",
    "411060.KS": "ACE KRX금현물",
}


# ---------------------------------------------------------------------------
# 삼성전자 — 기존 prices 테이블 재사용
# ---------------------------------------------------------------------------
def load_samsung_series(conn) -> pd.Series:
    """prices 테이블에서 삼성전자(005930) 종가 시계열을 날짜 오름차순 Series로 반환."""
    rows = conn.execute(
        "SELECT date, close FROM prices WHERE stock_code=? AND close IS NOT NULL ORDER BY date ASC",
        (SAMSUNG_CODE,),
    ).fetchall()
    if not rows:
        return pd.Series(dtype="float64")
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.Series([float(r[1]) for r in rows], index=idx, name=SAMSUNG_TICKER)


# ---------------------------------------------------------------------------
# yfinance 종가 (삼성 제외 3종목)
# ---------------------------------------------------------------------------
def _yf_close_series(ticker: str) -> pd.Series:
    """yfinance 수정종가 Series (지연 import — us_prices.py 패턴). auto_adjust=True 명시."""
    import yfinance as yf

    df = yf.Ticker(ticker).history(period="max", auto_adjust=True)
    return df["Close"].dropna()


def fetch_yf_close_series(ticker: str, fetch_fn: Callable[[str], pd.Series] | None = None) -> pd.Series:
    """티커의 종가 Series를 반환한다(DatetimeIndex). fetch_fn 주입 시 그것을 쓴다(테스트용)."""
    fetch_fn = fetch_fn or _yf_close_series
    s = fetch_fn(ticker)
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    # 실제 yfinance는 tz-aware DatetimeIndex를 반환한다(티커별 거래소 시간대가 다름 —
    # 예: QQQ/TLT=America/New_York, 411060.KS=Asia/Seoul). 삼성전자(DB, tz-naive)와
    # 섞으면 build_price_panel의 pd.DataFrame(dict) 생성 시점에 죽으므로 여기서 미리 통일한다.
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s


def build_price_panel(
    conn,
    fetch_fn: Callable[[str], pd.Series] | None = None,
    tickers: dict[str, str] | None = None,
) -> pd.DataFrame:
    """4종목 종가를 공통 거래일 기준 하나의 DataFrame(컬럼=티커)으로 합친다.

    삼성전자는 conn의 prices 테이블에서, 나머지는 fetch_fn(yfinance)로 가져온다(AC1/AC2).
    각 종목이 모두 존재하는 거래일만 남긴다(dropna) — 몬테카를로 pct_change 계산 전제.
    """
    tickers = tickers or TICKERS
    series: dict[str, pd.Series] = {}
    for tk in tickers:
        if tk == SAMSUNG_TICKER:
            series[tk] = load_samsung_series(conn)
        else:
            series[tk] = fetch_yf_close_series(tk, fetch_fn=fetch_fn)
    panel = pd.DataFrame(series).dropna()
    panel = panel.sort_index()
    # DatetimeIndex를 tz-naive로 통일(yfinance는 tz-aware일 수 있어 DB(naive)와 정렬 어긋남 방지).
    if getattr(panel.index, "tz", None) is not None:
        panel.index = panel.index.tz_localize(None)
    return panel


# ---------------------------------------------------------------------------
# 무위험이자율 (^IRX) — 시점별 과거값 (AC6/AC7)
# ---------------------------------------------------------------------------
def _yf_irx_close(ticker: str) -> pd.Series:
    import yfinance as yf

    df = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    return df["Close"].dropna()


def fetch_irx_series(fetch_fn: Callable[[str], pd.Series] | None = None) -> pd.Series:
    """^IRX(미국 3개월 국채 금리, 퍼센트) 종가 시계열을 반환한다(AC6)."""
    fetch_fn = fetch_fn or _yf_irx_close
    s = fetch_fn(IRX_TICKER)
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s.sort_index()


def risk_free_rate_at(irx_series: pd.Series, asof) -> float:
    """asof 시점의 무위험이자율(소수, 예: 0.052)을 반환한다(AC7).

    그 시점 '이하' 가장 가까운 과거 ^IRX 값(퍼센트)을 100으로 나눠 소수로 바꾼다 — 미래참조 없음.
    asof 이전 데이터가 전혀 없으면(아주 이른 시점) 가장 이른 값으로 폴백한다(억지 추정 대신 근사).
    """
    asof = pd.Timestamp(asof)
    if getattr(irx_series.index, "tz", None) is not None:
        irx_series = irx_series.copy()
        irx_series.index = irx_series.index.tz_localize(None)
    past = irx_series[irx_series.index <= asof]
    value = float(past.iloc[-1]) if len(past) else float(irx_series.iloc[0])
    return value / 100.0
