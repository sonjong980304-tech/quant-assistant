"""올웨더 4종목 가격 패널 + 무위험이자율(^IRX) 조회.

.omc/specs/brainstorming-all-weather-portfolio.md AC1/AC2/AC6/AC7 참고.

- 4종목(QQQ/삼성전자/TLT/ACE KRX금현물) 모두 yfinance로 수집한다.
- 삼성전자(005930.KS)는 yfinance 자체가 2000년부터 제공한다(기존 DB의 prices 테이블은 2014년부터라
  더 짧았음 — 20년 이상 백테스트 확보를 위해 DB 대신 yfinance로 직접 조회하도록 바꿨다).
- ACE KRX금현물(411060.KS)은 2021년 말에야 상장돼 그 이전 데이터가 존재하지 않는다. 이를 GLD(달러
  표시 실물담보 금ETF, 2004-11~)×USDKRW=X(원/달러 환율, 2003-12~)로 만든 합성 원화금가격으로
  보강하되, 411060.KS 실데이터가 존재하는 구간은 항상 진짜 시세를 우선한다(스플라이스) —
  국내 금현물 가격도 결국 국제 금값×환율로 움직이는 구조라 이 합성이 실제 노출과 근접하다.
- 무위험이자율은 미국 3개월 국채(^IRX)를 리밸런싱 시점별 과거값으로 조회한다(AC6/AC7).

실제 yfinance 호출은 지연 import + 주입 가능한 fetch_fn 으로 분리한다(기존 DI 관례).
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

# 종목/지표 티커.
SAMSUNG_TICKER = "005930.KS"
GOLD_ETF_TICKER = "GLD"  # 실물담보 금 ETF(달러 표시) — 411060.KS 합성 프록시의 원천
FX_TICKER = "USDKRW=X"  # 원/달러 환율 — 금 가격을 원화로 환산할 때 사용
KRX_GOLD_TICKER = "411060.KS"  # ACE KRX금현물 — 실제 보유·모니터링 종목
IRX_TICKER = "^IRX"  # 미국 3개월 국채 금리(퍼센트 표기)

TICKERS: dict[str, str] = {
    "QQQ": "QQQ (나스닥 ETF)",
    SAMSUNG_TICKER: "삼성전자",
    "TLT": "TLT (미국 장기채)",
    KRX_GOLD_TICKER: "ACE KRX금현물",
}


# ---------------------------------------------------------------------------
# yfinance 종가 공통 조회
# ---------------------------------------------------------------------------
def _yf_close_series(ticker: str) -> pd.Series:
    """yfinance 수정종가 Series (지연 import — 무거운 라이브러리는 필요 시점에만 로드). auto_adjust=True 명시."""
    import yfinance as yf

    df = yf.Ticker(ticker).history(period="max", auto_adjust=True)
    return df["Close"].dropna()


def fetch_yf_close_series(ticker: str, fetch_fn: Callable[[str], pd.Series] | None = None) -> pd.Series:
    """티커의 종가 Series를 반환한다(DatetimeIndex, tz-naive, 날짜 오름차순).

    fetch_fn 주입 시 그것을 쓴다(테스트용). 실제 yfinance는 tz-aware DatetimeIndex를 반환하는데
    (티커별 거래소 시간대가 다름 — 예: QQQ/TLT=America/New_York, 411060.KS=Asia/Seoul), 서로 다른
    티커를 pd.DataFrame(dict)로 합칠 때 tz-naive/aware가 섞이면 죽으므로 여기서 미리 통일한다.
    """
    fetch_fn = fetch_fn or _yf_close_series
    s = fetch_fn(ticker)
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s.sort_index()


# ---------------------------------------------------------------------------
# ACE KRX금현물 — GLD×환율 합성 + 실데이터 스플라이스 (20년 이력 확보)
# ---------------------------------------------------------------------------
def build_synthetic_krx_gold_series(fetch_fn: Callable[[str], pd.Series] | None = None) -> pd.Series:
    """GLD(달러)×USDKRW(환율) 합성 원화금가격에 411060.KS 실데이터를 스플라이스한다.

    411060.KS가 실제 존재하는 구간(상장일 이후)은 항상 그 진짜 시세를 쓰고, 그 이전 과거는
    합성값으로 채운다. 411060.KS fetch가 빈 결과여도(네트워크 이슈 등) 합성값만으로 폴백한다.
    """
    gld = fetch_yf_close_series(GOLD_ETF_TICKER, fetch_fn=fetch_fn)
    fx = fetch_yf_close_series(FX_TICKER, fetch_fn=fetch_fn)
    synthetic = (gld * fx).dropna()

    real = fetch_yf_close_series(KRX_GOLD_TICKER, fetch_fn=fetch_fn)
    if len(real):
        cutover = real.index.min()
        synthetic = synthetic[synthetic.index < cutover]
        combined = pd.concat([synthetic, real]).sort_index()
    else:
        combined = synthetic
    combined.name = KRX_GOLD_TICKER
    return combined


def build_price_panel(
    fetch_fn: Callable[[str], pd.Series] | None = None,
    tickers: dict[str, str] | None = None,
) -> pd.DataFrame:
    """4종목 종가를 공통 거래일 기준 하나의 DataFrame(컬럼=티커)으로 합친다.

    ACE KRX금현물은 build_synthetic_krx_gold_series로, 나머지는 fetch_yf_close_series로 가져온다.
    각 종목이 모두 존재하는 거래일만 남긴다(dropna) — 몬테카를로 pct_change 계산 전제.
    """
    tickers = tickers or TICKERS
    series: dict[str, pd.Series] = {}
    for tk in tickers:
        if tk == KRX_GOLD_TICKER:
            series[tk] = build_synthetic_krx_gold_series(fetch_fn=fetch_fn)
        else:
            series[tk] = fetch_yf_close_series(tk, fetch_fn=fetch_fn)
    panel = pd.DataFrame(series).dropna()
    panel = panel.sort_index()
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
