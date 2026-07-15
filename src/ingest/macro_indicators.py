"""매크로 지표 수집(ingest) — FRED 장단기금리차/VIX + CNN 공포탐욕지수.

.omc/specs/brainstorming-macro-indicator-agent.md 참고 (접근방식 A: 단순 스크립트).

세 지표를 각각 조회해 macro_indicators 테이블에 upsert한다:
  - T10Y2Y (장단기금리차, %p)      : FRED, pandas_datareader
  - VIXCLS (VIX 변동성지수)         : FRED, pandas_datareader
  - CNN_FNG (CNN Fear&Greed, 0~100): CNN, Selenium (JS 동적 렌더링이라 예외적으로 사용)

설계 원칙(기존 us_prices.py/naver_prices.py 관례 준수)
----------------------------------------------------
- 실제 라이브러리/네트워크 호출은 모두 주입 가능한 함수 인자(fetch_fn/fetch_*)로 분리 —
  네트워크 없이 파싱·검증·재시도·upsert 순수 로직만 단위테스트한다(DI 관례).
- 과거 백필 없음(스펙 Non-Goal) — 날짜범위 파라미터를 받지 않고 항상 오늘 날짜만 조회한다.
- 지표별 수집 실패는 1회 재시도 후 격리 — 한 지표가 실패해도 나머지 지표는 계속 수집한다
  (us_prices.py의 "종목별 실패 격리, continue"와 같은 정신).
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Callable

from ..db import connect, init_db
from .notify import send_slack_alert

# ---------------------------------------------------------------------------
# sanity 임계값 (diagnose.py SANITY 딕셔너리와 동일 관례: (min,max) 범위, 벗어나면 이상치).
# CNN Fear&Greed Index는 0~100 정수 도메인이므로 그 밖의 값은 저장하지 않는다.
# T10Y2Y/VIXCLS는 열린 도메인이라 여기 넣지 않는다(passes_sanity가 기본 통과 처리).
# ---------------------------------------------------------------------------
SANITY: dict[str, tuple[float, float]] = {
    "CNN_FNG": (0, 100),
}

# FRED 일별 시계열은 최신 며칠이 결측/미갱신일 수 있어 오늘 기준 과거로 살짝 되돌아본 창을
# 조회한 뒤 마지막 유효(비-NaN) 행을 최신값으로 쓴다(휴장일/공표지연 대응). 백필 아님.
_FRED_LOOKBACK_DAYS = 30

# CNN Fear&Greed 게이지 숫자 요소. By.CSS_SELECTOR 상수는 문자열 "css selector"라
# 여기서 문자열로 고정하면 selenium 미설치 환경에서도 _extract_cnn_value 파싱 로직을
# fake driver로 테스트할 수 있다(실제 브라우저 구동은 검증 범위 밖).
CNN_FNG_URL = "https://www.cnn.com/markets/fear-and-greed"
_CNN_BY = "css selector"
_CNN_SELECTOR = "span.market-fng-gauge__dial-number-value"


# ---------------------------------------------------------------------------
# sanity 검증
# ---------------------------------------------------------------------------
def passes_sanity(indicator: str, value) -> bool:
    """지표 값이 SANITY 도메인 안이면 True. SANITY에 없는 지표는 제약 없음(True)."""
    bound = SANITY.get(indicator)
    if bound is None:
        return True
    lo, hi = bound
    return lo <= value <= hi


# ---------------------------------------------------------------------------
# FRED 조회 (T10Y2Y/VIXCLS) — pandas_datareader
# ---------------------------------------------------------------------------
def _fetch_fred_series(series_id: str, start, end):
    import pandas_datareader.data as pdr  # 지연 import — fama_french.py의 기존 패턴과 동일

    return pdr.DataReader(series_id, "fred", start, end)


def _fetch_fred_latest(
    series_id: str,
    fetch_fn: Callable | None = None,
    today: date | None = None,
) -> tuple[str, float]:
    """FRED 시계열에서 최신 유효값을 (날짜, 값)으로 반환. 결측(NaN) 최신 행은 건너뛴다."""
    fetch_fn = fetch_fn or _fetch_fred_series
    today = today or date.today()
    start = today - timedelta(days=_FRED_LOOKBACK_DAYS)
    df = fetch_fn(series_id, start, today)
    series = df[series_id].dropna()
    if series.empty:
        raise ValueError(f"{series_id}: 최근 유효값 없음")
    idx = series.index[-1]
    return idx.strftime("%Y-%m-%d"), float(series.iloc[-1])


def fetch_t10y2y(fetch_fn: Callable | None = None, today: date | None = None) -> tuple[str, float]:
    """FRED T10Y2Y(10년-2년 국채금리차, %p) 최신값을 (날짜, 값)으로 반환."""
    return _fetch_fred_latest("T10Y2Y", fetch_fn, today)


def fetch_vixcls(fetch_fn: Callable | None = None, today: date | None = None) -> tuple[str, float]:
    """FRED VIXCLS(VIX 변동성지수) 최신값을 (날짜, 값)으로 반환."""
    return _fetch_fred_latest("VIXCLS", fetch_fn, today)


# ---------------------------------------------------------------------------
# CNN Fear&Greed 조회 — Selenium (JS 동적 렌더링)
# ---------------------------------------------------------------------------
def _parse_cnn_value(text: str) -> int:
    """게이지 요소 텍스트에서 0~100 정수를 추출."""
    m = re.search(r"\d+", str(text))
    if not m:
        raise ValueError(f"CNN 값 파싱 실패: {text!r}")
    return int(m.group())


def _cnn_gauge_text(driver) -> str:
    """게이지 요소의 텍스트. textContent를 우선 쓴다 — 헤드리스 모드에서는 요소가
    CSS상 "보이는" 상태로 계산되지 않아 .text가 빈 문자열을 반환하는 경우가 있다."""
    el = driver.find_element(_CNN_BY, _CNN_SELECTOR)
    return (el.get_attribute("textContent") or "").strip() or el.text


def _extract_cnn_value(driver) -> int:
    """(주입된) Selenium driver에서 게이지 숫자 요소를 찾아 정수로 파싱한다."""
    return _parse_cnn_value(_cnn_gauge_text(driver))


def _selenium_fetch_cnn() -> int:
    """헤드리스 Chrome으로 CNN Fear&Greed 페이지를 열어 지수 값을 읽는다(실제 크롤링).

    CNN 페이지는 광고/추적 스크립트 때문에 브라우저의 "완전 로딩(load)" 이벤트가
    거의 발생하지 않는다. page_load_strategy="eager"(DOM 파싱 완료 시점에 반환)로
    무한 대기를 피하고, 게이지 숫자 요소가 실제로 그려질 때까지만 명시적으로 기다린다.
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.page_load_strategy = "eager"
    driver = webdriver.Chrome(options=options)
    try:
        driver.set_page_load_timeout(30)
        driver.get(CNN_FNG_URL)
        WebDriverWait(driver, 20).until(
            lambda d: _cnn_gauge_text(d) != ""
        )
        return _extract_cnn_value(driver)
    finally:
        driver.quit()


def fetch_cnn_fng(fetch_fn: Callable | None = None, today: date | None = None) -> tuple[str, int]:
    """CNN Fear&Greed Index(0~100)를 (오늘 날짜, 값)으로 반환.

    fetch_fn은 값 하나를 반환하는 무인자 콜러블 — 실제로는 Selenium 크롤러이며
    테스트에서는 fake로 주입한다. CNN은 페이지에 당일 값만 노출해 날짜는 today를 쓴다.
    """
    fetch_fn = fetch_fn or _selenium_fetch_cnn
    today = today or date.today()
    value = fetch_fn()
    return today.strftime("%Y-%m-%d"), value


# ---------------------------------------------------------------------------
# upsert (INSERT OR REPLACE, UNIQUE(indicator,date))
# ---------------------------------------------------------------------------
def upsert_indicator(conn, indicator: str, date_str: str, value, source: str) -> None:
    """macro_indicators에 (indicator,date) 기준 upsert. commit은 호출자 책임."""
    conn.execute(
        "INSERT OR REPLACE INTO macro_indicators(indicator, date, value, source) VALUES (?,?,?,?)",
        (indicator, date_str, value, source),
    )


# ---------------------------------------------------------------------------
# 오케스트레이션 — 세 지표 수집(재시도·격리)·검증·upsert
# ---------------------------------------------------------------------------
def _fetch_with_retry(fetch: Callable, today: date, retries: int = 1):
    """fetch(today=today)를 실패 시 retries회 재시도. 전부 실패하면 마지막 예외를 던진다."""
    last_exc: Exception | None = None
    for _ in range(retries + 1):
        try:
            return fetch(today=today)
        except Exception as exc:  # noqa: BLE001 — 전 지표 재시도 후 격리 처리
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def ingest_macro_indicators(
    db_path: str | None = None,
    fetch_spread: Callable | None = None,
    fetch_vix: Callable | None = None,
    fetch_cnn: Callable | None = None,
    today: date | None = None,
) -> dict:
    """세 매크로 지표를 오늘 날짜 기준으로 조회해 macro_indicators에 upsert.

    각 지표는 1회 재시도 후에도 실패하면 그 지표만 failed로 남기고 나머지는 계속 진행한다
    (부분실패 격리). sanity(범위) 검증에 걸린 값도 저장하지 않고 실패로 처리한다.
    반환: {"succeeded": [indicator...], "failed": [indicator...]}.
    """
    fetch_spread = fetch_spread or fetch_t10y2y
    fetch_vix = fetch_vix or fetch_vixcls
    fetch_cnn = fetch_cnn or fetch_cnn_fng
    today = today or date.today()

    # (indicator 키, 조회 함수, 출처)
    jobs = [
        ("T10Y2Y", fetch_spread, "FRED"),
        ("VIXCLS", fetch_vix, "FRED"),
        ("CNN_FNG", fetch_cnn, "CNN"),
    ]

    init_db(db_path)
    conn = connect(db_path)
    succeeded: list[str] = []
    failed: list[str] = []
    try:
        for indicator, fetch, source in jobs:
            try:
                d, value = _fetch_with_retry(fetch, today)
                if not passes_sanity(indicator, value):
                    raise ValueError(f"sanity 실패: {indicator}={value} 범위 밖")
                upsert_indicator(conn, indicator, d, value, source)
                conn.commit()
                succeeded.append(indicator)
            except Exception as exc:  # noqa: BLE001 — 지표별 실패 격리, 다음 지표로 계속
                failed.append(indicator)
                send_slack_alert(f"[macro_indicators] {indicator} 수집 실패: {exc}")
                continue
        return {"succeeded": succeeded, "failed": failed}
    finally:
        conn.close()
