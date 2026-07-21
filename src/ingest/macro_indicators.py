"""매크로 지표 수집(ingest) — FRED 장단기금리차/VIX + CNN 공포탐욕지수 + KRX VKOSPI.

.omc/specs/brainstorming-macro-indicator-agent.md 참고 (접근방식 A: 단순 스크립트).

네 지표를 각각 조회해 macro_indicators 테이블에 upsert한다:
  - T10Y2Y (장단기금리차, %p)      : FRED, pandas_datareader
  - VIXCLS (VIX 변동성지수)         : FRED, pandas_datareader
  - CNN_FNG (CNN Fear&Greed, 0~100): CNN, Selenium (JS 동적 렌더링이라 예외적으로 사용)
  - VKOSPI (코스피 200 변동성지수)  : KRX Data Marketplace, Selenium(로그인 필요)

VKOSPI는 참고 표시 전용이다 — macro_signal(GREEN/YELLOW/RED 판정)에는 반영하지 않는다
(macro_pipeline._INDICATORS에 미포함, run_signal 입력 아님. VIX/CNN과 동일한 취급).

설계 원칙(기존 수집기 관례 준수)
----------------------------------------------------
- 실제 라이브러리/네트워크 호출은 모두 주입 가능한 함수 인자(fetch_fn/fetch_*)로 분리 —
  네트워크 없이 파싱·검증·재시도·upsert 순수 로직만 단위테스트한다(DI 관례).
- 과거 백필 없음(스펙 Non-Goal) — 날짜범위 파라미터를 받지 않고 항상 오늘 날짜만 조회한다.
- 지표별 수집 실패는 1회 재시도 후 격리 — 한 지표가 실패해도 나머지 지표는 계속 수집한다
  (수집기의 "종목별 실패 격리, continue"와 같은 정신).
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
# VKOSPI(코스피 200 변동성지수) 조회 — KRX Data Marketplace(data.krx.co.kr), Selenium.
#
# pykrx의 인덱스 목록(IndexTicker, 168개)에는 VKOSPI가 없다(주식/섹터 지수만 커버, 파생상품
# 지수는 별도 게시판이라 미포함) — 부득이 KRX 공식 사이트를 직접 크롤링한다.
# 2026-07 기준 상세 통계 페이지 전체가 로그인 필수로 바뀌어(KRX Data Marketplace 회원가입
# 계정 필요), CNN Fear&Greed와 달리 로그인 단계가 추가된다. KRX_ID/KRX_PW(pykrx 전용, KRX
# 정보데이터시스템 시세 로그인)와는 별개 계정/시스템이라 KRX_DATA_ID/KRX_DATA_PW(.env)로
# 분리했다. 헤드리스 기본 User-Agent는 KRX가 차단하므로 데스크톱 UA로 위장해야 한다.
# ---------------------------------------------------------------------------
_KRX_LOGIN_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd?locale=ko_KR"
_KRX_SEARCH_KEYWORD = "변동성지수"
_KRX_VKOSPI_LINK_TEXT = "코스피 200 변동성지수"
_VKOSPI_GRID_ID = "jsGrid_MDCSTAT014"  # 상세페이지의 "일자별 시세" 그리드(최신행이 맨 위)
_KRX_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _parse_vkospi_row(date_text: str, close_text: str) -> tuple[str, float]:
    """V-KOSPI200 일자별 시세 표의 한 행("2026/07/21","84.89")을 (날짜,값)으로 변환."""
    d = date_text.strip().replace("/", "-")
    v = float(close_text.strip().replace(",", ""))
    return d, v


def _extract_vkospi_from_grid(driver) -> tuple[str, float]:
    """V-KOSPI200 일자별 시세 그리드(#jsGrid_MDCSTAT014)의 최신(첫) 행에서 (날짜,종가) 추출."""
    row = driver.find_element("css selector", f"#{_VKOSPI_GRID_ID} tbody tr")
    cells = row.find_elements("css selector", "td")
    date_text = (cells[0].get_attribute("textContent") or "").strip()
    close_text = (cells[1].get_attribute("textContent") or "").strip()
    return _parse_vkospi_row(date_text, close_text)


def _visible_confirm_button(driver):
    """화면에 실제로 보이는 "확인" 버튼/링크(없으면 None). DOM엔 숨겨진 동일 텍스트 요소가 있다."""
    for btn in driver.find_elements("css selector", "a, button"):
        if (btn.get_attribute("textContent") or "").strip() == "확인" and btn.is_displayed():
            return btn
    return None


def _krx_data_login(driver, uid: str, pw: str) -> None:
    """KRX Data Marketplace 로그인(로그인 폼은 iframe 안에 있다) + 중복로그인 확인모달 처리.

    로그인 클릭·모달 확인 클릭 모두 비동기(AJAX/리다이렉트)라 클릭 직후 바로 다음 조작을
    하면 아직 반영 전이라 실패한다(NoSuchElement/ElementNotInteractable) — 각 단계를
    명시적으로 대기(WebDriverWait)한다.
    """
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait

    driver.get(_KRX_LOGIN_URL)
    iframe = driver.find_element(By.CSS_SELECTOR, "iframe")
    driver.switch_to.frame(iframe)
    driver.find_element(By.ID, "mbrId").send_keys(uid)
    driver.find_element(By.NAME, "pw").send_keys(pw)
    driver.find_element(By.LINK_TEXT, "로그인").click()

    # 이미 다른 곳에서 로그인돼 있으면 "기존 계정을 로그아웃하고 새로 로그인하시겠습니까?"
    # 확인모달이 뜬다(네이티브 alert 아닌 페이지 내 커스텀 모달, 항상 뜨는 건 아님) — 최대
    # 3초만 기다려보고 있으면 누르고, 없으면(타임아웃) 정상 로그인으로 간주해 계속 진행한다.
    try:
        WebDriverWait(driver, 3, poll_frequency=0.2).until(lambda d: _visible_confirm_button(d))
        _visible_confirm_button(driver).click()
    except TimeoutException:
        pass

    driver.switch_to.default_content()
    # 로그인 성공 시 메인 페이지(통합검색창 jsTotSch 있음)로 리다이렉트되는데, 이 전환도
    # 즉시 반영되지 않아 대기가 필요하다.
    WebDriverWait(driver, 15).until(lambda d: d.find_elements(By.ID, "jsTotSch"))


def _selenium_fetch_vkospi() -> tuple[str, float]:
    """헤드리스 Chrome으로 KRX Data Marketplace에 로그인해 V-KOSPI200 최신 종가를 읽는다.

    로그인(KRX_DATA_ID/KRX_DATA_PW, .env) 필수 — 미설정이면 즉시 실패시킨다(브라우저 구동 전).
    """
    import os

    from selenium import webdriver
    from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait

    uid = os.environ.get("KRX_DATA_ID")
    pw = os.environ.get("KRX_DATA_PW")
    if not uid or not pw:
        raise RuntimeError("KRX_DATA_ID/KRX_DATA_PW 미설정(.env) — VKOSPI 조회 불가")

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"user-agent={_KRX_UA}")
    driver = webdriver.Chrome(options=options)
    try:
        driver.set_page_load_timeout(45)
        _krx_data_login(driver, uid, pw)

        box = driver.find_element(By.ID, "jsTotSch")
        box.click()
        box.send_keys(_KRX_SEARCH_KEYWORD)
        box.send_keys(Keys.RETURN)

        def _find_target(d):
            # 검색결과 페이지 전환(URL 변경) 후에도 결과 테이블은 뒤이어 렌더링되므로, URL만
            # 보고 넘어가면 아직 비어있는 이전 페이지 DOM을 순회하다 StaleElementReference나
            # "링크 못 찾음"이 난다 — 목표 링크 자체가 나타날 때까지 기다린다.
            for a in d.find_elements(By.CSS_SELECTOR, "a"):
                if (a.get_attribute("textContent") or "").strip() == _KRX_VKOSPI_LINK_TEXT:
                    return a
            return False

        try:
            target = WebDriverWait(driver, 15, ignored_exceptions=(StaleElementReferenceException,)).until(
                _find_target
            )
        except TimeoutException:
            raise RuntimeError(f"KRX 검색결과에서 '{_KRX_VKOSPI_LINK_TEXT}' 링크를 찾지 못함") from None
        driver.execute_script("arguments[0].click();", target)  # 통합검색 결과 링크는 JS 바인딩

        def _grid_row_ready(d):
            # 그리드는 빈 <tr>부터 먼저 삽입되고 셀 내용은 뒤이어 채워진다 — 행 존재만으론
            # 부족하고, 종가 셀까지 실제로 채워졌는지(2번째 td에 텍스트) 확인해야 한다.
            rows = d.find_elements("css selector", f"#{_VKOSPI_GRID_ID} tbody tr")
            if not rows:
                return False
            cells = rows[0].find_elements("css selector", "td")
            return len(cells) >= 2 and (cells[1].get_attribute("textContent") or "").strip() != ""

        try:
            WebDriverWait(driver, 20, ignored_exceptions=(StaleElementReferenceException,)).until(
                _grid_row_ready
            )
        except TimeoutException:
            raise RuntimeError("V-KOSPI200 시세 그리드가 시간 내 채워지지 않음") from None
        return _extract_vkospi_from_grid(driver)
    finally:
        driver.quit()


def fetch_vkospi_krx(fetch_fn: Callable | None = None, today: date | None = None) -> tuple[str, float]:
    """V-KOSPI200(코스피 200 변동성지수) 최신값을 (날짜, 값)으로 반환.

    fetch_fn은 (날짜문자열, 값) 튜플을 반환하는 무인자 콜러블 — 테스트에서는 fake 주입.
    KRX 페이지 자체가 거래일 날짜를 명시하므로(휴장일엔 직전 거래일) today는 값 계산에
    쓰지 않는다 — 다른 지표 fetcher(fetch_cnn_fng 등)와 시그니처만 통일해 오케스트레이터
    (_fetch_with_retry가 today=today로 호출)와 호환시킨다.
    """
    fetch_fn = fetch_fn or _selenium_fetch_vkospi
    return fetch_fn()


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
    fetch_vkospi: Callable | None = None,
    today: date | None = None,
) -> dict:
    """네 매크로 지표를 오늘 날짜 기준으로 조회해 macro_indicators에 upsert.

    각 지표는 1회 재시도 후에도 실패하면 그 지표만 failed로 남기고 나머지는 계속 진행한다
    (부분실패 격리). sanity(범위) 검증에 걸린 값도 저장하지 않고 실패로 처리한다.
    반환: {"succeeded": [indicator...], "failed": [indicator...]}.
    """
    fetch_spread = fetch_spread or fetch_t10y2y
    fetch_vix = fetch_vix or fetch_vixcls
    fetch_cnn = fetch_cnn or fetch_cnn_fng
    fetch_vkospi = fetch_vkospi or fetch_vkospi_krx
    today = today or date.today()

    # (indicator 키, 조회 함수, 출처)
    jobs = [
        ("T10Y2Y", fetch_spread, "FRED"),
        ("VIXCLS", fetch_vix, "FRED"),
        ("CNN_FNG", fetch_cnn, "CNN"),
        ("VKOSPI", fetch_vkospi, "KRX"),
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
