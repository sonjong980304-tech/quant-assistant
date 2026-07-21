"""매크로 지표 수집(ingest) 테스트 (MAC-2).

.omc/specs/brainstorming-macro-indicator-agent.md AC1/AC2/AC3/AC4/AC5/AC6/AC20/AC22.
FRED 조회(pandas_datareader)와 CNN 크롤링(Selenium)은 모두 주입 가능한 함수로 분리해
네트워크/실제 라이브러리 없이 순수 로직(파싱·검증·재시도·upsert)만 검증한다.
"""
from __future__ import annotations

import inspect
from datetime import date

import pandas as pd
import pytest

import src.ingest.macro_indicators as macro_indicators
from src.db import connect, init_db
from src.ingest.macro_indicators import (
    SANITY,
    _extract_cnn_value,
    _extract_vkospi_from_grid,
    _parse_cnn_value,
    _parse_vkospi_row,
    fetch_cnn_fng,
    fetch_t10y2y,
    fetch_vixcls,
    fetch_vkospi_krx,
    ingest_macro_indicators,
    passes_sanity,
    upsert_indicator,
)


# --------------------------------------------------------------------------
# FRED 조회 (AC1/AC2/AC22) — fake fetch_fn 주입, 네트워크 없음
# --------------------------------------------------------------------------
def _fred_df(series_id: str, values, dates):
    return pd.DataFrame({series_id: values}, index=pd.to_datetime(dates))


def test_fetch_t10y2y_returns_latest_date_and_value():
    df = _fred_df("T10Y2Y", [0.40, 0.45], ["2026-07-11", "2026-07-14"])
    d, v = fetch_t10y2y(fetch_fn=lambda sid, start, end: df)
    assert d == "2026-07-14"
    assert v == 0.45


def test_fetch_t10y2y_skips_trailing_nan_row():
    df = _fred_df("T10Y2Y", [0.40, float("nan")], ["2026-07-11", "2026-07-14"])
    d, v = fetch_t10y2y(fetch_fn=lambda sid, start, end: df)
    assert d == "2026-07-11"
    assert v == 0.40


def test_fetch_t10y2y_requests_correct_series_id():
    captured = {}

    def fake(sid, start, end):
        captured["sid"] = sid
        return _fred_df("T10Y2Y", [0.5], ["2026-07-14"])

    fetch_t10y2y(fetch_fn=fake)
    assert captured["sid"] == "T10Y2Y"


def test_fetch_vixcls_returns_latest_date_and_value():
    df = _fred_df("VIXCLS", [13.5, 14.2], ["2026-07-11", "2026-07-14"])
    d, v = fetch_vixcls(fetch_fn=lambda sid, start, end: df)
    assert d == "2026-07-14"
    assert v == 14.2


def test_fetch_vixcls_requests_correct_series_id():
    captured = {}

    def fake(sid, start, end):
        captured["sid"] = sid
        return _fred_df("VIXCLS", [14.0], ["2026-07-14"])

    fetch_vixcls(fetch_fn=fake)
    assert captured["sid"] == "VIXCLS"


# --------------------------------------------------------------------------
# AC20 — 날짜범위 파라미터 없음, 내부적으로 today만 사용
# --------------------------------------------------------------------------
def test_fred_fetchers_have_no_date_range_params():
    for fn in (fetch_t10y2y, fetch_vixcls, fetch_cnn_fng, fetch_vkospi_krx):
        params = set(inspect.signature(fn).parameters)
        assert "start" not in params
        assert "end" not in params


def test_fetch_t10y2y_uses_today_as_window_end():
    captured = {}

    def fake(sid, start, end):
        captured["start"] = start
        captured["end"] = end
        return _fred_df("T10Y2Y", [0.5], ["2026-07-14"])

    fetch_t10y2y(fetch_fn=fake, today=date(2026, 7, 14))
    assert captured["end"] == date(2026, 7, 14)
    assert captured["start"] < captured["end"]  # 과거로 조금 되돌아본 창(휴장일 대응), 오늘 기준


# --------------------------------------------------------------------------
# CNN 크롤링 (AC3/AC22) — fake driver로 값 추출 로직만 검증(브라우저 미구동)
# --------------------------------------------------------------------------
class _FakeElement:
    def __init__(self, text):
        self.text = text

    def get_attribute(self, name):
        return self.text if name == "textContent" else None


class _FakeDriver:
    def __init__(self, text):
        self._text = text
        self.calls = []

    def find_element(self, by, value):
        self.calls.append((by, value))
        return _FakeElement(self._text)


def test_extract_cnn_value_reads_number_from_driver_element():
    driver = _FakeDriver("42")
    assert _extract_cnn_value(driver) == 42
    assert len(driver.calls) == 1  # 셀렉터로 요소를 한 번 찾는다


def test_parse_cnn_value_parses_integer_string():
    assert _parse_cnn_value("73") == 73


def test_fetch_cnn_fng_returns_today_and_value_via_injected_fn():
    d, v = fetch_cnn_fng(fetch_fn=lambda: 55, today=date(2026, 7, 14))
    assert d == "2026-07-14"
    assert v == 55


# --------------------------------------------------------------------------
# VKOSPI(코스피 200 변동성지수) 조회 — KRX Data Marketplace(로그인 필요), Selenium.
# pykrx 인덱스 목록엔 VKOSPI가 없어(주식/섹터 지수만 커버) KRX 사이트를 직접 크롤링한다.
# 로그인+검색+클릭 전체 흐름(_selenium_fetch_vkospi)은 CNN의 실제 브라우저 구동과 동일하게
# 검증 범위 밖 — 여기서는 fake driver로 "그리드에서 값 추출" 로직만 검증한다.
# --------------------------------------------------------------------------
def test_parse_vkospi_row_converts_date_and_value():
    d, v = _parse_vkospi_row("2026/07/21", "84.89")
    assert d == "2026-07-21"
    assert v == 84.89


def test_parse_vkospi_row_strips_thousand_separator():
    _, v = _parse_vkospi_row("2026/07/21", "1,234.56")
    assert v == 1234.56


class _FakeVkospiCell:
    def __init__(self, text):
        self._text = text

    def get_attribute(self, name):
        return self._text if name == "textContent" else None


class _FakeVkospiRow:
    def __init__(self, date_text, close_text):
        self._cells = [_FakeVkospiCell(date_text), _FakeVkospiCell(close_text)]

    def find_elements(self, by, value):
        return self._cells


class _FakeVkospiDriver:
    """#jsGrid_MDCSTAT014 tbody tr(첫 행=최신 거래일) 하나만 흉내내는 fake driver."""

    def __init__(self, date_text, close_text):
        self._row = _FakeVkospiRow(date_text, close_text)

    def find_element(self, by, value):
        return self._row


def test_extract_vkospi_from_grid_reads_latest_row():
    driver = _FakeVkospiDriver("2026/07/21", "84.89")
    d, v = _extract_vkospi_from_grid(driver)
    assert d == "2026-07-21"
    assert v == 84.89


def test_fetch_vkospi_krx_returns_value_via_injected_fn():
    d, v = fetch_vkospi_krx(fetch_fn=lambda: ("2026-07-21", 84.89))
    assert d == "2026-07-21"
    assert v == 84.89


# --------------------------------------------------------------------------
# sanity-check (AC5) — CNN 0~100 범위, diagnose.py SANITY 딕셔너리 관례
# --------------------------------------------------------------------------
def test_sanity_dict_defines_cnn_domain():
    assert SANITY["CNN_FNG"] == (0, 100)


@pytest.mark.parametrize("value,expected", [(0, True), (100, True), (50, True), (-1, False), (101, False)])
def test_passes_sanity_cnn_boundaries(value, expected):
    assert passes_sanity("CNN_FNG", value) is expected


# --------------------------------------------------------------------------
# upsert (AC4) — INSERT OR REPLACE, UNIQUE(indicator,date)
# --------------------------------------------------------------------------
def test_upsert_indicator_replaces_same_indicator_date(tmp_path):
    db = str(tmp_path / "u.db")
    init_db(db)
    conn = connect(db)
    try:
        upsert_indicator(conn, "T10Y2Y", "2026-07-14", 0.40, "FRED")
        upsert_indicator(conn, "T10Y2Y", "2026-07-14", 0.52, "FRED")
        conn.commit()
        rows = conn.execute(
            "SELECT value FROM macro_indicators WHERE indicator='T10Y2Y' AND date='2026-07-14'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 0.52
    finally:
        conn.close()


# --------------------------------------------------------------------------
# 오케스트레이션 (AC4/AC5/AC6/AC20/AC22)
# --------------------------------------------------------------------------
def test_ingest_stores_all_four_indicators(tmp_path, monkeypatch):
    db = str(tmp_path / "ing1.db")
    init_db(db)
    monkeypatch.setattr(macro_indicators, "send_slack_alert", lambda *a, **k: None)
    result = ingest_macro_indicators(
        db_path=db,
        fetch_spread=lambda today=None: ("2026-07-14", 0.45),
        fetch_vix=lambda today=None: ("2026-07-14", 14.2),
        fetch_cnn=lambda today=None: ("2026-07-14", 55),
        fetch_vkospi=lambda today=None: ("2026-07-14", 84.89),
        today=date(2026, 7, 14),
    )
    assert set(result["succeeded"]) == {"T10Y2Y", "VIXCLS", "CNN_FNG", "VKOSPI"}
    assert result["failed"] == []

    conn = connect(db)
    rows = {r["indicator"]: r["value"] for r in conn.execute("SELECT indicator, value FROM macro_indicators")}
    conn.close()
    assert rows == {"T10Y2Y": 0.45, "VIXCLS": 14.2, "CNN_FNG": 55.0, "VKOSPI": 84.89}


def test_ingest_out_of_range_cnn_not_stored_others_continue(tmp_path, monkeypatch):
    # AC5: CNN 값이 0~100을 벗어나면 저장되지 않고, 나머지 지표는 계속 진행된다.
    db = str(tmp_path / "ing2.db")
    init_db(db)
    monkeypatch.setattr(macro_indicators, "send_slack_alert", lambda *a, **k: None)
    result = ingest_macro_indicators(
        db_path=db,
        fetch_spread=lambda today=None: ("2026-07-14", 0.45),
        fetch_vix=lambda today=None: ("2026-07-14", 14.2),
        fetch_cnn=lambda today=None: ("2026-07-14", 150),
        fetch_vkospi=lambda today=None: ("2026-07-14", 84.89),
        today=date(2026, 7, 14),
    )
    assert "CNN_FNG" in result["failed"]
    assert set(result["succeeded"]) == {"T10Y2Y", "VIXCLS", "VKOSPI"}

    conn = connect(db)
    stored = {r["indicator"] for r in conn.execute("SELECT indicator FROM macro_indicators")}
    conn.close()
    assert "CNN_FNG" not in stored
    assert {"T10Y2Y", "VIXCLS", "VKOSPI"} == stored


def test_ingest_retries_once_then_succeeds(tmp_path, monkeypatch):
    # AC6: 1차 예외/2차 성공하면 그 지표는 최종 성공한다(1회 재시도).
    db = str(tmp_path / "ing3.db")
    init_db(db)
    monkeypatch.setattr(macro_indicators, "send_slack_alert", lambda *a, **k: None)
    calls = {"n": 0}

    def flaky_spread(today=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("일시적 FRED 실패(mock)")
        return ("2026-07-14", 0.45)

    result = ingest_macro_indicators(
        db_path=db,
        fetch_spread=flaky_spread,
        fetch_vix=lambda today=None: ("2026-07-14", 14.2),
        fetch_cnn=lambda today=None: ("2026-07-14", 55),
        fetch_vkospi=lambda today=None: ("2026-07-14", 84.89),
        today=date(2026, 7, 14),
    )
    assert calls["n"] == 2  # 1차 실패 + 2차 성공
    assert "T10Y2Y" in result["succeeded"]
    assert result["failed"] == []


def test_ingest_isolates_persistent_failure_and_continues(tmp_path, monkeypatch):
    # AC6: 재시도도 실패하면 그 지표만 실패로 남고 나머지 지표 수집은 계속된다(부분실패 격리).
    db = str(tmp_path / "ing4.db")
    init_db(db)
    alerts = []
    monkeypatch.setattr(macro_indicators, "send_slack_alert", lambda msg, **k: alerts.append(msg))
    called = {"vix": 0, "cnn": 0, "vkospi": 0}

    def always_fail(today=None):
        raise ConnectionError("FRED 지속 실패(mock)")

    def vix(today=None):
        called["vix"] += 1
        return ("2026-07-14", 14.2)

    def cnn(today=None):
        called["cnn"] += 1
        return ("2026-07-14", 55)

    def vkospi(today=None):
        called["vkospi"] += 1
        return ("2026-07-14", 84.89)

    result = ingest_macro_indicators(
        db_path=db, fetch_spread=always_fail, fetch_vix=vix, fetch_cnn=cnn, fetch_vkospi=vkospi,
        today=date(2026, 7, 14),
    )
    assert result["failed"] == ["T10Y2Y"]
    assert set(result["succeeded"]) == {"VIXCLS", "CNN_FNG", "VKOSPI"}
    assert called["vix"] == 1 and called["cnn"] == 1 and called["vkospi"] == 1  # 한 지표 실패해도 나머지는 정상 호출
    assert len(alerts) == 1  # 실패 지표에 대해서만 알림


def test_ingest_passes_today_into_fetchers(tmp_path, monkeypatch):
    # AC20: ingest는 오늘 날짜만 조회 — 주입한 today가 각 fetcher로 전달된다.
    db = str(tmp_path / "ing5.db")
    init_db(db)
    monkeypatch.setattr(macro_indicators, "send_slack_alert", lambda *a, **k: None)
    seen = {}

    def spread(today=None):
        seen["today"] = today
        return ("2026-07-14", 0.45)

    ingest_macro_indicators(
        db_path=db,
        fetch_spread=spread,
        fetch_vix=lambda today=None: ("2026-07-14", 14.2),
        fetch_cnn=lambda today=None: ("2026-07-14", 55),
        fetch_vkospi=lambda today=None: ("2026-07-14", 84.89),
        today=date(2026, 7, 14),
    )
    assert seen["today"] == date(2026, 7, 14)
