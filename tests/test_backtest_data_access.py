"""src/backtest/data_access.py의 _is_alive() 회귀 테스트 (son-checker 이슈 #23 BUG-1).

_is_alive()의 판정 로직 자체는 이번에 바꾸지 않았다(모르면 살아있다고 보는 폴백은
의도된 정책). 다만 ingest_delisting()이 빈 문자열 대신 NULL을 쓰도록 바뀌었으므로,
NULL과 빈 문자열 둘 다 기존과 동일하게 "모름 → 살아있음"으로 처리되는지, 그리고
실제 delisting_date가 있을 때는 asof와 정확히 비교해 판정하는지 회귀로 고정한다.
이 파일이 생기기 전에는 _is_alive()를 직접 겨냥한 테스트가 없었다(metrics_at 경유뿐).
"""
from __future__ import annotations

import sqlite3

import pytest

from src.backtest.data_access import (
    _is_alive,
    _months_before,
    _one_year_before,
    price_return_over_months,
)
from src.db import init_db


def _conn(tmp_path) -> sqlite3.Connection:
    db = tmp_path / "alive.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def _seed_prices(conn, code: str, rows: list[tuple[str, float]]) -> None:
    for date_str, close in rows:
        conn.execute(
            "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,?)",
            (code, date_str, close, 1e12),
        )
    conn.commit()


def test_is_alive_true_before_delisting_date(tmp_path):
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO delisting(stock_code, name, delisting_date) VALUES(?,?,?)",
        ("000030", "우리은행", "2019-02-12"),
    )
    conn.commit()
    assert _is_alive(conn, "000030", "2018-01-01") is True


def test_is_alive_false_on_or_after_delisting_date(tmp_path):
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO delisting(stock_code, name, delisting_date) VALUES(?,?,?)",
        ("000030", "우리은행", "2019-02-12"),
    )
    conn.commit()
    assert _is_alive(conn, "000030", "2020-01-01") is False


def test_is_alive_true_when_no_delisting_row(tmp_path):
    conn = _conn(tmp_path)
    assert _is_alive(conn, "005930", "2026-01-01") is True


def test_is_alive_true_when_delisting_date_is_null(tmp_path):
    """정확한 상폐일을 모르는(NULL) 경우 — ingest_delisting()이 이제 NULL을 쓰므로,
    그런 행이 있어도 기존과 동일하게 '모름 → 살아있음'으로 처리돼야 한다."""
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO delisting(stock_code, name, delisting_date) VALUES(?,?,?)",
        ("999999", "신규상폐추정", None),
    )
    conn.commit()
    assert _is_alive(conn, "999999", "2026-01-01") is True


# ── _months_before(): _one_year_before의 임의 개월수 일반화 ─────────────────────
# return_12m(고정 12개월)만 풀던 _one_year_before를 임의 N개월로 일반화한다. 연/월 캐리와
# 말일 보정(존재하지 않는 날짜로 넘어가지 않게)을 검증하고, months=12일 때 기존
# _one_year_before와 항상 동일한지(회귀 일관성)를 여러 날짜(윤년 2월 포함)로 고정한다.

def test_months_before_simple_within_year():
    assert _months_before("2026-07-16", 3) == "2026-04-16"


def test_months_before_carries_across_year_boundary():
    assert _months_before("2026-01-15", 3) == "2025-10-15"


def test_months_before_clamps_to_month_end_non_leap():
    # 3/31에서 1개월 전 → 2월엔 31일이 없으므로 평년 말일 28일로 보정.
    assert _months_before("2026-03-31", 1) == "2026-02-28"


def test_months_before_clamps_to_month_end_leap_year():
    # 윤년 2월은 29일 → 3/31의 1개월 전은 2/29로 보정(존재하는 날짜).
    assert _months_before("2024-03-31", 1) == "2024-02-29"


@pytest.mark.parametrize(
    "asof",
    [
        "2026-07-16",
        "2026-01-01",
        "2024-02-29",  # 윤년 2/29 → 전년 2/28 보정 경로
        "2025-12-31",
        "2020-02-29",
        "2023-03-01",
    ],
)
def test_months_before_12_matches_one_year_before(asof):
    # 기존 12개월 로직과 완전히 일관돼야 한다(회귀): _months_before(asof, 12) == _one_year_before(asof).
    assert _months_before(asof, 12) == _one_year_before(asof)


# ── price_return_over_months(): 임의 개월수 가격 수익률(결정론) ──────────────────
# return_12m와 동일한 "date<=시점 최근 거래일 종가" 원칙을 임의 N개월로 일반화한다.
# LLM 코드생성 폴백(신뢰 불가) 대신 SQL로 결정론적으로 계산한다.

def test_price_return_over_months_computes_3_6_12_months(tmp_path):
    conn = _conn(tmp_path)
    _seed_prices(
        conn,
        "000001",
        [
            ("2025-07-16", 60.0),   # 12개월 전
            ("2026-01-16", 80.0),   # 6개월 전
            ("2026-04-16", 100.0),  # 3개월 전
            ("2026-07-16", 120.0),  # 기준시점(asof)
        ],
    )

    r3 = price_return_over_months(conn, "000001", "2026-07-16", 3)
    assert r3["months"] == 3
    assert r3["start_target_date"] == "2026-04-16"
    assert r3["start_date"] == "2026-04-16"
    assert r3["start_close"] == pytest.approx(100.0)
    assert r3["end_date"] == "2026-07-16"
    assert r3["end_close"] == pytest.approx(120.0)
    assert r3["return_pct"] == pytest.approx(20.0)  # (120-100)/100*100

    r6 = price_return_over_months(conn, "000001", "2026-07-16", 6)
    assert r6["start_date"] == "2026-01-16"
    assert r6["return_pct"] == pytest.approx(50.0)  # (120-80)/80*100

    r12 = price_return_over_months(conn, "000001", "2026-07-16", 12)
    assert r12["start_date"] == "2025-07-16"
    assert r12["return_pct"] == pytest.approx(100.0)  # (120-60)/60*100


def test_price_return_over_months_matches_nearest_prior_trading_day_for_start(tmp_path):
    conn = _conn(tmp_path)
    # 3개월 전 목표일(2026-04-16)이 휴장일이라 그 이하 가장 가까운 거래일(2026-04-13)이 매칭돼야 한다.
    _seed_prices(conn, "000001", [("2026-04-13", 100.0), ("2026-07-16", 120.0)])
    r = price_return_over_months(conn, "000001", "2026-07-16", 3)
    assert r["start_target_date"] == "2026-04-16"
    assert r["start_date"] == "2026-04-13"  # 요청일과 다른 실제 매칭 거래일
    assert r["return_pct"] == pytest.approx(20.0)


def test_price_return_over_months_none_when_no_start_data(tmp_path):
    conn = _conn(tmp_path)
    # 기준시점 종가만 있고 start 시점 이전 데이터가 아예 없음(예: 상장 전) → 조용히 None.
    _seed_prices(conn, "000001", [("2026-07-16", 120.0)])
    r = price_return_over_months(conn, "000001", "2026-07-16", 3)
    assert r["end_date"] == "2026-07-16"
    assert r["end_close"] == pytest.approx(120.0)
    assert r["start_date"] is None
    assert r["start_close"] is None
    assert r["return_pct"] is None  # 잘못된 값 대신 None


def test_price_return_over_months_all_none_when_code_has_no_data(tmp_path):
    conn = _conn(tmp_path)
    r = price_return_over_months(conn, "NOPE", "2026-07-16", 3)
    assert r["end_date"] is None
    assert r["end_close"] is None
    assert r["start_date"] is None
    assert r["start_close"] is None
    assert r["return_pct"] is None
