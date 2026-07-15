"""미국판 백테스트 DB 어댑터(data_access_us.py) 단위 테스트 (TDD).

한국판 data_access.py의 effective_quarter_at/_price_at/_is_alive/metrics_at/
build_callbacks 구조를 미국(us_company/us_prices/us_financials EAV)판으로 옮긴 것을 검증한다.
- look-ahead 방지: disclosed_date<=asof인 최신 quarterly만 사용.
- 생존편향: 미국은 상장폐지 데이터가 없어 항상 "unverifiable"(검증불가).
- S&P500 벤치마크: yfinance ^GSPC 히스토리를 DI로 주입해 네트워크 없이 검증.
DB 접근이 필요한 검사는 임시 SQLite에 시딩해 사용자 DB와 완전 격리한다.
"""
from __future__ import annotations

import sqlite3

import pytest

from src.backtest import data_access_us as dau
from src.db import init_db


def _seed_us_db(tmp_path) -> sqlite3.Connection:
    """AAPL 1종목을 4개 quarterly + BS + 주가로 시드한 임시 DB 연결을 반환한다.

    quarterly 4분기(2024-03-31~2024-12-31) Net Income 각 10 → TTM=40.
    최신 분기(2024-12-31): Total Revenue=100, Operating Income=30, Net Income=10,
    Stockholders Equity=200. us_company.market_cap=1000.
    disclosed_date는 기말+45일(예: 2024-12-31 → 2025-02-14).
    """
    db = tmp_path / "dau.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        ("AAPL", "Apple", "NASDAQ", "Technology", 1000.0, "2025-03-01"),
    )
    quarters = [
        ("2024-03-31", "2024-05-15"),
        ("2024-06-30", "2024-08-14"),
        ("2024-09-30", "2024-11-14"),
        ("2024-12-31", "2025-02-14"),
    ]
    for as_of, disclosed in quarters:
        conn.execute(
            "INSERT INTO us_financials(stock_code, as_of_date, period_type, statement_type, "
            "item_key, item_value, disclosed_date, source) VALUES (?,?,?,?,?,?,?,?)",
            ("AAPL", as_of, "quarterly", "income_stmt", "Net Income", 10.0, disclosed, "yfinance"),
        )
    # 최신 분기 단일값(Revenue/Operating Income) + 재무상태(Stockholders Equity)
    for item_key, value in [("Total Revenue", 100.0), ("Operating Income", 30.0)]:
        conn.execute(
            "INSERT INTO us_financials(stock_code, as_of_date, period_type, statement_type, "
            "item_key, item_value, disclosed_date, source) VALUES (?,?,?,?,?,?,?,?)",
            ("AAPL", "2024-12-31", "quarterly", "income_stmt", item_key, value, "2025-02-14", "yfinance"),
        )
    conn.execute(
        "INSERT INTO us_financials(stock_code, as_of_date, period_type, statement_type, "
        "item_key, item_value, disclosed_date, source) VALUES (?,?,?,?,?,?,?,?)",
        ("AAPL", "2024-12-31", "quarterly", "balance_sheet", "Stockholders Equity", 200.0, "2025-02-14", "yfinance"),
    )
    for date_str, close in [("2025-01-15", 140.0), ("2025-03-01", 150.0)]:
        conn.execute(
            "INSERT INTO us_prices(stock_code, date, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
            ("AAPL", date_str, close - 2, close + 2, close - 3, close, 1000.0),
        )
    conn.commit()
    return conn


# --------------------------------------------------------------------------
# effective_quarter_at_us — look-ahead 방지
# --------------------------------------------------------------------------
def test_effective_quarter_at_us_returns_latest_disclosed_quarter(tmp_path):
    conn = _seed_us_db(tmp_path)
    # 2025-03-01 시점엔 2024-12-31(공시 2025-02-14)까지 반영
    assert dau.effective_quarter_at_us(conn, "AAPL", "2025-03-01") == "2024-12-31"


def test_effective_quarter_at_us_excludes_not_yet_disclosed(tmp_path):
    conn = _seed_us_db(tmp_path)
    # 2025-02-01 시점엔 2024-12-31 공시(2025-02-14) 전 → 직전 2024-09-30까지만
    assert dau.effective_quarter_at_us(conn, "AAPL", "2025-02-01") == "2024-09-30"


def test_effective_quarter_at_us_none_when_no_disclosed_quarter(tmp_path):
    conn = _seed_us_db(tmp_path)
    assert dau.effective_quarter_at_us(conn, "AAPL", "2020-01-01") is None


# --------------------------------------------------------------------------
# _price_at_us — 종가 + us_company.market_cap 근사
# --------------------------------------------------------------------------
def test_price_at_us_returns_close_and_company_market_cap(tmp_path):
    conn = _seed_us_db(tmp_path)
    close, cap = dau._price_at_us(conn, "AAPL", "2025-03-01")
    assert close == 150.0
    assert cap == 1000.0  # us_company.market_cap 근사(주가 변동 미반영)


def test_price_at_us_none_when_no_price_before_asof(tmp_path):
    conn = _seed_us_db(tmp_path)
    assert dau._price_at_us(conn, "AAPL", "2020-01-01") == (None, None)


# --------------------------------------------------------------------------
# _is_alive_us — 미국은 상폐 데이터 없음 → 항상 검증불가
# --------------------------------------------------------------------------
def test_is_alive_us_always_unverifiable(tmp_path):
    conn = _seed_us_db(tmp_path)
    # KR의 bool(True/False)과 달리 문자열 "unverifiable"을 돌려준다(검증불가를 명확히 구분).
    assert dau._is_alive_us(conn, "AAPL", "2025-03-01") == "unverifiable"


# --------------------------------------------------------------------------
# metrics_at_us — PER/PBR/ROE/영업이익률/순이익률
# --------------------------------------------------------------------------
def test_metrics_at_us_computes_ratios(tmp_path):
    conn = _seed_us_db(tmp_path)
    rows = dau.metrics_at_us(conn, "2025-03-01")
    assert len(rows) == 1
    r = rows[0]
    assert r["stock_code"] == "AAPL"
    assert r["quarter"] == "2024-12-31"
    assert r["close"] == 150.0
    assert r["market_cap"] == 1000.0
    assert r["per"] == pytest.approx(25.0)              # 1000 / 40(TTM NI)
    assert r["pbr"] == pytest.approx(5.0)               # 1000 / 200(equity)
    assert r["roe"] == pytest.approx(20.0)              # 40 / 200 * 100
    assert r["operating_margin"] == pytest.approx(30.0)  # 30 / 100 * 100
    assert r["net_margin"] == pytest.approx(10.0)        # 10 / 100 * 100


def test_metrics_at_us_skips_when_not_yet_disclosed(tmp_path):
    conn = _seed_us_db(tmp_path)
    # 2025-02-01엔 최신분기 2024-09-30만 유효한데 그 분기엔 Revenue/Equity 단일값이 없어
    # 마진/PBR은 None이 되지만 종목 자체는 유효분기가 있으므로 행은 나온다.
    rows = dau.metrics_at_us(conn, "2025-02-01")
    assert len(rows) == 1
    assert rows[0]["quarter"] == "2024-09-30"


def test_metrics_at_us_skips_company_without_price(tmp_path):
    conn = _seed_us_db(tmp_path)
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        ("NOPX", "NoPrice", "NYSE", "Tech", 500.0, "2025-03-01"),
    )
    conn.commit()
    codes = {r["stock_code"] for r in dau.metrics_at_us(conn, "2025-03-01")}
    assert "NOPX" not in codes  # 주가 없는 종목은 제외


# --------------------------------------------------------------------------
# build_callbacks_us — (metrics_fn, price_fn) 생성 + 캐시
# --------------------------------------------------------------------------
def test_build_callbacks_us_returns_metrics_and_price_fns(tmp_path):
    conn = _seed_us_db(tmp_path)
    metrics_fn, price_fn = dau.build_callbacks_us(conn)
    rows = metrics_fn("2025-03-01")
    assert rows[0]["stock_code"] == "AAPL"
    assert price_fn("2025-03-01", "AAPL") == 150.0


def test_build_callbacks_us_caches_metrics_per_asof(tmp_path):
    conn = _seed_us_db(tmp_path)
    metrics_fn, _ = dau.build_callbacks_us(conn)
    assert metrics_fn("2025-03-01") is metrics_fn("2025-03-01")  # 같은 시점은 캐시 재사용


# --------------------------------------------------------------------------
# build_sp500_benchmark_fn — ^GSPC 레벨 시계열 (DI로 네트워크 없이)
# --------------------------------------------------------------------------
def test_build_sp500_benchmark_fn_builds_normalized_levels():
    dates = ["2026-01-31", "2026-02-28", "2026-03-31"]

    def fake_fetch(start, end):
        # 리밸런싱일이 주말/휴장이면 그 이전 최근 거래일 종가를 쓴다(on-or-before)
        return {"2026-01-30": 4000.0, "2026-02-27": 4400.0, "2026-03-31": 4800.0}

    bench = dau.build_sp500_benchmark_fn(dates, fetch_fn=fake_fetch)
    assert bench("2026-01-31") == pytest.approx(1.0)   # 기준(base)
    assert bench("2026-02-28") == pytest.approx(1.1)   # 4400/4000
    assert bench("2026-03-31") == pytest.approx(1.2)   # 4800/4000


def test_build_sp500_benchmark_fn_none_for_unknown_date():
    dates = ["2026-01-31", "2026-02-28"]
    bench = dau.build_sp500_benchmark_fn(
        dates, fetch_fn=lambda s, e: {"2026-01-31": 4000.0, "2026-02-28": 4200.0}
    )
    assert bench("2099-12-31") is None
