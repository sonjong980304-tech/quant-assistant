"""미국판 백테스트 DB 어댑터(data_access_us.py) 단위 테스트 (TDD).

한국판 data_access.py의 effective_quarter_at/_price_at/_is_alive/metrics_at/
build_callbacks 구조를 미국(us_company/us_prices/us_financials EAV)판으로 옮긴 것을 검증한다.
- look-ahead 방지: disclosed_date<=asof인 최신 quarterly만 사용.
- 생존편향: us_delisting(FMP, 구간 기반)으로 _is_alive_us가 bool을 반환(티커 재사용 오탐 방지).
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
# _is_alive_us — us_delisting 구간 기반 bool 판정(KR _is_alive와 동일 시그니처, AC4/AC15)
# --------------------------------------------------------------------------
def _seed_us_delisting(conn, code: str, episodes: list[tuple[str, str]]) -> None:
    """episodes = [(listing_date, delisting_date), ...] 을 us_delisting에 시드한다."""
    for listing, delisting in episodes:
        conn.execute(
            "INSERT INTO us_delisting(stock_code, company_name, exchange, listing_date, delisting_date) "
            "VALUES (?,?,?,?,?)", (code, code, "NYSE", listing, delisting))
    conn.commit()


def test_is_alive_us_returns_bool_not_string(tmp_path):
    # AC4: KR _is_alive와 동일하게 bool(True/False)을 반환한다(더 이상 "unverifiable" 문자열 아님).
    conn = _seed_us_db(tmp_path)
    result = dau._is_alive_us(conn, "AAPL", "2025-03-01")
    assert result is True  # 상폐 이력 없음 → 살아있음
    assert isinstance(result, bool)


def test_is_alive_us_true_when_no_delisting_row(tmp_path):
    conn = _seed_us_db(tmp_path)
    assert dau._is_alive_us(conn, "AAPL", "2010-01-01") is True


def test_is_alive_us_true_before_delisting(tmp_path):
    conn = _seed_us_db(tmp_path)
    _seed_us_delisting(conn, "DEAD", [("2013-11-07", "2022-10-27")])
    assert dau._is_alive_us(conn, "DEAD", "2018-01-01") is True


def test_is_alive_us_false_on_or_after_delisting(tmp_path):
    conn = _seed_us_db(tmp_path)
    _seed_us_delisting(conn, "DEAD", [("2013-11-07", "2022-10-27")])
    assert dau._is_alive_us(conn, "DEAD", "2023-01-01") is False
    # 상장폐지일 당일도 KR과 동일하게 죽은 것으로 본다(KR _is_alive: delisting_date>asof).
    assert dau._is_alive_us(conn, "DEAD", "2022-10-27") is False


def test_is_alive_us_true_after_relisting_ticker_reuse(tmp_path):
    # AC15: 티커 재사용(TWTR류). 구간A(2013~2020) 상폐 후 구간B(2023~2025) 재상장.
    # 구간B 안(2024)의 조회는 앞선 상폐(2020)에도 불구하고 오탐 없이 True여야 한다.
    conn = _seed_us_db(tmp_path)
    _seed_us_delisting(conn, "TWTR", [("2013-01-01", "2020-01-01"), ("2023-01-01", "2025-01-01")])
    assert dau._is_alive_us(conn, "TWTR", "2024-01-01") is True


def test_is_alive_us_false_in_gap_between_delisting_and_relisting(tmp_path):
    # AC15 대칭: 구간A 상폐(2020) 후 재상장(2023) 전 공백기(2021)는 죽은 상태여야 한다.
    conn = _seed_us_db(tmp_path)
    _seed_us_delisting(conn, "TWTR", [("2013-01-01", "2020-01-01"), ("2023-01-01", "2025-01-01")])
    assert dau._is_alive_us(conn, "TWTR", "2021-06-01") is False
    # 재상장 구간 자체도 상폐(2025) 뒤(2026)에는 다시 죽는다.
    assert dau._is_alive_us(conn, "TWTR", "2026-01-01") is False


def test_is_alive_us_true_for_reused_active_single_episode(tmp_path):
    # critic 실데이터 모양(AC15 핵심): 옛 상폐 구간 1개 + '현재 활성 마커'(delisting_date='').
    # 상폐(2022) 이후 시점은 활성 마커 덕에 오탐 없이 살아있음으로 판정돼야 한다.
    conn = _seed_us_db(tmp_path)
    _seed_us_delisting(conn, "TWTR", [("2013-11-07", "2022-10-27"), ("", "")])  # 2번째=활성 마커
    assert dau._is_alive_us(conn, "TWTR", "2024-01-01") is True
    assert dau._is_alive_us(conn, "TWTR", "2018-01-01") is True


def _seed_extra_company(conn, code: str, close: float = 150.0) -> None:
    """metrics_at_us에 잡히도록 code에 최소 재무(유효분기)+주가를 시드한다(상폐 필터 검증용)."""
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, updated_at) "
        "VALUES (?,?,?,?,?,?)", (code, code, "NYSE", "Tech", 1000.0, "2025-03-01"))
    conn.execute(
        "INSERT INTO us_financials(stock_code, as_of_date, period_type, statement_type, "
        "item_key, item_value, disclosed_date, source) VALUES (?,?,?,?,?,?,?,?)",
        (code, "2024-12-31", "quarterly", "income_stmt", "Net Income", 10.0, "2025-02-14", "yfinance"))
    conn.execute(
        "INSERT INTO us_prices(stock_code, date, open, high, low, close, volume) "
        "VALUES (?,?,?,?,?,?,?)", (code, "2025-03-01", close - 2, close + 2, close - 3, close, 1000.0))
    conn.commit()


# --------------------------------------------------------------------------
# metrics_at_us — 사전필터: 상장폐지 종목은 선정 후보에서 제외(AC6/AC12)
# --------------------------------------------------------------------------
def test_metrics_at_us_excludes_delisted_ticker(tmp_path):
    # 상폐된 GHOST를 주가·재무가 있는 가짜 종목으로 주입해도, us_delisting 구간에 걸리면
    # metrics_at_us 결과에서 실제로 빠져야 한다(사전필터). AAPL은 그대로 남는다.
    conn = _seed_us_db(tmp_path)
    _seed_extra_company(conn, "GHOST")
    conn.execute(
        "INSERT INTO us_delisting(stock_code, company_name, exchange, listing_date, delisting_date) "
        "VALUES (?,?,?,?,?)", ("GHOST", "유령", "NYSE", "2010-01-01", "2020-01-01"))
    conn.commit()
    codes = {r["stock_code"] for r in dau.metrics_at_us(conn, "2025-03-01")}
    assert "GHOST" not in codes   # 상폐 → 사전필터로 제외
    assert "AAPL" in codes         # 생존 종목은 그대로


def test_metrics_at_us_keeps_relisted_ticker(tmp_path):
    # AC15/AC6 결합: 재상장 구간 안(2024) 시점의 재사용 티커는 사전필터에서 제외되지 않는다.
    conn = _seed_us_db(tmp_path)
    _seed_extra_company(conn, "REUSE")
    for listing, delisting in [("2010-01-01", "2020-01-01"), ("2023-01-01", "2027-01-01")]:
        conn.execute(
            "INSERT INTO us_delisting(stock_code, company_name, exchange, listing_date, delisting_date) "
            "VALUES (?,?,?,?,?)", ("REUSE", "재사용", "NYSE", listing, delisting))
    conn.commit()
    codes = {r["stock_code"] for r in dau.metrics_at_us(conn, "2025-03-01")}
    assert "REUSE" in codes  # 2025-03-01은 재상장 구간(2023~2027) 안 → 살아있음


def test_metrics_at_us_keeps_reused_active_single_episode(tmp_path):
    # critic 실데이터 모양: 옛 상폐 구간 1개 + 활성 마커인 재사용 종목은 사전필터에서 제외되지 않는다.
    conn = _seed_us_db(tmp_path)
    _seed_extra_company(conn, "TWTR")
    conn.execute(
        "INSERT INTO us_delisting(stock_code, company_name, exchange, listing_date, delisting_date) "
        "VALUES (?,?,?,?,?)", ("TWTR", "Twitter", "NYSE", "2013-11-07", "2022-10-27"))
    conn.execute(
        "INSERT INTO us_delisting(stock_code, company_name, exchange, listing_date, delisting_date) "
        "VALUES (?,?,?,?,?)", ("TWTR", None, None, "", ""))  # 현재 활성 마커
    conn.commit()
    codes = {r["stock_code"] for r in dau.metrics_at_us(conn, "2025-03-01")}
    assert "TWTR" in codes  # 상폐(2022) 이후지만 활성 마커 → 후보 유지


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


# ==========================================================================
# metrics_at_us — 파생 지표 확장(psr/pcr/ev_ebitda/peg/roa/gp_a/debt_ratio/
# current_ratio/interest_coverage/revenue_growth/op_growth/ni_growth) (TDD).
#
# 배경: 백테스트 UI 체크박스(db.py METRIC_DEFS 20개)는 국가 구분 없이 노출되는데
# metrics_at_us는 그중 12개만 계산해, 미국에서 나머지를 선택하면 selection.py의
# _validate_criteria_keys가 "존재하지 않는 필드" ValueError로 백테스트를 크래시시켰다.
# us_financials(yfinance EAV)에는 Total Assets/Total Debt/Current Assets·Liabilities/
# Operating Cash Flow/EBITDA/Interest Expense/Gross Profit 원본이 이미 수집돼 있어
# (data/market.db 직접 조회로 확인), 공식만 추가하면 계산 가능하다.
# ==========================================================================
def _seed_us_full(tmp_path, financial_currency: str = "USD") -> sqlite3.Connection:
    """AAPL 1종목을 5개 quarterly(2023-12-31 전년동기 + 2024 4분기)로 시드한다.

    손익(TTM 4분기 + 성장률용 전년동기), 재무상태(최신분기 스냅샷), 현금흐름을 모두 채워
    새 파생 지표를 결정론적으로 검증할 수 있게 한다. 값은 검산 편의를 위해 딱 떨어지게 잡았다:
      · Net Income   2023Q4=8,   2024 각 분기=10  → TTM=40, 단일분기=10, YoY=(10-8)/8=25%
      · Total Revenue 2023Q4=80,  2024 각 분기=100 → TTM=400, 단일=100, YoY=25%
      · Operating Income 2023Q4=24, 2024 각 분기=30 → TTM=120, 단일=30, YoY=25%
      · Gross Profit 2024 각 분기=40 → TTM=160 · EBITDA 각 분기=50 → TTM=200
      · Interest Expense 각 분기=5 → TTM=20 · Operating Cash Flow 각 분기=25 → TTM=100
    최신분기(2024-12-31) 재무상태: Stockholders Equity=200, Total Assets=800,
    Total Liabilities Net Minority Interest=600, Current Assets=300, Current Liabilities=150,
    Total Debt=250, Cash And Cash Equivalents=50. market_cap=1000, 종가(2025-03-01)=150.
    """
    db = tmp_path / "dau_full.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, "
        "financial_currency, updated_at) VALUES (?,?,?,?,?,?,?)",
        ("AAPL", "Apple", "NASDAQ", "Technology", 1000.0, financial_currency, "2025-03-01"),
    )

    def ins(as_of, disclosed, stmt, key, val):
        conn.execute(
            "INSERT INTO us_financials(stock_code, as_of_date, period_type, statement_type, "
            "item_key, item_value, disclosed_date, source) VALUES (?,?,?,?,?,?,?,?)",
            ("AAPL", as_of, "quarterly", stmt, key, val, disclosed, "yfinance"),
        )

    ttm_quarters = [
        ("2024-03-31", "2024-05-15"), ("2024-06-30", "2024-08-14"),
        ("2024-09-30", "2024-11-14"), ("2024-12-31", "2025-02-14"),
    ]
    prev_year = ("2023-12-31", "2024-02-14")  # 성장률(YoY) 비교용 전년동기
    # 손익: 4개 TTM 분기 값
    for as_of, disclosed in ttm_quarters:
        ins(as_of, disclosed, "income_stmt", "Net Income", 10.0)
        ins(as_of, disclosed, "income_stmt", "Total Revenue", 100.0)
        ins(as_of, disclosed, "income_stmt", "Operating Income", 30.0)
        ins(as_of, disclosed, "income_stmt", "Gross Profit", 40.0)
        ins(as_of, disclosed, "income_stmt", "EBITDA", 50.0)
        ins(as_of, disclosed, "income_stmt", "Interest Expense", 5.0)
        ins(as_of, disclosed, "cashflow", "Operating Cash Flow", 25.0)
    # 전년동기(단일분기 YoY 분모): Net Income/Total Revenue/Operating Income만
    ins(prev_year[0], prev_year[1], "income_stmt", "Net Income", 8.0)
    ins(prev_year[0], prev_year[1], "income_stmt", "Total Revenue", 80.0)
    ins(prev_year[0], prev_year[1], "income_stmt", "Operating Income", 24.0)
    # 최신분기 재무상태(스냅샷)
    for key, val in [
        ("Stockholders Equity", 200.0), ("Total Assets", 800.0),
        ("Total Liabilities Net Minority Interest", 600.0),
        ("Current Assets", 300.0), ("Current Liabilities", 150.0),
        ("Total Debt", 250.0), ("Cash And Cash Equivalents", 50.0),
    ]:
        ins("2024-12-31", "2025-02-14", "balance_sheet", key, val)
    for date_str, close in [("2025-01-15", 140.0), ("2025-03-01", 150.0)]:
        conn.execute(
            "INSERT INTO us_prices(stock_code, date, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
            ("AAPL", date_str, close - 2, close + 2, close - 3, close, 1000.0),
        )
    conn.commit()
    return conn


def test_metrics_at_us_computes_all_derived_metrics(tmp_path):
    conn = _seed_us_full(tmp_path)
    rows = dau.metrics_at_us(conn, "2025-03-01")
    assert len(rows) == 1
    r = rows[0]
    # 밸류
    assert r["psr"] == pytest.approx(2.5)          # 1000 / 400(TTM 매출)
    assert r["pcr"] == pytest.approx(10.0)         # 1000 / 100(TTM 영업현금흐름)
    assert r["ev_ebitda"] == pytest.approx(6.0)    # EV(1000+250-50=1200) / 200(TTM EBITDA)
    assert r["peg"] == pytest.approx(1.0)          # per(25) / ni_growth(25)
    # 수익성
    assert r["roa"] == pytest.approx(5.0)          # 40(TTM 순이익) / 800(총자산) * 100
    assert r["gp_a"] == pytest.approx(20.0)        # 160(TTM 매출총이익) / 800 * 100
    # 안정성
    assert r["debt_ratio"] == pytest.approx(300.0)     # 600(부채) / 200(자본) * 100
    assert r["current_ratio"] == pytest.approx(200.0)  # 300(유동자산) / 150(유동부채) * 100
    assert r["interest_coverage"] == pytest.approx(6.0)  # 120(TTM 영업이익) / 20(TTM 이자비용)
    # 성장(YoY 단일분기)
    assert r["revenue_growth"] == pytest.approx(25.0)  # (100-80)/80 * 100
    assert r["op_growth"] == pytest.approx(25.0)       # (30-24)/24 * 100
    assert r["ni_growth"] == pytest.approx(25.0)       # (10-8)/8 * 100
    conn.close()


def test_metrics_at_us_new_fields_registered_in_descriptions():
    """새 파생 지표가 단일 정의처(METRIC_FIELD_DESCRIPTIONS_US)에 등록돼야 UI/스크리닝에 노출된다."""
    for key in (
        "psr", "pcr", "ev_ebitda", "peg", "roa", "gp_a", "debt_ratio",
        "current_ratio", "interest_coverage", "revenue_growth", "op_growth", "ni_growth",
    ):
        assert key in dau.METRIC_FIELD_DESCRIPTIONS_US, f"{key} 가 정의처에 없음"


def test_metrics_at_us_new_fields_none_when_raw_data_missing(tmp_path):
    """원본 항목(총자산/부채/현금흐름 등)이 없는 종목은 새 지표가 None (억지 추정 안 함)."""
    conn = _seed_us_db(tmp_path)  # 기존 최소 시드(Total Assets/Total Debt 등 없음)
    r = dau.metrics_at_us(conn, "2025-03-01")[0]
    for key in ("pcr", "ev_ebitda", "roa", "gp_a", "debt_ratio", "current_ratio", "interest_coverage"):
        assert r[key] is None, f"{key} 는 원본 없으면 None 이어야 하는데 {r[key]!r}"
    conn.close()


def test_metrics_at_us_new_derived_fields_nullified_for_non_usd(tmp_path):
    """비USD(예: KRW) 재무통화 종목은 새 재무 파생 지표도 전부 무효화(통화 불일치 방어)."""
    conn = _seed_us_full(tmp_path, financial_currency="KRW")
    r = dau.metrics_at_us(conn, "2025-03-01")[0]
    for key in (
        "psr", "pcr", "ev_ebitda", "peg", "roa", "gp_a", "debt_ratio",
        "current_ratio", "interest_coverage", "revenue_growth", "op_growth", "ni_growth",
    ):
        assert r[key] is None, f"{key} 는 비USD 라 None 이어야 하는데 {r[key]!r}"
    # 가격/시총은 정상 달러 데이터라 유지
    assert r["close"] == 150.0
    assert r["market_cap"] == 1000.0
    conn.close()
