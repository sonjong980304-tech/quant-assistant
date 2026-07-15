"""metrics_at_us()의 비USD 재무통화 무효화 검증 (TDD).

배경: SK텔레콤(SKM)처럼 us_financials 원본이 원화(KRW)인 종목은 시가총액(달러, 항상
정상)과 재무제표 숫자(원화)를 섞어 계산하면 PER=0.035류의 비현실적 값이 나온다.
us_company.financial_currency(scripts/backfill_us_financial_currency.py가 채움)가
'USD'가 아니면(NULL은 미수집이라 아직 판단 보류 — 기존 동작 유지) 재무 파생 필드
(per/pbr/roe/operating_margin/net_margin/operating_profit/revenue/net_income)를
None으로 무효화한다. 가격/시총(close/market_cap)은 항상 정상 달러 데이터이므로
건드리지 않는다 — "종목을 숨기지 않고 재무비율 계산에만 안 쓴다" 원칙
(src/data_quality.py의 "신뢰 못 할 건 계산에서 뺀다" 철학과 동일하되, 종목 전체가
아니라 필드 단위로 좁게 적용— 가격 데이터는 실제로 문제가 없기 때문).

검증:
(a) financial_currency='KRW'(비USD) 종목은 재무 파생 필드가 전부 None.
(b) financial_currency='USD'이거나 NULL(미수집, 기존 종목)인 종목은 전혀 영향 없음
    (회귀 — 지금까지 만든 것이 깨지면 안 됨).
(c) 실제 DB의 SKM 실데이터 패턴을 그대로 재현한 통합 케이스.
"""
from __future__ import annotations

import sqlite3

from src.backtest.data_access_us import metrics_at_us
from src.db import init_db


def _seed(tmp_path, code: str, financial_currency: str | None) -> sqlite3.Connection:
    """AAPL 시드 패턴(tests/test_data_access_us.py)과 동일 구조, financial_currency만 가변."""
    db = tmp_path / "curgate.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, "
        "financial_currency, updated_at) VALUES (?,?,?,?,?,?,?)",
        (code, code, "NYSE", "Telecommunications", 1000.0, financial_currency, "2026-07-01"),
    )
    quarters = [
        ("2025-03-31", "2025-05-15"), ("2025-06-30", "2025-08-14"),
        ("2025-09-30", "2025-11-14"), ("2025-12-31", "2026-02-14"),
    ]
    for as_of, disclosed in quarters:
        conn.execute(
            "INSERT INTO us_financials(stock_code, as_of_date, period_type, statement_type, "
            "item_key, item_value, disclosed_date, source) VALUES (?,?,?,?,?,?,?,?)",
            (code, as_of, "quarterly", "income_stmt", "Net Income", 10.0, disclosed, "yfinance"),
        )
    for item_key, value in [("Total Revenue", 100.0), ("Operating Income", 30.0)]:
        conn.execute(
            "INSERT INTO us_financials(stock_code, as_of_date, period_type, statement_type, "
            "item_key, item_value, disclosed_date, source) VALUES (?,?,?,?,?,?,?,?)",
            (code, "2025-12-31", "quarterly", "income_stmt", item_key, value, "2026-02-14", "yfinance"),
        )
    conn.execute(
        "INSERT INTO us_financials(stock_code, as_of_date, period_type, statement_type, "
        "item_key, item_value, disclosed_date, source) VALUES (?,?,?,?,?,?,?,?)",
        (code, "2025-12-31", "quarterly", "balance_sheet", "Stockholders Equity", 200.0, "2026-02-14", "yfinance"),
    )
    conn.execute(
        "INSERT INTO us_prices(stock_code, date, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
        (code, "2026-06-30", 148.0, 152.0, 147.0, 150.0, 1000.0),
    )
    conn.commit()
    return conn


_FINANCIAL_FIELDS = (
    "per", "pbr", "roe", "operating_margin", "net_margin",
    "operating_profit", "revenue", "net_income",
)


# ---------------------------------------------------------------------------
# (a) 비USD 재무통화 → 재무 파생 필드 전부 None
# ---------------------------------------------------------------------------
def test_non_usd_financial_currency_nullifies_financial_fields(tmp_path):
    conn = _seed(tmp_path, "SKM", "KRW")
    rows = metrics_at_us(conn, "2026-06-30")
    assert len(rows) == 1
    r = rows[0]
    for key in _FINANCIAL_FIELDS:
        assert r[key] is None, f"{key} 는 비USD 종목이라 None 이어야 하는데 {r[key]!r}"
    conn.close()


def test_non_usd_financial_currency_keeps_price_and_market_cap(tmp_path):
    """가격/시총은 항상 정상 달러 데이터이므로 영향받지 않는다."""
    conn = _seed(tmp_path, "SKM", "KRW")
    rows = metrics_at_us(conn, "2026-06-30")
    r = rows[0]
    assert r["close"] == 150.0
    assert r["market_cap"] == 1000.0
    conn.close()


def test_non_usd_financial_currency_stock_still_appears_in_results(tmp_path):
    """종목 자체를 숨기지 않는다(제외가 아니라 필드 무효화) — return_12m 등은 살아있어도 됨."""
    conn = _seed(tmp_path, "SKM", "KRW")
    rows = metrics_at_us(conn, "2026-06-30")
    assert {r["stock_code"] for r in rows} == {"SKM"}
    conn.close()


# ---------------------------------------------------------------------------
# (b) USD 또는 NULL(미수집)인 종목은 전혀 영향 없음 (회귀 — 매우 중요)
# ---------------------------------------------------------------------------
def test_usd_financial_currency_unaffected_regression(tmp_path):
    conn = _seed(tmp_path, "NVDA", "USD")
    rows = metrics_at_us(conn, "2026-06-30")
    r = rows[0]
    assert r["operating_profit"] == 30.0
    assert r["revenue"] == 100.0
    assert r["net_income"] == 10.0
    assert r["operating_margin"] is not None
    assert r["pbr"] is not None
    conn.close()


def test_null_financial_currency_unaffected_regression(tmp_path):
    """financial_currency 미수집(NULL, 배치 스크립트 실행 전 기존 종목)은 기존 동작 그대로."""
    conn = _seed(tmp_path, "AAPL", None)
    rows = metrics_at_us(conn, "2026-06-30")
    r = rows[0]
    assert r["operating_profit"] == 30.0
    assert r["revenue"] == 100.0
    assert r["net_income"] == 10.0
    assert r["operating_margin"] is not None
    assert r["pbr"] is not None
    conn.close()


# ---------------------------------------------------------------------------
# (c) 실제 SKM 데이터 패턴 재현 통합 케이스 (2026Q1, 4.39조원 매출 규모)
# ---------------------------------------------------------------------------
def test_skm_real_world_krw_scale_pattern_nullified(tmp_path):
    """실제 DB의 SKM 규모(매출 4.39조원)를 그대로 재현 — 비USD 판정 시 completely 무효화."""
    db = tmp_path / "skm.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, "
        "financial_currency, updated_at) VALUES (?,?,?,?,?,?,?)",
        ("SKM", "SK Telecom Co. Ltd. Common Stock", "NYSE", "Telecommunications",
         12_746_363_400.0, "KRW", "2026-07-12"),
    )
    for as_of, disclosed in [
        ("2025-03-31", "2025-05-15"), ("2025-06-30", "2025-08-14"),
        ("2025-09-30", "2025-11-14"), ("2025-12-31", "2026-02-14"),
    ]:
        conn.execute(
            "INSERT INTO us_financials(stock_code, as_of_date, period_type, statement_type, "
            "item_key, item_value, disclosed_date, source) VALUES (?,?,?,?,?,?,?,?)",
            ("SKM", as_of, "quarterly", "income_stmt", "Net Income", 300_000_000_000.0, disclosed, "yfinance"),
        )
    conn.execute(
        "INSERT INTO us_financials(stock_code, as_of_date, period_type, statement_type, "
        "item_key, item_value, disclosed_date, source) VALUES (?,?,?,?,?,?,?,?)",
        ("SKM", "2025-12-31", "quarterly", "income_stmt", "Total Revenue", 4_392_312_000_000.0, "2026-02-14", "yfinance"),
    )
    conn.execute(
        "INSERT INTO us_financials(stock_code, as_of_date, period_type, statement_type, "
        "item_key, item_value, disclosed_date, source) VALUES (?,?,?,?,?,?,?,?)",
        ("SKM", "2025-12-31", "quarterly", "income_stmt", "Operating Income", 537_591_000_000.0, "2026-02-14", "yfinance"),
    )
    conn.execute(
        "INSERT INTO us_prices(stock_code, date, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
        ("SKM", "2026-06-30", 20.0, 21.0, 19.5, 20.5, 100000.0),
    )
    conn.commit()

    rows = metrics_at_us(conn, "2026-06-30")
    r = rows[0]
    # 통화 무효화 없이는 PER=시총/순이익(KRW)이 0.035류 비현실값이 됐을 것 — 이제 None.
    assert r["per"] is None
    assert r["revenue"] is None
    assert r["operating_profit"] is None
    assert r["close"] == 20.5  # 가격은 정상 유지
    conn.close()
