"""미국 데이터 플레인 스키마 회귀 테스트.

.omc/specs/brainstorming-us-market-data-plane.md AC1(완전 분리 신설) 검증.
us_company/us_prices/us_financials는 기존 KR 테이블(company/prices/financials/
metrics)과 별개 테이블로 신설되며, company에는 country 컬럼을 얹지 않는다
(Round 1에서 명시적으로 기각된 방식).
"""
from __future__ import annotations

import sqlite3

from src.db import init_db


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_init_db_creates_us_company_us_prices_us_financials_tables(tmp_path):
    db = tmp_path / "new.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        tables = _tables(conn)
        assert {"us_company", "us_prices", "us_financials"} <= tables
    finally:
        conn.close()


def test_kr_company_table_does_not_gain_country_column(tmp_path):
    # Round 1에서 "기존 테이블에 country 컬럼 추가"안은 명시적으로 기각되고
    # "완전 분리" 신설로 결정됐다 — company 테이블은 그대로여야 한다.
    db = tmp_path / "new2.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        cols = _columns(conn, "company")
        assert "country" not in cols
    finally:
        conn.close()


def test_us_financials_columns_disjoint_from_kr_financials_and_metrics(tmp_path):
    db = tmp_path / "new3.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        us_cols = _columns(conn, "us_financials")
        kr_financials_cols = _columns(conn, "financials")
        kr_metrics_cols = _columns(conn, "metrics")
        # financials의 계정 EAV 전용 컬럼(quarter/account_key/account_name/amount)과
        # metrics의 비율 wide 컬럼(per/pbr/roe 등)이 us_financials와 겹치면 안 된다.
        kr_specific = {"quarter", "account_key", "account_name", "amount", "per", "pbr", "roe"}
        assert kr_specific <= (kr_financials_cols | kr_metrics_cols)
        assert not (kr_specific & us_cols)
    finally:
        conn.close()


def test_us_prices_upsert_and_unique_constraint(tmp_path):
    db = tmp_path / "new4.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT INTO us_prices(stock_code, date, open, high, low, close, volume) "
            "VALUES (?,?,?,?,?,?,?)",
            ("AAPL", "2026-07-10", 210.0, 212.5, 209.0, 211.0, 50_000_000),
        )
        conn.commit()

        conn.execute(
            "INSERT OR REPLACE INTO us_prices(stock_code, date, open, high, low, close, volume) "
            "VALUES (?,?,?,?,?,?,?)",
            ("AAPL", "2026-07-10", 210.0, 212.5, 209.0, 213.0, 51_000_000),
        )
        conn.commit()

        rows = conn.execute(
            "SELECT close FROM us_prices WHERE stock_code='AAPL' AND date='2026-07-10'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 213.0
    finally:
        conn.close()


def test_us_company_stores_numeric_market_cap(tmp_path):
    db = tmp_path / "new5.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            ("AAPL", "Apple Inc", "NASDAQ", "Technology", 2.32e12, "2026-07-12T00:00:00"),
        )
        conn.commit()
        row = conn.execute(
            "SELECT market_cap FROM us_company WHERE stock_code='AAPL'"
        ).fetchone()
        assert row[0] == 2.32e12
    finally:
        conn.close()
