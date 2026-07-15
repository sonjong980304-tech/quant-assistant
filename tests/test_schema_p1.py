"""P1(네이버/FnGuide 크롤러) 스키마 확장 회귀 테스트.

.omc/specs/brainstorming-naver-fnguide-crawlers.md AC1(prices OHLCV 확장) +
AC7(fnguide_metrics 신규 테이블, 기존 metrics와 완전 분리) 검증.
"""
from __future__ import annotations

import sqlite3

from src.db import init_db


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def test_init_db_creates_prices_with_ohlv_columns(tmp_path):
    db = tmp_path / "new.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        cols = _columns(conn, "prices")
        assert {"open", "high", "low", "volume"} <= cols
    finally:
        conn.close()


def test_migrate_adds_ohlv_columns_to_existing_db_without_data_loss(tmp_path):
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    try:
        # 구버전 스키마(OHLV 컬럼 없음) 흉내
        conn.execute(
            """CREATE TABLE prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT NOT NULL,
                date TEXT NOT NULL,
                close REAL,
                market_cap REAL,
                UNIQUE(stock_code, date)
            )"""
        )
        conn.execute(
            "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,?)",
            ("005930", "2026-01-01", 70000, 1_000_000_000_000),
        )
        conn.commit()
    finally:
        conn.close()

    # init_db가 내부적으로 _migrate를 호출해 무중단 이행한다
    init_db(str(db))

    conn = sqlite3.connect(str(db))
    try:
        cols = _columns(conn, "prices")
        assert {"open", "high", "low", "volume"} <= cols

        row = conn.execute(
            "SELECT close, market_cap FROM prices WHERE stock_code='005930' AND date='2026-01-01'"
        ).fetchone()
        assert row == (70000, 1_000_000_000_000)
    finally:
        conn.close()


def test_init_db_creates_fnguide_metrics_table_separate_from_metrics(tmp_path):
    db = tmp_path / "new2.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "fnguide_metrics" in tables

        metrics_cols = _columns(conn, "metrics")
        fnguide_cols = _columns(conn, "fnguide_metrics")
        # DART 계산치 전용 wide 컬럼(per/pbr/roe 등)을 fnguide_metrics가 그대로
        # 재사용하지 않는지 확인 — 완전히 분리된 스키마여야 한다.
        dart_specific = {"per", "pbr", "roe", "operating_margin", "debt_ratio"}
        assert dart_specific <= metrics_cols
        assert not (dart_specific & fnguide_cols)
    finally:
        conn.close()


def test_fnguide_metrics_upsert_and_query(tmp_path):
    db = tmp_path / "new3.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            """INSERT INTO fnguide_metrics
                 (stock_code, as_of_date, metric_key, metric_value, source, collected_at)
               VALUES (?,?,?,?,?,?)""",
            ("005930", "2026-07-11", "roe", 12.3, "fnguide", "2026-07-11T00:00:00"),
        )
        conn.commit()

        row = conn.execute(
            "SELECT metric_value FROM fnguide_metrics WHERE stock_code='005930' AND metric_key='roe'"
        ).fetchone()
        assert row[0] == 12.3

        # UNIQUE(stock_code, as_of_date, metric_key) — 재삽입 시 REPLACE로 갱신돼야 한다
        conn.execute(
            """INSERT OR REPLACE INTO fnguide_metrics
                 (stock_code, as_of_date, metric_key, metric_value, source, collected_at)
               VALUES (?,?,?,?,?,?)""",
            ("005930", "2026-07-11", "roe", 15.0, "fnguide", "2026-07-12T00:00:00"),
        )
        conn.commit()

        rows = conn.execute(
            "SELECT metric_value FROM fnguide_metrics WHERE stock_code='005930' AND metric_key='roe'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 15.0
    finally:
        conn.close()
