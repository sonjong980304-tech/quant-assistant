"""매크로 지표 에이전트 DB 스키마 회귀 테스트 (MAC-1).

.omc/specs/brainstorming-macro-indicator-agent.md AC4/AC12/AC21 관련.
신규 테이블 macro_indicators(원시 시계열)/macro_signal(판정 이력)이
init_db로 생성되고, 기존 metrics 테이블과 동일하게 QUERYABLE_TABLES에는
포함되지 않음을 고정한다(자연어 SQL 질의 대상 아님).
"""
from __future__ import annotations

import sqlite3

from src.db import QUERYABLE_TABLES, init_db, schema_catalog


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def test_init_db_creates_macro_tables(tmp_path):
    db = tmp_path / "macro.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        assert {"macro_indicators", "macro_signal"} <= _tables(conn)
    finally:
        conn.close()


def test_macro_indicators_has_expected_columns(tmp_path):
    db = tmp_path / "macro2.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        cols = _columns(conn, "macro_indicators")
        assert {"indicator", "date", "value", "source"} <= cols
    finally:
        conn.close()


def test_macro_signal_has_expected_columns(tmp_path):
    db = tmp_path / "macro3.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        cols = _columns(conn, "macro_signal")
        assert {
            "as_of", "spread", "spread_regime", "cnn_value", "cnn_band",
            "vix_value", "vix_band", "overall", "prev_overall", "created_at",
        } <= cols
    finally:
        conn.close()


def test_macro_tables_not_queryable(tmp_path):
    # AC21: metrics 테이블과 동일 관례 — 자연어 SQL 질의 대상에서 제외.
    assert "macro_indicators" not in QUERYABLE_TABLES
    assert "macro_signal" not in QUERYABLE_TABLES
    catalog = schema_catalog(str(tmp_path / "unused.db"))
    assert "macro_indicators" not in catalog
    assert "macro_signal" not in catalog


def test_macro_indicators_unique_indicator_date_insert_or_replace(tmp_path):
    # AC4: UNIQUE(indicator,date) 기준 INSERT OR REPLACE 로 같은 지표/날짜는 덮어쓴다.
    db = tmp_path / "macro4.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT OR REPLACE INTO macro_indicators(indicator, date, value, source) VALUES (?,?,?,?)",
            ("T10Y2Y", "2026-07-14", 0.45, "FRED"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO macro_indicators(indicator, date, value, source) VALUES (?,?,?,?)",
            ("T10Y2Y", "2026-07-14", 0.52, "FRED"),
        )
        conn.commit()
        rows = conn.execute(
            "SELECT value FROM macro_indicators WHERE indicator='T10Y2Y' AND date='2026-07-14'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 0.52
    finally:
        conn.close()


def test_macro_signal_appends_rows_not_updates(tmp_path):
    # AC12: 판정은 날짜별 1행 append(UPDATE 아님) — 같은 as_of로 두 번 넣으면 2행.
    db = tmp_path / "macro5.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        for _ in range(2):
            conn.execute(
                "INSERT INTO macro_signal(as_of, spread, spread_regime, cnn_value, cnn_band, "
                "vix_value, vix_band, overall, prev_overall, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                ("2026-07-14", 0.5, "정상", 50, "중립", 14.0, "안정", "GREEN", None, "2026-07-14T09:00:00"),
            )
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM macro_signal WHERE as_of='2026-07-14'").fetchone()[0]
        assert n == 2
    finally:
        conn.close()
