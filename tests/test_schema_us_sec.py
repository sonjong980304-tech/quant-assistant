"""SEC EDGAR 데이터 플레인 스키마 회귀 테스트 (TDD, AC3).

.omc/specs/brainstorming-sec-edgar-us-financials-backfill.md AC3 검증:
- 새 테이블 us_financials_sec 가 원시 XBRL 팩트(tag/value/unit/fy/fp/form/filed 등)를 저장.
- us_company 에 cik 컬럼 추가(기존 컬럼은 보존).
- 기존 us_financials 테이블의 스키마·제약이 이번 작업으로 변하지 않음(완전 분리).

기존 test_schema_us.py 의 _columns/_tables 헬퍼 패턴을 그대로 따른다.
"""
from __future__ import annotations

import sqlite3

from src.db import init_db


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_init_db_creates_us_financials_sec_table(tmp_path):
    db = tmp_path / "sec.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        assert "us_financials_sec" in _tables(conn)
    finally:
        conn.close()


def test_us_financials_sec_has_raw_xbrl_fact_columns(tmp_path):
    """원시 XBRL 팩트 필드가 전부 컬럼으로 존재해야 한다(축소 저장 금지, 스펙 Round 7)."""
    db = tmp_path / "sec2.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        cols = _columns(conn, "us_financials_sec")
        for required in (
            "stock_code", "cik", "tag", "taxonomy", "unit", "value",
            "period_start", "period_end", "fy", "fp", "form", "filed",
            "frame", "accn", "source", "collected_at",
        ):
            assert required in cols, f"us_financials_sec 에 {required} 컬럼 없음"
    finally:
        conn.close()


def test_us_company_gains_cik_column(tmp_path):
    db = tmp_path / "sec3.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        assert "cik" in _columns(conn, "us_company")
    finally:
        conn.close()


def test_existing_us_financials_schema_unchanged(tmp_path):
    """기존 us_financials(yfinance) 컬럼 집합은 그대로여야 한다(회귀 방지, 스펙 Constraints)."""
    db = tmp_path / "sec4.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        cols = _columns(conn, "us_financials")
        assert cols == {
            "id", "stock_code", "as_of_date", "period_type", "statement_type",
            "item_key", "item_value", "disclosed_date", "source", "collected_at",
        }
        # SEC 전용 컬럼(cik/tag/fp/form/filed)이 기존 테이블로 새지 않았는지 확인
        assert not ({"cik", "tag", "fp", "form", "filed"} & cols)
    finally:
        conn.close()


def test_us_financials_sec_upsert_and_unique_constraint(tmp_path):
    """동일 (stock_code,tag,unit,period_start,period_end,form,accn) 재적재 시 1행 유지(멱등)."""
    db = tmp_path / "sec5.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        for val in (42905000000.0, 42905000001.0):
            conn.execute(
                "INSERT OR REPLACE INTO us_financials_sec"
                "(stock_code, cik, tag, taxonomy, unit, value, period_start, period_end, "
                "fy, fp, form, filed, frame, accn, source, collected_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("AAPL", "0000320193", "Revenues", "us-gaap", "USD", val,
                 "2008-09-28", "2009-09-26", 2009, "FY", "10-K", "2009-10-27",
                 "CY2009", "0001193125-09-214859", "sec_companyfacts_zip", "2026-07-19T00:00:00"),
            )
        conn.commit()
        rows = conn.execute(
            "SELECT value FROM us_financials_sec WHERE stock_code='AAPL' AND tag='Revenues'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 42905000001.0  # 마지막 값으로 갱신
    finally:
        conn.close()


def test_us_financials_sec_not_queryable_by_nl_sql(tmp_path):
    """SEC 테이블은 자연어 SQL 노출 대상이 아니다(기존 metrics/macro 관례와 동일, 별도 작업)."""
    from src.db import QUERYABLE_TABLES

    assert "us_financials_sec" not in QUERYABLE_TABLES
