"""US 테이블(us_company/us_prices/us_financials) 스키마 노출 (TDD, C-5 AC1/AC3).

.omc/specs/brainstorming-us-nl-sql-integration.md 참고.
QUERYABLE_TABLES에 US 3개 테이블을 추가하면 schema_catalog()가 자동으로 그 DDL을
포함해야 한다(질문 내용과 무관하게 항상 KR+US 스키마를 동시 노출 — Round6 결정).
"""
from __future__ import annotations

from src.db import QUERYABLE_TABLES, schema_catalog


def test_queryable_tables_includes_all_three_us_tables():
    assert "us_company" in QUERYABLE_TABLES
    assert "us_prices" in QUERYABLE_TABLES
    assert "us_financials" in QUERYABLE_TABLES


def test_queryable_tables_still_includes_kr_tables():
    assert "company" in QUERYABLE_TABLES
    assert "financials" in QUERYABLE_TABLES
    assert "prices" in QUERYABLE_TABLES


def test_schema_catalog_includes_us_company_ddl():
    catalog = schema_catalog(":memory:")
    assert "CREATE TABLE us_company" in catalog


def test_schema_catalog_includes_us_prices_ddl():
    catalog = schema_catalog(":memory:")
    assert "CREATE TABLE us_prices" in catalog


def test_schema_catalog_includes_us_financials_ddl():
    catalog = schema_catalog(":memory:")
    assert "CREATE TABLE us_financials" in catalog


def test_schema_catalog_still_includes_kr_company_ddl():
    catalog = schema_catalog(":memory:")
    assert "CREATE TABLE company" in catalog
