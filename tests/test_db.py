"""db.schema_catalog 단위테스트 (TDD).

schema_catalog()는 LLM 프롬프트에 넣을 스키마 DDL을 만든다. 기본값은 QUERYABLE_TABLES
(company/financials/prices)만 노출하고, 사전계산 스냅샷 테이블 metrics는 제외한다
(백테스트 미래참조편향 방지 관례). include_metrics=True를 명시적으로 opt-in한 경로
(exec_fallback 자유코드)에서만 metrics DDL을 추가로 포함시킨다.
"""
from __future__ import annotations

from src.db import schema_catalog


def test_schema_catalog_default_excludes_metrics_table():
    # 기본 동작 회귀 방지: 인자 없이 부르면 metrics DDL이 포함되지 않아야 한다.
    catalog = schema_catalog()
    assert "CREATE TABLE metrics" not in catalog


def test_schema_catalog_default_still_includes_queryable_tables():
    catalog = schema_catalog()
    assert "CREATE TABLE company" in catalog
    assert "CREATE TABLE financials" in catalog
    assert "CREATE TABLE prices" in catalog


def test_schema_catalog_include_metrics_true_adds_metrics_ddl():
    catalog = schema_catalog(include_metrics=True)
    assert "CREATE TABLE metrics" in catalog
    # 파생지표 컬럼(예: pbr, gp_a)도 DDL 안에 실제로 들어와야 실질적으로 쓸 수 있다.
    assert "pbr" in catalog
    assert "gp_a" in catalog


def test_schema_catalog_include_metrics_false_matches_default():
    # False를 명시적으로 줘도 기본값과 동일(metrics 없음).
    assert "CREATE TABLE metrics" not in schema_catalog(include_metrics=False)
    assert schema_catalog(include_metrics=False) == schema_catalog()
