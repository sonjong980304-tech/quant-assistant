"""US 종목 오류 케이스도 기존 diagnose_node 경로로 처리되는지 (TDD, C-5 AC12).

.omc/specs/brainstorming-us-nl-sql-integration.md 참고. 존재하지 않는 US 종목명으로
SQL이 생성돼 결과가 0건이면, KR과 동일하게 classify_status가 'empty'로 분류되고
collect_evidence가 SQL 조건분해(probe_conditions)를 시도한다 — US 전용 별도
에러 처리 로직 없이 기존 sql/refine/data/none 분류 경로를 그대로 재사용한다.
"""
from __future__ import annotations

import sqlite3

from src.legacy.graph.diagnose import classify_status, collect_evidence
from src.db import init_db


def test_classify_status_empty_for_us_query_with_zero_rows():
    state = {
        "sql_source": "generated",
        "sql": "SELECT * FROM us_company WHERE name='존재하지않는회사'",
        "row_count": 0,
        "rows": [],
        "columns": [],
    }
    assert classify_status(state) == "empty"


def test_collect_evidence_probes_us_sql_where_clause_like_kr(tmp_path):
    """US 스키마 SQL도 WHERE절 조건분해가 정상 동작한다(테이블명과 무관한 텍스트 파싱)."""
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, updated_at) "
        "VALUES ('AAPL','Apple Inc.','NASDAQ','Technology',3000000000000,'2026-07-12T00:00:00')"
    )
    conn.commit()
    state = {
        "sql_source": "generated",
        "sql": "SELECT * FROM us_company WHERE name='존재하지않는회사' AND exchange='NASDAQ'",
        "row_count": 0,
        "expected_count": None,
    }

    evidence = collect_evidence(conn, state, "empty")

    assert evidence["type"] == "probe"
    assert len(evidence["steps"]) >= 1  # 조건분해가 US SQL에도 정상 동작(빈 리스트 아님)
