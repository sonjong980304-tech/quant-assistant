"""diagnose.collect_evidence가 pipeline 경로에서도 유의미한 증거를 모으는지 (TDD).

Fable5 서브그래프 진단(2026-07-12)에서 발견: pipeline 경로는 state["sql"]에 SQL이
아니라 파이프라인 JSON이 들어있는데, collect_evidence가 이를 probe_conditions(SQL
WHERE절 파싱)에 그대로 넘겨 파싱 실패로 증거가 항상 빈 리스트가 된다(진단 품질 저하).
sql_source=='pipeline'이면 SQL 조건분해 대신 원본 파이프라인 JSON 자체를 증거로 준다.
"""
from __future__ import annotations

from src.legacy.graph.diagnose import collect_evidence


def test_collect_evidence_returns_pipeline_json_for_pipeline_source_empty_result():
    state = {
        "sql_source": "pipeline",
        "sql": '{"pipeline": [{"op": "combine", "params": {"criteria": [{"key": "bad_field"}]}}]}',
        "row_count": 0,
        "expected_count": 20,
    }
    evidence = collect_evidence(None, state, "empty")

    assert evidence["type"] == "pipeline_probe"
    assert evidence["pipeline"] == state["sql"]
    assert evidence["expected"] == 20
    assert evidence["got"] == 0


def test_collect_evidence_still_uses_sql_probe_for_generated_sql_source():
    """route가 SQL이면(sql_source != 'pipeline') 기존 동작(probe_conditions 시도)을 유지한다."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t(x INTEGER)")
    conn.execute("INSERT INTO t VALUES (1), (2), (3)")
    conn.commit()
    state = {
        "sql_source": "generated",
        "sql": "SELECT x FROM t WHERE x > 1",
        "row_count": 2,
        "expected_count": 5,
    }
    evidence = collect_evidence(conn, state, "short")

    assert evidence["type"] == "probe"
    assert "steps" in evidence
