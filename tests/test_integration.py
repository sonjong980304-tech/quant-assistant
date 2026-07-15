"""기능 축 — NL→SQL→결과 통합 테스트 (AC10).

정상 입력에서 오프라인(휴리스틱) 파이프라인이 끝까지 돌아 결과 행을 반환하는지
검증한다. 결정론 보장을 위해 offline=True(휴리스틱 폴백)로 실행하며, 사용자 DB를
건드리지 않도록 시드된 임시 DB(seeded_db)를 사용한다.
"""
from __future__ import annotations

from src.legacy.pipeline import Pipeline


def test_offline_pipeline_returns_rows(seeded_db):
    """대표 질문이 SQL 생성→실행→결과행 반환까지 완주한다."""
    p = Pipeline(db_path=seeded_db, offline=True)
    try:
        s = p.run("PER이 가장 낮은 10개 회사")
    finally:
        p.close()

    assert isinstance(s, dict)
    assert s.get("error") is None, f"실행 오류: {s.get('error')}"
    assert s.get("sql"), "SQL이 생성되지 않음"
    # 오프라인이므로 반드시 휴리스틱 폴백을 써야 한다(결정론).
    assert s.get("sql_source") == "fallback"
    assert s.get("row_count", 0) > 0, "결과 행이 비어 있음"

    rows = s.get("rows", [])
    assert rows, "rows가 비어 있음"
    # 휴리스틱 PER 질의는 (name, per) 컬럼을 반환한다.
    assert "per" in rows[0]
    # PER 오름차순 상위 10개 → 12개 중 10행.
    assert s.get("row_count") == 10


def test_offline_pipeline_orders_ascending(seeded_db):
    """PER '가장 낮은' 질의는 오름차순 정렬 결과를 준다(결과 정합성)."""
    p = Pipeline(db_path=seeded_db, offline=True)
    try:
        s = p.run("PER이 가장 낮은 10개 회사")
    finally:
        p.close()
    pers = [r["per"] for r in s.get("rows", []) if r.get("per") is not None]
    assert pers == sorted(pers), "PER 오름차순이 아님"
