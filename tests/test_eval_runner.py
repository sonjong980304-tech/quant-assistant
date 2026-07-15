"""평가(run_evaluation) 격리 회귀 테스트.

run_evaluation()은 절대로 db_path가 가리키는 원본 DB를 직접 수정해서는 안 된다
(기존 wiki 기록 삭제도, 평가용 합성 기록 삽입도). 항상 격리된 사본에서 평가가
이뤄져야 프로덕션 질의 기록 로그(wiki 테이블)가 보존된다.
"""
from __future__ import annotations

import sqlite3

from src.wiki.store import WikiStore


def _wiki_rows(db_path: str) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT id, question, sql FROM wiki ORDER BY id").fetchall()
    finally:
        conn.close()


def test_run_evaluation_does_not_touch_source_db_wiki_table(seeded_db):
    from src.eval.runner import run_evaluation

    # 평가 전, 원본 DB에 이미 실사용자 질의 기록이 있다고 가정한다.
    conn = sqlite3.connect(seeded_db)
    try:
        WikiStore(conn).save_record(
            question="기존 기록 1", raw_question="기존 기록 1",
            sql="SELECT 1", route="financial",
        )
        WikiStore(conn).save_record(
            question="기존 기록 2", raw_question="기존 기록 2",
            sql="SELECT 2", route="financial",
        )
    finally:
        conn.close()

    before = _wiki_rows(seeded_db)
    assert len(before) == 2

    # GOLDSET[0]은 PER 질의라 오프라인 휴리스틱 폴백이 seeded_db의 metrics 테이블을
    # 맞혀 record_node가 실제로 기록을 저장하는 경로까지 탄다(회귀 재현에 필수).
    run_evaluation(db_path=seeded_db, offline=True, limit=1)

    after = _wiki_rows(seeded_db)
    assert after == before, (
        "run_evaluation()이 원본 DB의 wiki 테이블을 건드렸다"
        "(삭제 또는 평가용 합성 기록 삽입) — 반드시 격리된 사본에서 실행돼야 한다."
    )


def test_run_evaluation_excludes_gold_sql_errors_from_ex_denominator(seeded_db, monkeypatch):
    """골드 SQL 자체가 깨진 문항은 EX 분모에서 빠지고 gold_errors로 집계돼야 한다."""
    from src.eval import runner

    fake_goldset = [
        {"id": 1, "question": "존재하지 않는 컬럼 질의", "sql": "SELECT no_such_col FROM company", "tags": ""},
    ]
    monkeypatch.setattr(runner, "GOLDSET", fake_goldset)

    rep = runner.run_evaluation(db_path=seeded_db, offline=True, limit=1)

    assert rep["execution_accuracy"]["applicable"] == 0
    assert rep["execution_accuracy"]["gold_errors"] == 1
