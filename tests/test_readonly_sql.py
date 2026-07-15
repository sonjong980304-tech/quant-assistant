"""읽기전용 연결 — LLM이 생성한 신뢰불가 SQL의 쓰기를 엔진 레벨에서 차단.

정규식 필터(is_safe_select)는 1차 방어. connect_readonly(mode=ro)는 그걸 우회해도
DB 엔진이 모든 쓰기를 거부하는 최종 방어층(defense-in-depth)이다.
"""
from __future__ import annotations

import sqlite3

import pytest

from src.db import connect, connect_readonly, init_db


def _seed(tmp_path) -> str:
    db = tmp_path / "ro.db"
    init_db(str(db))
    c = connect(str(db))
    c.execute("INSERT INTO company(stock_code, name) VALUES('000001','테스트')")
    c.commit()
    c.close()
    return str(db)


def test_readonly_allows_select(tmp_path):
    """읽기전용 연결도 SELECT는 정상 동작."""
    c = connect_readonly(_seed(tmp_path))
    try:
        assert c.execute("SELECT COUNT(*) FROM company").fetchone()[0] == 1
    finally:
        c.close()


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO company(stock_code, name) VALUES('X','x')",
        "UPDATE company SET name='해킹'",
        "DELETE FROM company",
        "DROP TABLE company",
        "CREATE TABLE evil(x)",
    ],
)
def test_readonly_rejects_all_writes(tmp_path, sql):
    """어떤 쓰기/DDL도 엔진이 거부한다(readonly database)."""
    c = connect_readonly(_seed(tmp_path))
    try:
        with pytest.raises(sqlite3.Error):
            c.execute(sql)
    finally:
        c.close()


def test_readonly_cannot_be_toggled_off(tmp_path):
    """mode=ro는 PRAGMA로 되돌릴 수 없다(핸들 레벨) — query_only OFF 시도 후에도 쓰기 거부."""
    c = connect_readonly(_seed(tmp_path))
    try:
        try:
            c.execute("PRAGMA query_only = OFF")
        except sqlite3.Error:
            pass
        with pytest.raises(sqlite3.Error):
            c.execute("UPDATE company SET name='해킹'")
    finally:
        c.close()


# ── 사이클 5: Pipeline 배선 — 질의 경로는 읽기전용, 앱 쓰기는 쓰기 가능 ──
def test_pipeline_query_conn_is_readonly(seeded_db):
    """LLM SQL이 실행되는 deps.conn 은 읽기전용(쓰기 거부), 읽기는 정상."""
    from src.legacy.pipeline import Pipeline

    p = Pipeline(db_path=seeded_db, offline=True)
    try:
        with pytest.raises(sqlite3.Error):
            p.deps.conn.execute("UPDATE company SET name='해킹'")
        assert p.deps.conn.execute("SELECT COUNT(*) FROM company").fetchone()[0] >= 1
    finally:
        p.close()


def test_pipeline_wiki_conn_still_writable(seeded_db):
    """WikiStore용 연결(self.conn)은 정상 쓰기 가능해야 한다(성공 질의 기록 저장용)."""
    from src.legacy.pipeline import Pipeline

    p = Pipeline(db_path=seeded_db, offline=True)
    try:
        p.conn.execute("UPDATE company SET name=name WHERE 1=0")  # no-op write → 성공해야 함
        p.conn.commit()
    finally:
        p.close()
