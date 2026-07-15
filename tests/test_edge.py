"""안정성 축 — 이상 입력 엣지케이스 테스트 (AC10).

이상 입력(빈 질문/엉뚱한 질문/없는 회사/문법 틀린 SQL)에도 크래시 없이 graceful
하게 처리되는지 검증한다. 오프라인·시드 임시 DB로 격리한다.
"""
from __future__ import annotations

import pytest

from src.db import connect
from src.legacy.pipeline import Pipeline
from src.sql_exec import run_select


@pytest.mark.parametrize(
    "question",
    ["", "   ", "오늘 점심 뭐먹지", "!@#$%^&*()", "asdf qwer zxcv"],
)
def test_pipeline_survives_weird_questions(seeded_db, question):
    """빈/엉뚱한 질문에도 예외 없이 상태 dict를 반환한다(크래시 없음)."""
    p = Pipeline(db_path=seeded_db, offline=True)
    try:
        s = p.run(question)  # 예외가 나면 테스트 실패
    finally:
        p.close()
    assert isinstance(s, dict)
    # graceful: 에러 필드가 채워질 수는 있어도 예외로 죽지 않는다.
    assert "raw_question" in s


def test_missing_company_returns_empty_not_crash(seeded_db):
    """DB에 없는 회사 조회는 빈 결과로 graceful 처리(예외 없음)."""
    conn = connect(seeded_db)
    try:
        r = run_select(conn, "SELECT name FROM company WHERE name = '없는회사엑스1234'")
    finally:
        conn.close()
    assert r["ok"] is True
    assert r["row_count"] == 0
    assert r["rows"] == []
    assert r["error"] is None


def test_malformed_sql_returns_error_not_crash(seeded_db):
    """문법 틀린 SQL은 예외를 던지지 않고 error 필드로 반환된다."""
    conn = connect(seeded_db)
    try:
        r = run_select(conn, "SELECT * FROM company WHERE")  # 불완전한 WHERE → 문법오류
    finally:
        conn.close()
    assert r["ok"] is False
    assert r["error"], "error 메시지가 있어야 함"
    assert r["rows"] == []


def test_write_sql_is_blocked_not_crash(seeded_db):
    """쓰기/DDL SQL은 안전가드로 차단되며 예외 없이 거부된다."""
    conn = connect(seeded_db)
    try:
        r = run_select(conn, "DELETE FROM company")
    finally:
        conn.close()
    assert r["ok"] is False
    assert r["error"]
