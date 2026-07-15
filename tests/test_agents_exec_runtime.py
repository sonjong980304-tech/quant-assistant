"""데이터 에이전트 실행기(exec_runtime) 단위테스트 (TDD).

계층형 멀티에이전트 재설계용 신규 실행기. pipeline_exec의 고정 화이트리스트
(PRIMITIVE_OPS)와 달리 LLM이 직접 작성한 SQL/Python 코드를 그대로 실행한다
(사용자가 명시적으로 승인한 트레이드오프 — quant_trader와 무관한 별도 파일).

검증하는 안전 조건:
- SQL 경로는 읽기전용 연결(connect_readonly)만 받아 실행 → 엔진이 쓰기를 거부한다
  (attempt to write a readonly database). is_safe_select를 우회당해도 데이터 불변.
- 두 경로 모두 timeout(기본 120초) 초과 시 명시적으로 실패 처리한다(fake 지연으로 재현).
- 고정 dict 화이트리스트(PRIMITIVE_OPS) 디스패치가 없다(정적 검사).
"""
from __future__ import annotations

import time
from pathlib import Path

from src.agents.exec_runtime import MAX_TIMEOUT, execute_python, execute_sql
from src.db import connect, connect_readonly, init_db


def _seed(tmp_path) -> str:
    db = tmp_path / "agent.db"
    init_db(str(db))
    c = connect(str(db))
    c.execute("INSERT INTO company(stock_code, name) VALUES('000001','가나전자')")
    c.execute("INSERT INTO company(stock_code, name) VALUES('000002','다라화학')")
    c.commit()
    c.close()
    return str(db)


# ── SQL 실행 경로 ──────────────────────────────────────────────────────────
def test_execute_sql_returns_rows_and_columns(tmp_path):
    conn = connect_readonly(_seed(tmp_path))
    try:
        res = execute_sql("SELECT stock_code, name FROM company ORDER BY stock_code", conn)
    finally:
        conn.close()
    assert res["ok"] is True
    assert res["columns"] == ["stock_code", "name"]
    assert res["row_count"] == 2
    assert res["rows"][0] == {"stock_code": "000001", "name": "가나전자"}
    assert res["error"] is None


def test_execute_sql_uses_readonly_connection_write_is_rejected(tmp_path):
    """읽기전용 연결만 받으므로 write SQL은 엔진이 거부(attempt to write a readonly database)."""
    conn = connect_readonly(_seed(tmp_path))
    try:
        res = execute_sql("INSERT INTO company(stock_code, name) VALUES('X','x')", conn)
    finally:
        conn.close()
    assert res["ok"] is False
    assert "readonly" in res["error"].lower()


def test_execute_sql_error_is_captured_not_raised(tmp_path):
    conn = connect_readonly(_seed(tmp_path))
    try:
        res = execute_sql("SELECT * FROM no_such_table", conn)
    finally:
        conn.close()
    assert res["ok"] is False
    assert res["error"]


class _SlowConn:
    """.execute가 느리게 동작하는 fake 연결 — SQL 경로의 타임아웃을 결정론적으로 재현.

    실제 sqlite 연결/스레드/close 경합 없이 지연만 흉내낸다.
    """

    def execute(self, sql):  # noqa: ARG002 (인터페이스 호환용 인자)
        time.sleep(0.5)
        raise AssertionError("타임아웃 전에 반환되면 안 된다")


def test_execute_sql_timeout_with_fake_slow_connection():
    res = execute_sql("SELECT 1", _SlowConn(), timeout=0.05)
    assert res["ok"] is False
    assert res["error"] == "timeout"


# ── Python 실행 경로 ───────────────────────────────────────────────────────
def test_execute_python_returns_result_variable():
    res = execute_python("result = sum(range(5))")
    assert res["ok"] is True
    assert res["result"] == 10
    assert res["error"] is None


def test_execute_python_receives_context():
    res = execute_python("result = [r * 2 for r in rows]", context={"rows": [1, 2, 3]})
    assert res["ok"] is True
    assert res["result"] == [2, 4, 6]


def test_execute_python_exception_is_captured_not_raised():
    res = execute_python("result = 1 / 0")
    assert res["ok"] is False
    assert "ZeroDivisionError" in res["error"]
    assert res["result"] is None


def test_execute_python_timeout_via_fake_slow_function():
    """time.sleep을 흉내내는 느린 함수 주입 → 타임아웃 초과 시 실패 처리."""
    res = execute_python("slow()", context={"slow": lambda: time.sleep(0.5)}, timeout=0.05)
    assert res["ok"] is False
    assert res["error"] == "timeout"


# ── 정적 안전 검사 ─────────────────────────────────────────────────────────
def test_source_has_no_primitive_ops_whitelist():
    """신규 실행기는 고정 dict 화이트리스트 디스패치를 쓰지 않는다."""
    src = Path(__file__).resolve().parent.parent / "src" / "agents" / "exec_runtime.py"
    text = src.read_text(encoding="utf-8")
    assert "PRIMITIVE_OPS" not in text


def test_max_timeout_default_is_120():
    assert MAX_TIMEOUT == 120.0
