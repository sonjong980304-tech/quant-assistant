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

import threading
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


def test_execute_sql_timeout_interrupts_running_query_and_cleans_up_thread(tmp_path):
    """타임아웃 시 conn.interrupt()로 진행 중인 쿼리를 실제로 중단시켜, 워커 스레드가
    방치되지 않고 정리되는지 확인한다.

    실측 확인(check_same_thread=False 연결에서): 다른 스레드의 conn.interrupt() 호출이
    진행 중인 execute()를 sqlite3.OperationalError("interrupted")로 즉시 중단시키고,
    그 이후에도 연결은 정상 재사용 가능하다. web/app.py의 finally: conn.close()가 아직
    실행 중인 백그라운드 스레드와 겹치는 레이스를 막으려면, 타임아웃 시 워커 스레드가
    실제로 빨리 정리돼야 한다(방치되면 안 됨).
    """
    conn = connect_readonly(_seed(tmp_path))
    try:
        slow_sql = (
            "WITH RECURSIVE cnt(x) AS ("
            "SELECT 1 UNION ALL SELECT x+1 FROM cnt WHERE x < 500000000"
            ") SELECT COUNT(*) FROM cnt"
        )
        threads_before = threading.active_count()
        res = execute_sql(slow_sql, conn, timeout=0.2)
        assert res["ok"] is False
        assert res["error"]

        deadline = time.time() + 3.0
        while time.time() < deadline and threading.active_count() > threads_before:
            time.sleep(0.05)
        assert threading.active_count() <= threads_before, "타임아웃 후에도 워커 스레드가 방치됨"

        # interrupt 후에도 연결이 정상 재사용 가능해야 한다(다음 쿼리가 막히지 않음).
        res2 = execute_sql("SELECT stock_code FROM company ORDER BY stock_code LIMIT 1", conn)
        assert res2["ok"] is True
    finally:
        conn.close()


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


def test_execute_python_timeout_via_slow_code():
    """느린 코드(문자열 자체에 sleep) → 타임아웃 초과 시 실패 처리.

    별도 프로세스로 실행하므로 context에 함수 객체(lambda 등)를 담아 넘길 수 없다
    (직렬화 불가) — 느린 동작은 code 문자열 안에서 직접 표현한다.
    """
    res = execute_python("import time; time.sleep(0.5)", timeout=0.05)
    assert res["ok"] is False
    assert res["error"] == "timeout"


def test_execute_python_timeout_actually_kills_child_process(tmp_path):
    """타임아웃 후에도 스레드처럼 백그라운드에 방치되지 않고 프로세스 자체가 죽는지 확인.

    자식이 하트비트 파일에 계속 시각을 기록하게 하고, 타임아웃 반환 후 잠깐 기다렸다가
    하트비트가 더 이상 갱신되지 않으면 프로세스가 실제로 종료된 것으로 판단한다.
    """
    heartbeat = tmp_path / "heartbeat.txt"
    code = (
        "import time\n"
        f"path = {str(heartbeat)!r}\n"
        "while True:\n"
        "    with open(path, 'a') as f:\n"
        "        f.write('x')\n"
        "    time.sleep(0.02)\n"
    )
    res = execute_python(code, timeout=0.2)
    assert res["ok"] is False
    assert res["error"] == "timeout"

    size_at_timeout = heartbeat.stat().st_size if heartbeat.exists() else 0
    time.sleep(0.3)  # 프로세스가 살아있었다면 이 사이 하트비트가 더 늘어났을 것
    size_after_wait = heartbeat.stat().st_size if heartbeat.exists() else 0
    assert size_after_wait == size_at_timeout, "타임아웃 이후에도 자식 프로세스가 계속 실행 중임"


def test_execute_python_runs_in_separate_process_not_main():
    """메인 프로세스가 아니라 별도 프로세스에서 실행되는지 확인(pid 비교)."""
    import os

    res = execute_python("import os; result = os.getpid()")
    assert res["ok"] is True
    assert res["result"] != os.getpid()


def test_execute_python_cpu_limit_kills_busy_loop():
    """CPU 사용시간 상한(RLIMIT_CPU)이 걸려 있어, 오래 도는 순수 연산 루프가
    (타임아웃보다 먼저) 상한 초과로 실패 처리된다.

    실측 확인(macOS): RLIMIT_CPU는 실제로 강제되어 SIGXCPU로 프로세스가 죽는다
    (RLIMIT_AS 메모리 상한과 달리 이 플랫폼에서 신뢰성 있게 동작함).
    """
    from src.agents import exec_runtime

    code = "x = 0\nwhile True:\n    x += 1\n"
    res = execute_python(code, timeout=exec_runtime.MAX_TIMEOUT, cpu_seconds=1)
    assert res["ok"] is False
    assert res["error"] != "timeout"  # CPU 상한이 먼저 걸려야 함(전체 타임아웃 120s보다 훨씬 빨리)


# ── 정적 안전 검사 ─────────────────────────────────────────────────────────
def test_source_has_no_primitive_ops_whitelist():
    """신규 실행기는 고정 dict 화이트리스트 디스패치를 쓰지 않는다."""
    src = Path(__file__).resolve().parent.parent / "src" / "agents" / "exec_runtime.py"
    text = src.read_text(encoding="utf-8")
    assert "PRIMITIVE_OPS" not in text


def test_max_timeout_default_is_120():
    assert MAX_TIMEOUT == 120.0
