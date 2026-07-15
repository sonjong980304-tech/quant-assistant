"""데이터 에이전트 실행기 — LLM이 직접 작성한 SQL/Python 코드를 실행한다.

pipeline_exec.py(고정 dict 연산 화이트리스트 + eval/exec 금지)와의 관계:
    이 파일은 그 실행기와 **완전히 별개**다. 계층형 멀티에이전트 재설계에서 LLM이 고정
    블록을 조립하는 대신 코드를 직접 쓰도록 하기로 사용자가 결정했고, 그에 따라 고정
    화이트리스트 없이 임의 코드를 실행한다. **이는 사용자가 명시적으로 승인한 트레이드오프**다.
    pipeline_exec.py는 import하지도, 수정하지도 않는다(결합 회피 — quant_trader 안전원칙 유지).

두 개의 안전 장치만 남긴다(고정 화이트리스트 대신):
1. SQL 경로는 **읽기전용 연결(connect_readonly, src/db.py)만** 받는다. LLM이 생성한
   신뢰불가 SQL을 읽기전용 연결에서만 실행하면, 필터를 우회당해도 sqlite 엔진이 모든
   쓰기/DDL을 거부한다("attempt to write a readonly database"). src/pipeline.py의
   defense-in-depth 패턴(self.roconn)을 그대로 따른다.
2. 두 경로 모두 timeout(기본 120초)을 넘기면 명시적으로 실패 처리한다. pipeline_exec가
   쓰는 ThreadPoolExecutor + FuturesTimeout 백스톱 패턴을 (import 없이) 새로 구현한다.
   파이썬 스레드는 강제 종료할 수 없으므로 초과한 작업은 백그라운드에 남아 자연히 끝난다
   — 호출자에게는 즉시 실패를 반환한다.
"""
from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Callable

# 타임아웃 상한(초). pipeline_exec.MAX_TIMEOUT과 같은 철학이지만 결합을 피하려고
# 이 파일에 독립적으로 정의한다(그 상수를 import하지 않는다).
MAX_TIMEOUT: float = 120.0


def _run_with_timeout(fn: Callable[[], Any], timeout: float) -> Any:
    """fn()을 워커 스레드에서 실행하고 timeout(초) 내 완료를 요구한다.

    초과 시 TimeoutError를 던진다. (스레드는 강제 종료할 수 없어 초과한 작업은
    백그라운드에 남지만, 호출자에게는 상한 내에 제어를 돌려준다 — 백스톱.)
    fn 내부에서 발생한 예외는 future.result()가 호출자 스레드로 그대로 다시 던진다.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout as exc:
            raise TimeoutError(f"실행 타임아웃({timeout}s 초과)") from exc


def execute_sql(
    sql: str,
    conn: sqlite3.Connection,
    timeout: float = MAX_TIMEOUT,
    max_rows: int = 1000,
) -> dict:
    """LLM이 생성한 SQL을 실행하고 결과를 반환한다.

    Args:
        sql: 실행할 SQL(신뢰불가). 안전은 conn의 읽기전용 여부에 의존한다.
        conn: **읽기전용 연결(connect_readonly)** 을 넘겨야 한다. 쓰기 SQL이 들어와도
            엔진이 거부하며(ok=False, error에 'readonly ...'), 그 에러가 결과로 잡힌다.
        timeout: 실행 상한(초). 초과 시 ok=False, error="timeout".
        max_rows: 가져올 최대 행 수.

    Returns:
        {"ok": bool, "columns": list[str], "rows": list[dict], "row_count": int,
         "error": str | None}
    """
    def _work() -> dict:
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        fetched = cur.fetchmany(max_rows)
        rows = [dict(zip(columns, r)) for r in fetched]
        return {
            "ok": True,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "error": None,
        }

    try:
        return _run_with_timeout(_work, timeout)
    except TimeoutError:
        return {"ok": False, "columns": [], "rows": [], "row_count": 0, "error": "timeout"}
    except sqlite3.Error as e:
        # 읽기전용 연결에 대한 쓰기 시도("attempt to write a readonly database") 등을 여기서 잡는다.
        return {
            "ok": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": f"{type(e).__name__}: {e}",
        }


def execute_python(
    code: str,
    context: dict | None = None,
    timeout: float = MAX_TIMEOUT,
    result_var: str = "result",
) -> dict:
    """LLM이 생성한 Python 코드를 실행한다(고정 화이트리스트 없음).

    코드는 exec()로 그대로 실행된다 — **신뢰불가 코드의 임의 실행이며 사용자가 승인한
    트레이드오프**다. 실행 네임스페이스에 context를 주입하고, 코드가 result_var(기본
    "result") 변수에 할당한 값을 결과로 돌려준다.

    Args:
        code: 실행할 Python 소스(신뢰불가).
        context: exec 네임스페이스에 미리 넣어줄 변수들(예: 앞 단계 SQL 결과 rows).
        timeout: 실행 상한(초). 초과 시 ok=False, error="timeout".
        result_var: 코드가 결과를 담아야 하는 변수명.

    Returns:
        {"ok": bool, "result": Any, "error": str | None}
    """
    namespace: dict = dict(context or {})

    def _work() -> Any:
        # 신뢰불가 코드의 임의 실행(사용자 승인 트레이드오프). SQL 경로와 달리 엔진 레벨
        # 방어가 없으므로 위험을 감수한 실행이라는 점을 명시한다.
        exec(code, namespace)
        return namespace.get(result_var)

    try:
        value = _run_with_timeout(_work, timeout)
    except TimeoutError:
        return {"ok": False, "result": None, "error": "timeout"}
    except Exception as e:  # noqa: BLE001 — 신뢰불가 코드가 어떤 예외든 던질 수 있어 광범위 포착
        return {"ok": False, "result": None, "error": f"{type(e).__name__}: {e}"}
    return {"ok": True, "result": value, "error": None}
