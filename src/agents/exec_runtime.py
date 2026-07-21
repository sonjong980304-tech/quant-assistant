"""데이터 에이전트 실행기 — LLM이 직접 작성한 SQL/Python 코드를 실행한다.

pipeline_exec.py(고정 dict 연산 화이트리스트 + eval/exec 금지)와의 관계:
    이 파일은 그 실행기와 **완전히 별개**다. 계층형 멀티에이전트 재설계에서 LLM이 고정
    블록을 조립하는 대신 코드를 직접 쓰도록 하기로 사용자가 결정했고, 그에 따라 고정
    화이트리스트 없이 임의 코드를 실행한다. **이는 사용자가 명시적으로 승인한 트레이드오프**다.
    pipeline_exec.py는 import하지도, 수정하지도 않는다(결합 회피 — quant_trader 안전원칙 유지).

안전 장치:
1. SQL 경로는 **읽기전용 연결(connect_readonly, src/db.py)만** 받는다. LLM이 생성한
   신뢰불가 SQL을 읽기전용 연결에서만 실행하면, 필터를 우회당해도 sqlite 엔진이 모든
   쓰기/DDL을 거부한다("attempt to write a readonly database"). src/pipeline.py의
   defense-in-depth 패턴(self.roconn)을 그대로 따른다. timeout 초과 시
   threading.Timer로 conn.interrupt()를 걸어 진행 중인 쿼리를 실제로 중단시킨다 —
   그냥 future.result(timeout=...)만 쓰면 워커 스레드 자체는 계속 실행되며 방치되고,
   web/app.py의 finally: conn.close()와 겹치면 아직 그 연결을 쓰는 백그라운드 스레드를
   메인이 닫아버리는 레이스가 날 수 있었다(실측: check_same_thread=False 연결에서
   다른 스레드의 conn.interrupt()가 진행 중인 execute()를 즉시 중단시키고, 그 뒤에도
   연결이 정상 재사용 가능함을 확인).
2. Python 경로는 **별도 프로세스**(multiprocessing, spawn)에서 실행한다. 이
   컴퓨터엔 실거래봇이 상주하고 웹서버가 외부에 노출돼 있어, 신뢰불가 코드를 메인
   프로세스 스레드에서 그대로 돌리던 이전 방식은 문제가 그 프로세스 전체로 번질 위험이
   있었다. 별도 프로세스로 격리하면 (a) 문제가 생겨도 그 자식 프로세스 안에서만 터지고
   (b) 타임아웃 시 process.kill()로 실제 강제 종료할 수 있다(스레드와 달리 방치되지
   않음). 자식 프로세스엔 resource.setrlimit으로 CPU/메모리 상한도 건다 — 단, 실측
   확인 결과(2026-07-15, macOS/Darwin) RLIMIT_AS(메모리)는 이 플랫폼 커널이 지원하지
   않아 설정 자체가 실패한다("current limit exceeds maximum limit"). 리눅스에서는
   두 상한이 모두 정상 동작할 것으로 예상되므로 설정은 시도하되, 실패하면 조용히
   무시한다(타임아웃에 의한 강제 종료가 이 플랫폼의 실질적 안전장치가 된다).
"""
from __future__ import annotations

import multiprocessing
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Callable

# 타임아웃 상한(초). pipeline_exec.MAX_TIMEOUT과 같은 철학이지만 결합을 피하려고
# 이 파일에 독립적으로 정의한다(그 상수를 import하지 않는다).
MAX_TIMEOUT: float = 120.0

# execute_python 자식 프로세스의 기본 CPU 시간 상한(초). RLIMIT_CPU는 실측으로 신뢰성
# 있게 동작함을 확인했다(SIGXCPU로 강제 종료).
DEFAULT_CPU_SECONDS: int = 30

# execute_python 자식 프로세스의 기본 가상메모리 상한(바이트). 리눅스에서는 유효하나
# macOS에서는 설정 자체가 실패할 수 있다(위 모듈 docstring 참고) — 그 경우 조용히 무시.
DEFAULT_MEMORY_BYTES: int = 1024 * 1024 * 1024  # 1GB

# execute_python이 parent_conn.recv() 이후(정상 수신 또는 EOFError) 자식 프로세스를
# 회수(reap)하기 위해 기다리는 짧은 상한(초). 이 시점엔 자식이 이미 결과를 보냈거나
# 메시지 없이 죽은 뒤라 사실상 즉시 종료돼야 한다 — 여기서 원래의 timeout을 그대로
# 다시 쓰면 poll(timeout)+join(timeout)으로 "최대 timeout초" 계약이 병리적인 경우
# (예: 논데몬 스레드가 남아 프로세스 종료가 지연되는 경우) 최대 2배까지 늘어날 수 있다.
_REAP_TIMEOUT: float = 5.0


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


def _interrupt_connection(conn: sqlite3.Connection) -> None:
    """timeout 도달 시 진행 중인 쿼리를 실제로 중단시킨다(워커 스레드 정리 목적).

    fake 커넥션(테스트용) 등 interrupt()가 없는 객체가 들어와도 조용히 무시한다 —
    이 타이머의 실패가 execute_sql의 반환 흐름(정상/타임아웃 처리)을 막으면 안 된다.
    """
    try:
        conn.interrupt()
    except Exception:  # noqa: BLE001
        pass


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
        timeout: 실행 상한(초). 초과 시 ok=False, error="timeout"(또는 인터럽트로 인한
            sqlite3 에러) — 어느 쪽이든 진행 중이던 쿼리는 conn.interrupt()로 실제
            중단되며 워커 스레드가 방치되지 않는다.
        max_rows: 가져올 최대 행 수.

    Returns:
        {"ok": bool, "columns": list[str], "rows": list[dict], "row_count": int,
         "error": str | None}
    """
    def _work() -> dict:
        # 실행 전 스키마 유효성(존재하는 테이블/컬럼) 사전검증(SoT [SQL] 원칙).
        # SQL을 직접 파싱하지 않고 SQLite 엔진에 위임한다: EXPLAIN은 실행 계획(opcode)만
        # 컴파일하고 실제로 행을 읽지 않으므로, 존재하지 않는 테이블/컬럼이면 무거운 스캔이
        # 시작되기 전에 이 컴파일 단계에서 곧바로 OperationalError가 난다.
        try:
            conn.execute(f"EXPLAIN {sql}")
        except sqlite3.Error as e:
            return {
                "ok": False,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": f"스키마 검증 실패: {type(e).__name__}: {e}",
            }

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

    # timeout 시점에 진행 중인 쿼리를 실제로 중단시켜 워커 스레드가 방치되지 않게 한다
    # (future.result(timeout=...)만으로는 스레드 자체는 계속 실행됨 — 그 상태로 호출부가
    # conn.close()를 하면 아직 그 연결을 쓰는 스레드와 레이스가 날 수 있었다). 정상
    # 완료 시에는 반드시 취소해야 다음 쿼리에 뒤늦게 인터럽트가 걸리는 걸 막는다.
    timer = threading.Timer(timeout, _interrupt_connection, args=(conn,))
    timer.start()
    try:
        return _run_with_timeout(_work, timeout)
    except TimeoutError:
        return {"ok": False, "columns": [], "rows": [], "row_count": 0, "error": "timeout"}
    except sqlite3.Error as e:
        # 읽기전용 연결에 대한 쓰기 시도("attempt to write a readonly database"),
        # 또는 위 타이머의 conn.interrupt()로 인한 중단("interrupted") 등을 여기서 잡는다.
        return {
            "ok": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": f"{type(e).__name__}: {e}",
        }
    finally:
        timer.cancel()


def _kill_if_alive(process: multiprocessing.process.BaseProcess) -> None:
    """자식이 아직 살아있으면 강제 종료하고 회수(reap)한다(execute_python의 3개 종료 경로 공유)."""
    if process.is_alive():
        process.kill()  # 스레드와 달리 진짜로 강제 종료된다(방치되지 않음)
        process.join()


def _execute_python_child(
    code: str, context: dict, result_var: str, conn, cpu_seconds: int, memory_bytes: int,
    extra_vars: list[str] | None = None,
) -> None:
    """execute_python의 자식 프로세스 진입점(모듈 top-level 함수여야 spawn으로 pickle 가능).

    conn(multiprocessing.Pipe 자식 끝)으로 3-tuple을 보낸다: 성공 시 ("ok", value, extra),
    실패 시 ("error", message, {}). extra는 extra_vars로 지정한 보조 변수({이름: 값})로,
    코드가 그 변수를 설정하지 않았으면 None이다(예: chart_base64를 안 만든 경우).
    extra_vars 미지정(기본 None)이면 extra는 빈 dict — 기존 동작과 동일하다.
    """
    try:
        import resource

        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except (ValueError, OSError):
            pass  # 플랫폼이 CPU 상한을 지원하지 않으면 조용히 무시(타임아웃이 백스톱)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except (ValueError, OSError):
            # macOS 등 일부 플랫폼은 RLIMIT_AS 자체를 지원하지 않는다(실측 확인) — 무시.
            pass
    except ImportError:
        pass  # resource 모듈은 POSIX 전용(Windows엔 없음) — 없어도 타임아웃이 백스톱

    if "matplotlib" in code:
        # LLM이 자유롭게 쓴 차트 코드(build_chart_freeform)는 charting.py의 기존 4개
        # 헬퍼와 달리 matplotlib.use("Agg")를 스스로 호출한다는 보장이 없다 — 실측
        # 확인(macOS): backend 미설정 시 기본값이 대화형 백엔드(macosx)로 잡혀 실제 GUI
        # 창이 뜨고 실행이 막힌다. 프롬프트 안내에만 기대지 않고(LLM이 빠뜨릴 수 있음)
        # 여기서 exec() 전에 강제한다 — pyplot이 아직 import되기 전이므로 이 설정이
        # 이후 코드의 `import matplotlib.pyplot`에도 그대로 적용된다. 한글 폰트도
        # charting.py와 동일한 안전장치(_configure_korean_font, 못 찾으면 조용히 폴백)를
        # 재사용해 "Glyph ... missing" 경고로 한글이 깨지는 문제도 함께 막는다.
        try:
            import matplotlib

            matplotlib.use("Agg")
            from src.agents.charting import _configure_korean_font

            _configure_korean_font(matplotlib)
        except Exception:  # noqa: BLE001 — 안전장치 설정 실패로 전체 실행을 막지 않는다
            pass

    namespace: dict = dict(context)
    try:
        exec(code, namespace)  # noqa: S102 — 신뢰불가 코드의 임의 실행(사용자 승인 트레이드오프)
        extra = {name: namespace.get(name) for name in (extra_vars or [])}
        conn.send(("ok", namespace.get(result_var), extra))
    except Exception as e:  # noqa: BLE001 — 신뢰불가 코드가 어떤 예외든 던질 수 있어 광범위 포착
        conn.send(("error", f"{type(e).__name__}: {e}", {}))
    finally:
        conn.close()


def execute_python(
    code: str,
    context: dict | None = None,
    timeout: float = MAX_TIMEOUT,
    result_var: str = "result",
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
    memory_bytes: int = DEFAULT_MEMORY_BYTES,
    extra_vars: list[str] | None = None,
) -> dict:
    """LLM이 생성한 Python 코드를 별도 프로세스에서 실행한다(고정 화이트리스트 없음).

    코드는 자식 프로세스에서 exec()로 그대로 실행된다 — **신뢰불가 코드의 임의 실행이며
    사용자가 승인한 트레이드오프**다. 다만 메인 프로세스(실거래봇과 같은 머신에서 도는
    웹서버)와는 프로세스 경계로 격리된다: 문제가 생겨도 자식 프로세스 안에서만 터지고,
    타임아웃 시 강제 종료(kill)할 수 있다.

    Args:
        code: 실행할 Python 소스(신뢰불가).
        context: exec 네임스페이스에 미리 넣어줄 변수들(예: 앞 단계 SQL 결과 rows).
            **직렬화 가능한 값만 담을 수 있다**(별도 프로세스로 전달되므로) — 함수/람다 등
            pickle 불가능한 객체는 담을 수 없다.
        timeout: 실행 상한(초). 초과 시 ok=False, error="timeout"(프로세스 강제 종료).
        result_var: 코드가 결과를 담아야 하는 변수명.
        cpu_seconds: 자식 프로세스 CPU 시간 상한(RLIMIT_CPU, 실측상 신뢰성 있게 동작).
        memory_bytes: 자식 프로세스 가상메모리 상한(RLIMIT_AS) — 플랫폼이 지원하면 적용,
            아니면(macOS 등) 조용히 무시된다.
        extra_vars: result_var 외에 추가로 회수할 보조 변수명 리스트(예: ["chart_base64",
            "chart_title"]). 코드가 설정하지 않은 변수는 반환 extra에서 None이다. 미지정
            (기본 None)이면 extra는 빈 dict이며 기존 동작과 완전히 동일하다.

    Returns:
        {"ok": bool, "result": Any, "error": str | None, "extra": dict}
        · extra: extra_vars로 요청한 보조 변수({이름: 값}, 코드가 안 채웠으면 None). 성공
          시에만 값이 담기고, 실패/타임아웃/비정상종료 시에는 빈 dict다.
    """
    ctx = dict(context or {})
    mp_ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
    try:
        process = mp_ctx.Process(
            target=_execute_python_child,
            args=(code, ctx, result_var, child_conn, cpu_seconds, memory_bytes, extra_vars),
        )
        process.start()
    except Exception as e:  # noqa: BLE001 — context 직렬화 실패 등 프로세스 기동 자체의 실패
        return {"ok": False, "result": None,
                "error": f"실행 프로세스 기동 실패: {type(e).__name__}: {e}", "extra": {}}
    finally:
        child_conn.close()  # 부모 쪽 자식-끝 핸들은 즉시 닫는다(자식이 자기 것을 따로 들고 있음)

    # join()을 먼저 하지 않는다: OS 파이프 버퍼(보통 64KB 안팎)보다 큰 결과(예: 차트
    # base64 PNG)를 자식이 conn.send()로 보내면, 버퍼가 가득 찬 순간부터 부모가 읽어
    # 비워주기 전까지 send()가 블로킹된다. 그런데 join()이 먼저면 부모는 "자식이 끝나기"만
    # 기다리고, 자식은 "부모가 읽어주기"만 기다려 서로 영원히 대기하는 데드락이 된다
    # (실측: dpi=150·10개 막대·값 라벨 있는 실제 차트에서 100% 재현, base64 111KB).
    # 그래서 poll(timeout)으로 먼저 응답을 기다렸다가 recv()로 파이프를 비운 뒤에
    # join()한다 — 이러면 자식의 send()가 즉시 풀려 정상 종료하고 join은 사실상 즉시 끝난다.
    if not parent_conn.poll(timeout):
        parent_conn.close()
        _kill_if_alive(process)
        return {"ok": False, "result": None, "error": "timeout", "extra": {}}

    try:
        # CPU/메모리 상한 초과로 시그널에 의해 즉사하면(SIGXCPU 등) 메시지를 보낼 기회 없이
        # 파이프만 닫히므로 EOFError를 "메시지 없이 종료"로 처리한다.
        status, value, extra = parent_conn.recv()
    except EOFError:
        # 여기서도 timeout을 그대로 다시 쓰면(join(timeout)) "최대 timeout초" 계약이
        # 최악의 경우(예: 논데몬 스레드가 남아 프로세스 종료가 지연되는 경우) 최대 2배까지
        # 늘어난다(poll(timeout) + join(timeout)) — architect 리뷰로 실측 확인된 지점.
        # recv 이후엔 자식이 이미 죽었거나 죽는 중이므로 짧은 상한이면 충분하다.
        process.join(_REAP_TIMEOUT)
        _kill_if_alive(process)
        return {
            "ok": False, "result": None,
            "error": f"실행 프로세스가 비정상 종료됨(exitcode={process.exitcode})", "extra": {},
        }
    finally:
        parent_conn.close()

    # 자식은 send() 직후 conn.close()만 하고 반환하므로 이 시점엔 사실상 이미 끝났거나
    # 곧 끝난다 — join은 그 정리를 기다리는 짧은 대기일 뿐이다(같은 이유로 timeout이 아닌
    # _REAP_TIMEOUT을 쓴다 — 위 EOFError 분기 주석 참고).
    process.join(_REAP_TIMEOUT)
    _kill_if_alive(process)

    if status == "ok":
        return {"ok": True, "result": value, "error": None, "extra": extra}
    return {"ok": False, "result": None, "error": value, "extra": {}}
