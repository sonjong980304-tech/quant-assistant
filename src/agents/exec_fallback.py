"""자유 실행 폴백 — 정형 도메인 경로가 3회 검증에 실패했을 때 쓰는 마지막 안전망.

answer_with_verification(supervisor.py)의 재시도 루프(정확히 max_retries회, 무한루프 없음)와는
별개다. 고정 스크리닝 기준(criteria key/direction/top_n)이나 백테스트 파이프라인 연산
(pipeline_exec.PRIMITIVE_OPS)의 표현력 한계로 반복 실패하는 질문(예: "시장별로 나눠 각각
상위 10개") — 즉 검증 로직이 아니라 **정형 어휘 자체가 질문을 표현 못 하는 경우** — 을 위해
마지막에만 쓴다. LLM이 (1) 원본 데이터를 폭넓게 가져오는 SQL, (2) 그 데이터를 원하는 모양으로
가공하는 Python 코드를 직접 작성하고, 각각 exec_runtime.py(HA-1 실행기)의 이미 검증된
안전장치로 실행한다 — SQL은 읽기전용 연결에서만, Python은 별도 프로세스에서 격리 실행(타임아웃
시 강제종료). 새 안전장치를 만들지 않는다: 이 폴백의 안전성은 exec_runtime.py에 전적으로
의존한다.

정형 검증(verify_answer)을 다시 거치지 않는다 — 폴백 자체가 "정형 검증이 반복 실패한 뒤의
최후 수단"이므로 같은 검증을 다시 적용하면 같은 이유로 다시 실패해 무한 루프에 가까워진다.
대신 SQL/Python 실행이 실제로 성공했고 결과가 비어있지 않은지만 확인한다(최소한의 성공 판정).
"""
from __future__ import annotations

import re
from typing import Callable

from src.agents.exec_runtime import execute_python, execute_sql
from src.db import schema_catalog
from src.llm import extract_sql

_PY_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _is_meaningfully_empty(result) -> bool:
    """result가 "껍데기는 있지만 속은 빈" 값인지 재귀적으로 판단한다.

    최상위 None/[]/{}뿐 아니라, {"코스피": [], "코스닥": []}처럼 컨테이너 안이 전부
    비었을 때도 실질적으로 빈 결과다(실사용 재현: 계정명/시장명 필터가 실제 데이터와
    안 맞아 그룹은 있는데 원소가 하나도 없던 경우). 0/False/""처럼 falsy하지만 실제
    답일 수 있는 스칼라 값은 비었다고 보지 않는다 — 컨테이너(dict/list)만 재귀한다.
    """
    if result in (None, [], {}):
        return True
    if isinstance(result, dict):
        return all(_is_meaningfully_empty(v) for v in result.values())
    if isinstance(result, list):
        return all(_is_meaningfully_empty(v) for v in result)
    return False


def _extract_python_code(text: str) -> str:
    """LLM 응답에서 Python 코드만 추출(코드펜스 제거).

    extract_sql(src/llm.py)과 달리 세미콜론 기준으로 자르지 않는다 — Python 문장은
    세미콜론으로 끝나지 않으므로 그 로직을 재사용하면 코드가 잘린다.
    """
    text = (text or "").strip()
    m = _PY_CODE_FENCE.search(text)
    if m:
        return m.group(1).strip()
    return text


def _sql_prompt(question: str, last_reason: str | None) -> str:
    reason_note = (
        f"\n\n정형 방식이 다음 이유로 반복 실패했습니다: {last_reason}" if last_reason else ""
    )
    return (
        "아래 스키마를 참고해, 이 질문에 답하는 데 필요한 원본 데이터를 폭넓게 가져오는 "
        "읽기전용 SELECT 쿼리 하나만 작성하세요. top_n으로 개수를 미리 제한하거나 그룹별로 "
        "분리하는 등의 최종 가공은 이 쿼리에서 하지 마세요 — 그건 다음 단계 Python 코드가 "
        "처리합니다. 필요 이상으로 좁게 필터링하지 말고, 이후 단계에서 쓸 수 있도록 관련 "
        "컬럼을 넉넉히 포함하세요.\n"
        "SQL은 단순하게 유지하세요 — 상관 서브쿼리(correlated subquery, 바깥 쿼리의 각 행마다 "
        "반복 실행되는 하위 쿼리, 예: 조인 조건 안에 'WHERE x = (SELECT ... WHERE y = 바깥행)' "
        "형태)는 prices처럼 큰 테이블에서 타임아웃을 유발하니 쓰지 마세요. 종목별 '최신 시점' "
        "같은 매칭이 필요하면 그 매칭 자체를 SQL로 정교하게 계산하려 하지 말고, 관련 원본 행을 "
        "그대로 가져와 다음 단계 Python(pandas)이 groupby/최댓값 등으로 처리하게 하세요.\n"
        "\n[metrics 테이블 활용 안내] 스키마에 포함된 metrics 테이블에는 PBR·GPA·ROE 등 "
        "파생지표가 이미 계산돼 있습니다. 단, 이 테이블은 '오늘 기준 최신 분기 1개'만 담은 "
        "스냅샷 캐시이며 과거 여러 시점의 이력이 아닙니다. 그래서 사용 규칙이 나뉩니다:\n"
        "  - '지금 기준' 스크리닝/비교/차트/분석(예: PBR·GPA 등 파생지표 조회, 분위수 계산)에는 "
        "financials/prices에서 직접 재계산하지 말고 metrics를 우선 사용하세요 — 더 정확하고 "
        "빠르며, 조인 실패로 NaN이 생겨 분위수 계산이 깨지는 문제를 피할 수 있습니다.\n"
        "  - 반대로 '여러 과거 시점에 걸친 백테스트/시뮬레이션'류 질문에는 metrics를 절대 쓰지 "
        "마세요 — 모든 시점에 오늘 값이 잘못 적용됩니다. 그런 경우엔 financials/prices에서 "
        "시점별로 직접 계산해야 합니다.\n"
        f"\n{schema_catalog(include_metrics=True)}"
        f"{reason_note}\n\n질문: {question}\nSQL:"
    )


def _infer_latest_date(rows: list[dict], columns: list[str]) -> str | None:
    """rows에 날짜형 컬럼(이름에 'date' 포함)이 있으면 그중 가장 최근 값을 찾는다.

    "최근 N개월/N년" 같은 상대적 기간 표현의 기준일(anchor)로 쓴다 — 없으면 LLM이 캘린더
    오늘을 알 방법이 없어 rows를 오름차순으로 훑을 때 맨 앞(가장 오래된 시점)을 "최근"으로
    착각해 잘못된 구간을 계산하는 사례가 실사용에서 확인됐다(예: 8년 전 데이터를 "최근
    12개월"이라며 답함). domain_kr.py의 _default_screening_asof와 동일한 원칙(캘린더가
    아니라 데이터에 실재하는 최신 시점을 쓴다)을 이 자유 코드 경로에도 적용한다.
    """
    date_cols = [c for c in columns if "date" in c.lower()]
    if not date_cols:
        return None
    values = [r.get(date_cols[0]) for r in rows if r.get(date_cols[0])]
    return max(values) if values else None


def _python_prompt(
    question: str,
    columns: list[str],
    row_count: int,
    last_reason: str | None,
    code_error: str | None = None,
    latest_date: str | None = None,
) -> str:
    reason_note = (
        f"\n\n정형 방식이 다음 이유로 반복 실패했습니다: {last_reason}" if last_reason else ""
    )
    retry_note = (
        f"\n\n[직전 코드 실행 실패] 방금 작성한 코드가 다음 이유로 실패했습니다: {code_error}\n"
        "같은 실수를 반복하지 말고 원인을 고쳐 다시 작성하세요(예: rows에 실제로 존재하는 "
        "컬럼명만 참조하고, 아직 계산 안 된 값을 이름만으로 미리 참조하지 마세요)."
        if code_error else ""
    )
    date_note = (
        f"\nrows에서 확인되는 가장 최근 시점(기준일)은 {latest_date}입니다. '최근 N개월/N년/"
        "직전 12개월' 같은 상대적 기간 표현은 반드시 이 기준일에서 과거 방향으로 계산하세요 "
        "— rows를 정렬했을 때 맨 앞(가장 오래된 시점)부터 N개를 세면 안 됩니다."
        if latest_date else ""
    )
    return (
        "방금 실행한 SQL로 가져온 원본 데이터가 `rows`라는 이름의 dict 리스트로 이미 "
        f"주어집니다(실제 컬럼 목록: {columns}, {row_count}행). 이 데이터를 가공해 질문에 "
        "정확히 답하는 Python 코드를 작성하세요. pandas 등 필요한 라이브러리는 코드 안에서 "
        "직접 import 하세요(별도 프로세스에서 그대로 실행되므로 미리 주입된 것 없이 자유롭게 "
        "써도 됩니다). 최종 답은 반드시 `result` 라는 변수에 담으세요(리스트/딕트 등 JSON "
        "직렬화 가능한 값).\n"
        "질문에 나온 지표 이름(예: '이익수익률', '투하자본수익률', PER/PBR/ROE 등 비율·수익률"
        "스러운 이름)이 위 실제 컬럼 목록에 그대로 없다면, 그건 원본에 저장돼 있지 않은 "
        "파생(계산) 값입니다 — account_name/account_key 같은 컬럼 안에서 그 이름을 문자열로 "
        "찾으려 하지 마세요(존재하지 않습니다). 대신 원본의 raw 항목들(예: 영업이익/매출/자산/"
        f"부채/시가총액 등)을 pandas로 조합해 표준적인 계산식으로 직접 산출하세요.{date_note}\n"
        f"{reason_note}{retry_note}\n\n질문: {question}\nPython 코드:"
    )


def run_free_exec_fallback(
    question: str,
    conn,
    llm_fn: Callable[[str], str] | None,
    last_reason: str | None = None,
    execute_sql_fn: Callable | None = None,
    execute_python_fn: Callable | None = None,
    max_code_attempts: int = 2,
) -> dict:
    """정형 3회 실패 후 폴백 — LLM이 SQL+Python을 직접 작성해 최종 답을 만든다.

    Args:
        question: 원본 사용자 질문(재시도 피드백이 섞이지 않은 순수 원본).
        conn: 읽기전용 DB 연결(connect_readonly) — execute_sql_fn에 그대로 전달된다.
        llm_fn: Callable[[str], str]. None이면 코드를 생성할 방법이 없으므로 즉시 실패.
        last_reason: 직전 정형 검증 실패 사유(있으면 프롬프트에 포함 — 왜 정형 방식이
            안 통했는지 알려줘야 LLM이 같은 실수를 반복하지 않는다).
        execute_sql_fn: 기본 exec_runtime.execute_sql(테스트 주입용).
        execute_python_fn: 기본 exec_runtime.execute_python(테스트 주입용).
        max_code_attempts: Python 코드 생성·실행을 정확히 이 횟수까지만 시도한다(무한루프
            없음). SQL은 이미 성공한 뒤라 재조회하지 않고 매 시도마다 재사용한다 — 실패
            사유(KeyError 등)를 다음 시도 프롬프트에 피드백해 자가 수정 기회를 준다
            (answer_with_verification의 "직전 실패 피드백" 재시도와 동일한 철학).

    Returns:
        {"ok": bool, "result": Any, "sql": str|None, "code": str|None, "error": str|None}
    """
    if llm_fn is None:
        return {
            "ok": False, "result": None, "sql": None, "code": None,
            "error": "llm_fn 없음 — 자유 코드 생성 폴백을 쓸 수 없습니다.",
        }

    execute_sql_fn = execute_sql_fn or execute_sql
    execute_python_fn = execute_python_fn or execute_python

    sql_raw = llm_fn(_sql_prompt(question, last_reason)) or ""
    sql = extract_sql(sql_raw)
    if not sql:
        return {
            "ok": False, "result": None, "sql": None, "code": None,
            "error": "LLM이 SQL을 생성하지 못했습니다.",
        }

    sql_result = execute_sql_fn(sql, conn)
    if not sql_result.get("ok"):
        return {
            "ok": False, "result": None, "sql": sql, "code": None,
            "error": f"SQL 실행 실패: {sql_result.get('error')}",
        }

    rows = sql_result.get("rows", [])
    columns = sql_result.get("columns", [])
    latest_date = _infer_latest_date(rows, columns)
    code: str | None = None
    code_error: str | None = None
    for _attempt in range(max_code_attempts):
        code_raw = llm_fn(
            _python_prompt(question, columns, len(rows), last_reason, code_error, latest_date)
        ) or ""
        code = _extract_python_code(code_raw)
        if not code:
            code_error = "LLM이 Python 코드를 생성하지 못했습니다."
            continue

        py_result = execute_python_fn(code, context={"rows": rows}, result_var="result")
        if not py_result.get("ok"):
            code_error = f"Python 실행 실패: {py_result.get('error')}"
            continue

        result = py_result.get("result")
        if _is_meaningfully_empty(result):
            code_error = (
                "생성된 코드가 빈 결과를 반환했습니다(그룹은 있는데 원소가 하나도 없다면 "
                "계정명/시장명 등 필터 조건이 rows의 실제 값과 정확히 일치하는지 확인하세요)."
            )
            continue

        return {"ok": True, "result": result, "sql": sql, "code": code, "error": None}

    return {"ok": False, "result": None, "sql": sql, "code": code, "error": code_error}
