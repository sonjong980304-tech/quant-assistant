"""자유 실행 폴백(exec_fallback) 단위테스트 (TDD).

총괄 에이전트의 정형 재시도(answer_with_verification, 최대 3회)가 모두 실패했을 때 쓰는
마지막 안전망 — LLM이 SQL(1단계)과 Python(2단계)을 직접 작성하고, 각각 이미 검증된
exec_runtime.py 안전장치(읽기전용 SQL 엔진 / 별도 프로세스 격리 Python)로 실행한다.

이 테스트는 exec_runtime.execute_sql/execute_python 을 fake로 주입해 격리한다(실제 프로세스
스폰/DB 연결 없이 오케스트레이션 로직만 검증) — 실제 안전장치 자체는 test_agents_exec_runtime.py
가 이미 검증했다.
"""
from __future__ import annotations

from src.agents.exec_fallback import _infer_latest_date, _python_prompt, _sql_prompt, run_free_exec_fallback


def _fake_llm_sequence(*responses):
    """호출될 때마다 순서대로 다음 응답을 돌려주는 fake llm_fn. 호출 인자도 기록한다."""
    calls: list[str] = []
    it = iter(responses)

    def fn(prompt: str) -> str:
        calls.append(prompt)
        return next(it)

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


def test_run_free_exec_fallback_happy_path_returns_ok_with_result():
    llm_fn = _fake_llm_sequence(
        "```sql\nSELECT stock_code, market, earnings_yield FROM cross_section;\n```",
        "```python\nresult = {'KOSPI': ['005930'], 'KOSDAQ': ['036000']}\n```",
    )
    sql_calls: list[tuple] = []
    py_calls: list[tuple] = []

    def fake_execute_sql(sql, conn):
        sql_calls.append((sql, conn))
        return {
            "ok": True,
            "columns": ["stock_code", "market", "earnings_yield"],
            "rows": [{"stock_code": "005930", "market": "KOSPI", "earnings_yield": 4.09}],
            "row_count": 1,
            "error": None,
        }

    def fake_execute_python(code, context=None, result_var="result"):
        py_calls.append((code, context, result_var))
        return {"ok": True, "result": {"KOSPI": ["005930"], "KOSDAQ": ["036000"]}, "error": None}

    res = run_free_exec_fallback(
        "코스피 코스닥 각각 상위 10개씩",
        conn=object(),
        llm_fn=llm_fn,
        last_reason="시장별 분리가 안 됨",
        execute_sql_fn=fake_execute_sql,
        execute_python_fn=fake_execute_python,
    )

    assert res["ok"] is True
    assert res["result"] == {"KOSPI": ["005930"], "KOSDAQ": ["036000"]}
    assert "SELECT" in res["sql"]
    assert "result =" in res["code"]
    assert res["error"] is None
    # SQL 실행 결과의 rows가 그대로 Python 실행 컨텍스트로 전달됐는지
    assert py_calls[0][1] == {"rows": [{"stock_code": "005930", "market": "KOSPI", "earnings_yield": 4.09}]}
    # 실패 피드백(last_reason)이 두 프롬프트 모두에 반영됐는지
    assert all("시장별 분리가 안 됨" in p for p in llm_fn.calls)


# 실사용 재현: LLM이 "종목별 최신 시세"를 상관 서브쿼리(각 행마다 반복 실행되는 하위쿼리)로
# 조인해 120초 타임아웃에 걸렸다(prices가 대용량 테이블). SQL은 단순하게 유지하고 복잡한
# 매칭/정렬은 Python 단계로 미루도록 프롬프트가 명시적으로 안내해야 한다.
def test_sql_prompt_discourages_correlated_subqueries():
    prompt = _sql_prompt("질문", None)
    assert "상관" in prompt or "서브쿼리" in prompt or "subquery" in prompt.lower()


# 실사용 재현(2회 연속): "이익수익률"/"투하자본수익률" 같은 파생(계산) 지표 이름을 AI가
# account_name 안의 리터럴 문자열로 찾으려다 KeyError. Python 프롬프트가 "질문에 나온
# 지표 이름이 실제 컬럼 목록에 없으면 파생값이니 원본 항목으로 직접 계산하라"고 명시해야
# 자가수정 재시도 1~2회 안에 근본 원인을 스스로 알아챌 확률이 올라간다.
def test_python_prompt_warns_that_missing_columns_may_be_derived_metrics():
    prompt = _python_prompt("질문", ["stock_code", "account_key", "amount"], 10, None)
    assert "파생" in prompt or "계산" in prompt
    assert "컬럼 목록" in prompt or "실제 컬럼" in prompt


# 실사용 재현: "SK하이닉스 12개월 누적 수익률"을 물으면 자유코드가 rows(전체 가격이력,
# 오름차순)의 맨 앞 12개월(2017~2018년, DB에서 가장 오래된 구간)을 "최근 12개월"로 착각해
# 계산했다 — LLM이 캘린더 오늘을 모르니 rows 순서만으로 "최근"을 잘못 추측한 것. rows에서
# 실제 최신 날짜를 뽑아 프롬프트에 기준일로 명시해야 이 착각을 막을 수 있다.
def test_infer_latest_date_picks_max_of_date_column():
    rows = [{"date": "2017-05-16", "close": 100}, {"date": "2026-07-16", "close": 200}, {"date": "2020-01-01", "close": 150}]
    assert _infer_latest_date(rows, ["date", "close"]) == "2026-07-16"


def test_infer_latest_date_returns_none_without_date_like_column():
    rows = [{"stock_code": "005930", "amount": 100}]
    assert _infer_latest_date(rows, ["stock_code", "amount"]) is None


def test_infer_latest_date_ignores_missing_values():
    rows = [{"date": None}, {"date": "2025-03-01"}]
    assert _infer_latest_date(rows, ["date"]) == "2025-03-01"


def test_python_prompt_includes_latest_date_as_anchor_when_given():
    prompt = _python_prompt("질문", ["date", "close"], 10, None, latest_date="2026-07-16")
    assert "2026-07-16" in prompt
    assert "최근" in prompt


def test_python_prompt_omits_date_anchor_note_when_no_date_column():
    prompt = _python_prompt("질문", ["stock_code", "amount"], 10, None, latest_date=None)
    assert "기준일은 None" not in prompt


def test_run_free_exec_fallback_passes_latest_date_from_rows_into_python_prompt():
    llm_fn = _fake_llm_sequence(
        "```sql\nSELECT date, close FROM prices;\n```",
        "```python\nresult = {'ret': 1.0}\n```",
    )
    captured_prompts: list[str] = []

    def capturing_llm(prompt):
        captured_prompts.append(prompt)
        return llm_fn(prompt)

    def fake_execute_sql(sql, conn):
        return {
            "ok": True, "columns": ["date", "close"],
            "rows": [{"date": "2017-05-16", "close": 100}, {"date": "2026-07-16", "close": 200}],
            "row_count": 2, "error": None,
        }

    def fake_execute_python(code, context=None, result_var="result"):
        return {"ok": True, "result": {"ret": 1.0}, "error": None}

    run_free_exec_fallback(
        "SK하이닉스 12개월 누적 수익률", conn=object(), llm_fn=capturing_llm,
        execute_sql_fn=fake_execute_sql, execute_python_fn=fake_execute_python,
    )

    python_prompt = captured_prompts[1]  # [0]=SQL 프롬프트, [1]=Python 프롬프트
    assert "2026-07-16" in python_prompt


def test_run_free_exec_fallback_without_llm_fn_short_circuits():
    sql_calls: list = []
    py_calls: list = []

    res = run_free_exec_fallback(
        "질문", conn=None, llm_fn=None,
        execute_sql_fn=lambda *a, **k: sql_calls.append(1),
        execute_python_fn=lambda *a, **k: py_calls.append(1),
    )

    assert res["ok"] is False
    assert "llm_fn" in res["error"]
    assert sql_calls == []
    assert py_calls == []


def test_run_free_exec_fallback_empty_sql_from_llm_does_not_call_python_step():
    # extract_sql(src/llm.py)은 세미콜론/코드펜스가 전혀 없는 순수 공백만 빈 문자열로
    # 취급한다 — "정말 SQL이 아예 안 나온" 경우를 재현한다.
    llm_fn = _fake_llm_sequence("   \n  ")
    py_calls: list = []

    res = run_free_exec_fallback(
        "질문", conn=object(), llm_fn=llm_fn,
        execute_sql_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("호출되면 안 됨")),
        execute_python_fn=lambda *a, **k: py_calls.append(1),
    )

    assert res["ok"] is False
    assert res["sql"] is None
    assert py_calls == []


def test_run_free_exec_fallback_sql_execution_failure_stops_before_python_step():
    llm_fn = _fake_llm_sequence("SELECT * FROM no_such_table;")
    py_calls: list = []

    def fake_execute_sql(sql, conn):
        return {"ok": False, "columns": [], "rows": [], "row_count": 0, "error": "no such table"}

    res = run_free_exec_fallback(
        "질문", conn=object(), llm_fn=llm_fn,
        execute_sql_fn=fake_execute_sql,
        execute_python_fn=lambda *a, **k: py_calls.append(1),
    )

    assert res["ok"] is False
    assert "SQL" in res["error"]
    assert "no such table" in res["error"]
    assert res["sql"] == "SELECT * FROM no_such_table;"
    assert py_calls == []


def test_run_free_exec_fallback_python_execution_failure_preserves_sql_and_code():
    # 기본 max_code_attempts=2 — 두 번 다 실패하는 코드를 생성하도록 fake llm에 응답 2개 준비.
    llm_fn = _fake_llm_sequence("SELECT 1;", "result = 1 / 0", "result = 1 / 0")

    def fake_execute_sql(sql, conn):
        return {"ok": True, "columns": ["1"], "rows": [{"1": 1}], "row_count": 1, "error": None}

    def fake_execute_python(code, context=None, result_var="result"):
        return {"ok": False, "result": None, "error": "ZeroDivisionError: division by zero"}

    res = run_free_exec_fallback(
        "질문", conn=object(), llm_fn=llm_fn,
        execute_sql_fn=fake_execute_sql,
        execute_python_fn=fake_execute_python,
    )

    assert res["ok"] is False
    assert "ZeroDivisionError" in res["error"]
    assert res["sql"] == "SELECT 1;"
    assert res["code"] == "result = 1 / 0"


# 실사용 재현: 코드가 예외 없이 실행됐지만(ok=True) 계정명/시장명 필터가 실제 데이터 값과
# 안 맞아 "껍데기는 있지만 속은 빈" 결과({"코스피": [], "코스닥": []})가 나왔다. 최상위
# None/[]/{}만 걸러내던 기존 체크는 이걸 놓친다 — 재귀적으로 "전부 비었는지"를 봐야 한다.
def test_run_free_exec_fallback_treats_dict_of_all_empty_lists_as_failure():
    llm_fn = _fake_llm_sequence(
        "SELECT 1;",
        "result = {'코스피': [], '코스닥': []}",
        "result = {'코스피': ['005930'], '코스닥': []}",  # 재시도: 일부라도 채움
    )

    def fake_execute_sql(sql, conn):
        return {"ok": True, "columns": ["1"], "rows": [{"1": 1}], "row_count": 1, "error": None}

    def fake_execute_python(code, context=None, result_var="result"):
        if "005930" in code:
            return {"ok": True, "result": {"코스피": ["005930"], "코스닥": []}, "error": None}
        return {"ok": True, "result": {"코스피": [], "코스닥": []}, "error": None}

    res = run_free_exec_fallback(
        "질문", conn=object(), llm_fn=llm_fn,
        execute_sql_fn=fake_execute_sql,
        execute_python_fn=fake_execute_python,
    )

    assert res["ok"] is True  # 일부(코스피)라도 채워졌으면 유효한 결과로 인정
    assert res["result"] == {"코스피": ["005930"], "코스닥": []}


def test_run_free_exec_fallback_all_empty_dict_gives_up_after_max_attempts():
    llm_fn = _fake_llm_sequence("SELECT 1;", "result = {'a': []}", "result = {'a': []}")

    def fake_execute_sql(sql, conn):
        return {"ok": True, "columns": ["1"], "rows": [{"1": 1}], "row_count": 1, "error": None}

    def fake_execute_python(code, context=None, result_var="result"):
        return {"ok": True, "result": {"a": []}, "error": None}

    res = run_free_exec_fallback(
        "질문", conn=object(), llm_fn=llm_fn,
        execute_sql_fn=fake_execute_sql,
        execute_python_fn=fake_execute_python,
        max_code_attempts=2,
    )

    assert res["ok"] is False
    assert res["error"]


def test_run_free_exec_fallback_empty_result_is_treated_as_failure():
    llm_fn = _fake_llm_sequence("SELECT 1;", "result = []", "result = []")

    def fake_execute_sql(sql, conn):
        return {"ok": True, "columns": ["1"], "rows": [{"1": 1}], "row_count": 1, "error": None}

    def fake_execute_python(code, context=None, result_var="result"):
        return {"ok": True, "result": [], "error": None}

    res = run_free_exec_fallback(
        "질문", conn=object(), llm_fn=llm_fn,
        execute_sql_fn=fake_execute_sql,
        execute_python_fn=fake_execute_python,
    )

    assert res["ok"] is False
    assert res["error"]


# ── Python 단계 자가 수정: 실사용 재현(KeyError: 계정별 long-format 원본을 코드가 아직
#    계산 안 된 컬럼명으로 바로 참조) — SQL 재시도(dispatch_fn 피드백)와 동일한 패턴으로
#    Python도 실패 사유를 다음 시도 프롬프트에 피드백하고 재시도한다(정확히 max_code_attempts
#    회까지, 무한루프 없음 — SQL은 이미 성공했으니 재사용하고 다시 조회하지 않는다). ─────────
def test_run_free_exec_fallback_retries_python_step_after_failure_and_succeeds():
    llm_fn = _fake_llm_sequence(
        "SELECT 1;",
        "result = row['이익수익률']",  # 1차: 존재하지 않는 컬럼 참조(KeyError 재현)
        "result = {'KOSPI': ['005930']}",  # 2차: 피드백 반영해 고친 코드
    )
    sql_calls: list = []
    py_attempts: list = []

    def fake_execute_sql(sql, conn):
        sql_calls.append(1)
        return {"ok": True, "columns": ["1"], "rows": [{"1": 1}], "row_count": 1, "error": None}

    def fake_execute_python(code, context=None, result_var="result"):
        py_attempts.append(code)
        if len(py_attempts) == 1:
            return {"ok": False, "result": None, "error": "KeyError: '이익수익률'"}
        return {"ok": True, "result": {"KOSPI": ["005930"]}, "error": None}

    res = run_free_exec_fallback(
        "질문", conn=object(), llm_fn=llm_fn,
        execute_sql_fn=fake_execute_sql,
        execute_python_fn=fake_execute_python,
    )

    assert res["ok"] is True
    assert res["result"] == {"KOSPI": ["005930"]}
    assert res["code"] == "result = {'KOSPI': ['005930']}"
    assert len(sql_calls) == 1  # SQL은 이미 성공했으므로 재시도 시 다시 조회하지 않음
    assert len(py_attempts) == 2
    # 두 번째(재시도) 프롬프트에 첫 실패의 에러 메시지가 피드백으로 포함됐는지
    assert "KeyError" in llm_fn.calls[2]


def test_run_free_exec_fallback_gives_up_after_max_code_attempts():
    llm_fn = _fake_llm_sequence("SELECT 1;", "result = bad1", "result = bad2")

    def fake_execute_sql(sql, conn):
        return {"ok": True, "columns": ["1"], "rows": [{"1": 1}], "row_count": 1, "error": None}

    calls: list = []

    def fake_execute_python(code, context=None, result_var="result"):
        calls.append(code)
        return {"ok": False, "result": None, "error": f"NameError: {code}"}

    res = run_free_exec_fallback(
        "질문", conn=object(), llm_fn=llm_fn,
        execute_sql_fn=fake_execute_sql,
        execute_python_fn=fake_execute_python,
        max_code_attempts=2,
    )

    assert res["ok"] is False
    assert len(calls) == 2  # 정확히 max_code_attempts회(무한루프 없음)
    assert res["code"] == "result = bad2"  # 마지막 시도의 코드가 남음
