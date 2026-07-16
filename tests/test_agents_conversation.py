"""멀티턴 대화 세션 (MT-1~MT-5, TDD).

.omc/specs/brainstorming-multiturn-conversation.md AC1-AC10 참고. 세션은 프로세스
메모리에만 존재하며, supervisor.answer_with_verification을 재사용한다(정형 도메인 검증
+ 필요 시 그 내부의 자유 코드 생성 폴백까지 자동) — 새 SQL/Python 생성 로직을 중복
구현하지 않는다. 신규 턴에 exec_fallback을 직접 쓰지 않는 이유는 conversation.py 모듈
docstring 참고 — "삼성전자 오늘 종가"류 쉬운 질문까지 매번 자유 코드부터 생성하다 실패하는
실사용 회귀가 있었다.
"""
from __future__ import annotations

import pytest

from src.agents.conversation import (
    ConversationSession,
    Turn,
    _classify_topic,
    _summarize_data_shape,
    get_history,
    get_or_create_session,
    get_session,
    reset_session,
    run_turn,
    turn_to_csv,
)


def _fresh_session(session_id="s1"):
    return ConversationSession(session_id=session_id)


# --------------------------------------------------------------------------
# MT-1 — 신규 턴(데이터 없음) → answer_with_verification 그대로 호출(정형 검증 + 필요 시
# 그 내부의 자유 코드 폴백까지 자동. exec_fallback을 신규 턴에서 직접 부르지 않는다 —
# "삼성전자 오늘 종가"류 쉬운 질문까지 매번 자유 코드부터 짜다 실패하는 회귀 방지).
# --------------------------------------------------------------------------
def test_run_turn_calls_verified_fn_when_session_has_no_data():
    calls = []

    def fake_verified(question, conn, llm_fn, **kwargs):
        calls.append((question, conn, llm_fn))
        return {"uncertain": False, "conclusion": "PBR 낮은 순 상위 종목", "domain_results": {"kr": {"result": [{"code": "005930", "pbr": 1.2}]}}}

    session = _fresh_session()
    turn = run_turn(session, "코스피 PBR 낮은 순", conn="fake_conn", llm_fn="fake_llm", verified_fn=fake_verified)

    assert len(calls) == 1
    assert calls[0][0] == "코스피 PBR 낮은 순"
    assert turn.status == "success"


def test_run_turn_updates_session_on_new_turn_success():
    domain_results = {"kr": {"result": [{"code": "005930", "pbr": 1.2}]}}

    def fake_verified(question, conn, llm_fn, **kwargs):
        return {"uncertain": False, "conclusion": "삼성전자 PBR은 1.2입니다", "domain_results": domain_results}

    session = _fresh_session()
    turn = run_turn(session, "코스피 PBR 낮은 순", conn=None, llm_fn="fake_llm", verified_fn=fake_verified)

    assert session.has_data is True
    assert session.current_data == domain_results  # 다음 턴이 이어받을 맥락은 원본 domain_results
    assert turn.answer == "삼성전자 PBR은 1.2입니다"  # 화면에 보이는 답은 종합결론 텍스트


def test_run_turn_keeps_session_empty_on_new_turn_failure():
    def fake_verified(question, conn, llm_fn, **kwargs):
        return {"uncertain": True, "reason": "질문을 이해하지 못했습니다", "domain_results": {}, "attempts": 3}

    session = _fresh_session()
    turn = run_turn(session, "이상한 질문", conn=None, llm_fn="fake_llm", verified_fn=fake_verified)

    assert turn.status == "fail"
    assert turn.error == "질문을 이해하지 못했습니다"
    assert session.has_data is False
    assert session.current_data is None


def test_run_turn_new_turn_success_keeps_tabular_data_for_csv():
    """실사용 재현: 신규 턴의 answer는 종합결론 텍스트라(그대로 CSV로는 못 바꿈) turn.data에
    원본 표를 따로 담아둬야 CSV 다운로드가 살아있다."""
    domain_results = {"kr": {"result": [{"code": "005930", "pbr": 1.2}, {"code": "000660", "pbr": 0.9}]}}

    def fake_verified(question, conn, llm_fn, **kwargs):
        return {"uncertain": False, "conclusion": "PBR 낮은 순 텍스트 답변", "domain_results": domain_results}

    session = _fresh_session()
    turn = run_turn(session, "코스피 PBR 낮은 순", conn=None, llm_fn="fake_llm", verified_fn=fake_verified)

    assert turn.answer == "PBR 낮은 순 텍스트 답변"  # 화면 표시는 여전히 텍스트
    assert turn.data == domain_results["kr"]["result"]  # CSV용 원본 표는 따로 보존됨

    session.turns = [turn]
    csv_text = turn_to_csv(session, 0)
    assert "005930" in csv_text
    assert "pbr" in csv_text


def test_run_turn_new_turn_without_tabular_result_has_no_csv():
    """단일값 조회(예: 종가 하나) 등 표가 아예 없는 도메인 결과면 data가 None이고 CSV 불가."""
    def fake_verified(question, conn, llm_fn, **kwargs):
        return {"uncertain": False, "conclusion": "삼성전자 종가는 255,000원", "domain_results": {"kr": {"stock_code": "005930", "close": 255000}}}

    session = _fresh_session()
    turn = run_turn(session, "삼성전자 오늘 종가", conn=None, llm_fn="fake_llm", verified_fn=fake_verified)

    assert turn.data is None
    session.turns = [turn]
    history = get_history(session)
    assert history[0]["csv_available"] is False


def test_run_turn_exposes_sql_code_when_verified_fn_used_internal_fallback():
    """answer_with_verification이 정형 검증 3회 실패 후 내부적으로 자유코드 폴백을 써서
    성공한 경우(used_fallback=True) — 그 sql/code를 Turn에 노출해 투명성을 유지한다."""
    def fake_verified(question, conn, llm_fn, **kwargs):
        return {
            "uncertain": False, "conclusion": "계산 결과입니다",
            "domain_results": {"free_exec": {"fallback_used": True, "sql": "SELECT *", "code": "result = 1", "result": 1}},
            "used_fallback": True,
        }

    session = _fresh_session()
    turn = run_turn(session, "복잡한 질문", conn=None, llm_fn="fake_llm", verified_fn=fake_verified)

    assert turn.sql == "SELECT *"
    assert turn.code == "result = 1"


# --------------------------------------------------------------------------
# 구조화 도메인 근거 노출 — 신규 턴이 성공하면 free_exec(자유코드) 여부와 무관하게
# 항상 domain_evidence(원본 domain_results)를 담는다. 정형 도메인(backtest/kr/us/macro)이
# 저장된 파이프라인으로 성공한 경우에도 "새로 실행한 SQL/코드가 없다"가 아니라 실제
# 근거 데이터를 보여주기 위함이다(오른쪽 패널 렌더링 근거, AC4 원본 그대로 병기).
# --------------------------------------------------------------------------
def test_run_turn_new_turn_exposes_domain_evidence_without_fallback():
    """저장된 함수(정형 도메인)로만 성공한 신규 턴 — free_exec가 없어도 domain_evidence가 채워진다."""
    domain_results = {"backtest": {"result": {"cagr": 0.12, "mdd": -0.2}}}

    def fake_verified(question, conn, llm_fn, **kwargs):
        return {"uncertain": False, "conclusion": "백테스트 결과입니다", "domain_results": domain_results}

    session = _fresh_session()
    turn = run_turn(session, "5일/20일 이동평균 백테스트", conn=None, llm_fn="fake_llm", verified_fn=fake_verified)

    assert turn.domain_evidence == domain_results  # 자유코드(free_exec) 없이도 근거가 채워짐
    assert turn.sql is None and turn.code is None   # free_exec 전용 필드는 그대로 비어 있음


def test_run_turn_fallback_turn_also_fills_domain_evidence():
    """자유코드 폴백으로 성공한 턴 — sql/code(free_exec)와 domain_evidence를 모두 담는다(중복 아님, 병기)."""
    domain_results = {"free_exec": {"fallback_used": True, "sql": "SELECT *", "code": "result = 1", "result": 1}}

    def fake_verified(question, conn, llm_fn, **kwargs):
        return {"uncertain": False, "conclusion": "계산 결과", "domain_results": domain_results, "used_fallback": True}

    session = _fresh_session()
    turn = run_turn(session, "복잡한 질문", conn=None, llm_fn="fake_llm", verified_fn=fake_verified)

    assert turn.sql == "SELECT *"
    assert turn.domain_evidence == domain_results


def test_run_turn_failed_turn_has_no_domain_evidence():
    def fake_verified(question, conn, llm_fn, **kwargs):
        return {"uncertain": True, "reason": "질문을 이해하지 못했습니다", "domain_results": {}}

    session = _fresh_session()
    turn = run_turn(session, "이상한 질문", conn=None, llm_fn="fake_llm", verified_fn=fake_verified)

    assert turn.domain_evidence is None


def test_get_history_includes_domain_evidence():
    session = _fresh_session()
    ev = {"kr": {"result": [{"code": "005930", "pbr": 1.2}]}}
    session.turns = [Turn(question="q", status="success", answer="답", domain_evidence=ev)]

    history = get_history(session)

    assert history[0]["domain_evidence"] == ev


# --------------------------------------------------------------------------
# MT-2 — 이어가기 턴(데이터 있음) → Python 가공만, SQL 재조회 없음, 요약정보만 프롬프트에
# --------------------------------------------------------------------------
def test_run_turn_does_not_call_sql_when_session_has_data():
    verified_calls = []

    def fake_verified(question, conn, llm_fn, **kwargs):
        verified_calls.append(question)
        return {"uncertain": False, "conclusion": "", "domain_results": {}}

    session = _fresh_session()
    session.has_data = True
    session.current_data = [{"code": "005930", "pbr": 1.2}, {"code": "000660", "pbr": 0.9}]

    def fake_llm(prompt):
        return "```python\nresult = sorted(data, key=lambda r: r['pbr'])\n```"

    def fake_execute_python(code, context, result_var):
        return {"ok": True, "result": sorted(context["data"], key=lambda r: r["pbr"])}

    run_turn(session, "이제 오름차순으로", conn=None, llm_fn=fake_llm,
              execute_python_fn=fake_execute_python, verified_fn=fake_verified,
              classify_fn=lambda *a, **k: False)

    assert verified_calls == []  # 신규 조회(answer_with_verification) 재호출 없음


def test_followup_prompt_contains_only_summary_not_raw_values():
    session = _fresh_session()
    session.has_data = True
    session.current_data = [{"code": "005930", "pbr": 1.2345678}]

    captured_prompts = []

    def fake_llm(prompt):
        captured_prompts.append(prompt)
        return "```python\nresult = data\n```"

    def fake_execute_python(code, context, result_var):
        return {"ok": True, "result": context["data"]}

    run_turn(session, "다시 보여줘", conn=None, llm_fn=fake_llm, execute_python_fn=fake_execute_python,
              classify_fn=lambda *a, **k: False)

    assert len(captured_prompts) == 1
    assert "1.2345678" not in captured_prompts[0]  # 원본 값이 프롬프트 텍스트에 없어야 함
    assert "005930" not in captured_prompts[0]
    assert "pbr" in captured_prompts[0]  # 컬럼명 같은 구조 요약은 있어야 함


def test_followup_execution_context_receives_full_prior_data():
    session = _fresh_session()
    session.has_data = True
    session.current_data = [{"code": "005930", "pbr": 1.2}, {"code": "000660", "pbr": 0.9}]

    captured_contexts = []

    def fake_llm(prompt):
        return "```python\nresult = data\n```"

    def fake_execute_python(code, context, result_var):
        captured_contexts.append(context)
        return {"ok": True, "result": context["data"]}

    run_turn(session, "그대로 다시", conn=None, llm_fn=fake_llm, execute_python_fn=fake_execute_python,
              classify_fn=lambda *a, **k: False)

    assert captured_contexts[0]["data"] == [{"code": "005930", "pbr": 1.2}, {"code": "000660", "pbr": 0.9}]


# --------------------------------------------------------------------------
# 무관한 새 주제 자동 감지 — 세션에 데이터가 있어도 새 질문이 그 데이터와 무관하면
# 이어가기 대신 신규 조회(_run_new_turn)로 처리한다(실사용 재현: "삼성전자 오늘 종가" 다음
# "코스피 전종목 PBR 히스토그램"을 물으면 무관한 데이터를 억지로 가공하려다 깨졌음).
# --------------------------------------------------------------------------
def test_classify_topic_continue_when_llm_says_continue():
    assert _classify_topic("오름차순으로 다시", [{"a": 1}], lambda prompt: "CONTINUE") is False


def test_classify_topic_new_when_llm_says_new():
    assert _classify_topic("완전히 다른 질문", [{"a": 1}], lambda prompt: "NEW") is True


def test_classify_topic_defaults_to_continue_when_llm_fn_missing():
    assert _classify_topic("질문", [{"a": 1}], None) is False


def test_run_turn_routes_unrelated_followup_question_to_new_turn():
    def fake_verified(question, conn, llm_fn, **kwargs):
        return {"uncertain": False, "conclusion": "새 데이터 답", "domain_results": {"kr": {"result": [{"b": 2}]}}}

    session = _fresh_session()
    session.has_data = True
    session.current_data = [{"a": 1}]  # 직전 턴(무관한 주제)의 데이터

    turn = run_turn(session, "완전히 다른 질문", conn=None, llm_fn="fake_llm",
                     verified_fn=fake_verified, classify_fn=lambda *a, **k: True)

    assert turn.answer == "새 데이터 답"
    assert session.current_data == {"kr": {"result": [{"b": 2}]}}  # 직전 무관 데이터를 덮어씀


def test_summarize_data_shape_describes_list_of_dict_without_values():
    summary = _summarize_data_shape([{"code": "005930", "pbr": 1.2}, {"code": "000660", "pbr": 0.9}])
    assert "005930" not in summary
    assert "pbr" in summary
    assert "2" in summary  # 행 개수


def test_summarize_data_shape_describes_dict():
    summary = _summarize_data_shape({"코스피": [1, 2], "코스닥": [3]})
    assert "코스피" in summary
    assert "코스닥" in summary


# --------------------------------------------------------------------------
# MT-3 — 실패 처리: 이어가기가 실패하면 무관한 새 주제였을 수 있으므로 신규 조회를 한 번
# 자동으로 재시도한다(_classify_topic이 LLM의 형식 미준수로 새 주제를 놓칠 수 있어서다,
# 실사용 확인됨). 그 재시도마저 uncertain(실패)이면 current_data는 오염되지 않는다.
# --------------------------------------------------------------------------
def _uncertain_verified(question, conn, llm_fn, **kwargs):
    return {"uncertain": True, "reason": "질문을 이해하지 못했습니다", "domain_results": {}}


def test_followup_failure_keeps_prior_current_data():
    session = _fresh_session()
    session.has_data = True
    session.current_data = [{"code": "005930", "pbr": 1.2}]

    def fake_llm(prompt):
        return "```python\nresult = data[0]['없는키']\n```"

    def fake_execute_python(code, context, result_var):
        return {"ok": False, "error": "KeyError: '없는키'"}

    turn = run_turn(session, "이상한 가공", conn=None, llm_fn=fake_llm, execute_python_fn=fake_execute_python,
                     max_code_attempts=1, classify_fn=lambda *a, **k: False, verified_fn=_uncertain_verified)

    assert turn.status == "fail"
    assert session.current_data == [{"code": "005930", "pbr": 1.2}]  # 오염 없음
    assert session.has_data is True


def test_followup_failure_retries_as_new_turn_and_can_succeed():
    """실사용 재현: 이전 턴과 무관한 새 주제라 이어가기가 깨졌을 때, 자동 재시도(신규 조회)로
    복구되는지 확인한다(더 이상 사용자가 '대화 초기화'를 직접 눌러야만 하는 게 아니다)."""
    session = _fresh_session()
    session.has_data = True
    session.current_data = [{"code": "005930", "close": 255000}]  # 무관한 직전 데이터

    def fake_llm(prompt):
        return "```python\nresult = data.columns\n```"  # dict에 .columns 접근 시도 → 실패

    def fake_execute_python(code, context, result_var):
        return {"ok": False, "error": "AttributeError: 'dict' object has no attribute 'columns'"}

    def fake_verified(question, conn, llm_fn, **kwargs):
        return {"uncertain": False, "conclusion": "새로 조회한 답", "domain_results": {"kr": {"result": [{"pbr": 0.5}]}}}

    turn = run_turn(session, "코스피 전종목 pbr 히스토그램", conn=None, llm_fn=fake_llm,
                     execute_python_fn=fake_execute_python, max_code_attempts=1,
                     classify_fn=lambda *a, **k: False,  # 분류가 "이어가기"로 놓쳤다고 가정
                     verified_fn=fake_verified)

    assert turn.status == "success"
    assert turn.answer == "새로 조회한 답"
    assert session.current_data == {"kr": {"result": [{"pbr": 0.5}]}}  # 새 데이터로 교체됨


def test_failure_reason_not_included_in_next_turn_prompt():
    session = _fresh_session()
    session.has_data = True
    session.current_data = [{"code": "005930", "pbr": 1.2}]

    def failing_llm(prompt):
        return "```python\nresult = 1/0\n```"

    def failing_execute_python(code, context, result_var):
        return {"ok": False, "error": "ZeroDivisionError: division by zero"}

    run_turn(session, "실패할 질문", conn=None, llm_fn=failing_llm, execute_python_fn=failing_execute_python,
              max_code_attempts=1, classify_fn=lambda *a, **k: False, verified_fn=_uncertain_verified)

    captured_prompts = []

    def next_llm(prompt):
        captured_prompts.append(prompt)
        return "```python\nresult = data\n```"

    def next_execute_python(code, context, result_var):
        return {"ok": True, "result": context["data"]}

    run_turn(session, "다음 질문", conn=None, llm_fn=next_llm, execute_python_fn=next_execute_python,
              classify_fn=lambda *a, **k: False)

    assert "ZeroDivisionError" not in captured_prompts[0]


def test_success_after_two_failures_bases_on_last_success_not_failed_attempts():
    session = _fresh_session()
    session.has_data = True
    session.current_data = [{"code": "005930", "pbr": 1.2}]

    fail_llm = lambda prompt: "```python\nresult = 1/0\n```"
    fail_exec = lambda code, context, result_var: {"ok": False, "error": "boom"}
    classify_continue = lambda *a, **k: False
    run_turn(session, "실패1", conn=None, llm_fn=fail_llm, execute_python_fn=fail_exec, max_code_attempts=1,
              classify_fn=classify_continue, verified_fn=_uncertain_verified)
    run_turn(session, "실패2", conn=None, llm_fn=fail_llm, execute_python_fn=fail_exec, max_code_attempts=1,
              classify_fn=classify_continue, verified_fn=_uncertain_verified)

    captured_contexts = []

    def ok_llm(prompt):
        return "```python\nresult = data\n```"

    def ok_exec(code, context, result_var):
        captured_contexts.append(context)
        return {"ok": True, "result": context["data"]}

    run_turn(session, "이제 성공", conn=None, llm_fn=ok_llm, execute_python_fn=ok_exec,
              classify_fn=classify_continue)

    assert captured_contexts[0]["data"] == [{"code": "005930", "pbr": 1.2}]


# --------------------------------------------------------------------------
# MT-4 — 세션 생성/조회/초기화 (메모리 저장)
# --------------------------------------------------------------------------
def test_get_or_create_session_creates_new_when_none_given():
    session = get_or_create_session(None)
    assert session.session_id
    assert session.has_data is False
    assert session.turns == []


def test_get_or_create_session_returns_same_object_for_existing_id():
    first = get_or_create_session(None)
    second = get_or_create_session(first.session_id)
    assert first is second


def test_reset_session_clears_accumulated_state():
    session = get_or_create_session(None)
    session.has_data = True
    session.current_data = [{"a": 1}]
    session.turns.append(Turn(question="q", status="success"))

    reset_session(session.session_id)
    after = get_session(session.session_id)

    assert after.has_data is False
    assert after.current_data is None
    assert after.turns == []


def test_get_session_returns_none_for_unknown_id():
    assert get_session("존재하지-않는-id-xyz") is None


# --------------------------------------------------------------------------
# MT-5 — 히스토리 조회 + CSV 재다운로드 가능여부
# --------------------------------------------------------------------------
def test_get_history_returns_all_turns_in_order():
    session = _fresh_session()
    session.turns = [
        Turn(question="q1", status="success", answer=[{"a": 1}]),
        Turn(question="q2", status="fail", error="err"),
    ]

    history = get_history(session)

    assert [h["question"] for h in history] == ["q1", "q2"]
    assert history[0]["status"] == "success"
    assert history[1]["status"] == "fail"
    assert history[1]["error"] == "err"


def test_turn_to_csv_succeeds_for_tabular_answer():
    session = _fresh_session()
    session.turns = [Turn(question="q1", status="success", answer=[{"code": "005930", "pbr": 1.2}])]

    csv_text = turn_to_csv(session, 0)

    assert "code" in csv_text
    assert "005930" in csv_text


def test_turn_to_csv_rejects_non_tabular_answer():
    session = _fresh_session()
    session.turns = [Turn(question="q1", status="success", answer=42)]

    with pytest.raises(ValueError, match="표 형태가 아니"):
        turn_to_csv(session, 0)


def test_turn_to_csv_rejects_failed_turn():
    session = _fresh_session()
    session.turns = [Turn(question="q1", status="fail", error="boom")]

    with pytest.raises(ValueError, match="실패한 턴"):
        turn_to_csv(session, 0)


# --------------------------------------------------------------------------
# on_progress — AC13(오른쪽 패널 실시간 진행상황) 뒷받침. exec_fallback.py는 건드리지
# 않으므로 신규 턴은 굵은 단위(단일 메시지)로, 이어가기(직접 구현)는 시도별로 세밀하게
# 보고한다. 콜백을 생략해도(기존 호출부는 그대로) 동작이 깨지지 않아야 한다.
# --------------------------------------------------------------------------
def test_run_turn_without_on_progress_still_works():
    def fake_verified(question, conn, llm_fn, **kwargs):
        return {"uncertain": False, "conclusion": "답", "domain_results": {"kr": {"result": [{"a": 1}]}}}

    session = _fresh_session()
    turn = run_turn(session, "질문", conn=None, llm_fn="fake_llm", verified_fn=fake_verified)
    assert turn.status == "success"  # on_progress 없이도 정상 동작(회귀 방지)


def test_run_turn_reports_progress_for_new_turn():
    def fake_verified(question, conn, llm_fn, **kwargs):
        return {"uncertain": False, "conclusion": "답", "domain_results": {"kr": {"result": [{"a": 1}]}}}

    messages = []
    session = _fresh_session()
    run_turn(session, "질문", conn=None, llm_fn="fake_llm", verified_fn=fake_verified,
              on_progress=messages.append)

    assert len(messages) >= 2  # 시작 신호 + 완료 신호 최소 2건(정적 스피너가 아님을 증명)
    assert any("완료" in m for m in messages)


def test_run_turn_bridges_verified_fn_progress_to_outer_callback():
    """answer_with_verification의 on_progress(step, summary) 시그니처를 conversation.py의
    on_progress(message) 단일 인자 콜백으로 브릿지하는지 확인한다."""
    def fake_verified(question, conn, llm_fn, on_progress=None, **kwargs):
        if on_progress:
            on_progress("kr", "국내 도메인 조회 중…")
        return {"uncertain": False, "conclusion": "답", "domain_results": {}}

    messages = []
    session = _fresh_session()
    run_turn(session, "질문", conn=None, llm_fn="fake_llm", verified_fn=fake_verified,
              on_progress=messages.append)

    assert any("국내 도메인 조회 중" in m for m in messages)


def test_run_turn_reports_progress_for_followup_per_attempt():
    session = _fresh_session()
    session.has_data = True
    session.current_data = [{"a": 1}]

    calls = {"n": 0}

    def flaky_llm(prompt):
        calls["n"] += 1
        return "```python\nresult = data\n```"

    def flaky_exec(code, context, result_var):
        if calls["n"] == 1:
            return {"ok": False, "error": "일시적 오류"}
        return {"ok": True, "result": context["data"]}

    messages = []
    run_turn(session, "질문", conn=None, llm_fn=flaky_llm, execute_python_fn=flaky_exec,
              on_progress=messages.append, classify_fn=lambda *a, **k: False)

    joined = " / ".join(messages)
    assert "생성" in joined  # 코드 생성 단계 보고
    assert "실행" in joined  # 코드 실행 단계 보고
    assert calls["n"] == 2   # 1차 실패 → 2차 재시도로 성공(전제 확인)
    assert any("재시도" in m or "실패" in m for m in messages)  # 재시도 자체도 보고됨


def test_get_history_flags_csv_availability_per_turn():
    session = _fresh_session()
    session.turns = [
        Turn(question="q1", status="success", answer=[{"a": 1}]),
        Turn(question="q2", status="success", answer=42),
        Turn(question="q3", status="fail", error="boom"),
    ]

    history = get_history(session)

    assert history[0]["csv_available"] is True
    assert history[1]["csv_available"] is False
    assert history[2]["csv_available"] is False
