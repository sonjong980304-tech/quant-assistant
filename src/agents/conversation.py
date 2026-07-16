"""멀티턴 대화 세션 — 기존 총괄 검증 엔진(answer_with_verification)을 재사용해 턴을 이어간다.

.omc/specs/brainstorming-multiturn-conversation.md 참고. 세션은 프로세스 메모리에만
존재한다(서버 재시작 시 소실 — Non-Goal, 설계상 허용된 트레이드오프). 세션에 데이터가
없으면(첫 턴) 기존 answer_with_verification을 그대로 실행하고(정형 도메인 라우팅+검증,
반복 실패 시 그 내부에서 자유 코드 생성 폴백까지 자동으로 시도), 이미 데이터가 있으면
먼저 "이 질문이 직전 데이터와 이어지는지, 무관한 새 주제인지"를 가볍게 LLM으로 분류한 뒤
(_classify_topic) — 이어지면 Python 가공 단계만 재실행하고, 무관한 새 주제면 신규 턴과
동일하게 answer_with_verification부터 다시 시작한다.

신규 턴에 exec_fallback(run_free_exec_fallback)을 직접 쓰지 않는 이유: 그건 "정형 검증이 3회
반복 실패한 뒤의 최후 수단"으로 설계된 것이라(exec_fallback.py 참고), "삼성전자 오늘 종가"처럼
기존 단발 질의 화면(/query)의 정형 도메인 엔진이 안정적으로 answer_with_verification 답하던
쉬운 질문까지 신규 턴에서 항상 자유 코드 생성부터 거치게 되면(구 default 화면에서는 없었던
안전장치 우회) LLM이 매번 원본 데이터를 처음부터 즉석으로 다시 계산하다 실패하는 경우가
생긴다(예: 이미 계산돼 있는 PBR/PER 등 파생지표를 원본 계정과목에서 다시 유도하려다 실패,
numpy 배열 진위값 오류 등) — 실사용 중 재현되어 answer_with_verification로 교체함.

세션 안에서 "신규/이어가기"를 매번 LLM에게 분류시키는 판단 로직은 브레인스토밍 Round 5에서
한 번 없앴었으나(있다/없다만 보는 단순 규칙), 실사용에서 "삼성전자 오늘 종가" 다음에 완전히
무관한 "코스피 전종목 PBR 히스토그램"을 물으면 직전 데이터(삼성전자 종가)를 억지로 가공하려다
깨지는 문제가 재현되어(AttributeError 등) 다시 들여왔다 — 단, "매 메시지 무조건 분류"가 아니라
"세션에 이미 데이터가 있을 때만" 가볍게 한 번 분류한다(첫 턴은 어차피 분류할 대상이 없다).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from src.agents.exec_fallback import _extract_python_code, _is_meaningfully_empty
from src.agents.exec_runtime import execute_python
from src.agents.supervisor import answer_with_verification


@dataclass
class Turn:
    question: str
    status: str  # "success" | "fail"
    answer: object = None
    data: object = None  # CSV 변환용 원본 표 데이터(있으면). answer는 신규 턴에서 사람이
    # 읽는 종합결론 텍스트라 표 형태가 아닐 수 있어 별도로 둔다 — _turn_csv_source 참고.
    sql: str | None = None
    code: str | None = None
    error: str | None = None
    # 정형 도메인(backtest/kr/us/macro)의 원본 결과(domain_results). free_exec(자유코드)를
    # 안 썼어도 신규 턴이 성공하면 항상 채운다 — 저장된 파이프라인으로만 성공한 경우에도
    # "새 SQL/코드 없음"이 아니라 실제 근거 데이터를 오른쪽 패널에 그대로 병기하기 위함이다.
    domain_evidence: dict | None = None


@dataclass
class ConversationSession:
    session_id: str
    turns: list = field(default_factory=list)
    current_data: object = None
    has_data: bool = False


_SESSIONS: dict[str, ConversationSession] = {}


def get_or_create_session(session_id: str | None) -> ConversationSession:
    if session_id and session_id in _SESSIONS:
        return _SESSIONS[session_id]
    sid = session_id or str(uuid.uuid4())
    session = ConversationSession(session_id=sid)
    _SESSIONS[sid] = session
    return session


def get_session(session_id: str) -> ConversationSession | None:
    return _SESSIONS.get(session_id)


def reset_session(session_id: str) -> ConversationSession:
    session = ConversationSession(session_id=session_id)
    _SESSIONS[session_id] = session
    return session


def _summarize_data_shape(data) -> str:
    """LLM 프롬프트용 구조 요약 — 실제 값은 넣지 않고 컬럼명·행수 등 구조만 설명한다."""
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            cols = sorted({k for row in data for k in row.keys()})
            return f"리스트[dict] {len(data)}개, 키: {cols}"
        return f"리스트 {len(data)}개"
    if isinstance(data, dict):
        return f"dict, 최상위 키: {sorted(data.keys())}"
    return f"{type(data).__name__} 값"


def _topic_classifier_prompt(question: str, prior_summary: str) -> str:
    return (
        "직전 대화에서 이미 만들어둔 데이터가 있습니다(구조: "
        f"{prior_summary}). 새 질문이 그 데이터를 이어서 가공/분석하는 질문인지, 아니면 그 "
        "데이터와 무관한 완전히 새로운 주제의 질문인지 판단하세요.\n"
        "이어서 가공하는 질문이면 CONTINUE 한 단어만, 무관한 새 주제면 NEW 한 단어만 답하세요. "
        "다른 설명은 절대 붙이지 마세요.\n\n"
        f"새 질문: {question}\n판정:"
    )


def _classify_topic(question: str, prior_data, llm_fn: Callable[[str], str] | None) -> bool:
    """직전 데이터와 무관한 새 주제 질문이면 True(=새로 조회해야 함).

    llm_fn이 없으면 판단할 방법이 없으므로 기존처럼 이어가기로 처리한다(False, 하위호환).
    LLM 응답이 예상 형식(CONTINUE/NEW)을 벗어나도 "NEW"라는 단어가 "CONTINUE"보다 뚜렷하게
    우세할 때만 새 주제로 판단해, 애매한 응답에서는 기존 동작(이어가기)을 그대로 유지한다.
    """
    if llm_fn is None:
        return False
    summary = _summarize_data_shape(prior_data)
    raw = (llm_fn(_topic_classifier_prompt(question, summary)) or "").strip().upper()
    return "NEW" in raw and "CONTINUE" not in raw


def _followup_python_prompt(question: str, prior_summary: str, code_error: str | None = None) -> str:
    retry_note = (
        f"\n\n[직전 코드 실행 실패] 방금 작성한 코드가 다음 이유로 실패했습니다: {code_error}\n"
        "같은 실수를 반복하지 말고 원인을 고쳐 다시 작성하세요."
        if code_error else ""
    )
    return (
        "직전 턴에서 만든 데이터가 `data`라는 이름의 변수로 이미 주어집니다"
        f"(구조: {prior_summary}). 이 데이터를 가공해 새 질문에 정확히 답하는 Python 코드를 "
        "작성하세요. pandas 등 필요한 라이브러리는 코드 안에서 직접 import 하세요. "
        "최종 답은 반드시 `result` 변수에 담으세요(리스트/딕트 등 JSON 직렬화 가능한 값)."
        f"{retry_note}\n\n질문: {question}\nPython 코드:"
    )


def _run_followup_step(
    question: str,
    prior_data,
    llm_fn: Callable[[str], str] | None,
    execute_python_fn: Callable | None = None,
    max_code_attempts: int = 2,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """이전 턴 결과(prior_data)를 `data` 변수로 실행 컨텍스트에 주입해 Python만 재실행한다.

    exec_fallback.py의 자가수정 재시도(코드 실패 사유를 다음 프롬프트에 피드백)와 빈 결과
    감지(_is_meaningfully_empty) 패턴을 그대로 재사용한다 — 새 재시도 로직을 만들지 않는다.
    on_progress는 이 함수가 직접 구현한 로직이라 시도(attempt) 단위로 세밀하게 보고할 수
    있다(exec_fallback.py는 건드리지 않으므로 신규 턴 쪽은 run_turn에서 굵은 단위로 보고).
    """
    if llm_fn is None:
        return {"ok": False, "result": None, "code": None, "error": "llm_fn 없음"}

    execute_python_fn = execute_python_fn or execute_python
    summary = _summarize_data_shape(prior_data)
    code: str | None = None
    code_error: str | None = None
    for attempt in range(max_code_attempts):
        if on_progress:
            on_progress(f"이전 데이터를 이어서 가공할 Python 코드 생성 중 (시도 {attempt + 1}/{max_code_attempts})")
        code_raw = llm_fn(_followup_python_prompt(question, summary, code_error)) or ""
        code = _extract_python_code(code_raw)
        if not code:
            code_error = "LLM이 Python 코드를 생성하지 못했습니다."
            if on_progress:
                on_progress(f"코드 생성 실패, 재시도 준비 중: {code_error}")
            continue

        if on_progress:
            on_progress("생성된 코드 실행 중")
        py_result = execute_python_fn(code, context={"data": prior_data}, result_var="result")
        if not py_result.get("ok"):
            code_error = f"Python 실행 실패: {py_result.get('error')}"
            if on_progress:
                on_progress(f"코드 실행 실패, 재시도 준비 중: {code_error}")
            continue

        result = py_result.get("result")
        if _is_meaningfully_empty(result):
            code_error = "생성된 코드가 빈 결과를 반환했습니다."
            if on_progress:
                on_progress(f"빈 결과라 재시도 준비 중: {code_error}")
            continue

        if on_progress:
            on_progress("완료")
        return {"ok": True, "result": result, "code": code, "error": None}

    return {"ok": False, "result": None, "code": code, "error": code_error}


def _extract_tabular_data(domain_results: dict):
    """domain_results(도메인별 원본 결과 dict)에서 CSV로 내려받을 수 있는 표 데이터를 하나
    찾는다. 여러 도메인이 섞인 복합질문이면 처음 찾은 표를 쓴다(질문당 CSV 1개, _build_charts
    의 "질문당 차트" 관례와 동일한 단순화). 표가 하나도 없으면(단일값 조회 등) None."""
    if not isinstance(domain_results, dict):
        return None
    for value in domain_results.values():
        if not isinstance(value, dict):
            continue
        candidate = value.get("result")
        if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
            return candidate
    return None


def _turn_csv_source(turn: Turn):
    """CSV 변환에 쓸 원본을 고른다. 신규 턴은 answer가 사람이 읽는 종합결론 텍스트라
    표가 아니므로 data(원본 표)를 우선 쓰고, 이어가기 턴은 data가 없으므로 answer(가공
    결과 자체가 표)로 자연히 폴백한다."""
    return turn.data if turn.data is not None else turn.answer


def _run_new_turn(
    session: ConversationSession,
    question: str,
    conn,
    llm_fn: Callable[[str], str] | None,
    verified_fn: Callable,
    on_progress: Callable[[str], None] | None,
) -> Turn:
    """정형 도메인 검증(필요시 내부 자유코드 폴백까지 자동)으로 새로 조회해 세션 상태를 갱신한다.

    session.has_data가 False인 첫 턴, 그리고 has_data가 True여도 _classify_topic이 "무관한
    새 주제"로 판단한 턴 양쪽에서 재사용된다(직전 데이터는 덮어쓴다 — 새 주제이므로).
    """
    if on_progress:
        on_progress("신규 질문 처리 중 (정형 도메인 조회 → 검증, 필요 시 자유 코드 폴백)")

    def _bridge(step: str, summary: str, detail: dict | None = None) -> None:
        # 스크리닝/백테스트 도메인은 on_progress를 (step, summary, detail=...) 3인자로도
        # 호출한다(domain_kr.py/domain_backtest.py의 실시간 트리 코드블록 통지) — detail을
        # 안 받으면 TypeError가 dispatch_domains에서 조용히 삼켜져 "유효 데이터 없음"으로
        # 오판되고, 스크리닝/백테스트가 걸린 질문은 신규 턴에서 전부 실패했다(실사용 재현).
        if on_progress:
            on_progress(f"{step}: {summary}")

    result = verified_fn(question, conn, llm_fn, on_progress=_bridge if on_progress else None)
    if not result.get("uncertain"):
        domain_results = result.get("domain_results") or {}
        session.current_data = domain_results
        session.has_data = True
        free_exec = domain_results.get("free_exec", {}) if result.get("used_fallback") else {}
        turn = Turn(
            question=question, status="success", answer=result.get("conclusion"),
            data=_extract_tabular_data(domain_results),
            sql=free_exec.get("sql"), code=free_exec.get("code"),
            domain_evidence=domain_results or None,
        )
        if on_progress:
            on_progress("완료")
    else:
        turn = Turn(question=question, status="fail", error=result.get("reason"))
        if on_progress:
            on_progress(f"실패: {result.get('reason')}")
    session.turns.append(turn)
    return turn


def run_turn(
    session: ConversationSession,
    question: str,
    conn,
    llm_fn: Callable[[str], str] | None,
    execute_python_fn: Callable | None = None,
    max_code_attempts: int = 2,
    verified_fn: Callable | None = None,
    classify_fn: Callable | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> Turn:
    """세션에 데이터가 없으면(또는 있어도 무관한 새 주제로 판단되면) 신규(정형 도메인 검증,
    필요시 내부 자유코드 폴백까지 자동)로, 직전 데이터와 이어지는 질문이면 이어가기(Python만)로
    턴을 실행한다.

    실패한 턴은 session.current_data/has_data를 건드리지 않는다 — 마지막 성공 데이터까지만
    다음 턴의 맥락으로 이어진다(실패 사유는 그 즉시 재시도 프롬프트에만 쓰이고, 다음 턴에는
    전달되지 않는다). on_progress(선택)는 오른쪽 패널 실시간 진행상황 표시용 콜백이다 —
    생략해도(기존 호출부) 동작은 그대로다.
    """
    verified_fn = verified_fn or answer_with_verification
    classify_fn = classify_fn or _classify_topic

    if not session.has_data:
        return _run_new_turn(session, question, conn, llm_fn, verified_fn, on_progress)

    if classify_fn(question, session.current_data, llm_fn):
        if on_progress:
            on_progress("이전 대화와 무관한 새 주제로 판단 — 새로 조회합니다")
        return _run_new_turn(session, question, conn, llm_fn, verified_fn, on_progress)

    followup = _run_followup_step(
        question, session.current_data, llm_fn, execute_python_fn, max_code_attempts, on_progress
    )
    if followup.get("ok"):
        session.current_data = followup["result"]
        turn = Turn(question=question, status="success", answer=followup["result"], code=followup.get("code"))
        session.turns.append(turn)
        return turn

    # 이어가기 실패 — _classify_topic이 "새 주제"를 놓쳤을 가능성이 높다(LLM이 분류 프롬프트의
    # CONTINUE/NEW 지시를 항상 정확히 따르지는 않는다, 실사용 확인됨). 사용자에게 실패만
    # 보여주고 끝내는 대신, 신규 조회(_run_new_turn)로 한 번 더 자동 시도한다 — 정말 무관한
    # 새 주제였다면 이걸로 성공하고, 진짜 이어가기 질문인데 코드가 실패한 거라면(가공 로직
    # 자체의 버그) 신규 조회도 맥락 없이는 어차피 못 풀어 결국 실패로 끝나되, 이번엔 raw
    # Python 에러 대신 "질문을 이해하지 못했습니다" 같은 더 명확한 사유로 끝난다.
    if on_progress:
        on_progress(f"이어가기 실패({followup.get('error')}) — 무관한 새 주제일 수 있어 새로 조회합니다")
    return _run_new_turn(session, question, conn, llm_fn, verified_fn, on_progress)


def get_history(session: ConversationSession) -> list[dict]:
    return [
        {
            "question": turn.question,
            "status": turn.status,
            "answer": turn.answer if turn.status == "success" else None,
            "error": turn.error,
            "sql": turn.sql,
            "code": turn.code,
            "domain_evidence": turn.domain_evidence,
            "csv_available": turn.status == "success" and _to_dataframe(_turn_csv_source(turn)) is not None,
        }
        for turn in session.turns
    ]


def _to_dataframe(data) -> pd.DataFrame | None:
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return None
    try:
        return pd.DataFrame(data)
    except Exception:
        return None


def turn_to_csv(session: ConversationSession, turn_index: int) -> str:
    if turn_index < 0 or turn_index >= len(session.turns):
        raise ValueError(f"턴 인덱스 {turn_index}가 존재하지 않습니다(세션에 {len(session.turns)}개 턴).")
    turn = session.turns[turn_index]
    if turn.status != "success":
        raise ValueError("실패한 턴은 CSV로 내려받을 수 없습니다.")
    df = _to_dataframe(_turn_csv_source(turn))
    if df is None:
        raise ValueError("이 턴의 결과는 표 형태가 아니라 CSV로 변환할 수 없습니다.")
    return df.to_csv(index=False)
