"""FastAPI 웹 서버.

엔드포인트
  GET    /api/models       : 선택 가능한 LLM 목록 + 가용성
  POST   /api/query        : 자연어 질의 → 계층형 총괄 그래프 실행(종합결론 + 도메인별 원본결과).
                             uncertain=True 로 끝나도 legacy 로 자동 폴백하지 않고 그대로 반환한다.
  GET    /api/query/stream : 계층형 총괄 그래프 스트리밍(SSE, text/event-stream)
                             — 노드 진행 이벤트({"step","summary"})를 실시간 방출(HA-12)
  GET    /api/wiki        : 기록 목록 (?tag=)   ← 질의 기록 로그
  GET    /api/wiki/{id}   : 기록 상세
  PUT    /api/wiki/{id}   : SQL 수정 + 검증/평가 표시 (+태그)
  DELETE /api/wiki/{id}   : 삭제
  GET    /api/stats       : 기록 통계
  GET    /api/macro        : 거시지표(환율/해외·국내 지수/파마프렌치 5팩터)
  GET    /api/macro/signal : 거시지표 파생 신호
  GET    /api/macro/history: 거시지표 히스토리
  GET    /api/metric-defs  : 백테스트 UI용 지표 정의 목록
  GET    /api/sectors      : 백테스트 UI용 업종(KRX 업종분류) 목록
  POST   /api/backtest     : 백테스트 실행
  GET    /api/backtest-runs: 저장된 백테스트 이력
  GET    /                : 프론트(질의/기록 단일 페이지)
  GET    /macro            : 매크로 전용 페이지

참고: /api/wiki 경로는 하위호환을 위해 유지하지만, 의미는 '질의 기록 로그'다.
      신규 계층형 질의 경로(POST /api/query)는 이 기록에 자동 저장하지 않는다.

실행: uvicorn web.app:app --reload
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agents.conversation import get_history, get_or_create_session, get_session, reset_session, run_turn, turn_to_csv
from src.agents.domain_backtest import answer_backtest_question
from src.agents.domain_kr import answer_kr_screening
from src.agents.graph import run_hierarchical, run_streaming
from src.config import CONFIG
from src.db import connect, connect_readonly
from src.ingest.exchange_rate import fetch_usdkrw_rate_live
from src.ingest.macro_signal import classify_vkospi_band
from src.wiki.store import WikiStore

app = FastAPI(title="Quant Assistant")

STATIC_DIR = Path(__file__).parent / "static"

# sqlite conn은 스레드 귀속이므로 요청마다 새 읽기전용 연결을 연다(계층형 그래프가 소비).


# ---------------------------------------------------------------------------
# 스키마
# ---------------------------------------------------------------------------
class QueryReq(BaseModel):
    question: str
    model: Optional[str] = None


class RerunReq(BaseModel):
    """휴먼인더루프 재실행 요청 — 실시간 트리에서 본 조건JSON/파이프라인을 사용자가
    편집한 뒤 그대로 재실행한다(LLM 생성 단계는 건너뛴다).

    kind="screening": domain(kr) + spec(criteria/top_n/sectors/markets 등, 편집된 값)
      + asof(선택, 기간 재지정).
    kind="backtest": steps(편집된 파이프라인 JSON) + market(KR).
    """
    kind: str
    question: str
    model: Optional[str] = None
    domain: Optional[str] = None
    spec: Optional[dict] = None
    asof: Optional[str] = None
    steps: Optional[list] = None
    market: str = "KR"


# 선택 가능한 LLM 후보 (OpenAI + 로컬 Ollama)
LLM_MODELS = [
    {"id": "gpt-5.4-mini", "label": "GPT-5.4-mini (OpenAI)", "provider": "openai"},
    {"id": "gpt-5.5", "label": "GPT-5.5 (OpenAI)", "provider": "openai"},
    {"id": "exaone3.5:7.8b", "label": "EXAONE (로컬)", "provider": "ollama"},
    {"id": "qwen2.5-coder:7b-instruct-q4_K_M", "label": "Qwen2.5-coder (로컬)", "provider": "ollama"},
]


class WikiUpdate(BaseModel):
    sql: Optional[str] = None
    verified: Optional[bool] = None
    tags: Optional[str] = None


class ChatReq(BaseModel):
    session_id: Optional[str] = None
    question: str
    model: Optional[str] = None


class ChatResetReq(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
# /api/models 캐시 — 매 요청마다 각 모델의 가용성을 네트워크로 확인하면 짧은 간격의
# 반복 요청(프론트 폴링 등)에서 낭비가 크다. 60초 동안은 캐시된 값을 그대로 돌려준다.
_MODELS_CACHE: dict = {"data": None, "fetched_at": 0.0}
_MODELS_CACHE_TTL_SECONDS = 60.0


@app.get("/api/models")
def api_models():
    """선택 가능한 LLM 후보 + 각 가용성(60초 캐싱)."""
    now = time.time()
    if _MODELS_CACHE["data"] is not None and (now - _MODELS_CACHE["fetched_at"]) < _MODELS_CACHE_TTL_SECONDS:
        return _MODELS_CACHE["data"]

    from src.llm import LLMClient

    out = []
    for m in LLM_MODELS:
        try:
            available = LLMClient(model=m["id"]).available
        except Exception:
            available = False
        out.append({**m, "available": available})

    _MODELS_CACHE["data"] = out
    _MODELS_CACHE["fetched_at"] = now
    return out


@app.post("/api/query")
def api_query(req: QueryReq):
    """자연어 질의를 계층형 총괄 그래프(run_hierarchical)로 실행한다.

    총괄→도메인→검증→(최대 3회)재시도를 수행하고, 종합결론(conclusion)과 라우팅된
    도메인별 원본 결과(domain_results, 가공 없음)를 함께 담은 dict를 그대로 돌려준다
    (AC4). 3회 재시도 후에도 uncertain=True 로 끝나면 그 사실 그대로 반환한다(프론트가
    "확신 못함" 경고로 렌더링) — legacy 6단계 Pipeline 으로 자동 폴백하지 않는다.

    (과거 HA-16에서 legacy 자동 폴백 안전망을 붙였으나, 신규 구조가 스스로 정확히
    답하지 못하는 원인을 legacy 우회로 가리는 대신 근본 원인을 고치는 쪽으로 방향을
    바꿔 폴백 로직을 제거했다. legacy 파이프라인(src/legacy/, cli.py)은 그 뒤로도
    한동안 별도 CLI 진입점으로 남아 있었으나, 그마저 쓰임이 없어져 완전히 삭제했다.)

    conn 은 요청마다 새 읽기전용 연결(connect_readonly) — 도메인/데이터 에이전트가 LLM 생성
    SQL 을 실행하므로 읽기전용 연결을 요구한다. llm_fn 은 HA-12 가 만든 _build_llm_fn 어댑터를
    재사용한다(가용하지 않으면 None → 결정론적 휴리스틱 폴백).
    """
    if not req.question.strip():
        raise HTTPException(400, "질문이 비어 있습니다.")
    conn = connect_readonly()
    try:
        llm_fn = _build_llm_fn(req.model)
        result = run_hierarchical(
            req.question, conn, llm_fn=llm_fn,
            chart_llm_fn=_build_llm_fn(req.model, role="chart"),
        )
    except Exception as exc:  # noqa: BLE001 — 원인을 감추지 않고 그대로 500으로 노출
        raise HTTPException(500, f"계층형 에이전트 실행 실패: {type(exc).__name__}: {exc}") from exc
    finally:
        conn.close()
    return {**result, "answered_by": "hierarchical"}


@app.post("/api/query/rerun")
def api_query_rerun(req: RerunReq):
    """휴먼인더루프 재실행: 사용자가 편집한 조건JSON(스크리닝) 또는 파이프라인(백테스트)을
    LLM 생성 단계 없이 그대로 실행한다.

    실시간 트리(GET /api/query/stream)가 detail로 노출한 "AI가 만든 코드"를 사용자가
    직접 고쳐 이 엔드포인트로 재실행 버튼을 누르면, override_spec/steps로 그대로 넘어와
    LLM을 다시 부르지 않고 결정론적 실행기만 탄다 — 안전장치(존재하지 않는 지표명 거부,
    감사 배선의 하드/소프트 검사 등)는 정상 질의 경로와 완전히 동일하게 적용된다.
    """
    if not req.question.strip():
        raise HTTPException(400, "질문이 비어 있습니다.")
    if req.kind == "screening":
        if req.domain != "kr":
            raise HTTPException(400, "domain은 'kr'여야 합니다.")
        if not isinstance(req.spec, dict):
            raise HTTPException(400, "재실행할 spec(조건 JSON)이 필요합니다.")
    elif req.kind == "backtest":
        if not isinstance(req.steps, list):
            raise HTTPException(400, "재실행할 steps(파이프라인 JSON)가 필요합니다.")
    else:
        raise HTTPException(400, f"알 수 없는 kind: {req.kind!r} (screening|backtest만 가능)")

    conn = connect_readonly()
    try:
        llm_fn = _build_llm_fn(req.model)
        if req.kind == "screening":
            result = answer_kr_screening(
                req.question, conn, llm_fn=llm_fn, override_spec=req.spec, asof=req.asof,
            )
        else:
            result = answer_backtest_question(
                req.question, req.steps, conn, llm_fn=llm_fn, market=req.market,
            )
    except Exception as exc:  # noqa: BLE001 — 원인을 감추지 않고 그대로 500으로 노출
        raise HTTPException(500, f"재실행 실패: {type(exc).__name__}: {exc}") from exc
    finally:
        conn.close()
    return result


# ---------------------------------------------------------------------------
# 멀티턴 대화 (.omc/specs/brainstorming-multiturn-conversation.md, MT-6)
# 세션은 src.agents.conversation의 프로세스 메모리 저장소에만 존재한다(서버 재시작 시
# 소실 — 설계상 허용, AC8). run_turn이 세션에 데이터가 없으면 신규(SQL+Python), 있으면
# 이어가기(Python만)로 자동 분기하므로 여기서는 얇게 배선만 한다.
# ---------------------------------------------------------------------------
@app.post("/api/chat")
def api_chat(req: ChatReq):
    """멀티턴 대화 턴 실행. conn은 요청마다 새 읽기전용 연결(신규 턴에서만 실제 SQL 실행)."""
    if not req.question.strip():
        raise HTTPException(400, "질문이 비어 있습니다.")
    session = get_or_create_session(req.session_id)
    conn = connect_readonly()
    try:
        llm_fn = _build_llm_fn(req.model)
        turn = run_turn(
            session, req.question, conn, llm_fn,
            chart_llm_fn=_build_llm_fn(req.model, role="chart"),
        )
    except Exception as exc:  # noqa: BLE001 — 원인을 감추지 않고 그대로 500으로 노출(api_query와 동일 관례)
        raise HTTPException(500, f"멀티턴 턴 실행 실패: {type(exc).__name__}: {exc}") from exc
    finally:
        conn.close()
    return {
        "session_id": session.session_id,
        "status": turn.status,
        "answer": turn.answer,
        "error": turn.error,
        "sql": turn.sql,
        "code": turn.code,
        "domain_evidence": turn.domain_evidence,
        "chart_base64": turn.chart_base64,
        "chart_title": turn.chart_title,
    }


@app.post("/api/chat/reset")
def api_chat_reset(req: ChatResetReq):
    """대화 초기화 — 누적 맥락(메모리 데이터)을 폐기하고 새 세션 상태로 되돌린다(AC7)."""
    reset_session(req.session_id)
    return {"session_id": req.session_id, "reset": True}


@app.get("/api/chat/history")
def api_chat_history(session_id: str):
    """턴별 질문/답변 + CSV 재다운로드 가능여부(AC9-AC10). 모르는 session_id는 빈 이력."""
    session = get_session(session_id)
    if session is None:
        return {"session_id": session_id, "turns": []}
    return {"session_id": session_id, "turns": get_history(session)}


@app.get("/api/chat/csv")
def api_chat_csv(session_id: str, turn_index: int):
    """턴의 최종 데이터를 CSV로 다운로드. 세션이 메모리에서 사라졌으면(재시작 등) 404 —
    이것이 "서버 재시작 후 CSV 재다운로드 불가"(AC10)의 실제 동작 근거다."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(404, "세션을 찾을 수 없습니다(서버 재시작 등으로 소실됐을 수 있습니다).")
    try:
        csv_text = turn_to_csv(session, turn_index)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return StreamingResponse(
        iter([csv_text]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=turn_{turn_index}.csv"},
    )


def _chat_event_stream(session_id: Optional[str], question: str, model: Optional[str]):
    """run_turn의 on_progress 콜백을 SSE 프레임으로 흘려보낸다(AC13).

    run_turn 자체는 동기 함수라, 별도 스레드에서 실행하며 진행 메시지를 큐에 쌓고
    이 제너레이터가 그 큐를 순서대로 소비해 SSE로 방출한다(_sse_message 프레이밍 재사용).
    스레드 안에서 connect_readonly()로 자체 연결을 새로 여므로 스레드 간 커넥션 공유가
    없다(sqlite3 기본 스레드 제약 회피). exec_fallback.py 자체는 건드리지 않으므로 신규
    턴(SQL+Python)은 굵은 단위로, 이어가기(conversation.py 자체 구현)는 시도별로 세밀하게
    보고된다 — src/agents/conversation.py의 on_progress 참고.
    """
    import queue
    import threading

    session = get_or_create_session(session_id)
    q: queue.Queue = queue.Queue()

    def worker():
        conn = None
        try:
            conn = connect_readonly()
            llm_fn = _build_llm_fn(model)
            turn = run_turn(
                session, question, conn, llm_fn, on_progress=lambda msg: q.put({"step": msg}),
                chart_llm_fn=_build_llm_fn(model, role="chart"),
            )
            q.put({"_done": True, "status": turn.status, "answer": turn.answer, "error": turn.error,
                   "sql": turn.sql, "code": turn.code, "domain_evidence": turn.domain_evidence,
                   "chart_base64": turn.chart_base64, "chart_title": turn.chart_title})
        except Exception as exc:  # noqa: BLE001 — 스트림 도중 실패를 클라이언트에 명시적으로 알림
            q.put({"_fail": True, "message": str(exc)})
        finally:
            if conn is not None:
                conn.close()
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    yield _sse_message({"step": f"세션 {session.session_id} — 처리 시작"})
    while True:
        item = q.get()
        if item is None:
            break
        if item.pop("_done", False):
            yield f"event: done\ndata: {json.dumps({**item, 'session_id': session.session_id}, ensure_ascii=False)}\n\n"
        elif item.pop("_fail", False):
            yield f"event: fail\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
        else:
            yield _sse_message(item)


@app.get("/api/chat/stream")
def api_chat_stream(question: str, session_id: Optional[str] = None, model: Optional[str] = None):
    """멀티턴 대화 턴을 SSE로 스트리밍 실행 — 진행 이벤트가 도착하는 즉시 방출된다(AC13)."""
    if not question.strip():
        raise HTTPException(400, "질문이 비어 있습니다.")
    return StreamingResponse(
        _chat_event_stream(session_id, question, model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# 계층형 총괄 그래프 스트리밍 (HA-12) — SSE(text/event-stream)
# ---------------------------------------------------------------------------
# src/agents/graph.py 의 run_streaming(question, conn, llm_fn=..., steps=...) 이
# 노드 완료마다 방출하는 진행 이벤트({"step","summary"})를 그대로 SSE 프레임으로 흘린다.
# 이벤트 스키마는 이미 상세(SQL 전문/원본 rows/결론 본문)를 제외하도록 설계돼 있으므로
# (AC15) 여기서 추가 가공 없이 순서대로 전달만 한다. 프론트(index.html)는 EventSource 로
# 구독해 들여쓰기 트리로 실시간 렌더한다.
#
# 동기 POST /api/query 는 HA-13 에서 이 run_hierarchical 기반으로 전환됐다(위 api_query 참고).
def _sse_message(event: dict) -> str:
    """진행 이벤트({"step","summary"}) 한 건을 SSE data 프레임으로 직렬화한다.

    ensure_ascii=False 로 한글이 \\uXXXX 로 깨지지 않게 한다. SSE 프레임 구분은 빈 줄(\\n\\n).
    """
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _build_llm_fn(model: Optional[str], role: str = "sql"):
    """LLMClient 를 도메인 에이전트 규약(Callable[[str], str])으로 감싼다.

    가용하지 않으면(키/데몬 없음) None 을 반환해 총괄·도메인 로직이 결정론적 휴리스틱으로
    폴백하게 한다. 총괄→도메인이 쓰는
    주 작업이 SQL·파이썬 코드 생성이라 기본 role="sql"(openai_model_sql 티어)로 모델을 고른다. 차트
    종류 판정처럼 저가 모델로 충분한 호출부는 role="chart"를 넘긴다 — LLMClient.model_for가
    sql/judge/diagnose 어디에도 안 걸리는 role을 기본 저가 모델(openai_model)로 떨어뜨리므로,
    별도 티어 신설 없이 차트 판단만 저가 모델을 쓰게 된다(sql/judge 티어는 그대로).
    """
    from src.llm import LLMClient

    client = LLMClient(model=model)
    if not client.available:
        return None
    return lambda prompt: (client.complete(prompt, role=role).text or "")


def _query_event_stream(question: str, model: Optional[str]):
    """run_streaming 의 진행 이벤트를 SSE 프레임으로 순서대로 방출하는 제너레이터.

    conn 은 요청마다 새 읽기전용 연결(connect_readonly) — 도메인/데이터 에이전트가
    LLM 생성 SQL 을 실행하므로 읽기전용 연결을 요구한다(src/agents/exec_runtime.py).
    스트림이 정상 종료되면 `event: done` 마커에 이 실행의 최종 상태(conclusion/domain_results/
    routes/uncertain/attempts/chart_*/...)를 실어 보낸다 — 프론트가 EventSource 를 닫게 하고
    (미종료 시 EventSource 가 자동 재연결→재실행하는 것을 막는다) 동시에 최종 답변도 함께
    받는다. (과거에는 done이 빈 데이터만 보내, 프론트가 최종 답변을 얻으려 POST /api/query를
    한 번 더 호출해 동일 질문을 두 번 계산했다 — 비용 2배 + 화면 진행상황과 실제 답이 달라질
    수 있는 문제였다. run_streaming(out_final=...)로 이 실행 한 번에서 진행 이벤트와 최종
    상태를 모두 얻어, 실행 경로를 하나로 합쳤다.) 도중 실패는 `event: fail`로 알린다
    (api_macro 의 "실패 격리, 응답은 유지" 관례와 동일 철학).
    """
    conn = None
    final: Dict[str, Any] = {}
    try:
        conn = connect_readonly()
        llm_fn = _build_llm_fn(model)
        for event in run_streaming(
            question, conn, llm_fn=llm_fn, out_final=final,
            chart_llm_fn=_build_llm_fn(model, role="chart"),
        ):
            yield _sse_message(event)
        payload = {**final, "answered_by": "hierarchical"}
        yield f"event: done\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
    except Exception as exc:  # noqa: BLE001 — 스트림 도중 실패를 클라이언트에 명시적으로 알림
        yield f"event: fail\ndata: {json.dumps({'message': str(exc)}, ensure_ascii=False)}\n\n"
    finally:
        if conn is not None:
            conn.close()


@app.get("/api/query/stream")
def api_query_stream(question: str, model: Optional[str] = None):
    """계층형 총괄 그래프를 스트리밍 실행해 노드 진행 이벤트를 SSE 로 방출한다.

    요청:  GET /api/query/stream?question=<질문>&model=<선택 LLM id>
    응답:  text/event-stream — 이벤트마다 `data: {"step","summary"}\\n\\n`,
           정상 종료 `event: done`(data에 최종 상태 — conclusion/domain_results/routes/
           uncertain/attempts/... 포함, POST /api/query 응답과 동일 형태), 실패 `event: fail`.
           이 스트리밍 실행 한 번이 진행 이벤트와 최종 답변을 모두 내어주므로, 프론트는
           이 응답만으로 렌더링을 완결할 수 있다(POST /api/query 를 별도로 또 호출해
           동일 질문을 두 번 계산할 필요가 없다).
    """
    if not question.strip():
        raise HTTPException(400, "질문이 비어 있습니다.")
    return StreamingResponse(
        _query_event_stream(question, model),
        media_type="text/event-stream",
        # 프록시/브라우저 버퍼링으로 실시간성이 깨지지 않게 캐시·버퍼링을 끈다.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/wiki")
def api_wiki_list(tag: Optional[str] = None):
    conn = connect()
    try:
        return WikiStore(conn).list_pages(tag=tag)
    finally:
        conn.close()


@app.get("/api/wiki/{wiki_id}")
def api_wiki_get(wiki_id: int):
    conn = connect()
    try:
        page = WikiStore(conn).get_page(wiki_id)
        if not page:
            raise HTTPException(404, "없는 항목")
        return page
    finally:
        conn.close()


@app.put("/api/wiki/{wiki_id}")
def api_wiki_update(wiki_id: int, body: WikiUpdate):
    conn = connect()
    try:
        store = WikiStore(conn)
        if store.get_page(wiki_id) is None:
            raise HTTPException(404, "없는 항목")
        if body.sql is not None:
            store.update_sql(wiki_id, body.sql, verified=body.verified if body.verified is not None else True)
        elif body.verified is not None:
            store.set_verified(wiki_id, body.verified)
        if body.tags is not None:
            store.set_tags(wiki_id, body.tags)
        return store.get_page(wiki_id)
    finally:
        conn.close()


@app.delete("/api/wiki/{wiki_id}")
def api_wiki_delete(wiki_id: int):
    conn = connect()
    try:
        ok = WikiStore(conn).delete(wiki_id)
        if not ok:
            raise HTTPException(404, "없는 항목")
        return {"deleted": wiki_id}
    finally:
        conn.close()


@app.get("/api/stats")
def api_stats():
    conn = connect()
    try:
        return WikiStore(conn).stats()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 거시지표 티커 (환율 / 해외·국내 지수 / 파마프렌치 5팩터)
# ---------------------------------------------------------------------------
_MACRO_INDICES = [
    {"ticker": "^IXIC", "label": "나스닥종합"},
    {"ticker": "^DJI", "label": "다우존스"},
    {"ticker": "^GSPC", "label": "S&P500"},
    {"ticker": "^SOX", "label": "필라델피아반도체"},
    {"ticker": "^KS11", "label": "코스피"},
    {"ticker": "^KQ11", "label": "코스닥"},
]


def _fetch_index_quote(ticker: str) -> dict:
    import yfinance as yf  # 지연 import — 무거운 라이브러리는 필요 시점에만 로드한다

    hist = yf.Ticker(ticker).history(period="5d")
    closes = hist["Close"].dropna()
    if len(closes) < 2:
        raise ValueError("최근 종가 2개를 확보하지 못함")
    last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
    return {"close": last, "change_pct": round((last - prev) / prev * 100, 2)}


# 코스피/코스닥은 yfinance(^KS11/^KQ11)가 며칠씩 데이터가 안 붙는 지연·결측이 실서버에서
# 확인됐다(2026-07-13 이후 결측 → 등락률이 -8.95%로 왜곡, 실제 당일 등락은 +0.73%).
# 개별종목 가격에 이미 적용 중인 "네이버가 source of truth" 원칙(naver_prices.py)을
# 지수에도 동일하게 적용 — 네이버 실시간 지수 API(delayTime=0, 장마감 즉시 확정치 반영)로
# 대체한다. 나머지 4개(나스닥/다우/S&P500/필라델피아반도체) 미국 지수는 기존 yfinance
# 경로를 그대로 쓴다(같은 지연 문제가 보고된 바 없어 스코프를 좁힘).
_KR_INDEX_NAVER_CODES = {"^KS11": "KOSPI", "^KQ11": "KOSDAQ"}


def _fetch_krx_index_quote(naver_code: str) -> dict:
    """네이버 금융 실시간 지수 API로 코스피/코스닥 종가·등락률을 가져온다.

    응답 필드가 "6,856.83" 같은 쉼표 포함 문자열이라 그대로 float 변환하면 깨진다.
    """
    url = f"https://polling.finance.naver.com/api/realtime/domestic/index/{naver_code}"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
    resp.raise_for_status()
    d = resp.json()["datas"][0]
    close = float(d["closePrice"].replace(",", ""))
    change_pct = float(d["fluctuationsRatio"].replace(",", ""))
    return {"close": close, "change_pct": change_pct}


@app.get("/api/macro")
def api_macro():
    """거시지표 티커 응답. 항목별로 실패를 격리해 하나가 죽어도 전체 응답은 유지한다
    (수집기 전반의 "종목별 실패 격리, 계속 진행" 관례와 동일).

    파마프렌치 팩터는 웹 티커에서 뺐다(사용자 요청) — 프롬프트로 직접 물어보면
    fama_french.py의 LLM 의도판단 경로로 여전히 조회 가능하다.
    """
    from datetime import datetime

    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    try:
        usdkrw: dict = {"rate": fetch_usdkrw_rate_live()}
    except Exception as exc:  # noqa: BLE001 — 환율 실패해도 나머지는 계속 응답
        usdkrw = {"error": str(exc)}

    indices = []
    for item in _MACRO_INDICES:
        naver_code = _KR_INDEX_NAVER_CODES.get(item["ticker"])
        try:
            quote = _fetch_krx_index_quote(naver_code) if naver_code else _fetch_index_quote(item["ticker"])
            indices.append({**item, **quote})
        except Exception as exc:  # noqa: BLE001 — 지수별 실패 격리
            indices.append({**item, "error": str(exc)})

    return {
        "fetched_at": fetched_at,
        "usdkrw": usdkrw,
        "indices": indices,
        "note": {"quotes": "코스피/코스닥은 네이버 실시간 시세(장중 실시간·장마감 후 확정치), 해외지수는 약 15-20분 지연"},
    }


# ---------------------------------------------------------------------------
# 매크로 신호 (장단기금리차 레짐 기반 GREEN/YELLOW/RED) — /macro 전용 페이지 지원 API.
# 위 /api/macro(환율+지수 티커)와 이름이 겹치지 않도록 하위 경로(/signal, /history)로
# 분리한다. FastAPI는 리터럴 경로를 정확 매칭하므로 /api/macro가 이 둘을 가로채지 않는다.
# 종합신호·밴드는 macro_signal 테이블에 이미 정규화돼 저장되므로 그 한 행을 그대로 노출한다.
# ---------------------------------------------------------------------------
def _signal_payload(row) -> dict:
    """macro_signal 한 행(sqlite Row 또는 None)을 프론트용 구조(지표별 값+밴드)로 변환.
    이력이 없으면(row=None) available=False에 전 필드 None인 동일 구조를 반환한다."""
    if row is None:
        return {
            "available": False, "as_of": None, "overall": None,
            "spread": {"value": None, "regime": None},
            "cnn": {"value": None, "band": None},
            "vix": {"value": None, "band": None},
            "created_at": None,
        }
    return {
        "available": True,
        "as_of": row["as_of"],
        "overall": row["overall"],
        "spread": {"value": row["spread"], "regime": row["spread_regime"]},
        "cnn": {"value": row["cnn_value"], "band": row["cnn_band"]},
        "vix": {"value": row["vix_value"], "band": row["vix_band"]},
        "created_at": row["created_at"],
    }


@app.get("/api/macro/signal")
def api_macro_signal():
    """최신 매크로 신호(overall)와 각 지표 현재값+밴드. 이력이 없으면 available=False로 200."""
    conn = connect()
    try:
        row = conn.execute(
            "SELECT as_of, spread, spread_regime, cnn_value, cnn_band, "
            "vix_value, vix_band, overall, created_at "
            "FROM macro_signal ORDER BY as_of DESC, id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    return _signal_payload(row)


@app.get("/api/macro/vkospi")
def api_macro_vkospi():
    """VKOSPI(코스피 200 변동성지수) 최신값 — 참고 표시 전용, macro_signal(종합신호)엔 미반영.

    macro_indicators(원시 지표 저장소)에서 직접 읽는다(macro_signal 경유 안 함) — VKOSPI는
    _INDICATORS(macro_pipeline.py)에 없어 신호 판정에 들어가지 않기 때문. 이력이 없으면
    available=False로 200(다른 macro API와 동일한 "미수집 시 조용히 빈 값" 관례).
    """
    conn = connect()
    try:
        row = conn.execute(
            "SELECT date, value FROM macro_indicators WHERE indicator='VKOSPI' "
            "ORDER BY date DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return {"available": False, "date": None, "value": None, "band": None}
    return {
        "available": True,
        "date": row["date"],
        "value": row["value"],
        "band": classify_vkospi_band(row["value"]),
    }


@app.get("/api/macro/history")
def api_macro_history(days: int = 30):
    """최근 N일 신호 이력(스파크라인용). 조회 후 과거→최신 순으로 정렬해 반환한다."""
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT as_of, spread, cnn_value, vix_value, overall "
            "FROM macro_signal ORDER BY as_of DESC, id DESC LIMIT ?",
            (days,),
        ).fetchall()
    finally:
        conn.close()
    series = [
        {"as_of": r["as_of"], "overall": r["overall"], "spread": r["spread"],
         "cnn": r["cnn_value"], "vix": r["vix_value"]}
        for r in reversed(rows)  # 최신순 조회 결과를 뒤집어 과거→최신(스파크라인 그리기 순서)
    ]
    return {"days": days, "series": series}


# ---------------------------------------------------------------------------
# 올웨더 포트폴리오 모니터링 — 저장된 스냅샷 읽기 전용 (AC11/AC14).
# 매달 1일 배치(scripts/run_all_weather.py)가 계산해 all_weather_snapshot에 저장한 값을
# 그대로 노출한다. 조회 시 즉석 재계산을 하지 않는다(macro_signal과 동일 캐싱 원칙).
# .omc/specs/brainstorming-all-weather-portfolio.md 참고.
# ---------------------------------------------------------------------------
@app.get("/api/allweather")
def api_allweather():
    """최신 올웨더 스냅샷(비중/MDD/샤프/누적수익률/CAGR + 자산곡선)을 DB에서 읽어 반환한다.

    화면은 이 저장값을 읽기만 하고 즉석 재계산하지 않는다(AC11). 이력이 없으면 available=False로 200.
    """
    from src.allweather.store import get_latest_snapshot

    conn = connect()
    try:
        snap = get_latest_snapshot(conn)
    finally:
        conn.close()
    if snap is None:
        return {
            "available": False, "computed_at": None, "weights": {},
            "cagr": None, "mdd": None, "sharpe": None, "sortino": None,
            "cumulative_return": None,
            "backtest_curve": [],
        }
    return {
        "available": True,
        "computed_at": snap["computed_at"],
        "weights": snap["weights"],
        "cagr": snap["cagr"],
        "mdd": snap["mdd"],
        "sharpe": snap["sharpe"],
        "sortino": snap["sortino"],
        "cumulative_return": snap["cumulative_return"],
        "backtest_curve": snap["backtest_curve"],
        "created_at": snap["created_at"],
    }


# ---------------------------------------------------------------------------
# 백테스트 (모듈 B)
# ---------------------------------------------------------------------------
class BacktestReq(BaseModel):
    domain: str = "kr"            # 'kr'(KOSPI/KOSDAQ). 기본 kr
    start_year: int = 2024
    end_year: int = 2026
    rebalance: str = "quarterly"
    n: int = 10
    criteria: list = []           # [{"key","direction","weight"}]
    combine: str = "zscore"
    sectors: Optional[list] = None
    markets: Optional[list] = None     # kr: ['KOSPI','KOSDAQ'] / None(전체)
    winsorize_z: Optional[float] = None  # z-score 이상치 완화 임계값(None이면 미적용, 기존 동작)
    winsorize_pct: Optional[float] = None  # 원본값 퍼센타일 winsorize(예: 0.01=상하위 1%, None이면 미적용)
    name: str = "전략"
    fee_rate: Optional[float] = None       # 수수료율(비율). None이면 서버 기본값
    tax_rate: Optional[float] = None       # 매도 거래세율(비율)
    slippage_rate: Optional[float] = None  # 슬리피지율(비율)


@app.get("/api/metric-defs")
def api_metric_defs():
    conn = connect()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT key,label,category,direction,description FROM metric_def")]
    finally:
        conn.close()


@app.get("/api/sectors")
def api_sectors(domain: str = "kr"):
    # company(KRX 업종분류). 그 외 값은 사용자 입력 오류(400).
    if domain != "kr":
        raise HTTPException(400, "domain은 'kr'여야 합니다.")
    table = "company"
    conn = connect()
    try:
        return [r["sector"] for r in conn.execute(
            f"SELECT DISTINCT sector FROM {table} WHERE sector IS NOT NULL AND sector != '' ORDER BY sector")]
    finally:
        conn.close()


@app.post("/api/backtest")
def api_backtest(req: BacktestReq):
    from src.backtest.data_access import build_benchmark_fn, build_callbacks, rebalance_dates
    from src.backtest.engine import run_backtest, save_backtest_run

    # domain(kr): KOSPI/KOSDAQ 국내 종목만 백테스트한다. 기본 'kr'로 기존 동작 100% 보존.
    if req.domain != "kr":
        raise HTTPException(400, "domain은 'kr'여야 합니다.")
    if not req.criteria:
        raise HTTPException(400, "지표를 1개 이상 선택하세요.")
    # markets 검증: 허용 시장 외 값이 섞이면 사용자 입력 오류(400). markets=None(전체)은 통과.
    allowed_markets = {"KOSPI", "KOSDAQ"}
    if req.markets:
        invalid = [m for m in req.markets if m not in allowed_markets]
        if invalid:
            raise HTTPException(
                400, f"허용되지 않는 시장 {invalid} (domain={req.domain}, 허용: {sorted(allowed_markets)})")
    # metric_def의 UI 체크박스 key는 이제 'return_12m'로 통일돼 있어 momentum을 안 보낸다.
    # 이 치환은 과거에 저장된 백테스트 설정(criteria에 momentum이 박혀 있는 경우) 등을 위한
    # 하위호환 방어 코드로만 남아있다 — 없어도 새 UI 흐름엔 영향 없다.
    criteria = [{**c, "key": "return_12m"} if c.get("key") == "momentum" else c
                for c in req.criteria]
    conn = connect()
    try:
        # 가격 최신일(리밸런싱 날짜 절단 기준)은 prices 테이블에서 뽑는다.
        maxd = conn.execute("SELECT MAX(date) FROM prices").fetchone()[0]
        full = rebalance_dates(req.start_year, req.end_year, req.rebalance)
        dates = [d for d in full if maxd and d <= maxd]
        # 진행 중(미완결) 구간 이어붙이기: 사용자가 요청한 마지막 리밸런싱이 아직 오지 않은
        # 미래라 잘렸고(예: 반기 전략에서 다음 리밸런싱 2026-07-31이 미도래) 마지막 완결
        # 리밸런싱 이후에도 가격 데이터가 더 있으면(dates[-1] < maxd), 현재 보유 중인
        # 포트폴리오의 미완결 구간을 데이터 최신일(maxd)까지 NAV에 이어 붙인다. 이렇게 하지
        # 않으면 차트가 마지막 완결 리밸런싱(예: 2026-01-31)에서 멈춰 사용자에게는 "왜 최근
        # 데이터까지 안 보이지?"로 보인다. 종료연도가 과거인 백테스트(잘린 미래 리밸런싱이
        # 없어 len(dates)==len(full))에는 영향이 없다.
        if maxd and len(dates) < len(full) and dates and dates[-1] < maxd:
            dates = dates + [maxd]
        if len(dates) < 2:
            raise HTTPException(400, "선택 기간에 데이터가 부족합니다(주가 시계열 범위 확인).")
        # 엔진 콜백/벤치마크(company/prices/financials 어댑터). 코스피·코스닥을 각각 독립된
        # 비교선으로 보여준다(사용자가 고른 markets 필터와 무관하게 항상 둘 다 계산).
        mfn, pfn = build_callbacks(conn)
        bench_fn = build_benchmark_fn(dates, mfn, pfn, code="KOSPI")
        bench_fn2 = build_benchmark_fn(dates, mfn, pfn, code="KOSDAQ")
        names_sql = "SELECT stock_code,name FROM company"
        params = {
            "n": req.n, "criteria": criteria, "combine": req.combine,
            "sectors": req.sectors, "markets": req.markets, "winsorize_z": req.winsorize_z,
            "winsorize_pct": req.winsorize_pct, "rebalance": req.rebalance,
            "fee_rate": req.fee_rate if req.fee_rate is not None else CONFIG.fee_rate,
            "tax_rate": req.tax_rate if req.tax_rate is not None else CONFIG.tax_rate,
            "slippage_rate": req.slippage_rate if req.slippage_rate is not None else CONFIG.slippage_rate,
        }
        res = run_backtest(dates, mfn, pfn, params, benchmark_fn=bench_fn, benchmark_fn2=bench_fn2)
        save_backtest_run(conn, req.name, params, res["performance"], req.start_year, req.end_year)
        names = {r["stock_code"]: r["name"] for r in conn.execute(names_sql)}
        holdings = [{"date": h["date"], "names": [names.get(c, c) for c in h["codes"]]}
                    for h in res["holdings"]]
        return {"dates": res["dates"], "navs": res["navs"], "benchmark": res.get("benchmark"),
                "benchmark2": res.get("benchmark2"),
                "performance": res["performance"], "holdings": holdings,
                "empty_periods": res.get("empty_periods", [])}
    except ValueError as e:
        # 존재하지 않는 지표 선택 등 사용자 입력 오류는 500(평문 'Internal Server Error')이 아니라
        # 400(JSON detail)으로 반환한다. 500 평문 본문은 프런트의 r.json() 파싱을 실패시켜
        # 브라우저가 'The string did not match the expected pattern.'라는 엉뚱한 에러를 띄웠다.
        raise HTTPException(400, str(e)) from e
    finally:
        conn.close()


@app.get("/api/backtest-runs")
def api_backtest_runs():
    import json

    conn = connect()
    try:
        rows = conn.execute(
            "SELECT id,name,start_year,end_year,result_json,created_at "
            "FROM backtest_runs ORDER BY id DESC LIMIT 20").fetchall()
        return [{"id": r["id"], "name": r["name"], "period": f'{r["start_year"]}~{r["end_year"]}',
                 "result": json.loads(r["result_json"]), "created_at": r["created_at"]}
                for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 프론트
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    """기본 화면 — 멀티턴 대화(chat.html). 기존 단발성 질의 화면은 /query로 이동했다."""
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/chat")
def chat_page():
    """멀티턴 대화 화면 — /의 별칭(북마크·기존 링크 호환용, 내용은 동일)."""
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/query")
def query_page():
    """기존 단발성 질의 화면(지표 조회 등, index.html) — 기본 화면 자리를 /에 내주고 이 경로로 이동."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/macro")
def macro_page():
    """매크로 신호 전용 페이지(index.html과 동일 패턴으로 정적 HTML 서빙)."""
    return FileResponse(STATIC_DIR / "macro.html")


@app.get("/allweather")
def allweather_page():
    """올웨더 포트폴리오 모니터링 페이지(macro.html과 동일 패턴으로 정적 HTML 서빙)."""
    return FileResponse(STATIC_DIR / "allweather.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
