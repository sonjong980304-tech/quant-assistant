"""계층형 총괄 그래프 SSE 스트리밍 엔드포인트 테스트 (HA-12, TDD).

GET /api/query/stream?question=...&model=... 을 FastAPI TestClient로 검증한다.
- src.agents.graph.run_streaming 의 진행 이벤트({"step","summary"})를 SSE data 프레임으로
  순서대로 방출하는지.
- 요청 파싱(빈 질문 400 / 누락 422 / question·model 전달).
- run_streaming 을 fake로 주입해 실제 LLM/DB 없이 단위 검증(webapp.run_streaming monkeypatch).

fastapi 미설치 환경(셸 기본 python3)에서는 통째로 skip 한다(test_macro_web 관례와 동일).
공유 venv(/Users/gyuyeong/projects/.venv)에서 실행하면 통과한다.
"""
from __future__ import annotations

import json
import types

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import web.app as webapp  # noqa: E402


def _dummy_conn():
    """connect_readonly 대체용 최소 더미(닫기만 지원). fake run_streaming 은 conn 을 안 쓴다."""
    return types.SimpleNamespace(close=lambda: None)


@pytest.fixture
def client(monkeypatch):
    # DB/LLM 없이 단위 검증: 읽기전용 연결과 llm_fn 빌더를 더미/None 으로 대체(네트워크 차단).
    monkeypatch.setattr(webapp, "connect_readonly", lambda *a, **k: _dummy_conn())
    monkeypatch.setattr(webapp, "_build_llm_fn", lambda model: None)
    return TestClient(webapp.app)


# ── _sse_message : 진행 이벤트 → SSE data 프레임(한글 미이스케이프) ──────────────

def test_sse_message_formats_event_frame():
    out = webapp._sse_message({"step": "supervisor", "summary": "한국 도메인 라우팅, 검증 통과(2회 시도)"})
    assert out.startswith("data: ")
    assert out.endswith("\n\n")
    payload = json.loads(out[len("data: "):].strip())
    assert payload == {"step": "supervisor", "summary": "한국 도메인 라우팅, 검증 통과(2회 시도)"}
    # 한글이 \uXXXX 로 이스케이프되지 않는다.
    assert "한국" in out


# ── 스트리밍 : run_streaming 이벤트를 SSE 로 순서대로 방출 + done 종료 ───────────

def test_query_stream_emits_events_as_sse(client, monkeypatch):
    def fake_run_streaming(question, conn, llm_fn=None, steps=None):
        yield {"step": "supervisor", "summary": "한국+미국 도메인 라우팅, 검증 통과(2회 시도)"}

    monkeypatch.setattr(webapp, "run_streaming", fake_run_streaming)
    r = client.get("/api/query/stream", params={"question": "삼성전자 vs 애플"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    body = r.text
    assert 'data: {"step": "supervisor"' in body
    assert "한국+미국 도메인 라우팅" in body
    # 종료 마커(done)로 끝난다 → 프론트가 EventSource 를 닫아 재연결(재실행) 방지.
    assert "event: done" in body


def test_query_stream_emits_multiple_events_in_order(client, monkeypatch):
    seq = [
        {"step": "supervisor", "summary": "A"},
        {"step": "kr", "summary": "B"},
        {"step": "verify", "summary": "C"},
    ]
    monkeypatch.setattr(webapp, "run_streaming",
                        lambda q, conn, llm_fn=None, steps=None: iter(seq))
    body = client.get("/api/query/stream", params={"question": "q"}).text
    ia, ib, ic = body.index('"A"'), body.index('"B"'), body.index('"C"')
    assert ia < ib < ic   # 방출 순서 보존


# ── 요청 파싱 : 빈 질문 400 / 누락 422 / question·model 전달 ────────────────────

def test_query_stream_empty_question_returns_400(client, monkeypatch):
    monkeypatch.setattr(webapp, "run_streaming", lambda *a, **k: iter([]))
    r = client.get("/api/query/stream", params={"question": "   "})
    assert r.status_code == 400


def test_query_stream_missing_question_returns_422(client):
    r = client.get("/api/query/stream")   # question 은 필수 쿼리 파라미터
    assert r.status_code == 422


def test_query_stream_threads_question_to_run_streaming(client, monkeypatch):
    captured = {}

    def fake_run_streaming(question, conn, llm_fn=None, steps=None):
        captured["question"] = question
        captured["conn"] = conn
        captured["llm_fn"] = llm_fn
        return iter([{"step": "supervisor", "summary": "ok"}])

    monkeypatch.setattr(webapp, "run_streaming", fake_run_streaming)
    client.get("/api/query/stream", params={"question": "삼성전자 PER"})
    assert captured["question"] == "삼성전자 PER"
    assert captured["conn"] is not None       # connect_readonly 더미가 전달됨
    assert captured["llm_fn"] is None          # 픽스처가 _build_llm_fn→None 으로 스텁


def test_query_stream_builds_llm_fn_with_selected_model(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(webapp, "_build_llm_fn",
                        lambda model: seen.setdefault("model", model) or None)
    monkeypatch.setattr(webapp, "run_streaming",
                        lambda q, conn, llm_fn=None, steps=None: iter([]))
    client.get("/api/query/stream", params={"question": "q", "model": "gpt-5.5"})
    assert seen["model"] == "gpt-5.5"


# ── 에러 격리 : run_streaming 도중 예외 → fail 이벤트로 알림(연결은 정상 종료) ──

def test_query_stream_emits_fail_event_on_exception(client, monkeypatch):
    def boom(question, conn, llm_fn=None, steps=None):
        yield {"step": "supervisor", "summary": "시작"}
        raise RuntimeError("도메인 폭발")

    monkeypatch.setattr(webapp, "run_streaming", boom)
    body = client.get("/api/query/stream", params={"question": "q"}).text
    assert "event: fail" in body
    assert "도메인 폭발" in body


# ── 라우팅 회귀 : 기존 /api/query 는 그대로, 신규 경로는 별개(가로채지 않음) ────

def test_query_stream_route_is_distinct_from_api_query():
    paths = {getattr(rt, "path", None): getattr(rt, "endpoint", None) for rt in webapp.app.routes}
    assert "/api/query" in paths
    assert "/api/query/stream" in paths
    # 기존 동기 엔드포인트 함수가 그대로 바인딩됨(대체/수정 아님).
    assert paths["/api/query"].__name__ == "api_query"
    assert paths["/api/query/stream"].__name__ == "api_query_stream"
