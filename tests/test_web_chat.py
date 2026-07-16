"""멀티턴 대화 API 배선 테스트 (MT-6).

.omc/specs/brainstorming-multiturn-conversation.md 대응 웹 계층. src.agents.conversation의
run_turn 등 핵심 로직은 tests/test_agents_conversation.py에서 이미 단위테스트로 검증됐으므로,
여기서는 API 요청/응답 배선(세션 유지, 히스토리, CSV 다운로드)만 확인한다 — run_turn 자체는
fake로 대체한다.

fastapi 미설치 환경(셸 기본 python3)에서는 통째로 skip한다 — test_macro_web.py와 동일 관례.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import web.app as webapp  # noqa: E402
from src.agents.conversation import Turn, get_session  # noqa: E402


@pytest.fixture
def client():
    return TestClient(webapp.app)


def test_api_chat_calls_run_turn_and_returns_session_id(client, monkeypatch):
    captured = {}

    def fake_run_turn(session, question, conn, llm_fn, **kwargs):
        captured["question"] = question
        return Turn(question=question, status="success", answer=[{"a": 1}])

    monkeypatch.setattr(webapp, "run_turn", fake_run_turn)
    monkeypatch.setattr(webapp, "_build_llm_fn", lambda model: None)

    r = client.post("/api/chat", json={"question": "코스피 PBR 낮은 순"})

    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["answer"] == [{"a": 1}]
    assert data["session_id"]
    assert captured["question"] == "코스피 PBR 낮은 순"


def test_api_chat_reuses_same_session_across_requests(client, monkeypatch):
    seen_sessions = []

    def fake_run_turn(session, question, conn, llm_fn, **kwargs):
        seen_sessions.append(session.session_id)
        return Turn(question=question, status="success", answer=[{"a": 1}])

    monkeypatch.setattr(webapp, "run_turn", fake_run_turn)
    monkeypatch.setattr(webapp, "_build_llm_fn", lambda model: None)

    first = client.post("/api/chat", json={"question": "첫 질문"}).json()
    second = client.post("/api/chat", json={"session_id": first["session_id"], "question": "이어서"}).json()

    assert first["session_id"] == second["session_id"]
    assert seen_sessions[0] == seen_sessions[1]


def test_api_chat_rejects_empty_question(client):
    r = client.post("/api/chat", json={"question": "   "})
    assert r.status_code == 400


def test_api_chat_reset_clears_session(client, monkeypatch):
    def fake_run_turn(session, question, conn, llm_fn, **kwargs):
        session.has_data = True
        session.current_data = [{"a": 1}]
        return Turn(question=question, status="success", answer=[{"a": 1}])

    monkeypatch.setattr(webapp, "run_turn", fake_run_turn)
    monkeypatch.setattr(webapp, "_build_llm_fn", lambda model: None)

    first = client.post("/api/chat", json={"question": "질문1"}).json()
    sid = first["session_id"]

    r = client.post("/api/chat/reset", json={"session_id": sid})
    assert r.status_code == 200

    session = get_session(sid)
    assert session.has_data is False
    assert session.current_data is None


def test_api_chat_history_returns_turns_in_order(client, monkeypatch):
    def fake_run_turn(session, question, conn, llm_fn, **kwargs):
        turn = Turn(question=question, status="success", answer=[{"code": "005930"}])
        session.turns.append(turn)
        session.has_data = True
        session.current_data = turn.answer
        return turn

    monkeypatch.setattr(webapp, "run_turn", fake_run_turn)
    monkeypatch.setattr(webapp, "_build_llm_fn", lambda model: None)

    first = client.post("/api/chat", json={"question": "질문1"}).json()
    sid = first["session_id"]
    client.post("/api/chat", json={"session_id": sid, "question": "질문2"})

    r = client.get(f"/api/chat/history?session_id={sid}")
    assert r.status_code == 200
    turns = r.json()["turns"]
    assert [t["question"] for t in turns] == ["질문1", "질문2"]


def test_api_chat_history_unknown_session_returns_empty(client):
    r = client.get("/api/chat/history?session_id=존재하지않음")
    assert r.status_code == 200
    assert r.json()["turns"] == []


def test_api_chat_csv_downloads_tabular_turn(client, monkeypatch):
    def fake_run_turn(session, question, conn, llm_fn, **kwargs):
        turn = Turn(question=question, status="success", answer=[{"code": "005930", "pbr": 1.2}])
        session.turns.append(turn)
        session.has_data = True
        session.current_data = turn.answer
        return turn

    monkeypatch.setattr(webapp, "run_turn", fake_run_turn)
    monkeypatch.setattr(webapp, "_build_llm_fn", lambda model: None)

    first = client.post("/api/chat", json={"question": "질문1"}).json()
    sid = first["session_id"]

    r = client.get(f"/api/chat/csv?session_id={sid}&turn_index=0")
    assert r.status_code == 200
    assert "005930" in r.text
    assert r.headers["content-type"].startswith("text/csv")


def test_api_chat_csv_unknown_session_returns_404(client):
    r = client.get("/api/chat/csv?session_id=존재하지않음&turn_index=0")
    assert r.status_code == 404


def test_api_chat_csv_non_tabular_turn_returns_400(client, monkeypatch):
    def fake_run_turn(session, question, conn, llm_fn, **kwargs):
        turn = Turn(question=question, status="success", answer=42)
        session.turns.append(turn)
        session.has_data = True
        session.current_data = 42
        return turn

    monkeypatch.setattr(webapp, "run_turn", fake_run_turn)
    monkeypatch.setattr(webapp, "_build_llm_fn", lambda model: None)

    first = client.post("/api/chat", json={"question": "질문1"}).json()
    sid = first["session_id"]

    r = client.get(f"/api/chat/csv?session_id={sid}&turn_index=0")
    assert r.status_code == 400


def test_chat_route_serves_html(client):
    r = client.get("/chat")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


# --------------------------------------------------------------------------
# 기본 화면(/) 전환 — 멀티턴 대화가 기본, 기존 단발성 질의 화면은 /query로 이동.
# --------------------------------------------------------------------------
def test_root_route_serves_chat_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "chatHistory" in r.text  # chat.html 고유 마커 — index.html이 아님을 확인


def test_query_route_serves_original_index_html(client):
    r = client.get("/query")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "chatHistory" not in r.text  # chat.html이 아니라 기존 index.html임을 확인


# --------------------------------------------------------------------------
# GET /api/chat/stream — AC13(오른쪽 패널 실시간 진행상황). run_turn의 on_progress 콜백을
# SSE 프레임으로 흘려보낸다(기존 /api/query/stream의 _sse_message 프레이밍 재사용).
# --------------------------------------------------------------------------
def test_api_chat_stream_emits_progress_frames_then_done(client, monkeypatch):
    def fake_run_turn(session, question, conn, llm_fn, on_progress=None, **kwargs):
        if on_progress:
            on_progress("1단계 진행 중")
            on_progress("2단계 진행 중")
        return Turn(question=question, status="success", answer=[{"a": 1}], sql="SELECT 1", code="result=rows")

    monkeypatch.setattr(webapp, "run_turn", fake_run_turn)
    monkeypatch.setattr(webapp, "_build_llm_fn", lambda model: None)

    with client.stream("GET", "/api/chat/stream", params={"question": "질문"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())

    assert "1단계 진행 중" in body
    assert "2단계 진행 중" in body
    assert "event: done" in body
    assert body.index("1단계 진행 중") < body.index("event: done")  # 진행 이벤트가 완료보다 먼저


def test_api_chat_stream_emits_fail_event_on_exception(client, monkeypatch):
    def failing_run_turn(session, question, conn, llm_fn, on_progress=None, **kwargs):
        raise RuntimeError("의도된 실패")

    monkeypatch.setattr(webapp, "run_turn", failing_run_turn)
    monkeypatch.setattr(webapp, "_build_llm_fn", lambda model: None)

    with client.stream("GET", "/api/chat/stream", params={"question": "질문"}) as r:
        body = "".join(r.iter_text())

    assert "event: fail" in body
    assert "의도된 실패" in body


def test_api_chat_stream_rejects_empty_question(client):
    r = client.get("/api/chat/stream", params={"question": "   "})
    assert r.status_code == 400


# --------------------------------------------------------------------------
# 회귀 — 기존 /api/query 등 단발성 질문 경로가 이번 멀티턴 API 추가로 변경되지 않았다.
# --------------------------------------------------------------------------
def test_existing_query_endpoint_untouched_by_chat_wiring(client, monkeypatch):
    monkeypatch.setattr(webapp, "run_hierarchical", lambda q, conn, llm_fn: {"uncertain": False, "conclusion": "ok"})
    monkeypatch.setattr(webapp, "_build_llm_fn", lambda model: None)
    r = client.post("/api/query", json={"question": "삼성전자 PER"})
    assert r.status_code == 200
    assert r.json()["conclusion"] == "ok"
