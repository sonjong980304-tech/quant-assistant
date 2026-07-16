"""동기 /api/query 엔드포인트 — 계층형 총괄 그래프 전환 테스트 (HA-13, TDD).

POST /api/query 가 legacy 6단계 Pipeline 이 아니라 신규 계층형 그래프
(src.agents.graph.run_hierarchical)를 호출하고, 그 최종 상태 dict
(conclusion/domain_results/routes/uncertain/attempts)를 그대로 반환하는지 검증한다.

- run_hierarchical 을 fake 로 주입해 실제 LLM/DB 없이 단위 검증(webapp.run_hierarchical monkeypatch).
- connect_readonly / _build_llm_fn 은 더미/None 으로 스텁(네트워크·DB 차단) — HA-12 스트리밍
  테스트(test_web_query_stream.py)와 동일 관례.

fastapi 미설치 환경(셸 기본 python3)에서는 통째로 skip 한다(test_macro_web 관례와 동일).
공유 venv(/Users/gyuyeong/projects/.venv)에서 실행하면 통과한다.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import web.app as webapp  # noqa: E402


def _dummy_conn():
    """connect_readonly 대체용 최소 더미(close 호출 추적). fake run_hierarchical 은 conn 을 안 쓴다."""
    ns = types.SimpleNamespace(closed=False)
    ns.close = lambda: setattr(ns, "closed", True)
    return ns


@pytest.fixture
def client(monkeypatch):
    # DB/LLM 없이 단위 검증: 읽기전용 연결과 llm_fn 빌더를 더미/None 으로 대체(네트워크 차단).
    monkeypatch.setattr(webapp, "connect_readonly", lambda *a, **k: _dummy_conn())
    monkeypatch.setattr(webapp, "_build_llm_fn", lambda model: None)
    return TestClient(webapp.app)


# ── 신규 응답 스키마 : run_hierarchical 최종 상태(conclusion/domain_results/routes...) 반환 ──

def test_api_query_returns_hierarchical_shape(client, monkeypatch):
    fake = {
        "question": "삼성전자 vs 애플 PER",
        "routes": ["kr", "us"],
        "domain_results": {
            "kr": {"stock_code": "005930", "financial": {"value": 12.3, "source": "DART"}},
            "us": {"ok": True, "stock_code": "AAPL", "financial": {"value": 30.1}},
        },
        "conclusion": "삼성전자 PER 12.3배, 애플 30.1배로 애플이 더 높게 평가됨.",
        "uncertain": False,
        "attempts": 1,
    }
    monkeypatch.setattr(webapp, "run_hierarchical", lambda *a, **k: fake)
    r = client.post("/api/query", json={"question": "삼성전자 vs 애플 PER"})
    assert r.status_code == 200
    body = r.json()
    assert body["conclusion"] == fake["conclusion"]
    assert body["routes"] == ["kr", "us"]
    assert body["uncertain"] is False
    assert body["attempts"] == 1
    # 각 도메인 원본 결과가 가공 없이 그대로 병기된다(AC4).
    assert body["domain_results"]["kr"]["financial"]["source"] == "DART"
    assert body["domain_results"]["us"]["stock_code"] == "AAPL"


# ── legacy 자동 폴백 제거 확인 ────────────────────────────────────────────────
# 한때(HA-16) uncertain/예외 시 legacy 6단계 Pipeline 으로 자동 폴백했으나, 신규 구조가
# 스스로 정확히 답 못하는 근본 원인(검증 정보 누락 등)을 legacy 우회로 가리는 대신
# 직접 고치는 쪽으로 방향을 바꿔 그 폴백 로직을 제거했다(2026-07-14). 이제 uncertain
# 이어도, 예외가 나도 legacy 는 절대 호출되지 않는다.

def test_api_query_success_marks_answered_by_hierarchical(client, monkeypatch):
    fake = {
        "question": "q", "routes": ["kr"], "domain_results": {"kr": {}},
        "conclusion": "삼성전자 PER 12.3배", "uncertain": False, "attempts": 1,
    }
    monkeypatch.setattr(webapp, "run_hierarchical", lambda *a, **k: fake)
    body = client.post("/api/query", json={"question": "q"}).json()
    assert body["answered_by"] == "hierarchical"
    assert body["conclusion"] == "삼성전자 PER 12.3배"
    assert body["uncertain"] is False


def test_api_query_uncertain_does_not_fall_back_to_legacy(client, monkeypatch):
    # 신규 구조가 uncertain=True(3회 재시도 소진)로 끝나도 그 결과를 그대로 반환한다 —
    # legacy 로 넘어가지 않는다(더 이상 _run_legacy_fallback 자체가 존재하지 않음).
    fake = {
        "question": "q", "routes": ["kr"],
        "domain_results": {"kr": {"errors": ["종목을 찾을 수 없습니다"]}},
        "uncertain": True, "reason": "3회 검증에 모두 실패했습니다.", "attempts": 3,
    }
    monkeypatch.setattr(webapp, "run_hierarchical", lambda *a, **k: fake)
    assert not hasattr(webapp, "_run_legacy_fallback")

    body = client.post("/api/query", json={"question": "q", "model": "gpt-5.5"}).json()
    assert body["answered_by"] == "hierarchical"
    assert body["uncertain"] is True
    assert body["attempts"] == 3
    assert body["reason"] == "3회 검증에 모두 실패했습니다."


def test_api_query_exception_surfaces_as_500_without_legacy_fallback(client, monkeypatch):
    # 신규 구조 실행 중 예외가 나면 원인을 감추지 않고 500 으로 그대로 노출한다.
    def boom(*a, **k):
        raise RuntimeError("신규 구조 폭발")

    monkeypatch.setattr(webapp, "run_hierarchical", boom)
    r = client.post("/api/query", json={"question": "애플 PER 알려줘"})
    assert r.status_code == 500
    assert "신규 구조 폭발" in r.json()["detail"]


def test_api_query_closes_conn_on_exception(client, monkeypatch):
    # 예외로 끝나도 읽기전용 연결은 finally 에서 닫힌다(연결 누수 방지).
    holder = {}

    def capturing_connect(*a, **k):
        holder["conn"] = _dummy_conn()
        return holder["conn"]

    monkeypatch.setattr(webapp, "connect_readonly", capturing_connect)

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(webapp, "run_hierarchical", boom)
    client.post("/api/query", json={"question": "q"})
    assert holder["conn"].closed is True


# ── 요청 파싱 / 인자 전달 ──────────────────────────────────────────────────────

def test_api_query_empty_question_returns_400(client, monkeypatch):
    monkeypatch.setattr(webapp, "run_hierarchical", lambda *a, **k: {})
    r = client.post("/api/query", json={"question": "   "})
    assert r.status_code == 400


def test_api_query_threads_question_conn_llm_fn(client, monkeypatch):
    captured = {}

    def fake_run_hierarchical(question, conn, llm_fn=None, steps=None):
        captured["question"] = question
        captured["conn"] = conn
        captured["llm_fn"] = llm_fn
        return {"conclusion": "ok", "routes": [], "domain_results": {}, "uncertain": False, "attempts": 1}

    monkeypatch.setattr(webapp, "run_hierarchical", fake_run_hierarchical)
    client.post("/api/query", json={"question": "삼성전자 PER"})
    assert captured["question"] == "삼성전자 PER"
    assert captured["conn"] is not None       # connect_readonly 더미가 전달됨
    assert captured["llm_fn"] is None          # 픽스처가 _build_llm_fn→None 으로 스텁


def test_api_query_builds_llm_fn_with_selected_model(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(webapp, "_build_llm_fn",
                        lambda model: seen.setdefault("model", model) or None)
    monkeypatch.setattr(webapp, "run_hierarchical",
                        lambda *a, **k: {"conclusion": "", "routes": [], "domain_results": {},
                                         "uncertain": False, "attempts": 1})
    client.post("/api/query", json={"question": "q", "model": "gpt-5.5"})
    assert seen["model"] == "gpt-5.5"


def test_api_query_closes_conn(client, monkeypatch):
    holder = {}

    def capturing_connect(*a, **k):
        holder["conn"] = _dummy_conn()
        return holder["conn"]

    monkeypatch.setattr(webapp, "connect_readonly", capturing_connect)
    monkeypatch.setattr(webapp, "run_hierarchical",
                        lambda *a, **k: {"conclusion": "", "routes": [], "domain_results": {},
                                         "uncertain": False, "attempts": 1})
    client.post("/api/query", json={"question": "q"})
    assert holder["conn"].closed is True       # finally 에서 연결이 닫힌다


# ── legacy 제거 흔적 : eval 옵션·Pipeline 이 라이브 경로에서 사라졌다 ──────────────

def test_query_req_has_no_eval_field():
    # 라이브 질의 경로에서 legacy SQL-diff 평가(eval)는 제거됐다(신규 응답 구조와 비호환).
    assert "eval" not in webapp.QueryReq.model_fields


def test_app_no_longer_imports_pipeline():
    # web.app 이 더 이상 legacy Pipeline 을 노출/사용하지 않는다.
    assert not hasattr(webapp, "Pipeline")


def test_api_query_route_still_bound():
    paths = {getattr(rt, "path", None): getattr(rt, "endpoint", None) for rt in webapp.app.routes}
    assert paths["/api/query"].__name__ == "api_query"
    # 스트리밍 경로(HA-12)는 그대로 공존한다.
    assert paths["/api/query/stream"].__name__ == "api_query_stream"


# ── POST /api/query/rerun — 휴먼인더루프: 사용자가 편집한 조건JSON/파이프라인을
#    LLM 생성 단계 없이 그대로 실행한다(실시간 트리에서 본 코드를 고쳐 재실행). ────────
def test_api_query_rerun_route_bound():
    paths = {getattr(rt, "path", None): getattr(rt, "endpoint", None) for rt in webapp.app.routes}
    assert paths["/api/query/rerun"].__name__ == "api_query_rerun"


def test_api_query_rerun_screening_uses_override_spec(client, monkeypatch):
    captured: dict = {}

    def spy_kr_screening(question, conn, llm_fn=None, override_spec=None, asof=None, **kw):
        captured.update(question=question, override_spec=override_spec, asof=asof)
        return {"intent": "screening", "result": [{"stock_code": "005930"}], "errors": []}

    monkeypatch.setattr(webapp, "answer_kr_screening", spy_kr_screening)
    r = client.post("/api/query/rerun", json={
        "kind": "screening", "domain": "kr", "question": "PBR 낮은 3개",
        "spec": {"criteria": [{"key": "pbr", "direction": "low"}], "top_n": 3},
        "asof": "2025-12-30",
    })
    assert r.status_code == 200
    assert captured["question"] == "PBR 낮은 3개"
    assert captured["override_spec"] == {"criteria": [{"key": "pbr", "direction": "low"}], "top_n": 3}
    assert captured["asof"] == "2025-12-30"
    assert r.json()["result"] == [{"stock_code": "005930"}]


def test_api_query_rerun_screening_us_domain_routes_to_us_screening(client, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        webapp, "answer_us_screening",
        lambda question, conn, llm_fn=None, override_spec=None, asof=None, **kw:
            captured.update(called=True) or {"intent": "screening", "result": [], "errors": []},
    )
    r = client.post("/api/query/rerun", json={
        "kind": "screening", "domain": "us", "question": "PER 낮은 3개",
        "spec": {"criteria": [{"key": "per", "direction": "low"}], "top_n": 3},
    })
    assert r.status_code == 200
    assert captured.get("called") is True


def test_api_query_rerun_backtest_uses_edited_steps(client, monkeypatch):
    captured: dict = {}

    def spy_backtest(question, steps, conn, llm_fn=None, market="KR", **kw):
        captured.update(question=question, steps=steps, market=market)
        return {"blocked": False, "error": None, "result": {"cagr": 5.0}, "hard": [], "warnings": []}

    monkeypatch.setattr(webapp, "answer_backtest_question", spy_backtest)
    edited_steps = [{"op": "run_backtest", "params": {"n": 5}, "out": "bt"}]
    r = client.post("/api/query/rerun", json={
        "kind": "backtest", "question": "저PER 백테스트", "steps": edited_steps, "market": "KR",
    })
    assert r.status_code == 200
    assert captured["steps"] == edited_steps
    assert captured["market"] == "KR"
    assert r.json()["result"]["cagr"] == 5.0


def test_api_query_rerun_unknown_kind_returns_400(client):
    r = client.post("/api/query/rerun", json={"kind": "이상함", "question": "q"})
    assert r.status_code == 400


def test_api_query_rerun_screening_missing_spec_returns_400(client):
    r = client.post("/api/query/rerun", json={"kind": "screening", "domain": "kr", "question": "q"})
    assert r.status_code == 400


def test_api_query_rerun_backtest_missing_steps_returns_400(client):
    r = client.post("/api/query/rerun", json={"kind": "backtest", "question": "q"})
    assert r.status_code == 400


def test_api_query_rerun_closes_conn(client, monkeypatch):
    holder: dict = {}

    def capturing_connect(*a, **k):
        c = _dummy_conn()
        holder["conn"] = c
        return c

    monkeypatch.setattr(webapp, "connect_readonly", capturing_connect)
    monkeypatch.setattr(webapp, "answer_backtest_question",
                         lambda *a, **k: {"blocked": False, "result": {}})
    client.post("/api/query/rerun", json={
        "kind": "backtest", "question": "q", "steps": [{"op": "x"}],
    })
    assert holder["conn"].closed is True
