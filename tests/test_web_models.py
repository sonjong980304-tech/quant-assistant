"""GET /api/models 캐싱 테스트.

/api/models가 매 요청마다 LLMClient(model=...).available로 네트워크 가용성을
확인하고 있었다 — 60초 캐싱을 추가해 짧은 간격의 반복 요청이 매번 네트워크를
타지 않게 한다.

fastapi 미설치 환경(셸 기본 python3)에서는 통째로 skip 한다(test_web_query.py
관례와 동일). 공유 venv(/Users/gyuyeong/projects/.venv)에서 실행하면 통과한다.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import web.app as webapp  # noqa: E402


class _FakeLLMClient:
    calls: list[str] = []

    def __init__(self, model: str):
        _FakeLLMClient.calls.append(model)
        self.available = True


@pytest.fixture(autouse=True)
def _reset_cache_and_calls(monkeypatch):
    _FakeLLMClient.calls.clear()
    monkeypatch.setattr("src.llm.LLMClient", _FakeLLMClient)
    webapp._MODELS_CACHE["data"] = None
    webapp._MODELS_CACHE["fetched_at"] = 0.0


def test_api_models_second_request_within_ttl_hits_cache():
    client = TestClient(webapp.app)
    r1 = client.get("/api/models")
    assert r1.status_code == 200
    calls_after_first = len(_FakeLLMClient.calls)
    assert calls_after_first > 0

    r2 = client.get("/api/models")
    assert r2.status_code == 200
    assert len(_FakeLLMClient.calls) == calls_after_first  # 캐시 히트 — 추가 호출 없음
    assert r2.json() == r1.json()


def test_api_models_refetches_after_ttl_expires(monkeypatch):
    fake_now = [1_000_000.0]
    monkeypatch.setattr(webapp.time, "time", lambda: fake_now[0])

    client = TestClient(webapp.app)
    client.get("/api/models")
    calls_after_first = len(_FakeLLMClient.calls)

    fake_now[0] += webapp._MODELS_CACHE_TTL_SECONDS + 1  # TTL 초과
    client.get("/api/models")
    assert len(_FakeLLMClient.calls) > calls_after_first  # 캐시 만료 — 재조회됨
