"""LLMClient.complete_vision() 신규 메서드 테스트 (US-5 vision 호출, TDD).

.omc/specs/brainstorming-factcheck-eval.md Round 9/10: 차트 vision 판정은 gpt-5.4-mini를
재사용한다(judge용 gpt-5.5는 비용 때문에 쓰지 않기로 확정). role="sql"로 호출하면 기존
SQL 생성과 동일한 모델이 선택되는지, OpenAI Chat Completions의 content 배열에
text+image_url 블록이 올바르게 조립되는지 검증한다.

기존 complete()의 텍스트 전용 동작에는 손대지 않는다 — 회귀 여부는 tests/test_llm.py가
별도로 보장한다(이 파일과 함께 실행해 확인).
"""
from __future__ import annotations

from types import SimpleNamespace

from src.llm import LLMClient


class _FakeCompletions:
    def __init__(self, calls: list[dict]):
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="vision-ok"))]
        )


class _FakeClient:
    def __init__(self, calls: list[dict]):
        self.chat = SimpleNamespace(completions=_FakeCompletions(calls))


def _fake_cfg():
    return SimpleNamespace(
        has_openai_key=True,
        openai_api_key="fake-key",
        openai_base_url=None,
        openai_model_sql="gpt-5.4-mini",
        openai_model_judge="gpt-5.5",
    )


def test_complete_vision_sends_image_and_text_content(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    calls: list[dict] = []
    monkeypatch.setattr("openai.OpenAI", lambda **kw: _FakeClient(calls))

    client = LLMClient(cfg=_fake_cfg(), model="gpt-5.4-mini")
    result = client.complete_vision(
        "이 차트가 데이터를 올바르게 표현했나요?", "BASE64PNGDATA", role="sql"
    )

    assert result.ok
    assert result.text == "vision-ok"
    assert len(calls) == 1
    messages = calls[0]["messages"]
    assert messages[-1]["role"] == "user"
    content = messages[-1]["content"]
    assert isinstance(content, list)
    text_blocks = [c for c in content if c["type"] == "text"]
    image_blocks = [c for c in content if c["type"] == "image_url"]
    assert text_blocks[0]["text"] == "이 차트가 데이터를 올바르게 표현했나요?"
    assert image_blocks[0]["image_url"]["url"] == "data:image/png;base64,BASE64PNGDATA"


def test_complete_vision_uses_sql_role_model_not_judge(monkeypatch):
    # 비용 절감 결정(Round 9/10): role="sql"은 gpt-5.4-mini를 써야 하고, judge용
    # gpt-5.5로 새어나가면 안 된다. model_override 없이 provider+role 경로로 모델을
    # 선택하는 기본 흐름을 검증한다.
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    calls: list[dict] = []
    monkeypatch.setattr("openai.OpenAI", lambda **kw: _FakeClient(calls))

    client = LLMClient(cfg=_fake_cfg(), provider="openai")
    result = client.complete_vision("prompt", "imgdata", role="sql")

    assert result.ok
    assert calls[0]["model"] == "gpt-5.4-mini"


def test_complete_vision_unavailable_when_no_key(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    cfg = SimpleNamespace(
        has_openai_key=False,
        openai_api_key="",
        openai_base_url=None,
        openai_model_sql="gpt-5.4-mini",
        openai_model_judge="gpt-5.5",
    )
    client = LLMClient(cfg=cfg, provider="openai")

    result = client.complete_vision("prompt", "imgdata", role="sql")

    assert result.ok is False
    assert "unavailable" in (result.error or "")
