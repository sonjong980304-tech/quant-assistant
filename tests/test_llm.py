"""LLM 클라이언트의 OpenAI 폴백 경로 테스트.

src/llm.py의 _openai()는 1차 호출이 max_tokens 파라미터 미지원으로 실패하면
max_completion_tokens로 재시도한다. 이 재시도가 temperature까지 조용히 빼버리면
안 된다(실측 확인: gpt-5.4-mini는 temperature=0.0을 정상 지원함 — 문제는 오직
max_tokens라는 파라미터명 자체였다).
"""
from __future__ import annotations

from types import SimpleNamespace

from src.llm import LLMClient


class _FakeCompletions:
    def __init__(self, calls: list[dict]):
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise Exception(
                "Unsupported parameter: 'max_tokens' is not supported with this "
                "model. Use 'max_completion_tokens' instead."
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


class _FakeClient:
    def __init__(self, calls: list[dict]):
        self.chat = SimpleNamespace(completions=_FakeCompletions(calls))


def _fake_cfg():
    return SimpleNamespace(
        has_openai_key=True, openai_api_key="fake-key", openai_base_url=None
    )


def test_openai_retry_preserves_temperature_after_max_tokens_fallback(monkeypatch):
    # 이 테스트의 관심사(재시도 로직)와 무관한 LangSmith 트레이싱 경로를 확실히 끈다
    # (.env에 LANGCHAIN_TRACING_V2=true가 실제로 설정돼 있어 config 임포트 시 로드됨).
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    calls: list[dict] = []
    monkeypatch.setattr("openai.OpenAI", lambda **kw: _FakeClient(calls))

    client = LLMClient(cfg=_fake_cfg(), model="gpt-5.4-mini")
    result = client.complete("hi", role="sql", temperature=0.0, max_tokens=50)

    assert result.ok
    assert len(calls) == 2  # 1차 실패(max_tokens) + 재시도
    retry_kwargs = calls[1]
    assert retry_kwargs.get("temperature") == 0.0  # 재시도에서도 temperature가 유지돼야 함
    assert retry_kwargs.get("max_completion_tokens") == 50
    assert "max_tokens" not in retry_kwargs


# ── LangSmith 연동(_maybe_trace) — 켜고 끄는 스위치와 실패 시 안전 폴백 ──────────

def test_maybe_trace_returns_original_client_when_tracing_disabled(monkeypatch):
    from src.llm import _maybe_trace

    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    sentinel = object()
    assert _maybe_trace(sentinel, "sql") is sentinel


def test_maybe_trace_wraps_client_when_tracing_enabled(monkeypatch):
    from src.llm import _maybe_trace

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    wrapped_sentinel = object()
    captured: dict = {}

    def fake_wrap_openai(client, chat_name=None, **kw):
        captured["client"] = client
        captured["chat_name"] = chat_name
        return wrapped_sentinel

    import langsmith.wrappers
    monkeypatch.setattr(langsmith.wrappers, "wrap_openai", fake_wrap_openai)

    original = object()
    result = _maybe_trace(original, "judge")

    assert result is wrapped_sentinel
    assert captured["client"] is original
    assert captured["chat_name"] == "llm.judge"


def test_maybe_trace_falls_back_to_original_on_wrap_failure(monkeypatch):
    """감싸기 자체가 실패해도(예: 미설치·타입 불일치) 질의를 막지 않고 원본을 반환한다."""
    from src.llm import _maybe_trace

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    def boom(client, chat_name=None, **kw):
        raise RuntimeError("wrap 실패")

    import langsmith.wrappers
    monkeypatch.setattr(langsmith.wrappers, "wrap_openai", boom)

    original = object()
    assert _maybe_trace(original, "sql") is original
