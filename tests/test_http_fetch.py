"""http_fetch.py — 네이버/FnGuide 공용 딜레이 강제 HTTP fetch 테스트.

.omc/specs/brainstorming-naver-fnguide-crawlers.md AC5(네이버 최소 2초 딜레이)/
AC10(FnGuide 최소 2초 딜레이)/AC12(requests+BeautifulSoup만, Selenium 미사용) 검증.
sleep_fn/time_fn을 주입해 실제 대기 없이 딜레이 로직만 단위 검증한다.
"""
from __future__ import annotations

import requests

from src.ingest.http_fetch import MIN_DELAY_SECONDS, ThrottledFetcher


class _FakeResponse:
    status_code = 200


def test_min_delay_constant_is_at_least_two_seconds():
    assert MIN_DELAY_SECONDS >= 2.0


def test_first_call_does_not_sleep(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **kw: _FakeResponse())
    sleeps = []
    fetcher = ThrottledFetcher(sleep_fn=sleeps.append, time_fn=iter([100.0]).__next__)
    fetcher.get("https://example.com")
    assert sleeps == []


def test_second_call_sleeps_for_remaining_delay_when_too_soon(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **kw: _FakeResponse())
    sleeps = []
    # 1번째 호출 시각=100.0(호출 시작 시각을 last_call_at으로 기록),
    # 2번째 호출 시작 시각=100.5(경과 0.5초) → 남은 대기 1.5초를 sleep해야 함.
    times = iter([100.0, 100.5])
    fetcher = ThrottledFetcher(sleep_fn=sleeps.append, time_fn=lambda: next(times))
    fetcher.get("https://example.com")
    fetcher.get("https://example.com")
    assert sleeps == [1.5]


def test_second_call_does_not_sleep_when_elapsed_exceeds_delay(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **kw: _FakeResponse())
    sleeps = []
    times = iter([100.0, 105.0])
    fetcher = ThrottledFetcher(sleep_fn=sleeps.append, time_fn=lambda: next(times))
    fetcher.get("https://example.com")
    fetcher.get("https://example.com")
    assert sleeps == []


def test_get_sets_default_user_agent_header(monkeypatch):
    captured = {}

    def fake_get(url, **kw):
        captured.update(kw)
        return _FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    fetcher = ThrottledFetcher(sleep_fn=lambda s: None, time_fn=iter([100.0]).__next__)
    fetcher.get("https://example.com")
    assert "User-Agent" in captured["headers"]
    assert captured["headers"]["User-Agent"]


def test_get_preserves_caller_provided_custom_header(monkeypatch):
    captured = {}

    def fake_get(url, **kw):
        captured.update(kw)
        return _FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    fetcher = ThrottledFetcher(sleep_fn=lambda s: None, time_fn=iter([100.0]).__next__)
    fetcher.get("https://example.com", headers={"X-Custom": "yes"})
    assert captured["headers"]["X-Custom"] == "yes"
    assert "User-Agent" in captured["headers"]
