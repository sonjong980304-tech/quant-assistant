"""Slack 알림 유틸 회귀 테스트.

P1(네이버/FnGuide)과 미국 데이터 플레인 크롤러가 실패/누락 시 공통으로
쓰는 알림 함수. 웹훅 URL 미설정이면 조용히 스킵하고(본 수집작업을 막지
않음), 네트워크 예외도 절대 상위로 전파하지 않는다.
"""
from __future__ import annotations

import requests

from src.ingest.notify import send_slack_alert


class _FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


def test_send_slack_alert_skips_when_webhook_not_configured(monkeypatch):
    called = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: called.append(1) or _FakeResponse(200))
    result = send_slack_alert("테스트 알림", webhook_url="")
    assert result is False
    assert called == []  # HTTP 호출 자체가 발생하면 안 됨


def test_send_slack_alert_posts_and_returns_true_on_success(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(200)

    monkeypatch.setattr(requests, "post", fake_post)
    result = send_slack_alert("네이버 크롤러 실패", webhook_url="https://hooks.slack.test/abc")
    assert result is True
    assert captured["url"] == "https://hooks.slack.test/abc"
    assert captured["json"] == {"text": "네이버 크롤러 실패"}


def test_send_slack_alert_returns_false_on_non_200(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(500))
    result = send_slack_alert("실패 케이스", webhook_url="https://hooks.slack.test/abc")
    assert result is False


def test_send_slack_alert_returns_false_on_network_exception(monkeypatch):
    def raise_conn_error(*a, **k):
        raise requests.RequestException("connection refused")

    monkeypatch.setattr(requests, "post", raise_conn_error)
    result = send_slack_alert("네트워크 오류 케이스", webhook_url="https://hooks.slack.test/abc")
    assert result is False
