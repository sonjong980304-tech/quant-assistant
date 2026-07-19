"""실패/누락 알림 — Slack Incoming Webhook.

크롤러(네이버/FnGuide 등)가 실패·누락 발생 시 공통으로
호출한다. 웹훅 URL 미설정이면 조용히 스킵한다(알림은 부가기능이라
본 수집 작업을 막으면 안 된다). 네트워크 예외도 상위로 전파하지 않는다.
"""
from __future__ import annotations

import requests

from ..config import CONFIG


def send_slack_alert(message: str, webhook_url: str | None = None) -> bool:
    """message를 Slack 웹훅으로 전송. 성공 시 True, 미설정/실패 시 False."""
    url = webhook_url if webhook_url is not None else CONFIG.slack_webhook_url
    if not url:
        return False
    try:
        resp = requests.post(url, json={"text": message}, timeout=10)
        return resp.status_code == 200
    except requests.RequestException:
        return False
