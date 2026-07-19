"""네이버/FnGuide 공용 HTTP fetch — 요청 간 최소 딜레이 강제 + User-Agent 헤더.

.omc/specs/brainstorming-naver-fnguide-crawlers.md AC5(네이버)/AC10(FnGuide) 최소
2초 딜레이, AC12(requests+BeautifulSoup만, Selenium 미추가) 대상. investing.com
Selenium 크롤러와는 별개 — 그쪽은 브라우저 자동화가 스펙상 허용됨.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import requests

MIN_DELAY_SECONDS = 2.0
_USER_AGENT = "Mozilla/5.0 (compatible; dart-text2sql-wiki-crawler/1.0)"


class ThrottledFetcher:
    """연속 호출 시작 시각 사이에 최소 min_delay초를 강제하는 GET 요청 래퍼."""

    def __init__(
        self,
        min_delay: float = MIN_DELAY_SECONDS,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._min_delay = min_delay
        self._sleep_fn = sleep_fn
        self._time_fn = time_fn
        self._last_call_at: Optional[float] = None

    def get(self, url: str, **kwargs) -> requests.Response:
        now = self._time_fn()
        if self._last_call_at is not None:
            wait = self._min_delay - (now - self._last_call_at)
            if wait > 0:
                self._sleep_fn(wait)
        self._last_call_at = now

        headers = kwargs.pop("headers", None) or {}
        headers.setdefault("User-Agent", _USER_AGENT)
        return requests.get(url, headers=headers, **kwargs)
