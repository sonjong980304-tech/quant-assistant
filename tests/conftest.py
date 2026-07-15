"""pytest 공용 픽스처 + 크롤러 테스트 공용 더블.

- 레포 루트를 sys.path에 넣어 `import src`가 어느 실행 방식(`pytest` / `python -m
  pytest`)에서도 되게 한다.
- seeded_db: 임시 SQLite DB를 스키마+소량 데이터로 시드해 반환한다. 기능/안정성
  테스트가 사용자 DB(data/market.db)를 절대 건드리지 않도록 완전 격리한다.
- FakeFetcher/FakeHttpResponse/FailingFetcher: ThrottledFetcher(.get(url)->response)
  인터페이스를 흉내낸 테스트 더블. 네이버/FnGuide 크롤러 테스트가 공유한다.
- seed_kr_companies/seed_us_companies: company/us_company 테이블에 최소 필드만
  채운 종목을 시드하는 헬퍼. ingest_* 오케스트레이터 테스트가 공유한다.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.db import init_db  # noqa: E402  (sys.path 조정 후 import)

# (stock_code, name, market, sector)
_COMPANIES = [
    ("000001", "가나전자", "KOSPI", "반도체"),
    ("000002", "다라화학", "KOSPI", "화학"),
    ("000003", "마바금융", "KOSPI", "금융"),
    ("000004", "사아자동차", "KOSPI", "자동차"),
    ("000005", "차카전지", "KOSDAQ", "2차전지"),
    ("000006", "타파게임", "KOSDAQ", "게임"),
    ("000007", "하가반도", "KOSPI", "반도체"),
    ("000008", "나다화학", "KOSPI", "화학"),
    ("000009", "라마금융", "KOSPI", "금융"),
    ("000010", "바사전자", "KOSDAQ", "반도체"),
    ("000011", "아자게임", "KOSDAQ", "게임"),
    ("000012", "카타전지", "KOSDAQ", "2차전지"),
]

# stock_code → (per, pbr, roe, operating_margin, debt_ratio)
_METRICS = {
    "000001": (8.0, 0.9, 12.0, 15.0, 80.0),
    "000002": (15.0, 1.5, 8.0, 10.0, 120.0),
    "000003": (5.0, 0.5, 20.0, 25.0, 60.0),
    "000004": (11.0, 1.1, 9.0, 8.0, 150.0),
    "000005": (25.0, 3.0, 6.0, 5.0, 90.0),
    "000006": (18.0, 2.2, 14.0, 20.0, 40.0),
    "000007": (7.5, 0.8, 16.0, 18.0, 70.0),
    "000008": (13.0, 1.3, 7.0, 9.0, 130.0),
    "000009": (6.0, 0.6, 18.0, 22.0, 55.0),
    "000010": (9.5, 1.0, 11.0, 13.0, 85.0),
    "000011": (30.0, 3.5, 4.0, 3.0, 45.0),
    "000012": (22.0, 2.8, 10.0, 12.0, 95.0),
}


@pytest.fixture
def seeded_db(tmp_path) -> str:
    """스키마+소량 데이터로 시드된 임시 DB 경로(str)를 반환한다(테스트별 격리)."""
    db = tmp_path / "test_market.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    try:
        conn.executemany(
            "INSERT INTO company(stock_code, name, market, sector) VALUES(?,?,?,?)",
            _COMPANIES,
        )
        for code, (per, pbr, roe, om, dr) in _METRICS.items():
            conn.execute(
                "INSERT INTO metrics("
                "stock_code, quarter, price_date, market_cap, per, pbr, roe, "
                "operating_margin, debt_ratio) VALUES(?,?,?,?,?,?,?,?,?)",
                (code, "2025Q1", "2026-06-22", 1e12, per, pbr, roe, om, dr),
            )
        for i, (code, *_rest) in enumerate(_COMPANIES):
            conn.execute(
                "INSERT INTO prices(stock_code, date, close, market_cap) VALUES(?,?,?,?)",
                (code, "2026-06-22", 10000.0 + i, (i + 1) * 1e12),
            )
        conn.commit()
    finally:
        conn.close()
    return str(db)


class FakeHttpResponse:
    """ThrottledFetcher.get()이 반환하는 requests.Response를 흉내낸 테스트 더블."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.encoding = None


class FakeFetcher:
    """ThrottledFetcher 인터페이스(.get(url) -> response)를 흉내낸 테스트 더블.

    모든 호출을 calls에 기록하고 last_response를 보관한다.
    """

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[str] = []
        self.last_response: FakeHttpResponse | None = None

    def get(self, url: str, **kwargs):
        self.calls.append(url)
        self.last_response = FakeHttpResponse(self._text)
        return self.last_response


class FailingFetcher:
    """URL에 fail_when 문자열이 포함된 요청에만 예외를 던지는 FakeFetcher.

    "실패 종목은 건너뛰고 알림만 보낸다" 계열 테스트가 공유한다.
    """

    def __init__(self, text: str, fail_when: str) -> None:
        self._text = text
        self._fail_when = fail_when
        self.calls: list[str] = []

    def get(self, url: str, **kwargs):
        self.calls.append(url)
        if self._fail_when in url:
            raise ConnectionError(f"요청 실패(mock): {url}")
        return FakeHttpResponse(self._text)


def seed_kr_companies(conn: sqlite3.Connection, codes: list[str]) -> None:
    """company 테이블에 최소 필드만 채운 종목을 시드한다(KR 파이프라인 테스트용)."""
    for code in codes:
        conn.execute(
            "INSERT INTO company(stock_code, name, market, sector) VALUES (?,?,?,?)",
            (code, code, "KOSPI", "기타"),
        )
    conn.commit()


def seed_us_companies(conn: sqlite3.Connection, codes: list[str]) -> None:
    """us_company 테이블에 최소 필드만 채운 종목을 시드한다(US 파이프라인 테스트용)."""
    for code in codes:
        conn.execute(
            "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (code, code, "NASDAQ", "Technology", 1.0e9, "2026-07-12T00:00:00"),
        )
    conn.commit()


class FakeLLM:
    """LLMClient 대역(파마프렌치/파이프라인 그래프 테스트 공용). 네트워크 호출 없음."""

    def __init__(self, text: str, ok: bool = True, available: bool = True):
        from src.llm import LLMResult

        self._result_cls = LLMResult
        self._text = text
        self._ok = ok
        self.available = available
        self.calls: list[dict] = []

    def complete(self, prompt, system=None, role="sql", **kwargs):
        self.calls.append({"prompt": prompt, "system": system, "role": role})
        return self._result_cls(self._text, ok=self._ok)

    def model_for(self, role):
        return "fake-model"
