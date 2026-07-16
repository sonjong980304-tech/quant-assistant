"""USD/KRW 환율 — 네이버 증권 크롤링 + 당일 캐시 (TDD, C-5 AC7/AC8).

.omc/specs/brainstorming-us-nl-sql-integration.md 참고. 절대금액(시가총액 등) 한미
비교 질의에서만 예외적으로 쓴다. 하루 1회만 크롤링해 재사용(price_live.py의
당일 캐시 패턴과 동일 — ingest_meta를 캐시 저장소로 재사용).
"""
from __future__ import annotations

from datetime import date

from src.db import connect, get_meta, init_db
from src.ingest.exchange_rate import (
    fetch_usdkrw_rate_live,
    get_usdkrw_rate,
    parse_naver_rate_response,
)


def test_parse_naver_rate_response_extracts_latest_close_price():
    raw = {"isSuccess": True, "result": [
        {"localTradedAt": "2026-07-10", "closePrice": "1,503.40"},
        {"localTradedAt": "2026-07-09", "closePrice": "1,507.00"},
    ]}
    assert parse_naver_rate_response(raw) == 1503.40


def test_get_usdkrw_rate_fetches_and_caches_when_no_cache(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    today = date(2026, 7, 12)

    rate = get_usdkrw_rate(conn, on=today, fetch_fn=lambda: 1503.40)

    assert rate == 1503.40
    assert get_meta(conn, "usdkrw_rate") == "1503.4"
    assert get_meta(conn, "usdkrw_rate_date") == "2026-07-12"


def test_get_usdkrw_rate_uses_cache_when_same_day(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    today = date(2026, 7, 12)
    get_usdkrw_rate(conn, on=today, fetch_fn=lambda: 1503.40)

    calls = {"n": 0}

    def fetch_fn():
        calls["n"] += 1
        return 9999.0

    rate = get_usdkrw_rate(conn, on=today, fetch_fn=fetch_fn)

    assert rate == 1503.40  # 캐시된 값 재사용
    assert calls["n"] == 0  # 재크롤링 안 함


def test_get_usdkrw_rate_refetches_on_new_day(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    get_usdkrw_rate(conn, on=date(2026, 7, 12), fetch_fn=lambda: 1503.40)

    rate = get_usdkrw_rate(conn, on=date(2026, 7, 13), fetch_fn=lambda: 1510.0)

    assert rate == 1510.0
    assert get_meta(conn, "usdkrw_rate_date") == "2026-07-13"


# --------------------------------------------------------------------------
# fetch_usdkrw_rate_live — /api/macro 실시간 티커 전용, 당일 캐시를 쓰지 않는다.
# 프론트가 60초마다 폴링하는데 get_usdkrw_rate()의 당일 캐시를 그대로 쓰면 하루 중
# 첫 요청 값이 그날 내내 고정돼버려 "실시간으로 안 바뀐다"는 버그가 났다(실사용 재현).
# --------------------------------------------------------------------------
def test_fetch_usdkrw_rate_live_refetches_every_call_no_cache():
    calls = {"n": 0}

    def fetch_fn():
        calls["n"] += 1
        return 1500.0 + calls["n"]

    first = fetch_usdkrw_rate_live(fetch_fn=fetch_fn)
    second = fetch_usdkrw_rate_live(fetch_fn=fetch_fn)

    assert first == 1501.0
    assert second == 1502.0  # 캐시됐다면 첫 값(1501.0)과 같았을 것
    assert calls["n"] == 2


def test_fetch_usdkrw_rate_live_defaults_to_naver_fetch(monkeypatch):
    import src.ingest.exchange_rate as mod

    monkeypatch.setattr(mod, "_fetch_usdkrw_rate", lambda: 1490.9)
    assert fetch_usdkrw_rate_live() == 1490.9
