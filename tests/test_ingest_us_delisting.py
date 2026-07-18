"""FMP 미국 상장폐지 수집·파싱 단위 테스트 (TDD).

.omc/specs/brainstorming-us-delisting-survivorship.md 참고.
- FMP delisted-companies / stock-list 응답을 mock(DI)으로 주입해 네트워크 없이 파싱 검증.
- us_delisting 스키마(구간 기반, 티커당 여러 행)와 upsert(멱등) 검증.
- API 호출 횟수를 mock으로 카운트해 하루 250회 한도 미만 설계인지 확인.
DB 접근이 필요한 검사는 임시 SQLite에 시딩해 사용자 DB와 완전 격리한다.
"""
from __future__ import annotations

import sqlite3

import pytest

from src.ingest import us_delisting as ud
from src.db import init_db


def _conn(tmp_path) -> sqlite3.Connection:
    db = tmp_path / "delist.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


# FMP delisted-companies 실제 응답 형식(공식문서/래퍼 교차검증):
# {symbol, companyName, exchange, ipoDate(YYYY-MM-DD), delistedDate(YYYY-MM-DD)}
_DELISTED_PAGE = [
    {"symbol": "TWTR", "companyName": "Twitter, Inc.", "exchange": "NYSE",
     "ipoDate": "2013-11-07", "delistedDate": "2022-10-27"},
    {"symbol": "LEHMQ", "companyName": "Lehman Brothers Holdings Inc.", "exchange": "NYSE",
     "ipoDate": "1994-05-01", "delistedDate": "2008-09-17"},
]

# FMP stock-list(활성 종목) 형식: {symbol, name, price, exchange, exchangeShortName, type}
_ACTIVE_LIST = [
    {"symbol": "AAPL", "name": "Apple Inc.", "price": 189.5,
     "exchange": "NASDAQ Global Select", "exchangeShortName": "NASDAQ", "type": "stock"},
    {"symbol": "TWTR", "name": "New Twitter Corp", "price": 12.0,
     "exchange": "NYSE", "exchangeShortName": "NYSE", "type": "stock"},
]


# --------------------------------------------------------------------------
# AC1: delisted-companies 파싱
# --------------------------------------------------------------------------
def test_parse_delisted_companies_extracts_all_fields():
    rows = ud.parse_delisted_companies(_DELISTED_PAGE)
    assert len(rows) == 2
    twtr = next(r for r in rows if r["stock_code"] == "TWTR")
    assert twtr["company_name"] == "Twitter, Inc."
    assert twtr["exchange"] == "NYSE"
    assert twtr["listing_date"] == "2013-11-07"
    assert twtr["delisting_date"] == "2022-10-27"


def test_parse_delisted_companies_missing_ipo_date_becomes_empty_sentinel():
    # ipoDate가 없거나 빈 값이면 '' 센티널로 정규화(UNIQUE 멱등성 보존).
    rows = ud.parse_delisted_companies([
        {"symbol": "XYZ", "companyName": "X", "exchange": "AMEX",
         "delistedDate": "2010-01-01"},
    ])
    assert rows[0]["listing_date"] == ""
    assert rows[0]["delisting_date"] == "2010-01-01"


def test_parse_delisted_companies_skips_rows_without_symbol_or_delisting():
    rows = ud.parse_delisted_companies([
        {"companyName": "no symbol", "delistedDate": "2010-01-01"},
        {"symbol": "NODATE", "companyName": "no delisting date"},
        {"symbol": "OK", "companyName": "ok", "exchange": "NYSE",
         "ipoDate": "2000-01-01", "delistedDate": "2005-01-01"},
    ])
    assert [r["stock_code"] for r in rows] == ["OK"]


# --------------------------------------------------------------------------
# AC2: 활성 종목 목록 파싱(티커 재사용 식별용 데이터)
# --------------------------------------------------------------------------
def test_parse_active_symbols_extracts_symbols():
    rows = ud.parse_active_symbols(_ACTIVE_LIST)
    codes = {r["stock_code"] for r in rows}
    assert codes == {"AAPL", "TWTR"}
    aapl = next(r for r in rows if r["stock_code"] == "AAPL")
    assert aapl["exchange"] == "NASDAQ"  # exchangeShortName 사용
    assert aapl["type"] == "stock"


def test_active_symbol_set_helper():
    s = ud.active_symbol_set(_ACTIVE_LIST)
    assert s == {"AAPL", "TWTR"}


# --------------------------------------------------------------------------
# AC10: 페이지네이션 + 호출 횟수 카운트(하루 250회 미만 설계)
# --------------------------------------------------------------------------
def test_fetch_delisted_companies_paginates_until_empty_and_counts_calls():
    pages = {0: _DELISTED_PAGE, 1: [{"symbol": "OLD", "companyName": "Old",
             "exchange": "NYSE", "ipoDate": "1990-01-01", "delistedDate": "1999-01-01"}], 2: []}
    calls = {"n": 0}

    def fake_fetch(url, params):
        calls["n"] += 1
        return pages.get(params["page"], [])

    rows, api_calls = ud.fetch_delisted_companies(api_key="dummy", fetch_fn=fake_fetch)
    assert api_calls == calls["n"]
    assert api_calls < 250  # AC10: 전체 백필도 250회 한참 미만
    # 3티커(TWTR/LEHMQ/OLD)를 두 페이지에서 모두 수집(3페이지째 빈 배열에서 종료)
    assert {r["symbol"] for r in rows} == {"TWTR", "LEHMQ", "OLD"}


def test_fetch_delisted_companies_respects_max_pages_guard():
    # 무한히 비지 않는(버그) 응답에서도 max_pages로 250회 한도를 넘지 않게 방어.
    def always_full(url, params):
        return _DELISTED_PAGE

    rows, api_calls = ud.fetch_delisted_companies(
        api_key="dummy", fetch_fn=always_full, max_pages=5)
    assert api_calls <= 5


# --------------------------------------------------------------------------
# AC9: 요청 실패 시에도 예외 메시지에 apikey가 새지 않는다
# --------------------------------------------------------------------------
def test_fetch_page_does_not_leak_apikey_on_error(monkeypatch):
    import requests

    class FakeResp:
        url = "https://financialmodelingprep.com/stable/delisted-companies?page=0&apikey=SECRET123"

        def raise_for_status(self):
            raise requests.HTTPError(f"403 Client Error: Forbidden for url: {self.url}")

        def json(self):
            return []

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp())
    with pytest.raises(RuntimeError) as ei:
        ud._fetch_page(ud.FMP_DELISTED_URL, {"page": 0, "apikey": "SECRET123"})
    msg = str(ei.value)
    assert "SECRET123" not in msg
    assert "apikey" not in msg.lower()


# --------------------------------------------------------------------------
# 실전 발견: FMP 무료플랜은 delisted-companies에서 page=0만 허용, 2페이지부터 402
# (Payment Required)를 반환한다(라이브 실행에서 실측). 진짜 장애가 아니라 '무료 범위
# 소진'이므로 페이지네이션이 크래시 대신 정상 종료해야 한다(주간 launchd 갱신이 매번
# 죽지 않도록). 반면 1페이지(page=0)조차 402면 엔드포인트 자체가 막힌 것이므로 그대로
# 에러로 던져야 한다(진짜 장애를 숨기지 않음).
# --------------------------------------------------------------------------
def test_fetch_page_returns_empty_on_402_beyond_first_page(monkeypatch):
    import requests

    class FakeResp:
        status_code = 402
        url = "https://financialmodelingprep.com/stable/delisted-companies?page=1&apikey=SECRET123"

        def raise_for_status(self):
            err = requests.HTTPError(f"402 Client Error: Payment Required for url: {self.url}")
            err.response = self
            raise err

        def json(self):
            return []

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp())
    result = ud._fetch_page(ud.FMP_DELISTED_URL, {"page": 1, "apikey": "SECRET123"})
    assert result == []


def test_fetch_page_raises_on_402_at_first_page(monkeypatch):
    import requests

    class FakeResp:
        status_code = 402
        url = "https://financialmodelingprep.com/stable/delisted-companies?page=0&apikey=SECRET123"

        def raise_for_status(self):
            err = requests.HTTPError(f"402 Client Error: Payment Required for url: {self.url}")
            err.response = self
            raise err

        def json(self):
            return []

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp())
    with pytest.raises(RuntimeError):
        ud._fetch_page(ud.FMP_DELISTED_URL, {"page": 0, "apikey": "SECRET123"})


def test_fetch_delisted_companies_stops_gracefully_when_free_plan_limits_pages(monkeypatch):
    import requests

    class FakeResp200:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return _DELISTED_PAGE

    class FakeResp402:
        status_code = 402
        url = "https://financialmodelingprep.com/stable/delisted-companies?page=1&apikey=k"
        def raise_for_status(self):
            err = requests.HTTPError("402 Client Error: Payment Required")
            err.response = self
            raise err
        def json(self):
            return []

    def fake_get(url, params=None, timeout=None):
        return FakeResp200() if params["page"] == 0 else FakeResp402()

    monkeypatch.setattr(requests, "get", fake_get)
    rows, api_calls = ud.fetch_delisted_companies(api_key="k")
    assert {r["symbol"] for r in rows} == {"TWTR", "LEHMQ"}
    assert api_calls == 2  # page 0(성공) + page 1(402 확인 후 종료), 크래시 없음


# --------------------------------------------------------------------------
# 실전 발견 2: FMP stock-list(활성목록)은 무료플랜에서 아예 막힘(402, "Restricted
# Endpoint"). 페이지 소진이 아니라 엔드포인트 자체가 유료 전용이라 위 402 완화로도
# 못 피한다. 이미 보유한 us_company(현재 추적 중인 활성 종목 전체, 이 프로젝트의
# 스크리닝 대상 그 자체)를 활성 목록 대용으로 써서 API 호출 없이 재사용 티커 판정을
# 계속 동작시킨다.
# --------------------------------------------------------------------------
def test_default_fetch_active_reads_from_us_company_table_without_api_call(tmp_path):
    db = str(tmp_path / "active.db")
    init_db(db)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO us_company(stock_code, name) VALUES ('AAPL','Apple'), ('TWTR','New Twitter')")
    conn.commit()
    conn.close()

    rows, calls = ud._default_fetch_active("dummy-key", db_path=db)
    assert {r["stock_code"] for r in rows} == {"AAPL", "TWTR"}
    assert calls == 0  # FMP 미호출(무료플랜 402 회피, us_company로 대체)


def test_ingest_us_delisting_detects_reuse_via_us_company_by_default(tmp_path):
    # fetch_active_fn을 주입하지 않아도(기본 경로) us_company만으로 재사용 판정이 된다.
    db = str(tmp_path / "reuse2.db")
    init_db(db)
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO us_company(stock_code, name) VALUES ('TWTR','New Twitter Corp')")
    conn.commit()
    conn.close()

    def delisted(api_key, fetch_fn=None, **kw):
        return ud.parse_delisted_companies([
            {"symbol": "TWTR", "companyName": "Twitter, Inc.", "exchange": "NYSE",
             "ipoDate": "2013-11-07", "delistedDate": "2022-10-27"}]), 1

    r = ud.ingest_us_delisting(db_path=db, fetch_delisted_fn=delisted, api_key="k")
    assert "TWTR" in r["reused_tickers"]
    assert r["api_calls"] == 1  # delisted만 1회 호출, active는 DB조회라 0회


# --------------------------------------------------------------------------
# AC3 + AC8: us_delisting 스키마(티커당 여러 행) + upsert 멱등
# --------------------------------------------------------------------------
def test_us_delisting_schema_allows_multiple_rows_per_ticker(tmp_path):
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO us_delisting(stock_code, company_name, exchange, listing_date, delisting_date) "
        "VALUES (?,?,?,?,?)", ("TWTR", "Twitter", "NYSE", "2013-11-07", "2022-10-27"))
    conn.execute(
        "INSERT INTO us_delisting(stock_code, company_name, exchange, listing_date, delisting_date) "
        "VALUES (?,?,?,?,?)", ("TWTR", "TWTR Reuse", "NASDAQ", "2024-01-01", "2025-06-30"))
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM us_delisting WHERE stock_code='TWTR'").fetchone()[0]
    assert n == 2  # 같은 티커에 여러 상장/폐지 이력 저장 가능


def test_ingest_us_delisting_upserts_and_is_idempotent(tmp_path):
    conn = _conn(tmp_path)
    conn.close()
    db = str(tmp_path / "delist.db")

    def fake_delisted(api_key, fetch_fn=None, **kw):
        return ud.parse_delisted_companies(_DELISTED_PAGE), 1

    def fake_active(api_key, fetch_fn=None, **kw):
        return ud.parse_active_symbols(_ACTIVE_LIST), 1

    r1 = ud.ingest_us_delisting(db_path=db, fetch_delisted_fn=fake_delisted,
                                fetch_active_fn=fake_active, api_key="dummy")
    r2 = ud.ingest_us_delisting(db_path=db, fetch_delisted_fn=fake_delisted,
                                fetch_active_fn=fake_active, api_key="dummy")

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    episodes = conn.execute(
        "SELECT COUNT(*) FROM us_delisting WHERE delisting_date<>''").fetchone()[0]
    markers = conn.execute(
        "SELECT COUNT(*) FROM us_delisting WHERE delisting_date=''").fetchone()[0]
    assert episodes == 2   # 실제 상폐 구간 2건, 두 번 돌려도 중복 없음(upsert 멱등, AC8)
    assert markers == 1    # TWTR(재사용)에 '현재 활성 마커' 1건, 멱등
    assert r1["episodes_upserted"] == 2
    # TWTR은 활성 목록에도 있으므로 재사용 티커로 식별(AC2 데이터 활용)
    assert "TWTR" in r1["reused_tickers"]
    # AC7 리포트: 가장 오래된 상장폐지일
    assert r1["oldest_delisting_date"] == "2008-09-17"
    conn.close()


def test_ingest_creates_active_marker_and_relisted_ticker_stays_alive(tmp_path):
    # critic 지적(실데이터 모양): 옛 상폐 구간 1개 + 활성목록 재등장(구간 2개 아님)에서도
    # 재상장 이후 시점이 오탐(잘못된 하드차단) 없이 살아있음으로 판정돼야 한다(AC15).
    from src.backtest.data_access_us import _is_alive_us

    db = str(tmp_path / "reuse.db")

    def delisted(api_key, fetch_fn=None, **kw):
        # TWTR: 옛 상폐 구간 1개만(재상장 구간 없음 — 새 회사는 아직 상폐목록에 없음)
        return ud.parse_delisted_companies([
            {"symbol": "TWTR", "companyName": "Twitter, Inc.", "exchange": "NYSE",
             "ipoDate": "2013-11-07", "delistedDate": "2022-10-27"}]), 1

    def active(api_key, fetch_fn=None, **kw):
        # 새 TWTR가 활성목록에 있음(재사용, 현재 활성)
        return ud.parse_active_symbols([
            {"symbol": "TWTR", "name": "New Twitter", "exchangeShortName": "NYSE", "type": "stock"}]), 1

    r = ud.ingest_us_delisting(db_path=db, fetch_delisted_fn=delisted, fetch_active_fn=active, api_key="k")
    assert "TWTR" in r["reused_tickers"]

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    # 상폐(2022) 이후 시점 → 활성 마커 덕에 살아있음(오탐 하드차단 없음)
    assert _is_alive_us(conn, "TWTR", "2024-01-01") is True
    # 옛 회사가 살아있던 상폐 이전 시점도 당연히 살아있음
    assert _is_alive_us(conn, "TWTR", "2018-01-01") is True
    conn.close()


def test_ingest_us_delisting_weekly_adds_only_new_episodes(tmp_path):
    # AC8: 주간 갱신은 이미 있는 레코드는 건드리지 않고 신규 상장폐지만 추가한다.
    db = str(tmp_path / "delist.db")

    def active(api_key, fetch_fn=None, **kw):
        return [], 1

    # 1주차: 2건
    def week1(api_key, fetch_fn=None, **kw):
        return ud.parse_delisted_companies(_DELISTED_PAGE), 1

    ud.ingest_us_delisting(db_path=db, fetch_delisted_fn=week1, fetch_active_fn=active, api_key="k")

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    twtr_id_before = conn.execute(
        "SELECT id FROM us_delisting WHERE stock_code='TWTR'").fetchone()["id"]
    conn.close()

    # 2주차: 기존 2건 + 신규 1건(NEWCO)
    def week2(api_key, fetch_fn=None, **kw):
        return ud.parse_delisted_companies(_DELISTED_PAGE + [
            {"symbol": "NEWCO", "companyName": "New Co", "exchange": "NASDAQ",
             "ipoDate": "2015-01-01", "delistedDate": "2026-01-01"}]), 1

    r2 = ud.ingest_us_delisting(db_path=db, fetch_delisted_fn=week2, fetch_active_fn=active, api_key="k")

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    total = conn.execute("SELECT COUNT(*) FROM us_delisting").fetchone()[0]
    assert total == 3  # 신규 1건만 늘어남
    assert "NEWCO" in {row["stock_code"] for row in conn.execute("SELECT stock_code FROM us_delisting")}
    # 기존 TWTR 구간의 id가 보존됨(건드리지 않음)
    twtr_id_after = conn.execute(
        "SELECT id FROM us_delisting WHERE stock_code='TWTR'").fetchone()["id"]
    assert twtr_id_after == twtr_id_before
    conn.close()


def test_ingest_us_delisting_total_api_calls_under_limit(tmp_path):
    # AC10: 백필 전체가 250회 한도 미만인지 수집 함수가 반환한 호출 수로 확인.
    db = str(tmp_path / "d.db")

    def fake_delisted(api_key, fetch_fn=None, **kw):
        return ud.parse_delisted_companies(_DELISTED_PAGE), 3

    def fake_active(api_key, fetch_fn=None, **kw):
        return ud.parse_active_symbols(_ACTIVE_LIST), 1

    r = ud.ingest_us_delisting(db_path=db, fetch_delisted_fn=fake_delisted,
                               fetch_active_fn=fake_active, api_key="dummy")
    assert r["api_calls"] == 4
    assert r["api_calls"] < 250
