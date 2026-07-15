"""네이버 fchart XML 파서 회귀 테스트.

.omc/specs/brainstorming-naver-fnguide-crawlers.md 네이버 크롤러 소스 섹션 참고.
소스: https://fchart.stock.naver.com/sise.nhn?symbol={code}&timeframe=day&count={n}&requestType=0
응답 형식: <item data="YYYYMMDD|open|high|low|close|volume" /> (파이프 구분, EUC-KR)

실제 curl 검증 시 확보한 샘플(삼성전자 005930, 2018-05-04 액면분할 전후 포함).
"""
from __future__ import annotations

import src.ingest.naver_prices as naver_prices
from src.db import connect, init_db
from src.ingest.naver_prices import fetch_daily_prices, ingest_naver_prices, parse_fchart_xml
from tests.conftest import FailingFetcher, FakeFetcher, seed_kr_companies


def test_parse_fchart_xml_normal_rows():
    xml = """<?xml version="1.0" encoding="EUC-KR" ?>
<protocol title="주식시세">
<chartdata symbol="005930" name="삼성전자" count="2">
<item data="20140417|27520|27600|27320|27400|144833" />
<item data="20140418|27399|27580|27399|27560|190367" />
</chartdata>
</protocol>"""
    rows = parse_fchart_xml(xml)
    assert rows == [
        {"date": "2014-04-17", "open": 27520, "high": 27600, "low": 27320, "close": 27400, "volume": 144833},
        {"date": "2014-04-18", "open": 27399, "high": 27580, "low": 27399, "close": 27560, "volume": 190367},
    ]


def test_parse_fchart_xml_zero_ohlv_treated_as_missing_on_halt_day():
    # 2018-05-02/05-03: 액면분할 직전 거래정지로 open/high/low/volume=0, close만 유효
    xml = """<?xml version="1.0" encoding="EUC-KR" ?>
<protocol title="주식시세">
<chartdata symbol="005930" name="삼성전자" count="2">
<item data="20180502|0|0|0|53000|0" />
<item data="20180504|53000|53900|51800|51900|39565391" />
</chartdata>
</protocol>"""
    rows = parse_fchart_xml(xml)
    assert rows[0] == {
        "date": "2018-05-02",
        "open": None,
        "high": None,
        "low": None,
        "close": 53000,
        "volume": None,
    }
    assert rows[1] == {
        "date": "2018-05-04",
        "open": 53000,
        "high": 53900,
        "low": 51800,
        "close": 51900,
        "volume": 39565391,
    }


def test_parse_fchart_xml_empty_chartdata_returns_empty_list():
    xml = """<?xml version="1.0" encoding="EUC-KR" ?>
<protocol title="주식시세">
<chartdata symbol="005930" name="삼성전자" count="0">
</chartdata>
</protocol>"""
    assert parse_fchart_xml(xml) == []


_SAMPLE_XML = """<?xml version="1.0" encoding="EUC-KR" ?>
<protocol title="주식시세">
<chartdata symbol="005930" name="삼성전자" count="1">
<item data="20140417|27520|27600|27320|27400|144833" />
</chartdata>
</protocol>"""


def test_fetch_daily_prices_builds_fchart_url_with_symbol_and_count():
    fetcher = FakeFetcher(_SAMPLE_XML)
    fetch_daily_prices("005930", count=3000, fetcher=fetcher)
    assert fetcher.calls == [
        "https://fchart.stock.naver.com/sise.nhn?symbol=005930&timeframe=day&count=3000&requestType=0"
    ]


def test_fetch_daily_prices_parses_response_via_parse_fchart_xml():
    fetcher = FakeFetcher(_SAMPLE_XML)
    rows = fetch_daily_prices("005930", count=3000, fetcher=fetcher)
    assert rows == [
        {"date": "2014-04-17", "open": 27520, "high": 27600, "low": 27320, "close": 27400, "volume": 144833},
    ]


def test_fetch_daily_prices_sets_euc_kr_response_encoding():
    fetcher = FakeFetcher(_SAMPLE_XML)
    fetch_daily_prices("005930", count=3000, fetcher=fetcher)
    assert fetcher.last_response.encoding == "euc-kr"


def test_fetch_daily_prices_uses_module_default_fetcher_when_omitted(monkeypatch):
    fake = FakeFetcher(_SAMPLE_XML)
    monkeypatch.setattr(naver_prices, "ThrottledFetcher", lambda: fake)
    rows = fetch_daily_prices("005930", count=3000)
    assert any("symbol=005930" in url for url in fake.calls)
    assert len(rows) == 1


def test_ingest_naver_prices_upserts_ohlcv_for_all_companies(tmp_path, monkeypatch):
    db = str(tmp_path / "ingest1.db")
    init_db(db)
    conn = connect(db)
    seed_kr_companies(conn, ["005930", "000660"])
    conn.close()

    monkeypatch.setattr(naver_prices, "send_slack_alert", lambda *a, **kw: None)
    fetcher = FakeFetcher(_SAMPLE_XML)
    result = ingest_naver_prices(db_path=db, fetcher=fetcher)

    conn = connect(db)
    rows = conn.execute("SELECT stock_code, date, close FROM prices ORDER BY stock_code").fetchall()
    conn.close()
    assert [(r["stock_code"], r["date"], r["close"]) for r in rows] == [
        ("000660", "2014-04-17", 27400),
        ("005930", "2014-04-17", 27400),
    ]
    assert result["succeeded"] == 2
    assert result["failed"] == []


def test_ingest_naver_prices_only_iterates_given_codes(tmp_path, monkeypatch):
    """codes를 주면 company 전체가 아니라 그 종목들만 재수집한다(오염 종가 복구용).

    company에 3종목이 있어도 codes=["005930","035420"]이면 그 둘만 fetch하고
    prices에도 그 둘만 들어간다(000660은 건드리지 않음)."""
    db = str(tmp_path / "ingest_codes.db")
    init_db(db)
    conn = connect(db)
    seed_kr_companies(conn, ["005930", "000660", "035420"])
    conn.close()

    monkeypatch.setattr(naver_prices, "send_slack_alert", lambda *a, **kw: None)
    fetcher = FakeFetcher(_SAMPLE_XML)
    result = ingest_naver_prices(db_path=db, fetcher=fetcher, codes=["005930", "035420"])

    fetched = {url.split("symbol=")[1].split("&")[0] for url in fetcher.calls}
    assert fetched == {"005930", "035420"}
    assert result["tickers"] == 2
    assert result["succeeded"] == 2

    conn = connect(db)
    codes_in_db = [r["stock_code"] for r in conn.execute("SELECT DISTINCT stock_code FROM prices ORDER BY stock_code")]
    conn.close()
    assert codes_in_db == ["005930", "035420"]


def test_ingest_naver_prices_codes_none_iterates_all_companies(tmp_path, monkeypatch):
    """codes=None(기본값)이면 기존 동작 그대로 company 전체를 순회한다(하위호환)."""
    db = str(tmp_path / "ingest_codes_none.db")
    init_db(db)
    conn = connect(db)
    seed_kr_companies(conn, ["005930", "000660"])
    conn.close()

    monkeypatch.setattr(naver_prices, "send_slack_alert", lambda *a, **kw: None)
    fetcher = FakeFetcher(_SAMPLE_XML)
    result = ingest_naver_prices(db_path=db, fetcher=fetcher)

    fetched = {url.split("symbol=")[1].split("&")[0] for url in fetcher.calls}
    assert fetched == {"005930", "000660"}
    assert result["tickers"] == 2


def test_ingest_naver_prices_skips_failing_ticker_and_alerts_without_pykrx_fallback(tmp_path, monkeypatch):
    db = str(tmp_path / "ingest2.db")
    init_db(db)
    conn = connect(db)
    seed_kr_companies(conn, ["005930", "000660"])
    conn.close()

    alerts: list[str] = []
    monkeypatch.setattr(naver_prices, "send_slack_alert", lambda msg, **kw: alerts.append(msg))
    fetcher = FailingFetcher(_SAMPLE_XML, fail_when="symbol=005930")
    result = ingest_naver_prices(db_path=db, fetcher=fetcher)

    assert result["failed"] == ["005930"]
    assert result["succeeded"] == 1
    assert len(alerts) == 1

    conn = connect(db)
    rows = conn.execute("SELECT stock_code FROM prices").fetchall()
    conn.close()
    assert [r["stock_code"] for r in rows] == ["000660"]


class _CommitCountingConn:
    """conn.commit() 호출 횟수를 세는 프록시(sqlite3.Connection은 속성 재할당 불가라 위임 프록시로
    감쌈 — test_us_financials.py의 동일 패턴)."""

    def __init__(self, real):
        self._real = real
        self.commit_calls = 0

    def execute(self, *a, **kw):
        return self._real.execute(*a, **kw)

    def commit(self):
        self.commit_calls += 1
        self._real.commit()

    def close(self):
        self._real.close()


def test_ingest_naver_prices_commits_periodically_not_only_at_end(tmp_path, monkeypatch):
    """맥미니 전원이 중간에 끊기는 등 프로세스가 죽어도 그때까지 진행이 날아가지 않도록,
    전체 종목 루프 끝에 딱 1번이 아니라 commit_every개 종목마다 주기적으로 commit해야 한다
    (실제로 이 문제로 크롤링 진행분이 통째로 유실된 사고가 있었음)."""
    db = str(tmp_path / "ingest_commit.db")
    init_db(db)
    conn = connect(db)
    seed_kr_companies(conn, ["005930", "000660", "035420", "051910", "005380"])
    conn.close()

    monkeypatch.setattr(naver_prices, "send_slack_alert", lambda *a, **kw: None)
    real_connect = naver_prices.connect
    counting = {}

    def fake_connect(db_path=None):
        c = _CommitCountingConn(real_connect(db_path))
        counting["conn"] = c
        return c

    monkeypatch.setattr(naver_prices, "connect", fake_connect)
    fetcher = FakeFetcher(_SAMPLE_XML)
    ingest_naver_prices(db_path=db, fetcher=fetcher, commit_every=2)
    assert counting["conn"].commit_calls > 1


def test_ingest_naver_prices_overwrites_existing_row_with_new_close(tmp_path, monkeypatch):
    db = str(tmp_path / "ingest3.db")
    init_db(db)
    conn = connect(db)
    seed_kr_companies(conn, ["005930"])
    conn.execute(
        "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,?)",
        ("005930", "2014-04-17", 99999, 123),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(naver_prices, "send_slack_alert", lambda *a, **kw: None)
    fetcher = FakeFetcher(_SAMPLE_XML)  # 2014-04-17 close=27400
    ingest_naver_prices(db_path=db, fetcher=fetcher)

    conn = connect(db)
    row = conn.execute(
        "SELECT close, market_cap FROM prices WHERE stock_code='005930' AND date='2014-04-17'"
    ).fetchone()
    conn.close()
    assert row["close"] == 27400
    # market_cap은 pykrx(krx.py)가 채우는 컬럼이라 네이버 크롤러가 덮어써서는 안 된다
    # (INSERT OR REPLACE는 행 전체를 갈아끼워 명시 안 한 컬럼을 NULL로 지운다 — 회귀 방지).
    assert row["market_cap"] == 123
