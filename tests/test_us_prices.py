"""us_prices.py yfinance 주가 매퍼 테스트.

.omc/specs/brainstorming-us-market-data-plane.md AC1/AC4/AC13 검증.
yfinance `Ticker(symbol).history(...)`가 반환하는 형태(DatetimeIndex,
Open/High/Low/Close/Volume 컬럼)를 흉내낸 pandas DataFrame을 입력으로 써서
순수 변환 로직만 검증한다(네트워크 호출/yfinance 패키지 자체는 이 사이클
범위 밖 — 별도 fetch 래퍼에서 다룬다).
"""
from __future__ import annotations

import pandas as pd

import src.ingest.us_prices as us_prices
from src.db import connect, init_db
from src.ingest.us_prices import ingest_us_prices, normalize_price_history
from tests.conftest import seed_us_companies


def test_normalize_price_history_maps_ohlcv_and_date():
    df = pd.DataFrame(
        {
            "Open": [210.0],
            "High": [212.5],
            "Low": [209.0],
            "Close": [211.0],
            "Volume": [50_000_000],
        },
        index=pd.to_datetime(["2026-07-10"]),
    )
    rows = normalize_price_history("AAPL", df)
    assert rows == [
        {
            "stock_code": "AAPL",
            "date": "2026-07-10",
            "open": 210.0,
            "high": 212.5,
            "low": 209.0,
            "close": 211.0,
            "volume": 50_000_000.0,
        }
    ]


def test_normalize_price_history_skips_rows_with_missing_close():
    df = pd.DataFrame(
        {
            "Open": [210.0, 220.0],
            "High": [212.5, 222.0],
            "Low": [209.0, 219.0],
            "Close": [211.0, float("nan")],
            "Volume": [50_000_000, 40_000_000],
        },
        index=pd.to_datetime(["2026-07-10", "2026-07-13"]),
    )
    rows = normalize_price_history("AAPL", df)
    assert len(rows) == 1
    assert rows[0]["date"] == "2026-07-10"


def test_normalize_price_history_treats_missing_volume_as_none():
    df = pd.DataFrame(
        {
            "Open": [210.0],
            "High": [212.5],
            "Low": [209.0],
            "Close": [211.0],
            "Volume": [float("nan")],
        },
        index=pd.to_datetime(["2026-07-10"]),
    )
    rows = normalize_price_history("AAPL", df)
    assert rows[0]["volume"] is None


def test_normalize_price_history_attaches_stock_code_to_every_row():
    df = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.0, 2.0],
            "Low": [1.0, 2.0],
            "Close": [1.0, 2.0],
            "Volume": [100, 200],
        },
        index=pd.to_datetime(["2026-07-10", "2026-07-13"]),
    )
    rows = normalize_price_history("MSFT", df)
    assert {r["stock_code"] for r in rows} == {"MSFT"}


_SAMPLE_HISTORY = pd.DataFrame(
    {
        "Open": [210.0],
        "High": [212.5],
        "Low": [209.0],
        "Close": [211.0],
        "Volume": [50_000_000],
    },
    index=pd.to_datetime(["2026-07-10"]),
)


def test_fetch_yf_history_passes_auto_adjust_true(monkeypatch):
    # 현재 라이브러리 기본값에 암묵 의존하지 말고 auto_adjust=True를 명시적으로 전달해야 한다
    # (수정주가 기준 일관성 — 배당·액면분할 반영). DI 없이 yfinance 모듈을 mock으로 주입해 검증.
    import sys
    import types

    captured = {}

    class _FakeTicker:
        def __init__(self, symbol):
            captured["symbol"] = symbol

        def history(self, **kwargs):
            captured["kwargs"] = kwargs
            return pd.DataFrame()

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=_FakeTicker))

    from src.ingest.us_prices import _fetch_yf_history

    _fetch_yf_history("AAPL", "10y")
    assert captured["symbol"] == "AAPL"
    assert captured["kwargs"].get("period") == "10y"
    assert captured["kwargs"].get("auto_adjust") is True


def test_ingest_us_prices_upserts_history_for_all_us_companies(tmp_path, monkeypatch):
    db = str(tmp_path / "usp1.db")
    init_db(db)
    conn = connect(db)
    seed_us_companies(conn, ["AAPL", "MSFT"])
    conn.close()

    monkeypatch.setattr(us_prices, "send_slack_alert", lambda *a, **kw: None)
    result = ingest_us_prices(db_path=db, fetch_history=lambda code, period: _SAMPLE_HISTORY)

    conn = connect(db)
    rows = conn.execute("SELECT stock_code, date, close FROM us_prices ORDER BY stock_code").fetchall()
    conn.close()
    assert [(r["stock_code"], r["date"], r["close"]) for r in rows] == [
        ("AAPL", "2026-07-10", 211.0),
        ("MSFT", "2026-07-10", 211.0),
    ]
    assert result["succeeded"] == 2
    assert result["failed"] == []


class _CommitCountingConn:
    """conn.commit() 호출 횟수를 세는 프록시(test_us_financials.py와 동일 패턴)."""

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


def test_ingest_us_prices_commits_periodically_not_only_at_end(tmp_path, monkeypatch):
    """수천 종목 규모 백필 중 프로세스가 죽어도 진행이 날아가지 않도록 주기적으로 commit한다."""
    db = str(tmp_path / "usp_commit.db")
    init_db(db)
    conn = connect(db)
    seed_us_companies(conn, ["AAA", "BBB", "CCC", "DDD", "EEE"])
    conn.close()

    monkeypatch.setattr(us_prices, "send_slack_alert", lambda *a, **kw: None)
    real_connect = us_prices.connect
    counting = {}

    def fake_connect(db_path=None):
        c = _CommitCountingConn(real_connect(db_path))
        counting["conn"] = c
        return c

    monkeypatch.setattr(us_prices, "connect", fake_connect)
    ingest_us_prices(
        db_path=db, fetch_history=lambda code, period: _SAMPLE_HISTORY, commit_every=2,
    )
    assert counting["conn"].commit_calls > 1


def test_ingest_us_prices_uses_full_backfill_period_for_new_ticker_and_short_period_for_existing(tmp_path, monkeypatch):
    db = str(tmp_path / "usp2.db")
    init_db(db)
    conn = connect(db)
    seed_us_companies(conn, ["AAPL", "MSFT"])
    conn.execute(
        "INSERT INTO us_prices(stock_code, date, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
        ("MSFT", "2026-07-01", 400.0, 405.0, 399.0, 402.0, 20_000_000),
    )
    conn.commit()
    conn.close()

    periods_used = {}

    def fake_fetch(code, period):
        periods_used[code] = period
        return _SAMPLE_HISTORY

    monkeypatch.setattr(us_prices, "send_slack_alert", lambda *a, **kw: None)
    ingest_us_prices(db_path=db, fetch_history=fake_fetch)

    assert periods_used["AAPL"] == "10y"  # 신규 종목(기존 데이터 없음) → 10년 백필
    assert periods_used["MSFT"] == "5d"  # 기존 데이터 있음 → 최근 구간만 증분


def test_ingest_us_prices_skips_failing_ticker_and_alerts(tmp_path, monkeypatch):
    db = str(tmp_path / "usp3.db")
    init_db(db)
    conn = connect(db)
    seed_us_companies(conn, ["AAPL", "MSFT"])
    conn.close()

    def failing_fetch(code, period):
        if code == "AAPL":
            raise ConnectionError("AAPL 요청 실패(mock)")
        return _SAMPLE_HISTORY

    alerts: list[str] = []
    monkeypatch.setattr(us_prices, "send_slack_alert", lambda msg, **kw: alerts.append(msg))
    result = ingest_us_prices(db_path=db, fetch_history=failing_fetch)

    assert result["failed"] == ["AAPL"]
    assert result["succeeded"] == 1
    assert len(alerts) == 1

    conn = connect(db)
    rows = conn.execute("SELECT DISTINCT stock_code FROM us_prices").fetchall()
    conn.close()
    assert [r["stock_code"] for r in rows] == ["MSFT"]
