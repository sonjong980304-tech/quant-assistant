"""US 종목 실시간 주가(yfinance) — KR ensure_live_prices와 동등한 US 버전 (TDD, C-5 AC5).

.omc/specs/brainstorming-us-nl-sql-integration.md 참고. "오늘 애플 주가 얼마야" 같은
질의는 yfinance로 실시간 1건을 조회해 us_prices에 당일 데이터로 저장 후 사용한다
(KR의 pykrx 실시간 경로 ensure_live_prices와 동등한 패턴, 데이터 소스만 yfinance).
"""
from __future__ import annotations

from datetime import date

from src.db import connect, init_db
from src.ingest.price_live import ensure_live_prices_us


def test_ensure_live_prices_us_fetches_when_no_today_row(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    today = date(2026, 7, 12)

    result = ensure_live_prices_us(conn, ["AAPL"], on=today, fetch_fn=lambda code: 210.5)

    assert result["fetched"] == 1
    assert result["cached"] == 0
    row = conn.execute(
        "SELECT close FROM us_prices WHERE stock_code='AAPL' AND date='2026-07-12'"
    ).fetchone()
    assert row["close"] == 210.5


def test_ensure_live_prices_us_uses_cache_when_today_row_exists(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    conn.execute(
        "INSERT INTO us_prices(stock_code, date, close) VALUES ('AAPL', '2026-07-12', 200.0)"
    )
    conn.commit()
    today = date(2026, 7, 12)

    calls = {"n": 0}

    def fetch_fn(code):
        calls["n"] += 1
        return 999.0

    result = ensure_live_prices_us(conn, ["AAPL"], on=today, fetch_fn=fetch_fn)

    assert result["fetched"] == 0
    assert result["cached"] == 1
    assert calls["n"] == 0  # 캐시 히트면 fetch_fn을 아예 호출하지 않는다
    row = conn.execute(
        "SELECT close FROM us_prices WHERE stock_code='AAPL' AND date='2026-07-12'"
    ).fetchone()
    assert row["close"] == 200.0  # 캐시된 값 유지(999.0으로 덮이지 않음)


def test_ensure_live_prices_us_skips_ticker_when_fetch_returns_none(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    today = date(2026, 7, 12)

    result = ensure_live_prices_us(conn, ["BADTICKER"], on=today, fetch_fn=lambda code: None)

    assert result["fetched"] == 0
    row = conn.execute(
        "SELECT close FROM us_prices WHERE stock_code='BADTICKER' AND date='2026-07-12'"
    ).fetchone()
    assert row is None


def test_ensure_live_prices_us_preserves_existing_volume_on_conflict(tmp_path, monkeypatch):
    """price_live.py(KR)와 동일하게 ON CONFLICT DO UPDATE로 다른 컬럼을 지우지 않는다(C-9 패턴).

    _has_today_us가 False를 반환하는 경쟁상황(동시 요청 등)을 강제 재현해, 이미
    volume이 채워진 오늘자 행이 있어도 close만 갱신되고 volume은 보존되는지 검증
    (us_prices에는 market_cap 컬럼이 없음 — us_company.market_cap만 존재).
    """
    import src.ingest.price_live as price_live

    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    today = date(2026, 7, 12)
    conn.execute(
        "INSERT INTO us_prices(stock_code, date, close, volume) VALUES "
        "('AAPL', '2026-07-12', 200.0, 55000000.0)"
    )
    conn.commit()
    monkeypatch.setattr(price_live, "_has_today_us", lambda conn, code, today: False)

    ensure_live_prices_us(conn, ["AAPL"], on=today, fetch_fn=lambda code: 210.5)

    row = conn.execute("SELECT close, volume FROM us_prices WHERE stock_code='AAPL'").fetchone()
    assert row["close"] == 210.5
    assert row["volume"] == 55000000.0
