"""price_live.py / backfill_prices.py의 UPSERT가 기존 market_cap을 지우지 않는지 검증.

기존에는 INSERT OR REPLACE(stock_code, date, close, market_cap=NULL)라서, 이미
market_cap이 채워진 행에 재적재가 겹치면(동시성/재실행) market_cap이 NULL로
지워지는 위험이 있었다. update_prices.py가 이미 쓰는 ON CONFLICT DO UPDATE SET
close=excluded.close 패턴으로 통일해 market_cap을 보존한다.
"""
from __future__ import annotations

from datetime import date

from src.db import connect, init_db


def _seed_price_with_marketcap(conn, code, d, close, market_cap):
    conn.execute(
        "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,?)",
        (code, d, close, market_cap),
    )
    conn.commit()


def test_ensure_live_prices_preserves_existing_market_cap_on_conflict(tmp_path, monkeypatch):
    import src.ingest.price_live as price_live

    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    today = date(2026, 7, 12)
    today_str = today.strftime("%Y-%m-%d")
    # 이미 market_cap이 채워진 오늘자 행이 있는 상태에서, _has_today가 False를
    # 반환하는 경쟁상황(동시 요청 등)을 강제 재현해 실제 ensure_live_prices()의
    # INSERT 경로가 기존 market_cap을 지우지 않는지 검증한다.
    _seed_price_with_marketcap(conn, "000001", today_str, 1000.0, 123456789.0)
    monkeypatch.setattr(price_live, "_has_today", lambda conn, code, today: False)
    monkeypatch.setattr(price_live, "_fetch_latest_close", lambda stock, code, on, lookback=10: 1100.0)
    monkeypatch.setitem(
        __import__("sys").modules,
        "pykrx",
        __import__("types").SimpleNamespace(stock=object()),
    )

    result = price_live.ensure_live_prices(conn, ["000001"], on=today)

    assert result["fetched"] == 1
    row = conn.execute(
        "SELECT close, market_cap FROM prices WHERE stock_code='000001' AND date=?", (today_str,)
    ).fetchone()
    # close도 기존 값(네이버 수정주가일 수 있음)을 보존한다 — pykrx 원본(1100.0)으로
    # 덮어쓰지 않는다(tests/test_price_upsert_preserves_close.py 참고).
    assert row["close"] == 1000.0
    assert row["market_cap"] == 123456789.0


def test_backfill_prices_ingest_one_preserves_existing_market_cap(tmp_path):
    from scripts.backfill_prices import _ingest_one

    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    _seed_price_with_marketcap(conn, "000002", "2026-01-05", 2000.0, 987654321.0)

    class _FakeIndex(list):
        pass

    import pandas as pd

    df = pd.DataFrame(
        {"종가": [2100.0]},
        index=pd.to_datetime(["2026-01-05"]),
    )

    class _FakeStock:
        def get_market_ohlcv_by_date(self, frm, to, code):
            return df

    n = _ingest_one(_FakeStock(), conn, "000002", "20260112")

    assert n == 1
    row = conn.execute(
        "SELECT close, market_cap FROM prices WHERE stock_code='000002' AND date='2026-01-05'"
    ).fetchone()
    # close도 기존 값(네이버 수정주가일 수 있음)을 보존한다 — pykrx 원본(2100.0)으로
    # 덮어쓰지 않는다(tests/test_price_upsert_preserves_close.py 참고).
    assert row["close"] == 2000.0
    assert row["market_cap"] == 987654321.0
