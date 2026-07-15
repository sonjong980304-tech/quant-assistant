"""pykrx 기반 주가 수집 5곳(krx.py 2곳, price_live.py, backfill_prices.py,
update_prices.py)이 이미 저장된 close(네이버의 수정주가일 수 있음)를 pykrx의
미조정 원본 종가로 덮어쓰지 않는지 검증한다.

배경: naver_prices.py의 문서화된 원칙(AC3)은 "네이버가 close의 source of truth"
(액면분할 반영된 수정주가)이다. 그런데 pykrx 기반 수집 5곳이 전부
"ON CONFLICT DO UPDATE SET close=excluded.close"로 무조건 덮어써서, 매일
실행되는 update_prices.py(launchd, 하루 3회) 등이 이 원칙을 어기고 액면분할
전후 불연속(가짜 급등락)을 만들어왔다. 이미 값이 있으면 보존하고, 없을 때만
채우도록(close=COALESCE(close, excluded.close)) 고친다.
"""
from __future__ import annotations

import types
from datetime import date

import pandas as pd

from src.db import connect, init_db


def _seed_close(conn, code, d, close):
    conn.execute(
        "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,NULL)",
        (code, d, close),
    )
    conn.commit()


def _fake_pykrx_module(df: pd.DataFrame):
    class _FakeStock:
        def get_market_ohlcv_by_date(self, frm, to, code, freq=None):
            return df

    return types.SimpleNamespace(stock=_FakeStock())


def test_krx_ingest_price_history_preserves_existing_close(tmp_path, monkeypatch):
    from src.ingest import krx

    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    _seed_close(conn, "005930", "2026-01-05", 53000.0)  # 네이버 수정주가로 이미 저장된 값
    conn.close()

    df = pd.DataFrame({"종가": [999999.0]}, index=pd.to_datetime(["2026-01-05"]))
    monkeypatch.setitem(__import__("sys").modules, "pykrx", _fake_pykrx_module(df))
    monkeypatch.setattr(krx, "COMPANIES", [("005930", "삼성전자", "KOSPI")])
    monkeypatch.setattr(krx, "_collect_shares", lambda on: {})

    krx.ingest_price_history(db_path=db_path, years=1, on=date(2026, 1, 12))

    conn = connect(db_path)
    row = conn.execute(
        "SELECT close FROM prices WHERE stock_code='005930' AND date='2026-01-05'"
    ).fetchone()
    assert row["close"] == 53000.0  # pykrx의 999999가 아니라 기존 값 보존


def test_krx_ingest_prices_preserves_existing_close(tmp_path, monkeypatch):
    from src.ingest import krx

    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    today_str = date(2026, 1, 12).strftime("%Y-%m-%d")
    _seed_close(conn, "005930", today_str, 53000.0)
    conn.close()

    monkeypatch.setattr(krx, "COMPANIES", [("005930", "삼성전자", "KOSPI")])
    monkeypatch.setattr(krx, "_collect_shares", lambda on: {})
    monkeypatch.setattr(krx, "_latest_close", lambda stock, code, on: (999999.0, today_str))
    monkeypatch.setitem(__import__("sys").modules, "pykrx", types.SimpleNamespace(stock=object()))

    krx.ingest_prices(db_path=db_path, on=date(2026, 1, 12), recompute=False)

    conn = connect(db_path)
    row = conn.execute(
        "SELECT close FROM prices WHERE stock_code='005930' AND date=?", (today_str,)
    ).fetchone()
    assert row["close"] == 53000.0


def test_ensure_live_prices_preserves_existing_close_on_race(tmp_path, monkeypatch):
    import src.ingest.price_live as price_live

    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    today = date(2026, 7, 12)
    today_str = today.strftime("%Y-%m-%d")
    _seed_close(conn, "000001", today_str, 53000.0)
    # _has_today가 False를 반환하는 경쟁상황을 강제 재현(기존 테스트와 동일한 방식)
    monkeypatch.setattr(price_live, "_has_today", lambda conn, code, today: False)
    monkeypatch.setattr(price_live, "_fetch_latest_close", lambda stock, code, on, lookback=10: 999999.0)
    monkeypatch.setitem(
        __import__("sys").modules, "pykrx",
        types.SimpleNamespace(stock=object()),
    )

    price_live.ensure_live_prices(conn, ["000001"], on=today)

    row = conn.execute(
        "SELECT close FROM prices WHERE stock_code='000001' AND date=?", (today_str,)
    ).fetchone()
    assert row["close"] == 53000.0


def test_backfill_prices_ingest_one_preserves_existing_close(tmp_path):
    from scripts.backfill_prices import _ingest_one

    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    _seed_close(conn, "000002", "2026-01-05", 53000.0)

    df = pd.DataFrame({"종가": [999999.0]}, index=pd.to_datetime(["2026-01-05"]))

    class _FakeStock:
        def get_market_ohlcv_by_date(self, frm, to, code):
            return df

    _ingest_one(_FakeStock(), conn, "000002", "20260112")

    row = conn.execute(
        "SELECT close FROM prices WHERE stock_code='000002' AND date='2026-01-05'"
    ).fetchone()
    assert row["close"] == 53000.0


def test_update_prices_preserves_existing_close(tmp_path, monkeypatch):
    import scripts.update_prices as update_prices_mod

    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    _seed_close(conn, "005930", "2026-07-13", 53000.0)
    conn.execute("INSERT INTO company(stock_code, name, market) VALUES ('005930','삼성전자','KOSPI')")
    conn.commit()
    conn.close()

    df = pd.DataFrame({"종가": [999999.0]}, index=pd.to_datetime(["2026-07-13"]))
    monkeypatch.setitem(__import__("sys").modules, "pykrx", _fake_pykrx_module(df))

    update_prices_mod.update_prices(db_path=db_path, on=date(2026, 7, 14))

    conn = connect(db_path)
    row = conn.execute(
        "SELECT close FROM prices WHERE stock_code='005930' AND date='2026-07-13'"
    ).fetchone()
    assert row["close"] == 53000.0


def test_update_prices_fills_close_when_missing(tmp_path, monkeypatch):
    """기존 값이 없는 날짜는 여전히 pykrx 값으로 정상 채워져야 한다(회귀 방지)."""
    import scripts.update_prices as update_prices_mod

    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    conn.execute("INSERT INTO company(stock_code, name, market) VALUES ('005930','삼성전자','KOSPI')")
    conn.commit()
    conn.close()

    df = pd.DataFrame({"종가": [263000.0]}, index=pd.to_datetime(["2026-07-13"]))
    monkeypatch.setitem(__import__("sys").modules, "pykrx", _fake_pykrx_module(df))

    update_prices_mod.update_prices(db_path=db_path, on=date(2026, 7, 14))

    conn = connect(db_path)
    row = conn.execute(
        "SELECT close FROM prices WHERE stock_code='005930' AND date='2026-07-13'"
    ).fetchone()
    assert row["close"] == 263000.0
