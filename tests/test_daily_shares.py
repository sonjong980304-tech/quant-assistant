"""일자별 상장주식수(daily_shares) 수집·적재 + backfill_marketcap 우선순위 검증.

액면분할·무상증자가 분기 결산일 사이에 나면 financials.shares_outstanding 이 옛 값에
멈춰 prices.market_cap 이 축소되는 버그를, pykrx 일자별 상장주식수(daily_shares)를
최우선 소스로 써서 해소한다. 실제 pykrx 네트워크 호출은 mock 처리한다.
"""
from __future__ import annotations

import sys
import types

from src.db import connect, init_db
from src.ingest.shares_daily import fetch_daily_shares, upsert_daily_shares


def _conn(tmp_path):
    db = str(tmp_path / "ds.db")
    init_db(db)
    return connect(db)


# ── upsert_daily_shares ──────────────────────────────────────────────────────

def test_upsert_daily_shares_inserts_and_skips_nonpositive(tmp_path):
    """정상 행은 적재, 0/음수/None/NaN 행은 스킵한다."""
    conn = _conn(tmp_path)
    rows = [
        {"date": "2026-04-01", "shares_outstanding": 2_199_268},
        {"date": "2026-04-13", "shares_outstanding": 21_992_680},
        {"date": "2026-04-14", "shares_outstanding": 0},          # 스킵
        {"date": "2026-04-15", "shares_outstanding": -5},         # 스킵
        {"date": "2026-04-16", "shares_outstanding": None},       # 스킵
        {"date": "2026-04-17", "shares_outstanding": float("nan")},  # 스킵
    ]
    n = upsert_daily_shares(conn, "134380", rows)
    assert n == 2
    got = conn.execute(
        "SELECT date, shares_outstanding FROM daily_shares WHERE stock_code='134380' ORDER BY date"
    ).fetchall()
    assert [(r["date"], r["shares_outstanding"]) for r in got] == [
        ("2026-04-01", 2_199_268.0),
        ("2026-04-13", 21_992_680.0),
    ]


def test_upsert_daily_shares_idempotent_replace(tmp_path):
    """같은 (code,date) 재적재는 INSERT OR REPLACE 로 값만 갱신(행 중복 없음)."""
    conn = _conn(tmp_path)
    upsert_daily_shares(conn, "134380", [{"date": "2026-04-13", "shares_outstanding": 21_992_680}])
    upsert_daily_shares(conn, "134380", [{"date": "2026-04-13", "shares_outstanding": 21_992_681}])
    got = conn.execute(
        "SELECT shares_outstanding FROM daily_shares WHERE stock_code='134380' AND date='2026-04-13'"
    ).fetchall()
    assert len(got) == 1
    assert got[0]["shares_outstanding"] == 21_992_681.0


# ── fetch_daily_shares (pykrx mock) ──────────────────────────────────────────

def test_fetch_daily_shares_parses_pykrx_df(tmp_path, monkeypatch):
    """pykrx get_market_cap_by_date 반환 DataFrame 을 [{date,shares_outstanding}] 로 변환."""
    import pandas as pd

    df = pd.DataFrame(
        {"시가총액": [1, 2], "거래량": [0, 0], "거래대금": [0, 0],
         "상장주식수": [2_199_268, 21_992_680]},
        index=pd.to_datetime(["2026-04-01", "2026-04-13"]),
    )
    fake_stock = types.SimpleNamespace(get_market_cap_by_date=lambda frm, to, code: df)
    monkeypatch.setitem(sys.modules, "pykrx", types.SimpleNamespace(stock=fake_stock))

    out = fetch_daily_shares("134380", "20260401", "20260413")
    assert out == [
        {"date": "2026-04-01", "shares_outstanding": 2_199_268},
        {"date": "2026-04-13", "shares_outstanding": 21_992_680},
    ]


def test_fetch_daily_shares_empty_df_returns_empty(tmp_path, monkeypatch):
    """빈 응답(상장 전 구간 등)은 빈 리스트(오류 아님)."""
    import pandas as pd

    fake_stock = types.SimpleNamespace(get_market_cap_by_date=lambda frm, to, code: pd.DataFrame())
    monkeypatch.setitem(sys.modules, "pykrx", types.SimpleNamespace(stock=fake_stock))
    assert fetch_daily_shares("000000", "20260401", "20260413") == []


# ── _daily_shares_at: date<=asof 최신값(bisect) ──────────────────────────────

def test_daily_shares_at_returns_latest_before_asof():
    from scripts.backfill_marketcap import _daily_shares_at

    dates = ["2026-04-01", "2026-04-13"]
    vals = [2_199_268, 21_992_680]
    assert _daily_shares_at(dates, vals, "2026-04-01") == 2_199_268   # 정확 일치
    assert _daily_shares_at(dates, vals, "2026-04-12") == 2_199_268   # 분할 전(직전값)
    assert _daily_shares_at(dates, vals, "2026-04-13") == 21_992_680  # 분할일(당일)
    assert _daily_shares_at(dates, vals, "2026-04-20") == 21_992_680  # 분할 후(최신값)
    assert _daily_shares_at(dates, vals, "2026-03-31") is None        # 첫 데이터 이전


# ── backfill_marketcap: daily_shares 최우선 사용 ──────────────────────────────

def test_backfill_marketcap_prefers_daily_shares_over_financials(tmp_path):
    """액면분할 후 시점의 시총은 financials(옛 주식수)가 아니라 daily_shares(분할 후)로 계산."""
    from scripts.backfill_marketcap import backfill_marketcap

    db_path = str(tmp_path / "m.db")
    init_db(db_path)
    conn = connect(db_path)
    # financials: 분할 전 주식수에 멈춰 있음(결산일이 분할일보다 이전)
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, account_name, amount) "
        "VALUES ('134380','2025Q4','2026-02-15','shares_outstanding','상장주식수',2199268)"
    )
    # daily_shares: 분할일(2026-04-13)에 10배로 정확히 발효
    upsert_daily_shares(conn, "134380", [
        {"date": "2026-04-01", "shares_outstanding": 2_199_268},
        {"date": "2026-04-13", "shares_outstanding": 21_992_680},
    ])
    # 분할 후 주가일: 저장 cap 은 옛(분할 전) 주식수로 잘못 계산돼 있음 → 교정돼야
    conn.execute(
        "INSERT INTO prices(stock_code, date, close, market_cap) VALUES ('134380','2026-04-20',10000,?)",
        (round(10000 * 2_199_268),),
    )
    conn.commit()
    conn.close()

    backfill_marketcap(db_path=db_path)

    conn = connect(db_path)
    row = conn.execute(
        "SELECT market_cap FROM prices WHERE stock_code='134380' AND date='2026-04-20'"
    ).fetchone()
    # daily_shares(분할 후 21,992,680)로 계산 — financials(2,199,268) 아님
    assert row["market_cap"] == round(10000 * 21_992_680)
    assert row["market_cap"] != round(10000 * 2_199_268)


# ── _fromdate_for: 증분 갱신(매일 예약 대비) ─────────────────────────────────

def test_fromdate_for_incremental_starts_day_after_last_covered_date(tmp_path):
    """daily_shares 에 이미 데이터가 있으면 그 다음날부터만(전체 재수집 아님)."""
    from scripts.backfill_shares_daily import _fromdate_for

    conn = _conn(tmp_path)
    upsert_daily_shares(conn, "134380", [
        {"date": "2026-07-21", "shares_outstanding": 21_992_680},
    ])
    conn.commit()
    assert _fromdate_for(conn, "134380") == "20260722"


def test_fromdate_for_falls_back_to_price_min_date_without_daily_shares(tmp_path):
    """daily_shares 가 아예 없는 종목(최초 백필)은 기존대로 prices 최소 date부터."""
    from scripts.backfill_shares_daily import _fromdate_for

    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO prices(stock_code, date, close, market_cap) VALUES ('134380','2020-03-02',100,NULL)"
    )
    conn.commit()
    assert _fromdate_for(conn, "134380") == "20200302"


def test_fromdate_for_falls_back_to_start_date_without_prices_or_daily_shares(tmp_path):
    """prices 도 daily_shares 도 없는 종목(신규상장 등)은 프로젝트 시작일(START_DATE)."""
    from scripts.backfill_shares_daily import START_DATE, _fromdate_for

    conn = _conn(tmp_path)
    assert _fromdate_for(conn, "999999") == START_DATE


def test_backfill_shares_daily_skips_already_up_to_date_code_without_fetching(tmp_path, monkeypatch):
    """daily_shares 가 이미 오늘 날짜까지 커버돼 있으면(증분분이 없으면) pykrx 호출 자체를 안 한다."""
    from scripts.backfill_shares_daily import backfill_shares_daily

    conn = _conn(tmp_path)
    conn_path = str(tmp_path / "ds.db")
    upsert_daily_shares(conn, "134380", [{"date": "2026-07-22", "shares_outstanding": 21_992_680}])
    conn.commit()
    conn.close()

    def _boom(code, frm, to):
        raise AssertionError("이미 오늘까지 커버된 종목인데 pykrx를 호출했다")

    monkeypatch.setattr("scripts.backfill_shares_daily.fetch_daily_shares", _boom)
    report = backfill_shares_daily(
        db_path=conn_path, codes=["134380"], on=__import__("datetime").date(2026, 7, 22),
    )
    assert report["ok"] == 0
    assert report["failed"] == 0
    assert report["skipped"] == 1


def test_backfill_marketcap_falls_back_to_financials_without_daily(tmp_path):
    """daily_shares 가 없는 종목은 기존대로 financials 기반으로 계산(회귀 보호)."""
    from scripts.backfill_marketcap import backfill_marketcap

    db_path = str(tmp_path / "m.db")
    init_db(db_path)
    conn = connect(db_path)
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, account_name, amount) "
        "VALUES ('005930','2024Q1','2024-05-15','shares_outstanding','상장주식수',1000000)"
    )
    conn.execute(
        "INSERT INTO prices(stock_code, date, close, market_cap) VALUES ('005930','2024-06-28',100,999999900)"
    )
    conn.commit()
    conn.close()

    backfill_marketcap(db_path=db_path)

    conn = connect(db_path)
    row = conn.execute(
        "SELECT market_cap FROM prices WHERE stock_code='005930' AND date='2024-06-28'"
    ).fetchone()
    assert row["market_cap"] == round(100 * 1_000_000)  # financials 폴백으로 교정
