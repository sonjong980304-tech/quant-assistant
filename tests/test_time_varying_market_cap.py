"""시점별 상장주식수 기반 시가총액 계산 (look-ahead 방지 + staleness + 이상치 가드).

배경: 기존 ingest_price_history/ingest_prices 는 _collect_shares 로 "오늘 기준 최신
상장주식수" 상수 하나를 구해 과거 전 기간 종가에 곱했다. 유상증자·자사주소각 등으로
상장주식수가 바뀐 종목은 과거 시점 시가총액이 부정확했다. financials 에 이미 분기별
shares_outstanding 이 disclosed_date 와 함께 적재돼 있으므로, 그 시점 값을 조회해
market_cap = 종가 × 그 시점 상장주식수 로 계산한다.

US(SEC) 선례 data_access_us_sec._shares_outstanding_at 과 동일 사상의 KR 버전이다.
"""
from __future__ import annotations

import sys
import types
from datetime import date

import pandas as pd

from src.db import connect, init_db
from src.ingest.metrics import _shares_outstanding_at


def _conn(tmp_path):
    db = str(tmp_path / "shares.db")
    init_db(db)
    return connect(db)


def _seed_shares(conn, code, rows):
    """rows: [(quarter, disclosed_date, amount)]"""
    for q, disc, amt in rows:
        conn.execute(
            "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, account_name, amount) "
            "VALUES (?,?,?,'shares_outstanding','상장주식수',?)",
            (code, q, disc, amt),
        )
    conn.commit()


def _fake_pykrx_module(df: pd.DataFrame):
    class _FakeStock:
        def get_market_ohlcv_by_date(self, frm, to, code, freq=None):
            return df

    return types.SimpleNamespace(stock=_FakeStock())


# ── _shares_outstanding_at: look-ahead 방지 ──────────────────────────────────

def test_shares_at_returns_latest_disclosed_before_asof(tmp_path):
    """asof 시점까지 공시된(disclosed_date<=asof) 최신 상장주식수. 삼성전자 50:1 분할 경계."""
    conn = _conn(tmp_path)
    _seed_shares(conn, "005930", [
        ("2017Q4", "2018-04-02", 129_098_494),
        ("2018Q1", "2018-05-15", 128_386_494),   # 분할 전
        ("2018Q2", "2018-08-14", 6_419_324_700),  # 50:1 액면분할 후
        ("2018Q3", "2018-11-14", 6_419_324_700),
    ])
    # 분할 공시(2018-08-14) 전 → 분할 전 주식수(2018Q1)
    assert _shares_outstanding_at(conn, "005930", "2018-06-01") == 128_386_494
    # 분할 공시 후 → 분할 후 주식수(2018Q2). 미래 공시가 새어나오지 않음
    assert _shares_outstanding_at(conn, "005930", "2018-09-01") == 6_419_324_700


def test_shares_at_none_before_first_disclosure(tmp_path):
    """asof 가 첫 공시보다 이르면 look-ahead 방지로 None(미래값 안 씀)."""
    conn = _conn(tmp_path)
    _seed_shares(conn, "005930", [("2018Q2", "2018-08-14", 6_419_324_700)])
    assert _shares_outstanding_at(conn, "005930", "2018-06-01") is None


def test_shares_at_missing_returns_none(tmp_path):
    conn = _conn(tmp_path)
    assert _shares_outstanding_at(conn, "999999", "2020-06-01") is None


# ── _shares_outstanding_at: staleness 폴백 ────────────────────────────────────

def test_shares_at_stale_returns_none(tmp_path):
    """마지막 공시가 asof 대비 너무 오래(>400일)면 None(3단계에서 상수 폴백)."""
    conn = _conn(tmp_path)
    _seed_shares(conn, "111111", [("2018Q2", "2018-08-14", 1_000_000)])
    assert _shares_outstanding_at(conn, "111111", "2020-06-01") is None  # ~657일 경과


# ── _shares_outstanding_at: 이상치(×1000 단위오류 원복형 스파이크) 가드 ──────────

def test_shares_at_rejects_transient_1000x_spike(tmp_path):
    """한 분기만 ×1000 튀었다 원복하는 단위오류(003240 실측 패턴)는 이상치로 None."""
    conn = _conn(tmp_path)
    _seed_shares(conn, "003240", [
        ("2021Q1", "2021-05-15", 1_113_400),
        ("2021Q2", "2021-08-14", 1_113_400_000),  # ×1000 오류
        ("2021Q3", "2021-11-14", 1_113_400),
        ("2021Q4", "2022-04-01", 1_113_400),
    ])
    # 스파이크 공시 직후 asof → 이상치로 None
    assert _shares_outstanding_at(conn, "003240", "2021-09-01") is None
    # 정상 분기 → 정상값(가드가 정상값까지 죽이지 않음)
    assert _shares_outstanding_at(conn, "003240", "2021-12-01") == 1_113_400


# ── ingest_price_history: 시점별 주식수 배선 ─────────────────────────────────

def test_ingest_price_history_uses_time_varying_shares(tmp_path, monkeypatch):
    """상장주식수가 중간에 2배로 바뀐 종목: 같은 종가라도 시점별 시총이 달라야 한다."""
    from src.ingest import krx

    db_path = str(tmp_path / "t.db")
    init_db(db_path)
    conn = connect(db_path)
    _seed_shares(conn, "005930", [
        ("2023Q4", "2024-04-01", 1_000_000),
        ("2024Q2", "2024-08-14", 2_000_000),  # 유상증자 시뮬레이션(2배)
    ])
    conn.close()

    df = pd.DataFrame({"종가": [100.0, 100.0]},
                      index=pd.to_datetime(["2024-06-28", "2024-09-30"]))
    monkeypatch.setitem(sys.modules, "pykrx", _fake_pykrx_module(df))
    monkeypatch.setattr(krx, "COMPANIES", [("005930", "삼성전자", "KOSPI")])
    monkeypatch.setattr(krx, "_collect_shares", lambda on: {"005930": 9_999_999})  # 폴백(안 쓰여야)

    krx.ingest_price_history(db_path=db_path, years=1, on=date(2024, 10, 1))

    conn = connect(db_path)
    early = conn.execute(
        "SELECT market_cap FROM prices WHERE stock_code='005930' AND date='2024-06-28'"
    ).fetchone()
    late = conn.execute(
        "SELECT market_cap FROM prices WHERE stock_code='005930' AND date='2024-09-30'"
    ).fetchone()
    assert early["market_cap"] == round(100.0 * 1_000_000)   # 그 시점(1M주) 시총
    assert late["market_cap"] == round(100.0 * 2_000_000)    # 증자 후(2M주) 시총


def test_ingest_price_history_falls_back_to_constant_when_no_series(tmp_path, monkeypatch):
    """financials 에 shares 시계열이 없으면 기존 _collect_shares 상수로 폴백(결측보다 나음)."""
    from src.ingest import krx

    db_path = str(tmp_path / "t.db")
    init_db(db_path)

    df = pd.DataFrame({"종가": [100.0]}, index=pd.to_datetime(["2024-06-28"]))
    monkeypatch.setitem(sys.modules, "pykrx", _fake_pykrx_module(df))
    monkeypatch.setattr(krx, "COMPANIES", [("005930", "삼성전자", "KOSPI")])
    monkeypatch.setattr(krx, "_collect_shares", lambda on: {"005930": 5_000_000})

    krx.ingest_price_history(db_path=db_path, years=1, on=date(2024, 10, 1))

    conn = connect(db_path)
    row = conn.execute(
        "SELECT market_cap FROM prices WHERE stock_code='005930' AND date='2024-06-28'"
    ).fetchone()
    assert row["market_cap"] == round(100.0 * 5_000_000)  # 폴백 상수 사용


def test_ingest_prices_uses_time_varying_shares(tmp_path, monkeypatch):
    """일일 스냅샷 경로(ingest_prices)도 그 시점 공시된 주식수를 써야 한다."""
    from src.ingest import krx

    db_path = str(tmp_path / "t.db")
    init_db(db_path)
    conn = connect(db_path)
    _seed_shares(conn, "005930", [
        ("2024Q1", "2024-05-15", 1_000_000),
        ("2024Q2", "2024-08-14", 2_000_000),
    ])
    conn.close()

    monkeypatch.setattr(krx, "COMPANIES", [("005930", "삼성전자", "KOSPI")])
    monkeypatch.setattr(krx, "_collect_shares", lambda on: {"005930": 9_999_999})
    monkeypatch.setattr(krx, "_latest_close", lambda stock, code, on: (100.0, "2024-09-30"))
    monkeypatch.setitem(sys.modules, "pykrx", types.SimpleNamespace(stock=object()))

    krx.ingest_prices(db_path=db_path, on=date(2024, 10, 1), recompute=False)

    conn = connect(db_path)
    row = conn.execute(
        "SELECT market_cap FROM prices WHERE stock_code='005930' AND date='2024-09-30'"
    ).fetchone()
    assert row["market_cap"] == round(100.0 * 2_000_000)  # 2024-08-14 공시 주식수(2M)


# ── 마이그레이션(backfill_marketcap): 재계산 + 확정 스파이크 오염 정리 ──────────

def _seed_price(conn, code, d, close, market_cap):
    conn.execute(
        "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,?)",
        (code, d, close, market_cap),
    )
    conn.commit()


def test_backfill_marketcap_recomputes_time_varying(tmp_path):
    """저장된 market_cap 이 시점별 값과 다르면 그 시점 주식수로 재계산(UPDATE)."""
    from scripts.backfill_marketcap import backfill_marketcap

    db_path = str(tmp_path / "m.db")
    init_db(db_path)
    conn = connect(db_path)
    _seed_shares(conn, "005930", [
        ("2024Q1", "2024-05-15", 1_000_000),
        ("2024Q2", "2024-08-14", 2_000_000),
    ])
    # 잘못된(옛 상수) cap 저장 → 시점별 값으로 교정돼야 한다
    _seed_price(conn, "005930", "2024-06-28", 100.0, 999_999_900)  # 틀린 값
    conn.close()

    backfill_marketcap(db_path=db_path)

    conn = connect(db_path)
    row = conn.execute(
        "SELECT market_cap FROM prices WHERE stock_code='005930' AND date='2024-06-28'"
    ).fetchone()
    assert row["market_cap"] == round(100.0 * 1_000_000)  # 그 시점(1M주)로 교정


def test_backfill_marketcap_nulls_confirmed_spike_pollution(tmp_path):
    """구버전 무가드 backfill 이 써둔 ×1000 스파이크 오염(저장 cap==종가×스파이크주식수)은 NULL 로 정리."""
    from scripts.backfill_marketcap import backfill_marketcap

    db_path = str(tmp_path / "m.db")
    init_db(db_path)
    conn = connect(db_path)
    _seed_shares(conn, "003240", [
        ("2021Q3", "2021-11-15", 1_113_400),
        ("2021Q4", "2022-02-25", 1_113_400_000),  # ×1000 단위오류(실측)
        ("2021Q4R", "2022-03-17", 1_113_400),       # 정정 원복
    ])
    # 정상 행(스파이크 전): 이미 올바른 시점별 cap → 유지
    _seed_price(conn, "003240", "2021-12-01", 1_000_000, round(1_000_000 * 1_113_400))
    # 오염 행(스파이크 공시 직후): 저장 cap == 종가×스파이크주식수 → NULL 로 정리돼야
    _seed_price(conn, "003240", "2022-03-01", 1_000_000, round(1_000_000 * 1_113_400_000))
    conn.close()

    backfill_marketcap(db_path=db_path)

    conn = connect(db_path)
    normal = conn.execute(
        "SELECT market_cap FROM prices WHERE stock_code='003240' AND date='2021-12-01'"
    ).fetchone()
    polluted = conn.execute(
        "SELECT market_cap FROM prices WHERE stock_code='003240' AND date='2022-03-01'"
    ).fetchone()
    assert normal["market_cap"] == round(1_000_000 * 1_113_400)  # 정상값 유지
    assert polluted["market_cap"] is None  # ×1000 오염 → NULL 정리


def test_backfill_marketcap_dry_run_makes_no_change(tmp_path):
    """dry_run=True 는 UPDATE 하지 않고 규모만 집계한다."""
    from scripts.backfill_marketcap import backfill_marketcap

    db_path = str(tmp_path / "m.db")
    init_db(db_path)
    conn = connect(db_path)
    _seed_shares(conn, "005930", [("2024Q1", "2024-05-15", 1_000_000)])
    _seed_price(conn, "005930", "2024-06-28", 100.0, 999_999_900)  # 틀린 값
    conn.close()

    report = backfill_marketcap(db_path=db_path, dry_run=True)

    conn = connect(db_path)
    row = conn.execute(
        "SELECT market_cap FROM prices WHERE stock_code='005930' AND date='2024-06-28'"
    ).fetchone()
    assert row["market_cap"] == 999_999_900  # 변경 없음
    assert report["mode"] == "dry-run"
    assert report["market_cap_changed"] == 1  # 바뀔 행 수는 집계
