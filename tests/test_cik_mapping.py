"""SEC 티커→CIK 매핑 단위 테스트 (TDD, AC1/AC2/AC13).

.omc/specs/brainstorming-sec-edgar-us-financials-backfill.md:
- AC1: us_company 전종목에 SEC 공식 company_tickers.json 로 CIK 매핑.
- AC2: 매핑률<95% 이면 매핑 실패 종목 목록을 포함한 명시적 경고 리포트(조용한 스킵 금지).
- AC13: CIK 매핑 로직을 TDD(RED→GREEN)로 구현.

네트워크는 fetch_tickers_fn 주입(DI)으로 분리한다(기존 us_universe/us_financials 관례와 동일).
"""
from __future__ import annotations

import sqlite3

from src.db import init_db
from src.ingest import cik_mapping as cm


# --------------------------------------------------------------------------
# format_cik — 10자리 zero-pad (SEC API 호출용 CIK 표준형)
# --------------------------------------------------------------------------
def test_format_cik_zero_pads_to_10_digits():
    assert cm.format_cik(320193) == "0000320193"
    assert cm.format_cik("320193") == "0000320193"
    assert cm.format_cik("0000320193") == "0000320193"  # 이미 패딩된 값도 안전


# --------------------------------------------------------------------------
# parse_company_tickers — SEC company_tickers.json 파싱
# --------------------------------------------------------------------------
def test_parse_company_tickers_maps_ticker_to_padded_cik():
    raw = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    }
    mapping = cm.parse_company_tickers(raw)
    assert mapping["AAPL"] == "0000320193"
    assert mapping["MSFT"] == "0000789019"


def test_parse_company_tickers_uppercases_ticker_keys():
    raw = {"0": {"cik_str": 1045810, "ticker": "nvda", "title": "NVIDIA CORP"}}
    mapping = cm.parse_company_tickers(raw)
    assert mapping["NVDA"] == "0001045810"


# --------------------------------------------------------------------------
# backfill_ciks — us_company 에 CIK UPDATE + 매핑 리포트
# --------------------------------------------------------------------------
def _seed_companies(tmp_path, tickers) -> str:
    db = tmp_path / "cik.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    for t in tickers:
        conn.execute(
            "INSERT INTO us_company(stock_code, name, exchange, updated_at) VALUES (?,?,?,?)",
            (t, f"{t} Inc", "NASDAQ", "2026-07-19"),
        )
    conn.commit()
    conn.close()
    return str(db)


def _fake_fetch(mapping):
    def _f():
        return mapping
    return _f


def test_backfill_ciks_writes_cik_to_us_company(tmp_path):
    db = _seed_companies(tmp_path, ["AAPL", "MSFT"])
    fetch = _fake_fetch({"AAPL": "0000320193", "MSFT": "0000789019"})
    cm.backfill_ciks(db_path=db, fetch_tickers_fn=fetch)
    conn = sqlite3.connect(db)
    rows = dict(conn.execute("SELECT stock_code, cik FROM us_company").fetchall())
    conn.close()
    assert rows["AAPL"] == "0000320193"
    assert rows["MSFT"] == "0000789019"


def test_backfill_ciks_reports_full_coverage_without_warning(tmp_path):
    db = _seed_companies(tmp_path, ["AAPL", "MSFT"])
    fetch = _fake_fetch({"AAPL": "0000320193", "MSFT": "0000789019"})
    report = cm.backfill_ciks(db_path=db, fetch_tickers_fn=fetch)
    assert report["total"] == 2
    assert report["matched"] == 2
    assert report["matched_rate"] == 1.0
    assert report["warning"] is False
    assert report["unmatched"] == []


def test_backfill_ciks_warns_and_lists_unmatched_below_threshold(tmp_path):
    """매핑률<95% 시 warning=True 이고 unmatched 목록이 채워져야 한다(AC2, 조용한 스킵 금지)."""
    db = _seed_companies(tmp_path, ["AAPL", "MSFT", "GOOGL", "ZZZZ"])
    # SEC 매핑에 ZZZZ 없음 → 3/4 = 75% < 95%
    fetch = _fake_fetch({
        "AAPL": "0000320193", "MSFT": "0000789019", "GOOGL": "0001652044",
    })
    report = cm.backfill_ciks(db_path=db, fetch_tickers_fn=fetch)
    assert report["total"] == 4
    assert report["matched"] == 3
    assert report["matched_rate"] == 0.75
    assert report["warning"] is True
    assert "ZZZZ" in report["unmatched"]
    assert report["threshold"] == 0.95


def test_backfill_ciks_no_warning_at_or_above_threshold(tmp_path):
    """정확히 95% 이상이면 경고하지 않는다(경계값)."""
    db = _seed_companies(tmp_path, [f"T{i}" for i in range(20)])  # 20종목
    mapping = {f"T{i}": cm.format_cik(1000 + i) for i in range(19)}  # 19/20 = 95%
    report = cm.backfill_ciks(db_path=db, fetch_tickers_fn=_fake_fetch(mapping))
    assert report["matched_rate"] == 0.95
    assert report["warning"] is False


def test_backfill_ciks_idempotent_skips_already_mapped(tmp_path):
    """이미 cik 가 채워진 종목은 재매핑 대상에서 제외(멱등, 기존 backfill 관례)."""
    db = _seed_companies(tmp_path, ["AAPL"])
    conn = sqlite3.connect(db)
    conn.execute("UPDATE us_company SET cik='0000320193' WHERE stock_code='AAPL'")
    conn.commit()
    conn.close()
    # fetch 가 호출되면 안 됨을 확인하기 위해 예외를 던지는 fetch 를 넣는다
    def _boom():
        raise AssertionError("이미 매핑된 종목만 있으면 fetch 를 부르지 말아야 함")
    report = cm.backfill_ciks(db_path=db, fetch_tickers_fn=_boom)
    assert report["total"] == 0  # 대상 없음
