"""us_universe.py 정규화 로직 + NASDAQ 스크리너 API 파서 테스트.

.omc/specs/brainstorming-us-market-data-plane.md AC2(거래소 필터+Symbol 중복제거),
AC3(market_cap 파싱), AC14(sector 원본 taxonomy 그대로 보존) 검증.

2026-07-12 소스 전환: investing.com Stock Screener는 2페이지 이상이 InvestingPro
유료 게이트로 막혀있음을 Playwright 브라우저 레벨(실제 클릭→XHR 캡처)로 확정했고,
로그인해도 동일함을 재확인했다(무료 계정 로그인 성공 상태에서도 totalItems:0).
대신 NASDAQ 공식 스크리너 API(`api.nasdaq.com/api/screener/stocks`)는 인증/로그인
없이 curl로 즉시 전체 결과를 반환함을 실측했다: NASDAQ 4,110 + NYSE 2,718 +
AMEX 295 = 7,123건, tableonly=true면 limit 무시하고 거래소당 1회 호출로 전량 수신.
sector도 12개 표준 카테고리(Technology/Health Care/Finance 등)라 investing.com
원문 taxonomy보다 다루기 쉽다. 이 파일은 investing.com 전용 파서(extract_screener_
results/parse_screener_rows, __NEXT_DATA__ 기반) 테스트를 전량 NASDAQ API 기반으로
교체한다. normalize_universe_rows는 소스 무관 순수 로직이라 그대로 유지한다.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.db import connect, init_db
from src.ingest import us_universe
from src.ingest.us_universe import (
    ingest_us_universe,
    normalize_universe_rows,
    parse_nasdaq_rows,
)

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# --------------------------------------------------------------------------
# normalize_universe_rows (소스 무관 순수 로직 — 그대로 유지)
# --------------------------------------------------------------------------
def test_normalize_filters_out_disallowed_exchanges():
    rows = [
        {"symbol": "AAPL", "name": "Apple Inc", "exchange": "NASDAQ", "sector": "Technology", "market_cap": "3.4T"},
        {"symbol": "VOD", "name": "Vodafone Group", "exchange": "LSE", "sector": "Telecom", "market_cap": "20.1B"},
    ]
    result = normalize_universe_rows(rows)
    symbols = {r["symbol"] for r in result}
    assert symbols == {"AAPL"}


def test_normalize_dedups_by_symbol_keeping_first_occurrence():
    rows = [
        {"symbol": "MSFT", "name": "Microsoft Corp", "exchange": "NASDAQ", "sector": "Software", "market_cap": "3.1T"},
        {"symbol": "MSFT", "name": "Microsoft Corp (dup page)", "exchange": "NASDAQ", "sector": "Software", "market_cap": "3.1T"},
    ]
    result = normalize_universe_rows(rows)
    assert len(result) == 1
    assert result[0]["name"] == "Microsoft Corp"


def test_normalize_drops_rows_with_missing_name():
    rows = [
        {"symbol": "XYZ", "name": "", "exchange": "NYSE", "sector": "Industrials", "market_cap": "500M"},
        {"symbol": "ABC", "name": None, "exchange": "NYSE", "sector": "Industrials", "market_cap": "500M"},
        {"symbol": "GE", "name": "General Electric", "exchange": "NYSE", "sector": "Industrials", "market_cap": "180B"},
    ]
    result = normalize_universe_rows(rows)
    symbols = {r["symbol"] for r in result}
    assert symbols == {"GE"}


def test_normalize_parses_market_cap_via_shared_parser():
    rows = [
        {"symbol": "AAPL", "name": "Apple Inc", "exchange": "NASDAQ", "sector": "Technology", "market_cap": "2.32T"},
    ]
    result = normalize_universe_rows(rows)
    assert result[0]["market_cap"] == 2.32e12


def test_normalize_preserves_sector_raw_taxonomy_unchanged():
    rows = [
        {"symbol": "NVDA", "name": "NVIDIA Corp", "exchange": "NASDAQ", "sector": "Technology", "market_cap": "3.0T"},
        {"symbol": "AMEX1", "name": "Amex Listed Co", "exchange": "NYSE Amex", "sector": "Technology", "market_cap": "1.0B"},
    ]
    result = normalize_universe_rows(rows)
    assert len(result) == 2
    assert {r["sector"] for r in result} == {"Technology"}


# --------------------------------------------------------------------------
# parse_nasdaq_rows: NASDAQ 스크리너 API 원시 rows → 정규화 입력 형태 변환
# --------------------------------------------------------------------------
def test_parse_nasdaq_rows_maps_fields_and_attaches_exchange_label():
    raw = [
        {"symbol": "AACG", "name": "ATA Creativity Global", "marketCap": "42169397.00", "sector": "Real Estate", "industry": "Other"},
    ]
    rows = parse_nasdaq_rows(raw, "NASDAQ")
    assert rows == [
        {"symbol": "AACG", "name": "ATA Creativity Global", "exchange": "NASDAQ", "sector": "Real Estate", "market_cap": "42169397.00"},
    ]


def test_parse_nasdaq_rows_empty_sector_becomes_empty_string_not_crash():
    # 블랭크체크 SPAC류는 sector가 빈 문자열("")로 옴 — normalize 단계에서 None으로 정리된다.
    raw = [{"symbol": "AACB", "name": "Artius II Acquisition", "marketCap": "0.00", "sector": "", "industry": ""}]
    rows = parse_nasdaq_rows(raw, "NASDAQ")
    assert rows[0]["sector"] == ""
    normalized = normalize_universe_rows(rows)
    assert normalized[0]["sector"] is None
    assert normalized[0]["market_cap"] == 0.0


def test_parse_nasdaq_rows_real_captured_fixture():
    # 2026-07-12 curl로 api.nasdaq.com/api/screener/stocks를 직접 받아 확보한 실제
    # 응답(exchange=nasdaq 상위 5건)을 리포에 커밋한 fixture. 합성 데이터가 아니라
    # 실측 구조로 회귀를 검증한다(CI 재현성 확보, P1 fixture 관례와 동일).
    data = json.loads((_FIXTURES_DIR / "nasdaq_screener_sample.json").read_text(encoding="utf-8"))
    raw_rows = data["data"]["rows"]
    rows = parse_nasdaq_rows(raw_rows, "NASDAQ")
    assert [r["symbol"] for r in rows] == ["AACB", "AACBR", "AACG", "AACI", "AACIU"]
    normalized = normalize_universe_rows(rows)
    aacg = next(r for r in normalized if r["symbol"] == "AACG")
    assert aacg["sector"] == "Real Estate"
    assert aacg["exchange"] == "NASDAQ"
    assert aacg["market_cap"] == 42169397.0


# --------------------------------------------------------------------------
# ingest_us_universe 오케스트레이터: 거래소(NASDAQ/NYSE/AMEX) 순회 — 페이지네이션 아님
# --------------------------------------------------------------------------
# NASDAQ API는 거래소당 1회 호출로 전량 반환하므로(2026-07-12 실측, tableonly=true가
# limit 무시), investing.com 페이지 순회 방식과 달리 오케스트레이터는 3개 거래소를
# 순회하며 각각 독립적으로 fetch한다. 한 거래소 실패는 격리하고 나머지는 계속
# 진행한다(페이지 기반과 달리 거래소 간 순서 의존성이 없으므로 중단이 아니라 격리가
# 맞는 전략 — yfinance 개별종목 실패 격리와 동일 원칙, AC6).
def test_ingest_us_universe_upserts_companies_from_all_exchanges(tmp_path, monkeypatch):
    db = str(tmp_path / "usu1.db")
    monkeypatch.setattr(us_universe, "send_slack_alert", lambda *a, **kw: None)
    monkeypatch.setattr(us_universe.time, "sleep", lambda *a, **kw: None)
    fixtures = {
        "nasdaq": [{"symbol": "NVDA", "name": "NVIDIA", "marketCap": "5110000000000.00", "sector": "Technology"}],
        "nyse": [{"symbol": "GE", "name": "General Electric", "marketCap": "180000000000.00", "sector": "Industrials"}],
        "amex": [{"symbol": "SMALL", "name": "Small Cap Co", "marketCap": "50000000.00", "sector": "Finance"}],
    }
    result = ingest_us_universe(db_path=db, fetch_exchange=lambda ex: fixtures[ex])

    conn = connect(db)
    rows = {r["stock_code"]: r for r in conn.execute(
        "SELECT stock_code, name, exchange, sector, market_cap FROM us_company")}
    conn.close()
    assert set(rows) == {"NVDA", "GE", "SMALL"}
    assert rows["NVDA"]["exchange"] == "NASDAQ"
    assert rows["GE"]["exchange"] == "NYSE"
    assert rows["SMALL"]["exchange"] == "NYSE Amex"
    assert result["exchanges"] == ["nasdaq", "nyse", "amex"]
    assert result["failed_exchanges"] == []
    assert result["companies"] == 3


def test_ingest_us_universe_isolates_single_exchange_failure(tmp_path, monkeypatch):
    db = str(tmp_path / "usu2.db")
    alerts: list[str] = []
    monkeypatch.setattr(us_universe, "send_slack_alert", lambda msg, **kw: alerts.append(msg))
    monkeypatch.setattr(us_universe.time, "sleep", lambda *a, **kw: None)

    def fetch(ex):
        if ex == "nyse":
            raise ConnectionError("NYSE 요청 실패(mock)")
        return [{"symbol": ex.upper(), "name": f"{ex} co", "marketCap": "1000000000.00", "sector": "Technology"}]

    result = ingest_us_universe(db_path=db, fetch_exchange=fetch)

    conn = connect(db)
    symbols = {r["stock_code"] for r in conn.execute("SELECT stock_code FROM us_company")}
    conn.close()
    # NYSE만 실패, 나머지(nasdaq/amex)는 계속 진행돼 upsert된다(격리, 중단 아님).
    assert symbols == {"NASDAQ", "AMEX"}
    assert result["failed_exchanges"] == ["nyse"]
    assert result["exchanges"] == ["nasdaq", "amex"]
    assert len(alerts) == 1
    assert "nyse" in alerts[0]


def test_ingest_us_universe_sleeps_at_least_two_seconds_between_exchange_calls(tmp_path, monkeypatch):
    db = str(tmp_path / "usu3.db")
    monkeypatch.setattr(us_universe, "send_slack_alert", lambda *a, **kw: None)
    sleep_calls: list[float] = []
    monkeypatch.setattr(us_universe.time, "sleep", lambda s: sleep_calls.append(s))

    ingest_us_universe(db_path=db, fetch_exchange=lambda ex: [])

    assert sleep_calls  # 거래소 호출 사이 딜레이가 최소 1회 이상 있어야 함(AC11)
    assert all(s >= 2.0 for s in sleep_calls)


def test_ingest_us_universe_applies_exchange_filter_and_dedup(tmp_path, monkeypatch):
    db = str(tmp_path / "usu4.db")
    monkeypatch.setattr(us_universe, "send_slack_alert", lambda *a, **kw: None)
    monkeypatch.setattr(us_universe.time, "sleep", lambda *a, **kw: None)

    def fetch(ex):
        return [{"symbol": "NVDA", "name": "NVIDIA", "marketCap": "5.0e12", "sector": "Technology"}] if ex == "nasdaq" else []

    ingest_us_universe(db_path=db, fetch_exchange=fetch)

    conn = connect(db)
    symbols = [r["stock_code"] for r in conn.execute("SELECT stock_code FROM us_company").fetchall()]
    conn.close()
    assert symbols == ["NVDA"]
