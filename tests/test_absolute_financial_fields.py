"""영업이익/매출/순이익 절대값(원화) 필드 노출 검증 (KR/US 대칭).

배경: "코스피 종목 중 26년 1분기 영업이익 가장 높은 기업과 나스닥에서 26년 1분기
영업이익 가장 높은 기업" 질문 검증이 3번 다 실패했다. metrics_at()/metrics_at_us()가
이미 op_q(영업이익)/rev_q(매출)/ni_q(순이익)를 DB에서 조회해 변수에 담아두고도 마진
(비율) 계산에만 쓰고 결과 dict에는 절대값 자체를 넣지 않았기 때문이다 — 데이터 수집
문제가 아니라 이미 메모리에 있는 값을 출력에 안 담는 문제였다. 새 계산 없이 기존
변수(op_q/rev_q/ni_q)를 그대로 노출만 한다. 단일 분기 값(TTM 아님)이라는 점에 유의
— "26년 1분기 영업이익"처럼 특정 분기를 묻는 질문 의도와 정확히 일치한다.

이 파일은 3개 층을 검증한다:
1) metrics_at / metrics_at_us: 반환 dict에 operating_profit/revenue/net_income 존재.
2) _heuristic_screening_spec: "영업이익"/"매출"/"순이익" 질문이 새 절대값 필드로
   매핑되고, 기존 "영업이익률"/"영업이익성장"/"매출성장"/"매출증가"/"순이익률"/
   "순이익성장" 질문은 여전히 원래 필드로 매핑된다(회귀, 서로 안 섞임 — 매우 중요).
3) _KR_SCREEN_FIELDS / _US_SCREEN_FIELDS: LLM 프롬프트에 새 필드가 노출된다.
"""
from __future__ import annotations

import sqlite3

import src.agents.domain_kr as kr
from src.agents.domain_kr import _KR_SCREEN_FIELDS
from src.agents.domain_us import _US_SCREEN_FIELDS
from src.backtest.data_access import metrics_at
from src.backtest.data_access_us import metrics_at_us
from src.db import init_db


# ---------------------------------------------------------------------------
# 1) metrics_at / metrics_at_us: 절대값 필드 노출
# ---------------------------------------------------------------------------
def _seed_kr(tmp_path) -> sqlite3.Connection:
    db = tmp_path / "abs_kr.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO company(stock_code, name, market, sector) VALUES (?,?,?,?)",
        ("005930", "삼성전자", "KOSPI", "반도체"),
    )
    q, disclosed = "2026Q1", "2026-05-15"
    for key, amount in (
        ("operating_profit", 6_000_000_000_000.0),
        ("revenue", 70_000_000_000_000.0),
        ("net_income", 5_000_000_000_000.0),
        ("total_equity", 100_000_000_000_000.0),
    ):
        conn.execute(
            "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
            "VALUES (?,?,?,?,?)",
            ("005930", q, disclosed, key, amount),
        )
    conn.execute(
        "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,?)",
        ("005930", "2026-06-30", 72000.0, 4.1e14),
    )
    conn.commit()
    return conn


def test_metrics_at_exposes_operating_profit_revenue_net_income(tmp_path):
    conn = _seed_kr(tmp_path)
    rows = metrics_at(conn, "2026-06-30")
    assert len(rows) == 1
    r = rows[0]
    assert r["operating_profit"] == 6_000_000_000_000.0
    assert r["revenue"] == 70_000_000_000_000.0
    assert r["net_income"] == 5_000_000_000_000.0
    conn.close()


def test_metrics_at_absolute_fields_are_single_quarter_not_ttm(tmp_path):
    """단일분기(2026Q1)만 넣었을 때 절대값이 그 분기 그대로여야 한다(TTM 합산 아님)."""
    conn = _seed_kr(tmp_path)
    # 이전 분기(2025Q4)에도 재무를 하나 더 심어, TTM 합산이었다면 값이 달라졌을 것을 확인.
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
        "VALUES (?,?,?,?,?)",
        ("005930", "2025Q4", "2026-02-15", "operating_profit", 999_000_000_000.0),
    )
    conn.commit()
    rows = metrics_at(conn, "2026-06-30")
    # effective_quarter_at은 asof 시점 최신 공시분기(2026Q1)만 골라 그 분기 값만 쓴다.
    assert rows[0]["operating_profit"] == 6_000_000_000_000.0
    conn.close()


def test_metrics_at_preserves_existing_ratio_fields_regression(tmp_path):
    """절대값 필드 추가가 기존 비율 필드(마진 등) 계산/존재를 깨지 않는다(회귀)."""
    conn = _seed_kr(tmp_path)
    rows = metrics_at(conn, "2026-06-30")
    r = rows[0]
    for key in ("operating_margin", "net_margin", "per", "pbr", "roe", "revenue_growth"):
        assert key in r
    # operating_margin = 6조/70조*100
    assert r["operating_margin"] == 6_000_000_000_000.0 / 70_000_000_000_000.0 * 100


def _seed_us(tmp_path) -> sqlite3.Connection:
    db = tmp_path / "abs_us.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        ("AAPL", "Apple", "NASDAQ", "Technology", 3.0e12, "2026-06-01"),
    )
    for item_key, value in (
        ("Total Revenue", 100_000_000_000.0),
        ("Operating Income", 30_000_000_000.0),
        ("Net Income", 25_000_000_000.0),
        ("Stockholders Equity", 60_000_000_000.0),
    ):
        conn.execute(
            "INSERT INTO us_financials(stock_code, as_of_date, period_type, statement_type, "
            "item_key, item_value, disclosed_date, source) VALUES (?,?,?,?,?,?,?,?)",
            ("AAPL", "2026-03-31", "quarterly", "income_stmt" if item_key != "Stockholders Equity"
             else "balance_sheet", item_key, value, "2026-05-15", "yfinance"),
        )
    conn.execute(
        "INSERT INTO us_prices(stock_code, date, open, high, low, close, volume) "
        "VALUES (?,?,?,?,?,?,?)",
        ("AAPL", "2026-06-30", 200.0, 200.0, 200.0, 200.0, 1000.0),
    )
    conn.commit()
    return conn


def test_metrics_at_us_exposes_operating_profit_revenue_net_income(tmp_path):
    conn = _seed_us(tmp_path)
    rows = metrics_at_us(conn, "2026-06-30")
    assert len(rows) == 1
    r = rows[0]
    assert r["operating_profit"] == 30_000_000_000.0
    assert r["revenue"] == 100_000_000_000.0
    assert r["net_income"] == 25_000_000_000.0
    conn.close()


def test_metrics_at_us_preserves_existing_ratio_fields_regression(tmp_path):
    conn = _seed_us(tmp_path)
    rows = metrics_at_us(conn, "2026-06-30")
    r = rows[0]
    for key in ("operating_margin", "net_margin", "per", "pbr", "roe"):
        assert key in r
    assert r["operating_margin"] == 30_000_000_000.0 / 100_000_000_000.0 * 100


# ---------------------------------------------------------------------------
# 2) _heuristic_screening_spec: 새 절대값 별칭 vs 기존 비율 별칭 (회귀, 충돌 금지)
# ---------------------------------------------------------------------------
def _metric_for(question: str) -> str | None:
    spec = kr._heuristic_screening_spec(question, domain="KR")
    return spec["criteria"][0]["key"] if spec else None


def test_heuristic_maps_operating_profit_absolute():
    assert _metric_for("영업이익 가장 높은 기업") == "operating_profit"
    assert _metric_for("코스피 종목 중 영업이익이 제일 높은 회사") == "operating_profit"


def test_heuristic_still_maps_operating_margin_not_absolute_regression():
    assert _metric_for("영업이익률 가장 높은 기업") == "operating_margin"


def test_heuristic_still_maps_operating_profit_growth_not_absolute_regression():
    assert _metric_for("영업이익성장 가장 높은 기업") == "op_growth"


def test_heuristic_maps_revenue_absolute():
    assert _metric_for("매출 가장 높은 기업") == "revenue"
    assert _metric_for("매출액이 가장 큰 회사") == "revenue"


def test_heuristic_still_maps_revenue_growth_not_absolute_regression():
    assert _metric_for("매출성장 가장 높은 기업") == "revenue_growth"
    assert _metric_for("매출증가 가장 높은 기업") == "revenue_growth"


def test_heuristic_maps_net_income_absolute():
    assert _metric_for("순이익 가장 높은 기업") == "net_income"


def test_heuristic_still_maps_net_margin_not_absolute_regression():
    assert _metric_for("순이익률 가장 높은 기업") == "net_margin"


def test_heuristic_still_maps_net_income_growth_not_absolute_regression():
    assert _metric_for("순이익성장 가장 높은 기업") == "ni_growth"


def test_heuristic_maps_operating_profit_absolute_for_us_domain_too():
    """별칭표(_SCREEN_METRIC_ALIASES)는 KR/US 공유이므로 US 도메인 호출에서도 동일하게 동작."""
    spec = kr._heuristic_screening_spec("나스닥에서 영업이익 가장 높은 기업", domain="US")
    assert spec["criteria"][0]["key"] == "operating_profit"


# ---------------------------------------------------------------------------
# 3) 필드 목록 노출 (LLM 프롬프트가 존재를 알게)
# ---------------------------------------------------------------------------
def test_kr_screen_fields_expose_absolute_financial_fields():
    assert "operating_profit" in _KR_SCREEN_FIELDS
    assert "revenue" in _KR_SCREEN_FIELDS
    assert "net_income" in _KR_SCREEN_FIELDS


def test_us_screen_fields_expose_absolute_financial_fields():
    assert "operating_profit" in _US_SCREEN_FIELDS
    assert "revenue" in _US_SCREEN_FIELDS
    assert "net_income" in _US_SCREEN_FIELDS
