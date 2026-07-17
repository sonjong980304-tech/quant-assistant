"""매출총이익률(gross_margin)·매출원가율(cogs_ratio) 크로스섹션/단일종목 노출 (TDD).

배경: 영업이익률(operating_margin)은 이미 KR/US metrics_at(_us)에 있으나, 매출총이익률
(=매출총이익÷매출액)과 매출원가율(=매출원가÷매출액)은 스크리닝/단일종목 어디에도 없었다.

주의(혼동 금지): 매출총이익률(gross_margin, 분모=매출액)은 기존 GPA(gp_a, 분모=총자산)와
분모가 완전히 다른 별개 지표다. operating_margin/net_margin과 동일하게 "단일분기" 기준으로
통일한다(data_access.py의 마진 SoT 관례).

원본 계정이 없을 때(서비스업 등 매출원가/매출총이익을 별도 표기 안 하는 종목)는 항등식
(매출액 = 매출원가 + 매출총이익)으로 유도하고, '{metric}_estimated' 컴패니언 필드로 근사
여부를 노출한다(마법공식 roc_estimated 패턴 그대로 — 원본 우선, 없을 때만 유도).
"""
from __future__ import annotations

import sqlite3

import pytest

import src.agents.domain_kr as kr
from src.agents.domain_kr import _KR_SCREEN_FIELDS, _extract_metric, resolve_computed_metric
from src.agents.domain_us import _US_SCREEN_FIELDS
from src.backtest.data_access import METRIC_FIELD_DESCRIPTIONS, metrics_at
from src.backtest.data_access_us import METRIC_FIELD_DESCRIPTIONS_US, metrics_at_us
from src.db import init_db
from src.ingest.normalize import normalize_account
from src.version import shift_quarter as _shift_quarter

_Q = "2026Q1"
_DISCLOSED = "2026-05-15"
_ASOF = "2026-06-30"


# ── KR 시딩 헬퍼 ─────────────────────────────────────────────────────────────
def _seed_kr(tmp_path, name, *, single_quarter: dict, ttm_per_quarter: dict | None = None,
             market_cap: float = 4.1e14) -> sqlite3.Connection:
    """단일 종목을 시드한다. single_quarter는 최신 분기(_Q) 1건만 시드(단일분기 마진 계산용),
    ttm_per_quarter는 분기당 값을 4분기 반복 시드(_sum_ttm 대상 — gp_a 등)."""
    db = tmp_path / f"{name}.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO company(stock_code, name, market, sector) VALUES (?,?,?,?)",
        ("005930", "삼성전자", "KOSPI", "반도체"),
    )
    for key, val in single_quarter.items():
        conn.execute(
            "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
            "VALUES (?,?,?,?,?)",
            ("005930", _Q, _DISCLOSED, key, float(val)),
        )
    for key, per_q in (ttm_per_quarter or {}).items():
        for i in range(1, 4):  # _Q 는 위에서 이미 시드했으므로 직전 3분기만 추가
            conn.execute(
                "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
                "VALUES (?,?,?,?,?)",
                ("005930", _shift_quarter(_Q, -i), _DISCLOSED, key, float(per_q)),
            )
    conn.execute(
        "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,?)",
        ("005930", _ASOF, 72000.0, float(market_cap)),
    )
    conn.commit()
    return conn


# ── KR: 정상 계산 ─────────────────────────────────────────────────────────────
def test_metrics_at_exposes_gross_margin_and_cogs_ratio_normal(tmp_path):
    """매출액 10000, 매출총이익 3000, 매출원가 7000 → 매출총이익률 30%, 매출원가율 70%."""
    conn = _seed_kr(
        tmp_path, "kr_ok",
        single_quarter={"revenue": 10_000.0, "gross_profit": 3_000.0, "cost_of_sales": 7_000.0},
    )
    r = metrics_at(conn, _ASOF)[0]
    assert r["gross_margin"] == pytest.approx(3_000.0 / 10_000.0 * 100)
    assert r["gross_margin_estimated"] is False  # 원본 매출총이익 사용 → 근사 아님
    assert r["cogs_ratio"] == pytest.approx(7_000.0 / 10_000.0 * 100)
    assert r["cogs_ratio_estimated"] is False   # 원본 매출원가 사용 → 근사 아님
    conn.close()


def test_cogs_ratio_estimated_when_cost_of_sales_missing(tmp_path):
    """매출원가 계정이 없으면 매출액-매출총이익으로 유도하고 cogs_ratio_estimated=True."""
    conn = _seed_kr(
        tmp_path, "kr_no_cogs",
        single_quarter={"revenue": 10_000.0, "gross_profit": 3_000.0},  # cost_of_sales 없음
    )
    r = metrics_at(conn, _ASOF)[0]
    # 유도: 매출원가 = 10000 - 3000 = 7000 → 70%
    assert r["cogs_ratio"] == pytest.approx((10_000.0 - 3_000.0) / 10_000.0 * 100)
    assert r["cogs_ratio_estimated"] is True
    # 매출총이익은 원본이 있으므로 매출총이익률은 근사 아님
    assert r["gross_margin"] == pytest.approx(30.0)
    assert r["gross_margin_estimated"] is False
    conn.close()


def test_gross_margin_estimated_when_gross_profit_missing(tmp_path):
    """매출총이익 계정이 없으면 매출액-매출원가로 유도하고 gross_margin_estimated=True."""
    conn = _seed_kr(
        tmp_path, "kr_no_gp",
        single_quarter={"revenue": 10_000.0, "cost_of_sales": 7_000.0},  # gross_profit 없음
    )
    r = metrics_at(conn, _ASOF)[0]
    # 유도: 매출총이익 = 10000 - 7000 = 3000 → 30%
    assert r["gross_margin"] == pytest.approx((10_000.0 - 7_000.0) / 10_000.0 * 100)
    assert r["gross_margin_estimated"] is True
    # 매출원가는 원본이 있으므로 매출원가율은 근사 아님
    assert r["cogs_ratio"] == pytest.approx(70.0)
    assert r["cogs_ratio_estimated"] is False
    conn.close()


def test_margins_none_when_revenue_missing(tmp_path):
    """매출액이 없으면(0으로 나누기 방지) 두 비율 모두 None, estimated도 None."""
    conn = _seed_kr(
        tmp_path, "kr_no_rev",
        single_quarter={"gross_profit": 3_000.0, "cost_of_sales": 7_000.0},  # revenue 없음
    )
    r = metrics_at(conn, _ASOF)[0]
    assert r["gross_margin"] is None
    assert r["gross_margin_estimated"] is None
    assert r["cogs_ratio"] is None
    assert r["cogs_ratio_estimated"] is None
    conn.close()


def test_gross_margin_distinct_from_gp_a(tmp_path):
    """회귀(혼동 방지): gross_margin(분모=매출액)과 gp_a(분모=총자산)는 값이 다르다."""
    conn = _seed_kr(
        tmp_path, "kr_distinct",
        single_quarter={
            "revenue": 10_000.0, "gross_profit": 3_000.0, "cost_of_sales": 7_000.0,
            "total_assets": 50_000.0,
        },
        ttm_per_quarter={"gross_profit": 3_000.0},  # 4분기 합 gp_ttm=12000 (gp_a용)
    )
    r = metrics_at(conn, _ASOF)[0]
    # gross_margin = 3000/10000*100 = 30 (단일분기 매출총이익 ÷ 단일분기 매출액)
    assert r["gross_margin"] == pytest.approx(30.0)
    # gp_a = 12000/50000*100 = 24 (TTM 매출총이익 ÷ 총자산) — 분모가 다르므로 값이 다르다
    assert r["gp_a"] == pytest.approx(24.0)
    assert r["gross_margin"] != r["gp_a"]
    conn.close()


def test_gross_margin_and_cogs_ratio_in_field_descriptions():
    """스크리닝 노출 단일 정의처에 두 지표가 있고, 설명에 한국어 용어가 포함된다."""
    assert "gross_margin" in METRIC_FIELD_DESCRIPTIONS
    assert "cogs_ratio" in METRIC_FIELD_DESCRIPTIONS
    assert "매출총이익률" in METRIC_FIELD_DESCRIPTIONS["gross_margin"]
    assert "매출원가율" in METRIC_FIELD_DESCRIPTIONS["cogs_ratio"]


def test_cost_of_sales_normalizes_without_revenue_contamination():
    """회귀: 매출원가/매출액 계정이 서로 오염 매핑되지 않는다(기존 매핑 재확인)."""
    assert normalize_account("매출원가") == "cost_of_sales"
    assert normalize_account("매출액") == "revenue"
    assert normalize_account("매출총이익") == "gross_profit"


# ── 스크리닝 후보 목록 노출 (KR/US) ───────────────────────────────────────────
def test_gross_margin_cogs_ratio_in_kr_screen_fields():
    assert "gross_margin" in _KR_SCREEN_FIELDS
    assert "cogs_ratio" in _KR_SCREEN_FIELDS


def test_gross_margin_cogs_ratio_in_us_screen_fields():
    assert "gross_margin" in _US_SCREEN_FIELDS
    assert "cogs_ratio" in _US_SCREEN_FIELDS


# ── 단일종목 조회 별칭(휴리스틱 폴백 + _extract_metric) ─────────────────────────
def test_heuristic_maps_gross_margin_alias_not_gp_a():
    """'매출총이익률'은 gross_margin으로 매핑돼야 한다(과거 gp_a로 오매핑되던 것 교정)."""
    spec = kr._heuristic_screening_spec("매출총이익률 가장 높은 기업", domain="KR")
    assert spec["criteria"][0]["key"] == "gross_margin"


def test_heuristic_maps_cogs_ratio_alias():
    spec = kr._heuristic_screening_spec("매출원가율 가장 낮은 기업", domain="KR")
    assert spec["criteria"][0]["key"] == "cogs_ratio"


def test_extract_metric_single_stock_gross_margin():
    assert _extract_metric("삼성전자 매출총이익률 알려줘") == "gross_margin"


def test_extract_metric_single_stock_cogs_ratio():
    assert _extract_metric("삼성전자 매출원가율 알려줘") == "cogs_ratio"


def test_resolve_computed_metric_returns_estimated_companion():
    """단일종목 조회에서도 gross_margin_estimated 컴패니언 필드가 함께 노출된다."""
    def fake_cross_section(conn, asof):
        return [{"stock_code": "005930", "gross_margin": 30.0, "gross_margin_estimated": True}]

    result = resolve_computed_metric(
        None, "005930", "gross_margin",
        execute_sql_fn=lambda sql, c: {"ok": True, "rows": [{"d": _ASOF}]},
        cross_section_fn=fake_cross_section,
    )
    assert result["value"] == 30.0
    assert result["estimated"] is True


# ── US: 정상 계산 + 통화 게이트 ───────────────────────────────────────────────
def _seed_us(tmp_path, *, financial_currency: str | None = None,
             income_items: dict) -> sqlite3.Connection:
    """AAPL 1종목을 4분기 Net Income + 최신분기 손익항목 + BS + 주가로 시드한다."""
    db = tmp_path / "us_margin.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, "
        "financial_currency, updated_at) VALUES (?,?,?,?,?,?,?)",
        ("AAPL", "Apple", "NASDAQ", "Technology", 1000.0, financial_currency, "2025-03-01"),
    )
    quarters = [
        ("2024-03-31", "2024-05-15"), ("2024-06-30", "2024-08-14"),
        ("2024-09-30", "2024-11-14"), ("2024-12-31", "2025-02-14"),
    ]
    for as_of, disclosed in quarters:
        conn.execute(
            "INSERT INTO us_financials(stock_code, as_of_date, period_type, statement_type, "
            "item_key, item_value, disclosed_date, source) VALUES (?,?,?,?,?,?,?,?)",
            ("AAPL", as_of, "quarterly", "income_stmt", "Net Income", 10.0, disclosed, "yfinance"),
        )
    for item_key, value in income_items.items():
        conn.execute(
            "INSERT INTO us_financials(stock_code, as_of_date, period_type, statement_type, "
            "item_key, item_value, disclosed_date, source) VALUES (?,?,?,?,?,?,?,?)",
            ("AAPL", "2024-12-31", "quarterly", "income_stmt", item_key, value, "2025-02-14", "yfinance"),
        )
    conn.execute(
        "INSERT INTO us_financials(stock_code, as_of_date, period_type, statement_type, "
        "item_key, item_value, disclosed_date, source) VALUES (?,?,?,?,?,?,?,?)",
        ("AAPL", "2024-12-31", "quarterly", "balance_sheet", "Stockholders Equity", 200.0, "2025-02-14", "yfinance"),
    )
    for date_str, close in [("2025-01-15", 140.0), ("2025-03-01", 150.0)]:
        conn.execute(
            "INSERT INTO us_prices(stock_code, date, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
            ("AAPL", date_str, close - 2, close + 2, close - 3, close, 1000.0),
        )
    conn.commit()
    return conn


def test_metrics_at_us_exposes_gross_margin_and_cogs_ratio(tmp_path):
    """매출 100, 매출총이익 40, 매출원가 60 → 매출총이익률 40%, 매출원가율 60%."""
    conn = _seed_us(
        tmp_path,
        income_items={"Total Revenue": 100.0, "Gross Profit": 40.0, "Cost Of Revenue": 60.0},
    )
    r = metrics_at_us(conn, "2025-03-01")[0]
    assert r["gross_margin"] == pytest.approx(40.0)
    assert r["gross_margin_estimated"] is False
    assert r["cogs_ratio"] == pytest.approx(60.0)
    assert r["cogs_ratio_estimated"] is False
    conn.close()


def test_us_gross_margin_estimated_when_gross_profit_missing(tmp_path):
    """미국도 매출총이익이 없으면 매출-매출원가로 유도하고 estimated=True."""
    conn = _seed_us(
        tmp_path,
        income_items={"Total Revenue": 100.0, "Cost Of Revenue": 60.0},  # Gross Profit 없음
    )
    r = metrics_at_us(conn, "2025-03-01")[0]
    assert r["gross_margin"] == pytest.approx(40.0)  # 100-60=40
    assert r["gross_margin_estimated"] is True
    conn.close()


def test_us_non_usd_nullifies_gross_margin_and_cogs_ratio(tmp_path):
    """비USD 재무통화 종목은 매출총이익률/매출원가율도 전부 None(통화 불일치 방어)."""
    conn = _seed_us(
        tmp_path, financial_currency="KRW",
        income_items={"Total Revenue": 100.0, "Gross Profit": 40.0, "Cost Of Revenue": 60.0},
    )
    r = metrics_at_us(conn, "2025-03-01")[0]
    assert r["gross_margin"] is None
    assert r["gross_margin_estimated"] is None
    assert r["cogs_ratio"] is None
    assert r["cogs_ratio_estimated"] is None
    conn.close()


def test_us_gross_margin_and_cogs_ratio_in_field_descriptions():
    assert "gross_margin" in METRIC_FIELD_DESCRIPTIONS_US
    assert "cogs_ratio" in METRIC_FIELD_DESCRIPTIONS_US
