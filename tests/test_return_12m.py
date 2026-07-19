"""가격 수익률(모멘텀) 필드 return_12m 단위 테스트 (TDD).

배경: "직전 12개월간 수익률이 가장 좋았던 상위 N개" 류 질문은 "12개월 가격 수익률"
숫자가 시스템 어디에도 없어 답할 수 없었다(KR은 revenue_growth로 오매핑, US는 존재하지
않는 return_12m을 참조해 검증 배선이 재시도로 잡음). 이 테스트는 KR/US 크로스섹션
(metrics_at/metrics_at_us)이 return_12m을 올바르게 계산하는지, 미래참조를 하지 않는지,
기존 필드가 그대로 유지되는지 검증한다. 계산식: (기준시점 종가 - 12개월전 종가)/12개월전 종가.
12개월 전 정확한 거래일이 없으면 가장 가까운 이전 거래일 종가를 쓴다(look-ahead 방지).
"""
from __future__ import annotations

import sqlite3

import pytest

from src.backtest.data_access import metrics_at
from src.db import init_db


# --------------------------------------------------------------------------
# KR: metrics_at return_12m
# --------------------------------------------------------------------------
def _seed_kr(tmp_path) -> sqlite3.Connection:
    db = tmp_path / "r12.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO company(stock_code, name, market, sector) VALUES (?,?,?,?)",
        ("000001", "가나전자", "KOSPI", "전기·전자"),
    )
    # 공시된 분기(effective_quarter_at이 유효분기를 돌려주도록) — disclosed_date<=asof.
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
        "VALUES (?,?,?,?,?)",
        ("000001", "2025Q1", "2025-05-15", "net_income", 1_000.0),
    )
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
        "VALUES (?,?,?,?,?)",
        ("000001", "2025Q1", "2025-05-15", "total_equity", 100_000.0),
    )
    conn.commit()
    return conn


def _add_price(conn, code, date_str, close):
    conn.execute(
        "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,?)",
        (code, date_str, close, 1e12),
    )
    conn.commit()


def test_metrics_at_computes_return_12m_from_prices(tmp_path):
    conn = _seed_kr(tmp_path)
    _add_price(conn, "000001", "2025-06-30", 100.0)   # 12개월 전
    _add_price(conn, "000001", "2026-06-30", 120.0)   # 기준시점 종가
    rows = metrics_at(conn, "2026-06-30")
    assert len(rows) == 1
    # (120-100)/100 * 100 = 20.0%
    assert rows[0]["return_12m"] == pytest.approx(20.0)


def test_metrics_at_return_12m_uses_nearest_prior_trading_day(tmp_path):
    conn = _seed_kr(tmp_path)
    # 12개월 전(2025-06-30)에 정확히 데이터가 없고, 가장 가까운 이전 거래일은 2025-06-27.
    _add_price(conn, "000001", "2025-06-27", 100.0)
    _add_price(conn, "000001", "2026-06-30", 130.0)
    rows = metrics_at(conn, "2026-06-30")
    assert rows[0]["return_12m"] == pytest.approx(30.0)  # (130-100)/100*100


def test_metrics_at_return_12m_ignores_future_prices(tmp_path):
    conn = _seed_kr(tmp_path)
    _add_price(conn, "000001", "2025-06-30", 100.0)
    _add_price(conn, "000001", "2026-06-30", 120.0)
    # asof 이후(미래) 종가가 있어도 return_12m은 이를 참조하지 않아야 한다(look-ahead 방지).
    # 150.0(120.0 대비 1.25배)은 데이터 품질 게이트(src/data_quality.py, ratio>=2.0 제외)에
    # 걸리지 않는 값 — "미래참조 무시" 검증과 "이상치 제외"가 서로 간섭하지 않게 한다.
    _add_price(conn, "000001", "2026-12-31", 150.0)
    rows = metrics_at(conn, "2026-06-30")
    assert rows[0]["return_12m"] == pytest.approx(20.0)


def test_metrics_at_return_12m_none_when_no_prior_year_price(tmp_path):
    conn = _seed_kr(tmp_path)
    # 기준시점 종가만 있고 12개월 전 종가가 아예 없으면 return_12m=None.
    _add_price(conn, "000001", "2026-06-30", 120.0)
    rows = metrics_at(conn, "2026-06-30")
    assert rows[0]["return_12m"] is None


def test_metrics_at_preserves_existing_fields(tmp_path):
    conn = _seed_kr(tmp_path)
    _add_price(conn, "000001", "2025-06-30", 100.0)
    _add_price(conn, "000001", "2026-06-30", 120.0)
    rows = metrics_at(conn, "2026-06-30")
    # 기존 필드는 그대로 유지(회귀 방지) — 값이 None이어도 키는 존재해야 한다.
    for key in ("per", "pbr", "roe", "revenue_growth", "operating_margin", "close"):
        assert key in rows[0]


# --------------------------------------------------------------------------
# 스크리닝/프롬프트 필드 목록에 return_12m이 노출된다 (LLM이 존재를 알게)
# --------------------------------------------------------------------------
def test_return_12m_exposed_in_screening_and_pipeline_field_lists():
    from src.agents.domain_backtest import _PIPELINE_PROMPT
    from src.agents.domain_kr import _KR_SCREEN_FIELDS

    assert "return_12m" in _KR_SCREEN_FIELDS
    assert "return_12m" in _PIPELINE_PROMPT
