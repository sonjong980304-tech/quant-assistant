"""GPA(매출총이익/자산) 크로스섹션 노출 + correlation/quantile_bucket_means 프리미티브 (TDD).

배경: "PBR·GPA 각각 5분위/오름·내림차순 정렬하고 상관관계, PBR 분위수별 평균 GPA" 질문에
답하려면 (1) GPA가 여러 종목을 한 번에 보는 크로스섹션(get_cross_section/metrics_at)에
노출돼야 하고, (2) 두 팩터 간 상관관계, (3) 한 팩터 기준 N분위 그룹별 다른 팩터의 평균을
구하는 프리미티브가 필요한데 셋 다 이전엔 없었다(GPA는 단일종목 조회 전용 metrics 테이블에만
존재, compute_ic는 팩터-미래수익률 IC이지 팩터간 상관관계가 아님).
"""
from __future__ import annotations

import sqlite3

import pytest

from src.agents.domain_kr import _KR_SCREEN_FIELDS
from src.backtest.data_access import METRIC_FIELD_DESCRIPTIONS, metrics_at
from src.backtest.primitives import correlation, quantile_bucket_means
from src.db import init_db
from src.version import shift_quarter as _shift_quarter


# ---------------------------------------------------------------------------
# A. GPA/매출총이익/자산 크로스섹션 노출
# ---------------------------------------------------------------------------
def _seed_kr(tmp_path) -> sqlite3.Connection:
    db = tmp_path / "gpa.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO company(stock_code, name, market, sector) VALUES (?,?,?,?)",
        ("005930", "삼성전자", "KOSPI", "반도체"),
    )
    q, disclosed = "2026Q1", "2026-05-15"
    # gross_profit은 _sum_ttm(최근 4분기 합, 하나라도 누락되면 None)이라 4개 분기 모두 필요.
    # total_assets/total_equity는 스냅샷(_fin, 최신 유효 분기만 조회)이라 2026Q1 하나면 충분.
    for i in range(4):
        conn.execute(
            "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
            "VALUES (?,?,?,?,?)",
            ("005930", _shift_quarter(q, -i), disclosed, "gross_profit", 5_000_000_000_000.0),
        )
    for key, amount in (
        ("total_assets", 400_000_000_000_000.0),
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


def test_metrics_at_exposes_gross_profit_total_assets_gp_a(tmp_path):
    conn = _seed_kr(tmp_path)
    rows = metrics_at(conn, "2026-06-30")
    r = rows[0]
    assert r["gross_profit"] == 20_000_000_000_000.0
    assert r["total_assets"] == 400_000_000_000_000.0
    # GPA = 매출총이익(TTM) / 총자산 * 100
    assert r["gp_a"] == pytest.approx(20_000_000_000_000.0 / 400_000_000_000_000.0 * 100)
    conn.close()


def test_metrics_at_gp_a_none_when_assets_missing(tmp_path):
    """total_assets가 없으면(0으로 나누기 방지) gp_a는 None이어야 한다."""
    db = tmp_path / "no_assets.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO company(stock_code, name, market, sector) VALUES (?,?,?,?)",
        ("000001", "종목1", "KOSPI", "화학"),
    )
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
        "VALUES (?,?,?,?,?)",
        ("000001", "2026Q1", "2026-05-15", "gross_profit", 1_000.0),
    )
    conn.execute(
        "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,?)",
        ("000001", "2026-06-30", 1000.0, 1_000_000.0),
    )
    conn.commit()
    rows = metrics_at(conn, "2026-06-30")
    assert rows[0]["gp_a"] is None
    conn.close()


def test_kr_screen_fields_expose_gpa_fields():
    assert "gp_a" in _KR_SCREEN_FIELDS
    assert "gross_profit" in _KR_SCREEN_FIELDS
    assert "total_assets" in _KR_SCREEN_FIELDS
    assert "gp_a" in METRIC_FIELD_DESCRIPTIONS


# ---------------------------------------------------------------------------
# B. correlation — 두 필드 간 피어슨 상관계수
# ---------------------------------------------------------------------------
def _rows_xy(pairs):
    return [{"stock_code": str(i), "x": x, "y": y} for i, (x, y) in enumerate(pairs)]


def test_correlation_perfect_positive_linear_relationship():
    rows = _rows_xy([(1, 2), (2, 4), (3, 6), (4, 8), (5, 10)])
    result = correlation(rows, "x", "y")
    assert result["correlation"] == pytest.approx(1.0)
    assert result["n"] == 5


def test_correlation_perfect_negative_linear_relationship():
    rows = _rows_xy([(1, 10), (2, 8), (3, 6), (4, 4), (5, 2)])
    result = correlation(rows, "x", "y")
    assert result["correlation"] == pytest.approx(-1.0)


def test_correlation_ignores_rows_with_none_values():
    rows = _rows_xy([(1, 2), (2, 4), (3, 6)])
    rows.append({"stock_code": "none1", "x": None, "y": 100})
    rows.append({"stock_code": "none2", "x": 4, "y": None})
    result = correlation(rows, "x", "y")
    assert result["n"] == 3
    assert result["correlation"] == pytest.approx(1.0)


def test_correlation_raises_when_fewer_than_two_valid_samples():
    rows = _rows_xy([(1, 2)])
    with pytest.raises(ValueError):
        correlation(rows, "x", "y")


# ---------------------------------------------------------------------------
# C. quantile_bucket_means — bucket_field 기준 N분위, value_field 그룹평균
# ---------------------------------------------------------------------------
def _rows_bucket(n=10):
    # pbr = 1..n (오름차순), gpa = 항상 pbr과 정반대(내림차순)로 설계 → 분위별 평균이 단조 감소해야 함
    return [{"stock_code": str(i), "pbr": float(i), "gpa": float(n - i)} for i in range(1, n + 1)]


def test_quantile_bucket_means_splits_into_equal_groups():
    rows = _rows_bucket(10)
    buckets = quantile_bucket_means(rows, "pbr", "gpa", n=5)
    assert len(buckets) == 5
    assert all(b["count"] == 2 for b in buckets)


def test_quantile_bucket_means_bucket_1_is_lowest_pbr_group():
    rows = _rows_bucket(10)
    buckets = quantile_bucket_means(rows, "pbr", "gpa", n=5)
    assert buckets[0]["bucket"] == 1
    assert buckets[0]["bucket_range"] == [1.0, 2.0]   # pbr 1~2인 두 종목
    assert buckets[-1]["bucket"] == 5
    assert buckets[-1]["bucket_range"] == [9.0, 10.0]


def test_quantile_bucket_means_computes_mean_of_value_field_per_bucket():
    rows = _rows_bucket(10)
    buckets = quantile_bucket_means(rows, "pbr", "gpa", n=5)
    # bucket1: pbr={1,2} → gpa={9,8} → 평균 8.5 / bucket5: pbr={9,10} → gpa={1,0} → 평균 0.5
    assert buckets[0]["mean_value"] == pytest.approx(8.5)
    assert buckets[-1]["mean_value"] == pytest.approx(0.5)
    # pbr 낮을수록 gpa 평균이 높게 설계했으므로 분위가 올라갈수록 평균이 단조 감소해야 함
    means = [b["mean_value"] for b in buckets]
    assert means == sorted(means, reverse=True)


def test_quantile_bucket_means_ignores_rows_with_none_bucket_or_value_field():
    rows = _rows_bucket(10)
    rows.append({"stock_code": "none1", "pbr": None, "gpa": 5.0})
    rows.append({"stock_code": "none2", "pbr": 3.0, "gpa": None})
    buckets = quantile_bucket_means(rows, "pbr", "gpa", n=5)
    assert sum(b["count"] for b in buckets) == 10  # None 2건은 제외


def test_quantile_bucket_means_raises_when_fewer_valid_rows_than_n():
    rows = _rows_bucket(3)
    with pytest.raises(ValueError):
        quantile_bucket_means(rows, "pbr", "gpa", n=5)
