"""백테스트 팩터 체크박스에는 있으나 metrics_at()이 노출하지 않던 지표 배선 (TDD).

배경: 백테스트 UI 체크박스는 src/db.py의 METRIC_DEFS(20개 키)에서 나오는데, 그중
pcr/ev_ebitda/peg/current_ratio/interest_coverage 는 metrics_at()(src/backtest/
data_access.py) 출력에 없어, 사용자가 체크하면 selection._validate_criteria_keys 가
"존재하지 않는 필드" ValueError 로 크래시났다. 이 다섯 지표는 이미 수집된 원본 데이터
(영업활동현금흐름/유동자산·부채/이자비용/감가상각비 + 이미 계산 중인 EV·EBIT)만으로
공식 계산이 가능하므로 metrics_at()에 배선한다.

공식(src/ingest/metrics.py::compute_metrics 의 기존 정의와 동일 관례):
  PCR             = 시가총액 / 영업활동현금흐름(TTM)
  EV/EBITDA       = EV / (EBIT + 감가상각비(TTM))   … EV·EBIT 는 마법공식 인프라 재사용
  PEG             = PER / 순이익성장률(YoY, %)
  유동비율        = 유동자산 / 유동부채 (%)
  이자보상배율    = 영업이익(TTM) / 이자비용(TTM)

손익계산서 항목(영업현금흐름/영업이익/이자비용/감가상각비)은 TTM(4분기 합),
재무상태표 항목(유동자산/유동부채)은 시점 스냅샷(_fin)이다.
"""
from __future__ import annotations

import sqlite3

import pytest

from src.backtest.data_access import METRIC_FIELD_DESCRIPTIONS, metrics_at
from src.db import init_db
from src.version import shift_quarter as _shift_quarter

_Q = "2026Q1"
_DISCLOSED = "2026-05-15"
_ASOF = "2026-06-30"


def _base_conn(tmp_path, name: str, market_cap: float) -> sqlite3.Connection:
    db = tmp_path / f"{name}.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO company(stock_code, name, market, sector) VALUES (?,?,?,?)",
        ("005930", "삼성전자", "KOSPI", "반도체"),
    )
    conn.execute(
        "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,?)",
        ("005930", _ASOF, 72000.0, float(market_cap)),
    )
    return conn


def _fin(conn, quarter: str, key: str, amount: float) -> None:
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
        "VALUES (?,?,?,?,?)",
        ("005930", quarter, _DISCLOSED, key, float(amount)),
    )


def _ttm(conn, key: str, per_quarter: float) -> None:
    """key를 최근 4분기(_Q ~ _Q-3)에 per_quarter로 시드 → TTM = 4×per_quarter."""
    for i in range(4):
        _fin(conn, _shift_quarter(_Q, -i), key, per_quarter)


# ── PCR = 시가총액 / 영업활동현금흐름(TTM) ────────────────────────────────────────
def test_metrics_at_exposes_pcr_normal_case(tmp_path):
    conn = _base_conn(tmp_path, "pcr_ok", market_cap=20_000.0)
    _ttm(conn, "operating_cashflow", 250.0)  # ocf_ttm = 1000
    conn.commit()
    r = metrics_at(conn, _ASOF)[0]
    assert r["pcr"] == pytest.approx(20_000.0 / 1_000.0)  # = 20.0
    conn.close()


def test_pcr_none_when_operating_cashflow_not_positive(tmp_path):
    """영업현금흐름 TTM<=0(현금 소진)이면 PCR 무의미 → None(PER의 음수순이익 처리와 동일)."""
    conn = _base_conn(tmp_path, "pcr_neg", market_cap=20_000.0)
    _ttm(conn, "operating_cashflow", -250.0)  # ocf_ttm = -1000
    conn.commit()
    assert metrics_at(conn, _ASOF)[0]["pcr"] is None
    conn.close()


# ── 유동비율 = 유동자산 / 유동부채 (%) ───────────────────────────────────────────
def test_metrics_at_exposes_current_ratio_normal_case(tmp_path):
    conn = _base_conn(tmp_path, "cr_ok", market_cap=10_000.0)
    _fin(conn, _Q, "current_assets", 5_000.0)
    _fin(conn, _Q, "current_liabilities", 2_000.0)
    conn.commit()
    r = metrics_at(conn, _ASOF)[0]
    assert r["current_ratio"] == pytest.approx(5_000.0 / 2_000.0 * 100)  # = 250.0
    conn.close()


def test_current_ratio_none_when_current_liabilities_zero(tmp_path):
    conn = _base_conn(tmp_path, "cr_zero", market_cap=10_000.0)
    _fin(conn, _Q, "current_assets", 5_000.0)
    _fin(conn, _Q, "current_liabilities", 0.0)
    conn.commit()
    assert metrics_at(conn, _ASOF)[0]["current_ratio"] is None
    conn.close()


# ── 이자보상배율 = 영업이익(TTM) / 이자비용(TTM) ────────────────────────────────
def test_metrics_at_exposes_interest_coverage_normal_case(tmp_path):
    conn = _base_conn(tmp_path, "ic_ok", market_cap=10_000.0)
    _ttm(conn, "operating_profit", 300.0)   # op_ttm = 1200
    _ttm(conn, "interest_expense", 100.0)    # int_ttm = 400
    conn.commit()
    r = metrics_at(conn, _ASOF)[0]
    assert r["interest_coverage"] == pytest.approx(1_200.0 / 400.0)  # = 3.0
    conn.close()


def test_interest_coverage_none_when_interest_expense_zero(tmp_path):
    conn = _base_conn(tmp_path, "ic_zero", market_cap=10_000.0)
    _ttm(conn, "operating_profit", 300.0)
    _ttm(conn, "interest_expense", 0.0)  # int_ttm = 0
    conn.commit()
    assert metrics_at(conn, _ASOF)[0]["interest_coverage"] is None
    conn.close()


# ── PEG = PER / 순이익성장률(YoY, %) ─────────────────────────────────────────────
def test_metrics_at_exposes_peg_normal_case(tmp_path):
    conn = _base_conn(tmp_path, "peg_ok", market_cap=10_000.0)
    _ttm(conn, "net_income", 250.0)          # ni_ttm = 1000 → PER = 10000/1000 = 10
    _fin(conn, _shift_quarter(_Q, -4), "net_income", 200.0)  # 전년동기 → ni_growth = (250-200)/200*100 = 25
    conn.commit()
    r = metrics_at(conn, _ASOF)[0]
    assert r["per"] == pytest.approx(10.0)
    assert r["ni_growth"] == pytest.approx(25.0)
    assert r["peg"] == pytest.approx(10.0 / 25.0)  # = 0.4
    conn.close()


def test_peg_none_when_growth_not_positive(tmp_path):
    """순이익성장률<=0이면 PEG 무의미 → None(성장 대비 밸류 팩터의 정의상)."""
    conn = _base_conn(tmp_path, "peg_neg", market_cap=10_000.0)
    _ttm(conn, "net_income", 250.0)
    _fin(conn, _shift_quarter(_Q, -4), "net_income", 300.0)  # ni_growth = (250-300)/300*100 < 0
    conn.commit()
    r = metrics_at(conn, _ASOF)[0]
    assert r["ni_growth"] < 0
    assert r["peg"] is None
    conn.close()


# ── EV/EBITDA = EV / (EBIT + 감가상각비 TTM) — 마법공식 EV·EBIT 인프라 재사용 ──────
def test_metrics_at_exposes_ev_ebitda_normal_case(tmp_path):
    conn = _base_conn(tmp_path, "eve_ok", market_cap=20_000.0)
    # EBIT(TTM) = (net_income + tax + interest)*4 = (250+50+25)*4 = 1300
    _ttm(conn, "net_income", 250.0)
    _ttm(conn, "tax_expense", 50.0)
    _ttm(conn, "interest_expense", 25.0)
    _ttm(conn, "depreciation", 125.0)  # dep_ttm = 500 → EBITDA = 1300+500 = 1800
    # EV 입력(스냅샷): excess_cash = 현금 - max(0, 유동부채-유동자산+현금)
    _fin(conn, _Q, "current_assets", 5_000.0)
    _fin(conn, _Q, "current_liabilities", 3_000.0)
    _fin(conn, _Q, "total_liabilities", 6_000.0)
    _fin(conn, _Q, "cash", 1_000.0)
    _fin(conn, _Q, "non_current_assets", 8_000.0)
    conn.commit()
    r = metrics_at(conn, _ASOF)[0]
    # excess_cash = 1000 - max(0, 3000-5000+1000)=1000 → EV = 20000+6000-1000 = 25000
    # EBITDA = EBIT(1300) + dep_ttm(500) = 1800 → EV/EBITDA = 25000/1800
    assert r["ev_ebitda"] == pytest.approx(25_000.0 / 1_800.0)
    conn.close()


def test_ev_ebitda_none_when_depreciation_missing(tmp_path):
    """감가상각비(TTM)가 없으면 EBITDA를 구할 수 없어 EV/EBITDA=None(추정 안 함).
    같은 종목의 earnings_yield/roc 는 감가상각비 없이도 계산되므로 영향 없다(가드 특정성)."""
    conn = _base_conn(tmp_path, "eve_no_dep", market_cap=20_000.0)
    _ttm(conn, "net_income", 250.0)
    _ttm(conn, "tax_expense", 50.0)
    _ttm(conn, "interest_expense", 25.0)
    # depreciation 미시드
    _fin(conn, _Q, "current_assets", 5_000.0)
    _fin(conn, _Q, "current_liabilities", 3_000.0)
    _fin(conn, _Q, "total_liabilities", 6_000.0)
    _fin(conn, _Q, "cash", 1_000.0)
    _fin(conn, _Q, "non_current_assets", 8_000.0)
    conn.commit()
    r = metrics_at(conn, _ASOF)[0]
    assert r["ev_ebitda"] is None
    assert r["earnings_yield"] is not None  # 감가상각비 없이도 EY는 계산됨
    conn.close()


# ── 스크리닝 단일 정의처(METRIC_FIELD_DESCRIPTIONS) 등록 확인 ─────────────────────
def test_new_factors_registered_in_field_descriptions():
    for key, ko in (
        ("pcr", "현금흐름"),
        ("ev_ebitda", "EBITDA"),
        ("peg", "PEG"),
        ("current_ratio", "유동비율"),
        ("interest_coverage", "이자보상배율"),
    ):
        assert key in METRIC_FIELD_DESCRIPTIONS
        assert ko in METRIC_FIELD_DESCRIPTIONS[key]
