"""SEC XBRL 소스 백테스트 지표 어댑터 단위 테스트 (TDD, AC6/AC7/AC13).

.omc/specs/brainstorming-sec-edgar-us-financials-backfill.md:
- AC6: 새 테이블(us_financials_sec)의 XBRL 태그로 기존과 동일한 지표군을 계산.
- AC7: SEC 실제 filed(제출일)를 disclosed_date 로 써서 asof 이후 제출/정정된 데이터는 제외.
- AC13: XBRL 파싱, TTM(직전4분기합), YoY(전년동기대비) 로직을 TDD 로 구현.

핵심: fixture 는 **실제 SEC 데이터 모양**을 그대로 반영한다 — Q1~Q3 는 단독 3개월(10-Q)
팩트로 존재하지만 Q4 는 단독으로 존재하지 않고 12개월(10-K) 연간(FY) 팩트만 있다. 어댑터는
Q4 = FY − (Q1+Q2+Q3) 로 역산해야 하며, 이 fixture 가 그 회귀를 가드한다(critic C1).
"""
from __future__ import annotations

import sqlite3

import pytest

from src.backtest import data_access_us_sec as das
from src.db import init_db


def _new_conn(tmp_path, name="sec_das.db") -> sqlite3.Connection:
    db = tmp_path / name
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def _ins_dur(conn, tag, p_start, p_end, filed, val, form="10-Q", accn=None):
    """duration(기간) 팩트 삽입 — 단독분기(10-Q, 3개월) 또는 연간(10-K, 12개월)."""
    conn.execute(
        "INSERT INTO us_financials_sec(stock_code, cik, tag, taxonomy, unit, value, "
        "period_start, period_end, form, filed, accn, source) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("AAPL", "0000320193", tag, "us-gaap", "USD", val, p_start, p_end, form,
         filed, accn or f"{tag}-{p_end}", "sec_companyfacts_zip"),
    )


def _ins_inst(conn, tag, p_end, filed, val, unit="USD", taxonomy="us-gaap"):
    """instant(시점) 팩트 삽입 — 재무상태 스냅샷/발행주식수(period_start='')."""
    conn.execute(
        "INSERT INTO us_financials_sec(stock_code, cik, tag, taxonomy, unit, value, "
        "period_start, period_end, form, filed, accn, source) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("AAPL", "0000320193", tag, taxonomy, unit, val, "", p_end, "10-Q",
         filed, f"{tag}-{p_end}", "sec_companyfacts_zip"),
    )


# 각 손익 태그의 (분기값 2024, FY 2024, 분기값 2023, FY 2023). Q4 = FY − 분기값×3.
_INCOME = {
    "NetIncomeLoss": (10.0, 40.0, 8.0, 32.0),                     # Q4'24=10, Q4'23=8 → YoY 25%
    "Revenues": (100.0, 400.0, 80.0, 320.0),                     # Q4'24=100, Q4'23=80
    "OperatingIncomeLoss": (30.0, 120.0, 24.0, 96.0),            # Q4'24=30, Q4'23=24
    "GrossProfit": (40.0, 160.0, None, None),                    # Q4'24=40, TTM=160
    "DepreciationDepletionAndAmortization": (20.0, 80.0, None, None),  # dep TTM=80
    "InterestExpense": (5.0, 20.0, None, None),                  # TTM=20
    "NetCashProvidedByUsedInOperatingActivities": (25.0, 100.0, None, None),  # OCF TTM=100
}

# 분기 (start, end, filed) — Q1~Q3 만 단독. filed=기말+~45일.
_Q_2024 = [
    ("2024-01-01", "2024-03-31", "2024-05-15"),
    ("2024-04-01", "2024-06-30", "2024-08-14"),
    ("2024-07-01", "2024-09-30", "2024-11-14"),
]
_FY_2024 = ("2024-01-01", "2024-12-31", "2025-02-14")   # 10-K, Q4 는 여기에만 포함
_Q_2023 = [
    ("2023-01-01", "2023-03-31", "2023-05-15"),
    ("2023-04-01", "2023-06-30", "2023-08-14"),
    ("2023-07-01", "2023-09-30", "2023-11-14"),
]
_FY_2023 = ("2023-01-01", "2023-12-31", "2024-02-14")


def _seed_sec_db(tmp_path) -> sqlite3.Connection:
    """실제 SEC 모양: Q1~Q3 단독 + FY 연간(단독 Q4 없음) + 재무상태 + 발행주식수 + 주가.

    시점별 시가총액 검증을 위해 us_company.market_cap 은 일부러 틀린 값(99999)으로 넣어,
    지표가 '종가×발행주식수'(시점별)를 쓰는지 확인한다. 발행주식수=5, 종가(2025-03-01)=200
    → 시점 시가총액=1000. (per=1000/40=25 등 기존 검증표와 동일해지도록 값 설계)
    """
    conn = _new_conn(tmp_path)
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, cik, updated_at) "
        "VALUES (?,?,?,?,?,?,?)",
        ("AAPL", "Apple", "NASDAQ", "Technology", 99999.0, "0000320193", "2025-03-01"),
    )
    for tag, (q24, fy24, q23, fy23) in _INCOME.items():
        for p_start, p_end, filed in _Q_2024:
            _ins_dur(conn, tag, p_start, p_end, filed, q24)
        _ins_dur(conn, tag, *_FY_2024[:2], _FY_2024[2], fy24, form="10-K")
        if q23 is not None:
            for p_start, p_end, filed in _Q_2023:
                _ins_dur(conn, tag, p_start, p_end, filed, q23)
            _ins_dur(conn, tag, *_FY_2023[:2], _FY_2023[2], fy23, form="10-K")
    # 최신분기(2024-12-31) 재무상태(instant)
    for tag, val in [
        ("StockholdersEquity", 200.0), ("Assets", 800.0), ("Liabilities", 600.0),
        ("AssetsCurrent", 300.0), ("LiabilitiesCurrent", 150.0),
        ("CashAndCashEquivalentsAtCarryingValue", 50.0), ("LongTermDebtNoncurrent", 250.0),
    ]:
        _ins_inst(conn, tag, "2024-12-31", "2025-02-14", val)
    # 발행주식수(dei, unit='shares') — 시점별 시가총액용
    _ins_inst(conn, "EntityCommonStockSharesOutstanding", "2024-12-31", "2025-02-14", 5.0,
              unit="shares", taxonomy="dei")
    for date_str, close in [("2025-01-15", 190.0), ("2025-03-01", 200.0)]:
        conn.execute(
            "INSERT INTO us_prices(stock_code, date, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
            ("AAPL", date_str, close - 2, close + 2, close - 3, close, 1000.0),
        )
    conn.commit()
    return conn


# --------------------------------------------------------------------------
# C1 회귀 가드: 실제 SEC 모양엔 단독 Q4 가 없다 (fixture 리얼리즘 확인)
# --------------------------------------------------------------------------
def test_fixture_has_no_standalone_q4(tmp_path):
    conn = _seed_sec_db(tmp_path)
    # 2024-12-31 로 끝나는 3개월(단독) 팩트는 존재하지 않아야 한다(FY 12개월만 존재).
    n = conn.execute(
        "SELECT COUNT(*) FROM us_financials_sec WHERE tag='NetIncomeLoss' "
        "AND period_end='2024-12-31' AND period_start<>'' "
        "AND julianday(period_end)-julianday(period_start) BETWEEN 80 AND 100"
    ).fetchone()[0]
    assert n == 0
    conn.close()


def test_q4_reconstructed_from_annual_minus_three_quarters(tmp_path):
    """Q4 = FY − (Q1+Q2+Q3) 역산. 단독 Q4 팩트가 없어도 연말 단일분기 값을 구한다(C1)."""
    conn = _seed_sec_db(tmp_path)
    q4 = das._single_q(conn, "AAPL", "2024-12-31", das.NET_INCOME_TAGS)
    assert q4 == pytest.approx(10.0)  # 40 − (10+10+10)
    conn.close()


def test_effective_quarter_finds_year_end_via_reconstruction(tmp_path):
    """10-K 제출 후 asof 에선 연말분기(2024-12-31)가 유효분기여야 한다(역산 없으면 못 찾음)."""
    conn = _seed_sec_db(tmp_path)
    assert das.effective_quarter_at_us_sec(conn, "AAPL", "2025-03-01") == "2024-12-31"
    conn.close()


def test_effective_quarter_excludes_not_yet_filed_annual(tmp_path):
    """10-K(2025-02-14) 제출 전 asof(2025-02-01)에선 직전 단독분기 Q3(2024-09-30)만."""
    conn = _seed_sec_db(tmp_path)
    assert das.effective_quarter_at_us_sec(conn, "AAPL", "2025-02-01") == "2024-09-30"
    conn.close()


def test_ttm_uses_reconstructed_q4_not_stale_prior_quarter(tmp_path):
    """TTM(2024-12-31)=40. 역산 없으면 Q3'24+Q2'24+Q1'24+Q3'23=38 이 되던 버그를 가드(C1)."""
    conn = _seed_sec_db(tmp_path)
    assert das._ttm_sec(conn, "AAPL", "2024-12-31", das.NET_INCOME_TAGS) == pytest.approx(40.0)
    conn.close()


def test_ttm_none_when_fewer_than_four_quarters(tmp_path):
    conn = _seed_sec_db(tmp_path)
    # 2023-06-30 기준: Q1'23,Q2'23 둘뿐(FY2023 는 기말 2023-12-31>기준 → 역산 불가) → None
    assert das._ttm_sec(conn, "AAPL", "2023-06-30", das.NET_INCOME_TAGS) is None
    conn.close()


def test_yoy_uses_offset_four_with_reconstructed_q4(tmp_path):
    conn = _seed_sec_db(tmp_path)
    # Q4'24(10, 역산) 대비 OFFSET4 = Q4'23(8, 역산) → 25%
    assert das._yoy_sec(conn, "AAPL", "2024-12-31", das.NET_INCOME_TAGS) == pytest.approx(25.0)
    conn.close()


# --------------------------------------------------------------------------
# M1 회귀 가드: look-ahead — 정정(10-K/A) 값이 과거 asof 조회에 새면 안 됨
# --------------------------------------------------------------------------
def test_ttm_excludes_later_restatement(tmp_path):
    """나중에 제출된 정정(10-K/A, filed 미래)은 과거 asof TTM 에 반영되면 안 된다(M1)."""
    conn = _seed_sec_db(tmp_path)
    # 정정: FY2024 순이익을 999로 바꾼 10-K/A 가 2026-06-01 에 제출됨(다른 accn)
    _ins_dur(conn, "NetIncomeLoss", "2024-01-01", "2024-12-31", "2026-06-01", 999.0,
             form="10-K/A", accn="restated-2024")
    conn.commit()
    # 정정 제출 전(2025-03-01) 시점 TTM 은 원본(40) — 정정값 999 를 쓰면 안 됨
    assert das._ttm_sec(conn, "AAPL", "2024-12-31", das.NET_INCOME_TAGS,
                        filed_max="2025-03-01") == pytest.approx(40.0)
    # 정정 제출 후(2026-07-01) 시점 TTM 은 정정본 반영
    assert das._ttm_sec(conn, "AAPL", "2024-12-31", das.NET_INCOME_TAGS,
                        filed_max="2026-07-01") == pytest.approx(999.0)
    conn.close()


# --------------------------------------------------------------------------
# M2 회귀 가드: 시점별 시가총액 = 종가 × 발행주식수 (현재 스냅샷 고정 금지)
# --------------------------------------------------------------------------
def test_market_cap_is_point_in_time_from_shares(tmp_path):
    conn = _seed_sec_db(tmp_path)
    r = das.metrics_at_us_sec(conn, "2025-03-01")[0]
    # 종가(200) × 발행주식수(5) = 1000. us_company.market_cap(99999)를 쓰면 틀림(M2).
    assert r["market_cap"] == pytest.approx(1000.0)
    conn.close()


def test_market_cap_falls_back_to_company_when_no_shares(tmp_path):
    """발행주식수 XBRL 이 없으면 us_company.market_cap 으로 폴백."""
    conn = _new_conn(tmp_path, "fallback.db")
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, market_cap, cik, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        ("AAPL", "Apple", "NASDAQ", 1000.0, "0000320193", "2025-03-01"),
    )
    for p_start, p_end, filed in _Q_2024:
        _ins_dur(conn, "NetIncomeLoss", p_start, p_end, filed, 10.0)
    _ins_dur(conn, "NetIncomeLoss", *_FY_2024[:2], _FY_2024[2], 40.0, form="10-K")
    conn.execute("INSERT INTO us_prices(stock_code, date, close) VALUES (?,?,?)",
                 ("AAPL", "2025-03-01", 150.0))
    conn.commit()
    r = das.metrics_at_us_sec(conn, "2025-03-01")[0]
    assert r["market_cap"] == pytest.approx(1000.0)  # 발행주식수 없음 → 회사 스냅샷 폴백
    assert r["per"] == pytest.approx(25.0)            # 1000/40
    conn.close()


# --------------------------------------------------------------------------
# 생존편향 사전필터: SEC 경로도 us_delisting 구간으로 상폐 종목을 제외한다(AC6/AC12)
# --------------------------------------------------------------------------
def test_metrics_at_us_sec_excludes_delisted_ticker(tmp_path):
    conn = _seed_sec_db(tmp_path)
    # AAPL을 asof(2025-03-01) 이전 상폐 구간으로 표시하면 사전필터로 제외돼야 한다.
    conn.execute(
        "INSERT INTO us_delisting(stock_code, company_name, exchange, listing_date, delisting_date) "
        "VALUES (?,?,?,?,?)", ("AAPL", "Apple", "NASDAQ", "2000-01-01", "2024-01-01"))
    conn.commit()
    assert das.metrics_at_us_sec(conn, "2025-03-01") == []
    conn.close()


# --------------------------------------------------------------------------
# AC6: 전체 지표군 (기존 data_access_us 검증표와 동일 값)
# --------------------------------------------------------------------------
def test_metrics_at_us_sec_computes_all_metrics(tmp_path):
    conn = _seed_sec_db(tmp_path)
    rows = das.metrics_at_us_sec(conn, "2025-03-01")
    assert len(rows) == 1
    r = rows[0]
    assert r["stock_code"] == "AAPL"
    assert r["quarter"] == "2024-12-31"
    assert r["close"] == 200.0
    assert r["market_cap"] == pytest.approx(1000.0)
    # 밸류
    assert r["per"] == pytest.approx(25.0)
    assert r["pbr"] == pytest.approx(5.0)
    assert r["psr"] == pytest.approx(2.5)
    assert r["pcr"] == pytest.approx(10.0)
    assert r["ev_ebitda"] == pytest.approx(6.0)     # EV(1200)/EBITDA_ttm(op120+dep80=200)
    assert r["peg"] == pytest.approx(1.0)
    # 수익성
    assert r["roe"] == pytest.approx(20.0)
    assert r["roa"] == pytest.approx(5.0)
    assert r["gp_a"] == pytest.approx(20.0)
    assert r["operating_margin"] == pytest.approx(30.0)
    assert r["net_margin"] == pytest.approx(10.0)
    # 안정성
    assert r["debt_ratio"] == pytest.approx(300.0)
    assert r["current_ratio"] == pytest.approx(200.0)
    assert r["interest_coverage"] == pytest.approx(6.0)
    # 성장(YoY 단일분기, Q4 역산 기반)
    assert r["revenue_growth"] == pytest.approx(25.0)
    assert r["op_growth"] == pytest.approx(25.0)
    assert r["ni_growth"] == pytest.approx(25.0)
    conn.close()


def test_metrics_at_us_sec_new_fields_none_when_raw_missing(tmp_path):
    """재무상태/현금흐름 원본이 없으면 관련 지표가 None(억지 추정 없음)."""
    conn = _new_conn(tmp_path, "min.db")
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, market_cap, cik, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        ("AAPL", "Apple", "NASDAQ", 1000.0, "0000320193", "2025-03-01"),
    )
    for p_start, p_end, filed in _Q_2024:
        _ins_dur(conn, "NetIncomeLoss", p_start, p_end, filed, 10.0)
    _ins_dur(conn, "NetIncomeLoss", *_FY_2024[:2], _FY_2024[2], 40.0, form="10-K")
    conn.execute("INSERT INTO us_prices(stock_code, date, close) VALUES (?,?,?)",
                 ("AAPL", "2025-03-01", 150.0))
    conn.commit()
    r = das.metrics_at_us_sec(conn, "2025-03-01")[0]
    for key in ("pbr", "psr", "pcr", "ev_ebitda", "roa", "gp_a", "debt_ratio",
                "current_ratio", "interest_coverage"):
        assert r[key] is None, f"{key} 는 원본 없으면 None 이어야 하는데 {r[key]!r}"
    assert r["per"] == pytest.approx(25.0)
    conn.close()


def test_metrics_at_us_sec_look_ahead_uses_prior_quarter(tmp_path):
    """asof 가 10-K 제출 전이면 직전 분기로 계산(미래참조 방지, AC7)."""
    conn = _seed_sec_db(tmp_path)
    rows = das.metrics_at_us_sec(conn, "2025-02-01")
    assert rows[0]["quarter"] == "2024-09-30"
    conn.close()


def test_build_callbacks_us_sec_returns_and_caches(tmp_path):
    conn = _seed_sec_db(tmp_path)
    metrics_fn, price_fn = das.build_callbacks_us_sec(conn)
    rows = metrics_fn("2025-03-01")
    assert rows[0]["stock_code"] == "AAPL"
    assert price_fn("2025-03-01", "AAPL") == 200.0
    assert metrics_fn("2025-03-01") is metrics_fn("2025-03-01")
    conn.close()


# --------------------------------------------------------------------------
# 부수: TTM 계산가능 커버리지 리포트(AC10 보완 — Q4 역산까지 성공한 종목 비율)
# --------------------------------------------------------------------------
def test_ttm_coverage_reports_computable_ratio(tmp_path):
    conn = _seed_sec_db(tmp_path)
    # 데이터 없는 추적종목 하나 추가 → 1/2
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, cik, updated_at) VALUES (?,?,?,?,?)",
        ("MSFT", "Microsoft", "NASDAQ", "0000789019", "2025-03-01"),
    )
    conn.commit()
    rep = das.ttm_coverage(conn)
    assert rep["total_tracked"] == 2
    assert rep["ttm_computable"] == 1   # AAPL 만 TTM(4분기, Q4 역산 포함) 가능
    assert rep["ttm_coverage_rate"] == 0.5
    conn.close()


# --------------------------------------------------------------------------
# MAJOR-1 회귀 가드: YoY 는 위치(series[4])가 아니라 '실제 1년 전 날짜'로 짝지어야 함
# --------------------------------------------------------------------------
def test_yoy_returns_none_when_year_ago_quarter_missing(tmp_path):
    """시계열에 구멍(전년동기 분기 누락)이 있으면 위치 인덱스로 엉뚱한 분기와 비교해선 안 된다.

    2023 회계연도에 FY 팩트가 없어 Q4'23 역산이 안 되면 series 에 2023-12-31 이 빠진다.
    이때 series[4]=2023-09-30(Q3'23) 을 '4분기 전'으로 오인해 (10-4)/4=150.0 처럼 그럴듯하게
    틀린 값을 내던 버그(critic MAJOR-1)를 가드 — 정답은 None(전년동기 분기 자체가 없으므로).
    """
    conn = _new_conn(tmp_path, "yoy_gap.db")
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, market_cap, cik, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        ("AAPL", "Apple", "NASDAQ", 1000.0, "0000320193", "2025-03-01"),
    )
    # 2024: Q1~Q3 단독 + FY(→Q4'24 역산 가능)
    for p_start, p_end, filed in _Q_2024:
        _ins_dur(conn, "NetIncomeLoss", p_start, p_end, filed, 10.0)
    _ins_dur(conn, "NetIncomeLoss", *_FY_2024[:2], _FY_2024[2], 40.0, form="10-K")
    # 2023: Q1~Q3 단독만(값 4). **FY2023 없음** → Q4'23 역산 불가 → 2023-12-31 이 series 에서 빠짐
    for p_start, p_end, filed in _Q_2023:
        _ins_dur(conn, "NetIncomeLoss", p_start, p_end, filed, 4.0)
    conn.commit()
    # 기준 Q4'24(10) 의 진짜 전년동기(2023-12-31)가 없으므로 YoY 는 None 이어야 한다.
    assert das._yoy_sec(conn, "AAPL", "2024-12-31", das.NET_INCOME_TAGS) is None
    conn.close()


# --------------------------------------------------------------------------
# MAJOR-2 실측 후속: 발행주식수 팩트가 asof 대비 오래되면(stale) 회사 스냅샷으로 폴백
# --------------------------------------------------------------------------
def test_stale_shares_fact_falls_back_to_company_market_cap(tmp_path):
    """발행주식수 XBRL 이 asof 보다 훨씬 오래됐으면(예: 버크셔 — 2011까지만·단일클래스) 그
    낡은 값으로 시총을 잡지 말고 us_company.market_cap 으로 폴백해야 한다(과소·오류 방지)."""
    # _ins_dur/_ins_inst 는 stock_code='AAPL' 고정이므로 회사도 AAPL 로 둔다(버크셔 시나리오 대입).
    conn = _new_conn(tmp_path, "stale.db")
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, market_cap, cik, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        ("AAPL", "DualClassCo", "NYSE", 7000.0, "0001067983", "2025-03-01"),
    )
    # 손익(2024 Q1~Q3 + FY) — 유효분기 확보
    for p_start, p_end, filed in _Q_2024:
        _ins_dur(conn, "NetIncomeLoss", p_start, p_end, filed, 10.0)
    _ins_dur(conn, "NetIncomeLoss", *_FY_2024[:2], _FY_2024[2], 40.0, form="10-K")
    # 발행주식수: 2011년(오래됨) 단일 팩트만 존재 → asof(2025-03-01) 대비 stale
    _ins_inst(conn, "EntityCommonStockSharesOutstanding", "2011-04-29", "2011-05-06", 941481.0,
              unit="shares", taxonomy="dei")
    conn.execute("INSERT INTO us_prices(stock_code, date, close) VALUES (?,?,?)",
                 ("AAPL", "2025-03-01", 700000.0))
    conn.commit()
    r = das.metrics_at_us_sec(conn, "2025-03-01")[0]
    # stale 주식수(941481×700000=거대)를 쓰면 안 됨 → 회사 스냅샷(7000) 폴백
    assert r["market_cap"] == pytest.approx(7000.0)
    conn.close()


def test_fresh_shares_fact_used_for_point_in_time_cap(tmp_path):
    """asof 근처(≈1년 이내)의 발행주식수는 정상적으로 시점 시총 계산에 쓴다(회귀 방지)."""
    conn = _new_conn(tmp_path, "fresh.db")
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, market_cap, cik, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        ("AAPL", "Apple", "NASDAQ", 99999.0, "0000320193", "2025-03-01"),
    )
    for p_start, p_end, filed in _Q_2024:
        _ins_dur(conn, "NetIncomeLoss", p_start, p_end, filed, 10.0)
    _ins_dur(conn, "NetIncomeLoss", *_FY_2024[:2], _FY_2024[2], 40.0, form="10-K")
    _ins_inst(conn, "EntityCommonStockSharesOutstanding", "2024-12-31", "2025-02-14", 5.0,
              unit="shares", taxonomy="dei")
    conn.execute("INSERT INTO us_prices(stock_code, date, close) VALUES (?,?,?)",
                 ("AAPL", "2025-03-01", 200.0))
    conn.commit()
    r = das.metrics_at_us_sec(conn, "2025-03-01")[0]
    assert r["market_cap"] == pytest.approx(1000.0)  # 200×5, stale 아님 → 주식수 사용
    conn.close()
