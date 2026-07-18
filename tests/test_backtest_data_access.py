"""src/backtest/data_access.py의 _is_alive() 회귀 테스트 (son-checker 이슈 #23 BUG-1).

_is_alive()의 판정 로직 자체는 이번에 바꾸지 않았다(모르면 살아있다고 보는 폴백은
의도된 정책). 다만 ingest_delisting()이 빈 문자열 대신 NULL을 쓰도록 바뀌었으므로,
NULL과 빈 문자열 둘 다 기존과 동일하게 "모름 → 살아있음"으로 처리되는지, 그리고
실제 delisting_date가 있을 때는 asof와 정확히 비교해 판정하는지 회귀로 고정한다.
이 파일이 생기기 전에는 _is_alive()를 직접 겨냥한 테스트가 없었다(metrics_at 경유뿐).
"""
from __future__ import annotations

import sqlite3

import pytest

from src.backtest.data_access import (
    _is_alive,
    _months_before,
    _one_year_before,
    mode_financial_quarter_at,
    momentum_12_1,
    momentum_12_1_batch,
    price_return_over_months,
)
from src.db import init_db


def _conn(tmp_path) -> sqlite3.Connection:
    db = tmp_path / "alive.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def _seed_prices(conn, code: str, rows: list[tuple[str, float]]) -> None:
    for date_str, close in rows:
        conn.execute(
            "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,?)",
            (code, date_str, close, 1e12),
        )
    conn.commit()


def test_is_alive_true_before_delisting_date(tmp_path):
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO delisting(stock_code, name, delisting_date) VALUES(?,?,?)",
        ("000030", "우리은행", "2019-02-12"),
    )
    conn.commit()
    assert _is_alive(conn, "000030", "2018-01-01") is True


def test_is_alive_false_on_or_after_delisting_date(tmp_path):
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO delisting(stock_code, name, delisting_date) VALUES(?,?,?)",
        ("000030", "우리은행", "2019-02-12"),
    )
    conn.commit()
    assert _is_alive(conn, "000030", "2020-01-01") is False


def test_is_alive_true_when_no_delisting_row(tmp_path):
    conn = _conn(tmp_path)
    assert _is_alive(conn, "005930", "2026-01-01") is True


def test_is_alive_true_when_delisting_date_is_null(tmp_path):
    """정확한 상폐일을 모르는(NULL) 경우 — ingest_delisting()이 이제 NULL을 쓰므로,
    그런 행이 있어도 기존과 동일하게 '모름 → 살아있음'으로 처리돼야 한다."""
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO delisting(stock_code, name, delisting_date) VALUES(?,?,?)",
        ("999999", "신규상폐추정", None),
    )
    conn.commit()
    assert _is_alive(conn, "999999", "2026-01-01") is True


# ── _months_before(): _one_year_before의 임의 개월수 일반화 ─────────────────────
# return_12m(고정 12개월)만 풀던 _one_year_before를 임의 N개월로 일반화한다. 연/월 캐리와
# 말일 보정(존재하지 않는 날짜로 넘어가지 않게)을 검증하고, months=12일 때 기존
# _one_year_before와 항상 동일한지(회귀 일관성)를 여러 날짜(윤년 2월 포함)로 고정한다.

def test_months_before_simple_within_year():
    assert _months_before("2026-07-16", 3) == "2026-04-16"


def test_months_before_carries_across_year_boundary():
    assert _months_before("2026-01-15", 3) == "2025-10-15"


def test_months_before_clamps_to_month_end_non_leap():
    # 3/31에서 1개월 전 → 2월엔 31일이 없으므로 평년 말일 28일로 보정.
    assert _months_before("2026-03-31", 1) == "2026-02-28"


def test_months_before_clamps_to_month_end_leap_year():
    # 윤년 2월은 29일 → 3/31의 1개월 전은 2/29로 보정(존재하는 날짜).
    assert _months_before("2024-03-31", 1) == "2024-02-29"


@pytest.mark.parametrize(
    "asof",
    [
        "2026-07-16",
        "2026-01-01",
        "2024-02-29",  # 윤년 2/29 → 전년 2/28 보정 경로
        "2025-12-31",
        "2020-02-29",
        "2023-03-01",
    ],
)
def test_months_before_12_matches_one_year_before(asof):
    # 기존 12개월 로직과 완전히 일관돼야 한다(회귀): _months_before(asof, 12) == _one_year_before(asof).
    assert _months_before(asof, 12) == _one_year_before(asof)


# ── price_return_over_months(): 임의 개월수 가격 수익률(결정론) ──────────────────
# return_12m와 동일한 "date<=시점 최근 거래일 종가" 원칙을 임의 N개월로 일반화한다.
# LLM 코드생성 폴백(신뢰 불가) 대신 SQL로 결정론적으로 계산한다.

def test_price_return_over_months_computes_3_6_12_months(tmp_path):
    conn = _conn(tmp_path)
    _seed_prices(
        conn,
        "000001",
        [
            ("2025-07-16", 60.0),   # 12개월 전
            ("2026-01-16", 80.0),   # 6개월 전
            ("2026-04-16", 100.0),  # 3개월 전
            ("2026-07-16", 120.0),  # 기준시점(asof)
        ],
    )

    r3 = price_return_over_months(conn, "000001", "2026-07-16", 3)
    assert r3["months"] == 3
    assert r3["start_target_date"] == "2026-04-16"
    assert r3["start_date"] == "2026-04-16"
    assert r3["start_close"] == pytest.approx(100.0)
    assert r3["end_date"] == "2026-07-16"
    assert r3["end_close"] == pytest.approx(120.0)
    assert r3["return_pct"] == pytest.approx(20.0)  # (120-100)/100*100

    r6 = price_return_over_months(conn, "000001", "2026-07-16", 6)
    assert r6["start_date"] == "2026-01-16"
    assert r6["return_pct"] == pytest.approx(50.0)  # (120-80)/80*100

    r12 = price_return_over_months(conn, "000001", "2026-07-16", 12)
    assert r12["start_date"] == "2025-07-16"
    assert r12["return_pct"] == pytest.approx(100.0)  # (120-60)/60*100


def test_price_return_over_months_matches_nearest_prior_trading_day_for_start(tmp_path):
    conn = _conn(tmp_path)
    # 3개월 전 목표일(2026-04-16)이 휴장일이라 그 이하 가장 가까운 거래일(2026-04-13)이 매칭돼야 한다.
    _seed_prices(conn, "000001", [("2026-04-13", 100.0), ("2026-07-16", 120.0)])
    r = price_return_over_months(conn, "000001", "2026-07-16", 3)
    assert r["start_target_date"] == "2026-04-16"
    assert r["start_date"] == "2026-04-13"  # 요청일과 다른 실제 매칭 거래일
    assert r["return_pct"] == pytest.approx(20.0)


def test_price_return_over_months_none_when_no_start_data(tmp_path):
    conn = _conn(tmp_path)
    # 기준시점 종가만 있고 start 시점 이전 데이터가 아예 없음(예: 상장 전) → 조용히 None.
    _seed_prices(conn, "000001", [("2026-07-16", 120.0)])
    r = price_return_over_months(conn, "000001", "2026-07-16", 3)
    assert r["end_date"] == "2026-07-16"
    assert r["end_close"] == pytest.approx(120.0)
    assert r["start_date"] is None
    assert r["start_close"] is None
    assert r["return_pct"] is None  # 잘못된 값 대신 None


def test_price_return_over_months_all_none_when_code_has_no_data(tmp_path):
    conn = _conn(tmp_path)
    r = price_return_over_months(conn, "NOPE", "2026-07-16", 3)
    assert r["end_date"] is None
    assert r["end_close"] is None
    assert r["start_date"] is None
    assert r["start_close"] is None
    assert r["return_pct"] is None


# ── momentum_12_1(): 12-1 모멘텀(최근 1개월 제외 12개월 수익률) ─────────────────
# 12개월 전 종가 대비 1개월 전 종가의 수익률(%). 최근 1개월(asof~1개월전)은 제외한다
# (단순 12M return_12m과 다름 — 사용자가 명시적으로 12-1을 선택). 배치 버전이 크로스섹션
# (수천 종목)에 쓰이므로 종목별 반복 SQL이 아니라 상수 회수 배치 SQL로 동작해야 한다.


class _CountingConn:
    """conn.execute 호출 횟수를 세는 얇은 래퍼(배치 SQL 성능 회귀 가드용)."""

    def __init__(self, real):
        self._real = real
        self.execute_calls = 0

    def execute(self, *args, **kwargs):
        self.execute_calls += 1
        return self._real.execute(*args, **kwargs)


def test_momentum_12_1_manual_value(tmp_path):
    conn = _conn(tmp_path)
    # asof=2026-07-16 → 1개월전=2026-06-16, 12개월전=2025-07-16.
    # 최근 1개월(asof 종가 150)은 반드시 제외되고, 1개월전(130)/12개월전(100)로 계산.
    _seed_prices(
        conn,
        "000001",
        [("2025-07-16", 100.0), ("2026-06-16", 130.0), ("2026-07-16", 150.0)],
    )
    m = momentum_12_1(conn, "000001", "2026-07-16")
    assert m == pytest.approx(30.0)  # (130-100)/100*100, asof(150)은 제외


def test_momentum_12_1_nearest_prior_trading_day(tmp_path):
    conn = _conn(tmp_path)
    # 목표일이 휴장이면 그 이하 가장 가까운 거래일 종가를 쓴다.
    _seed_prices(
        conn,
        "000001",
        [("2025-07-10", 100.0), ("2026-06-12", 120.0), ("2026-07-16", 999.0)],
    )
    m = momentum_12_1(conn, "000001", "2026-07-16")
    assert m == pytest.approx(20.0)  # (120-100)/100*100


def test_momentum_12_1_none_when_insufficient_history(tmp_path):
    conn = _conn(tmp_path)
    _seed_prices(conn, "000001", [("2026-07-16", 150.0)])  # 과거 없음
    assert momentum_12_1(conn, "000001", "2026-07-16") is None


def test_momentum_12_1_batch_matches_single(tmp_path):
    conn = _conn(tmp_path)
    _seed_prices(conn, "AAA", [("2025-07-16", 100.0), ("2026-06-16", 130.0), ("2026-07-16", 150.0)])
    _seed_prices(conn, "BBB", [("2025-07-16", 50.0), ("2026-06-16", 40.0), ("2026-07-16", 45.0)])
    _seed_prices(conn, "CCC", [("2026-07-16", 10.0)])  # 과거 없음 → None
    batch = momentum_12_1_batch(conn, ["AAA", "BBB", "CCC"], "2026-07-16")
    assert batch["AAA"] == pytest.approx(momentum_12_1(conn, "AAA", "2026-07-16"))
    assert batch["BBB"] == pytest.approx(momentum_12_1(conn, "BBB", "2026-07-16"))
    assert batch["AAA"] == pytest.approx(30.0)
    assert batch["BBB"] == pytest.approx(-20.0)  # (40-50)/50*100
    assert batch["CCC"] is None


def test_momentum_12_1_batch_uses_constant_sql_calls(tmp_path):
    """배치 SQL 성능 회귀 가드: execute 호출 횟수가 종목 수에 비례하지 않아야 한다."""
    conn = _conn(tmp_path)
    for i in range(8):
        _seed_prices(
            conn, f"S{i}",
            [("2025-07-16", 100.0 + i), ("2026-06-16", 120.0 + i), ("2026-07-16", 150.0 + i)],
        )

    c2 = _CountingConn(conn)
    momentum_12_1_batch(c2, ["S0", "S1"], "2026-07-16")
    c8 = _CountingConn(conn)
    momentum_12_1_batch(c8, [f"S{i}" for i in range(8)], "2026-07-16")
    # 2종목이든 8종목이든 execute 호출 횟수가 동일(상수) — N에 비례하지 않는다
    assert c2.execute_calls == c8.execute_calls
    assert c8.execute_calls <= 2  # 시작/끝 컷오프 각 1회 배치 SQL


def test_momentum_12_1_batch_empty_codes(tmp_path):
    conn = _conn(tmp_path)
    assert momentum_12_1_batch(conn, [], "2026-07-16") == {}


# --------------------------------------------------------------------------
# mode_financial_quarter_at — get_cross_section(_qvm)류 파이프라인(correlation/
# quantile_bucket_means 등)이 result에 종목별 quarter를 남기지 않아, backtest 도메인의
# data_asof에 "재무 기준분기"를 못 붙이던 문제 대응(domain_backtest._build_data_asof에서
# 재사용). effective_quarter_at을 종목마다 반복 호출하는 대신(전종목 순회 시 비용 큼),
# 동일한 disclosed_date<=asof 조건을 SQL 윈도우함수로 한 번에 계산해 최빈값만 뽑는다.
# --------------------------------------------------------------------------
def _seed_financials(conn, rows: list[tuple[str, str, str]]) -> None:
    """rows: (stock_code, quarter, disclosed_date). account_key는 더미로 채운다."""
    for code, quarter, disclosed in rows:
        conn.execute(
            "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
            "VALUES (?,?,?,?,?)",
            (code, quarter, disclosed, "revenue", 100.0),
        )
    conn.commit()


def test_mode_financial_quarter_at_returns_most_common_quarter_among_disclosed(tmp_path):
    conn = _conn(tmp_path)
    _seed_financials(conn, [
        ("000001", "2026Q1", "2026-05-15"),
        ("000002", "2026Q1", "2026-05-20"),
        ("000003", "2025Q4", "2026-03-31"),  # 아직 2026Q1 미공시
    ])
    assert mode_financial_quarter_at(conn, "2026-07-18") == "2026Q1"


def test_mode_financial_quarter_at_respects_lookahead_cutoff(tmp_path):
    """disclosed_date > asof인 분기는 무시한다(미래참조 방지, effective_quarter_at과 동일 원칙)."""
    conn = _conn(tmp_path)
    _seed_financials(conn, [
        ("000001", "2026Q2", "2026-08-14"),  # asof 이후 공시 — 아직 반영 안 됨
        ("000001", "2026Q1", "2026-05-15"),
    ])
    assert mode_financial_quarter_at(conn, "2026-07-18") == "2026Q1"


def test_mode_financial_quarter_at_none_when_no_data(tmp_path):
    conn = _conn(tmp_path)
    assert mode_financial_quarter_at(conn, "2026-07-18") is None


def test_mode_financial_quarter_at_counts_per_stock_not_raw_rows(tmp_path):
    """종목 하나가 계정과목(account_key)별로 여러 행을 가져도 quarter 표는 종목당 1표여야
    한다 — 원시 행수로 세면(버그) 소수 종목의 계정 수가 많다는 이유만으로 최빈값이 왜곡된다."""
    conn = _conn(tmp_path)
    for key in ("revenue", "net_income", "total_assets"):
        conn.execute(
            "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
            "VALUES (?,?,?,?,?)",
            ("000001", "2026Q1", "2026-05-15", key, 1.0),
        )
    for code in ("000002", "000003"):
        conn.execute(
            "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
            "VALUES (?,?,?,?,?)",
            (code, "2025Q4", "2026-03-31", "revenue", 1.0),
        )
    conn.commit()
    # 000001은 원시 행 3개(2026Q1)지만 종목 수로는 1표. 000002/000003은 원시행 1개씩(2025Q4)
    # 이지만 종목 수로는 2표 — 종목당 1표로 세면 2025Q4가 이겨야 한다(원시 행수로 세면 2026Q1이 이김).
    assert mode_financial_quarter_at(conn, "2026-07-18") == "2025Q4"
