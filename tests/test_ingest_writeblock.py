"""_write_reports_year — 재무 기록 블록(라이브 수집과 재파싱 공용) 테스트.

핵심: Q4 흐름항목(FLOW) 차분(연간−(Q1+Q2+Q3))과 재무상태(STOCK) 시점값 통과를 검증한다.
이 로직을 추출해 라이브 수집(_ingest_one_company)과 원문 재파싱이 동일하게 쓰도록 한다.
"""
from __future__ import annotations

from src.db import connect, init_db
from src.ingest.dart import _write_reports_year


def _amt(conn, quarter, key):
    r = conn.execute(
        "SELECT amount FROM financials WHERE stock_code='000001' "
        "AND quarter=? AND account_key=?",
        (quarter, key),
    ).fetchone()
    return r[0] if r else None


def test_q4_flow_differencing_and_stock_passthrough(tmp_path):
    """Q4 FLOW는 연간−누적 차분, STOCK(자산총계)은 시점값 그대로."""
    db = tmp_path / "t.db"
    init_db(str(db))
    conn = connect(str(db))
    # 분기별 3개월 값: revenue(FLOW), 연간 total_assets(STOCK)
    reports = {
        1: ({"revenue": (10.0, "매출액")}, "20240515"),
        2: ({"revenue": (20.0, "매출액")}, "20240814"),
        3: ({"revenue": (30.0, "매출액")}, "20241114"),
        4: ({"revenue": (100.0, "매출액"), "total_assets": (500.0, "자산총계")}, "20250320"),
    }
    wanted = {"2024Q1", "2024Q2", "2024Q3", "2024Q4"}

    n_rows, latest = _write_reports_year(conn, "000001", 2024, reports, wanted)
    conn.commit()

    assert _amt(conn, "2024Q1", "revenue") == 10          # 그대로
    assert _amt(conn, "2024Q4", "revenue") == 40           # FLOW 차분: 100-(10+20+30)
    assert _amt(conn, "2024Q4", "total_assets") == 500     # STOCK: 차분 안 함
    assert latest == "2024Q4"
    assert n_rows == 5                                      # Q1~Q4 revenue 4행 + Q4 자산 1행


# 실사용 재현: 효성화학(298000) 2025Q4 PER이 1분기 적자인데도 낮은 양수(0.47배)로 계산됐다.
# 원인 추적 결과 controlling_net_income(지배주주순이익, 손익계산서 FLOW 항목)이 FLOW_KEYS에
# 빠져 있어 Q4에 "연간−(Q1+Q2+Q3) 차분"이 아니라 연간 누적치가 그대로 저장됐다 — net_income
# (총순이익)은 정확히 차분됐는데 controlling_net_income만 차분 안 된 비대칭이 발생했다.
def test_q4_flow_differencing_applies_to_controlling_net_income(tmp_path):
    db = tmp_path / "t2.db"
    init_db(str(db))
    conn = connect(str(db))
    reports = {
        1: ({"controlling_net_income": (100.0, "지배기업의 소유주지분")}, "20240515"),
        2: ({"controlling_net_income": (50.0, "지배기업의 소유주지분")}, "20240814"),
        3: ({"controlling_net_income": (-20.0, "지배기업의 소유주지분")}, "20241114"),
        4: ({"controlling_net_income": (300.0, "지배기업의 소유주지분")}, "20250320"),
    }
    wanted = {"2024Q1", "2024Q2", "2024Q3", "2024Q4"}

    _write_reports_year(conn, "000001", 2024, reports, wanted)
    conn.commit()

    assert _amt(conn, "2024Q1", "controlling_net_income") == 100  # 그대로
    # FLOW 차분: 연간누적(300) - (Q1+Q2+Q3 = 100+50-20 = 130) = 170
    assert _amt(conn, "2024Q4", "controlling_net_income") == 170


# 실사용 재현(2건째): FLOW_KEYS 전수 점검 중 tax_expense(법인세비용, 마법공식 EBIT용)도
# controlling_net_income과 동일한 패턴으로 FLOW_KEYS에서 빠져 있었다 — 손익계산서 항목(매
# 분기 새로 발생)인데 Q4에 차분 없이 연간 누적치가 그대로 저장되고 있었다(2,786종목,
# 21,407건 영향 확인됨).
def test_q4_flow_differencing_applies_to_tax_expense(tmp_path):
    db = tmp_path / "t3.db"
    init_db(str(db))
    conn = connect(str(db))
    reports = {
        1: ({"tax_expense": (10.0, "법인세비용")}, "20240515"),
        2: ({"tax_expense": (5.0, "법인세비용")}, "20240814"),
        3: ({"tax_expense": (-2.0, "법인세비용")}, "20241114"),
        4: ({"tax_expense": (30.0, "법인세비용")}, "20250320"),
    }
    wanted = {"2024Q1", "2024Q2", "2024Q3", "2024Q4"}

    _write_reports_year(conn, "000001", 2024, reports, wanted)
    conn.commit()

    assert _amt(conn, "2024Q1", "tax_expense") == 10  # 그대로
    # FLOW 차분: 연간누적(30) - (Q1+Q2+Q3 = 10+5-2 = 13) = 17
    assert _amt(conn, "2024Q4", "tax_expense") == 17


def test_disclosed_date_from_rcept(tmp_path):
    """공시일은 rcept 앞 8자리(YYYYMMDD)를 YYYY-MM-DD로 기록한다."""
    db = tmp_path / "t.db"
    init_db(str(db))
    conn = connect(str(db))
    reports = {
        1: ({"revenue": (10.0, "매출액")}, "20240515"),
        2: ({}, None),
        3: ({}, None),
        4: ({}, None),
    }
    _write_reports_year(conn, "000001", 2024, reports, {"2024Q1"})
    conn.commit()
    r = conn.execute(
        "SELECT disclosed_date FROM financials WHERE stock_code='000001' AND quarter='2024Q1'"
    ).fetchone()
    assert r[0] == "2024-05-15"


def test_quarter_not_in_wanted_is_skipped(tmp_path):
    """wanted에 없는 분기는 기록하지 않는다."""
    db = tmp_path / "t.db"
    init_db(str(db))
    conn = connect(str(db))
    reports = {
        1: ({"revenue": (10.0, "매출액")}, "20240515"),
        2: ({"revenue": (20.0, "매출액")}, "20240814"),
        3: ({}, None),
        4: ({}, None),
    }
    _write_reports_year(conn, "000001", 2024, reports, {"2024Q1"})  # Q2 제외
    conn.commit()
    assert _amt(conn, "2024Q1", "revenue") == 10
    assert _amt(conn, "2024Q2", "revenue") is None
