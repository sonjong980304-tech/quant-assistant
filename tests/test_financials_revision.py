"""정정공시(재무제표 재작성) 이력 보존 + 백테스트 look-ahead 재현 테스트.

배경: financials 테이블은 UNIQUE(stock_code, quarter, account_key)라 (종목,분기,계정)당
1행뿐이다. DART가 정정공시를 내고 재수집하면 예전 값이 덮어써져 "정정 전엔 얼마였는지"
역사가 사라진다 → 백테스트가 과거 asof 시점에 "그때 실제로 알 수 있었던 숫자" 대신
"나중에야 알 수 있었던 정정값"을 잘못 쓰는 look-ahead(미래참조) 편향이 생긴다.

해결: append-only 새 테이블 financials_revision(+ rcept_no)에 매 적재마다 새 행을 INSERT
(같은 rcept_no 재수집은 멱등). 백테스트 look-ahead 크리티컬 경로만 이 테이블에서
"disclosed_date<=asof 중 가장 최근 disclosed_date" 버전을 고른다(SEC filed_max와 동일 사상).
기존 financials 테이블/소비자(goldset 등)는 그대로 둔다.
"""
from __future__ import annotations

import sqlite3

from src.backtest.data_access import effective_quarter_at, metrics_at
from src.db import connect, init_db
from src.ingest.dart import _write_reports_year
from src.ingest.metrics import _fin


def _conn(tmp_path, name="rev.db") -> sqlite3.Connection:
    db = tmp_path / name
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def _seed_rev(conn, code, quarter, key, disclosed, rcept, amount, name="계정"):
    conn.execute(
        "INSERT INTO financials_revision"
        "(stock_code, quarter, account_key, account_name, amount, disclosed_date, rcept_no) "
        "VALUES (?,?,?,?,?,?,?)",
        (code, quarter, key, name, amount, disclosed, rcept),
    )


# ---------------------------------------------------------------------------
# 1) 쓰기 경로: financials는 최신값 1행, financials_revision은 정정마다 append
# ---------------------------------------------------------------------------
def test_restatement_appends_revision_but_replaces_financials(tmp_path):
    db = tmp_path / "w.db"
    init_db(str(db))
    conn = connect(str(db))
    # 원본 공시: 2024Q1 revenue=100 (rcept 2024-05-15)
    reports1 = {1: ({"revenue": (100.0, "매출액")}, "20240515000001"),
                2: ({}, None), 3: ({}, None), 4: ({}, None)}
    _write_reports_year(conn, "000001", 2024, reports1, {"2024Q1"})
    conn.commit()
    # 정정 공시: 같은 2024Q1 revenue=80 (rcept 2024-11-01) → 재무제표 재작성
    reports2 = {1: ({"revenue": (80.0, "매출액")}, "20241101000002"),
                2: ({}, None), 3: ({}, None), 4: ({}, None)}
    _write_reports_year(conn, "000001", 2024, reports2, {"2024Q1"})
    conn.commit()

    # financials(기존): 최신값 하나만 남는다(REPLACE)
    frows = conn.execute(
        "SELECT amount FROM financials WHERE stock_code='000001' "
        "AND quarter='2024Q1' AND account_key='revenue'"
    ).fetchall()
    assert len(frows) == 1
    assert frows[0]["amount"] == 80

    # financials_revision: 두 버전이 모두 보존된다(append-only)
    rrows = conn.execute(
        "SELECT amount, disclosed_date, rcept_no FROM financials_revision "
        "WHERE stock_code='000001' AND quarter='2024Q1' AND account_key='revenue' "
        "ORDER BY disclosed_date"
    ).fetchall()
    assert len(rrows) == 2
    assert rrows[0]["amount"] == 100 and rrows[0]["disclosed_date"] == "2024-05-15"
    assert rrows[1]["amount"] == 80 and rrows[1]["disclosed_date"] == "2024-11-01"
    conn.close()


def test_reingest_same_rcept_is_idempotent(tmp_path):
    """같은 공시(같은 rcept_no)를 재수집하면 revision 행이 늘지 않는다(멱등)."""
    db = tmp_path / "w2.db"
    init_db(str(db))
    conn = connect(str(db))
    reports = {1: ({"revenue": (100.0, "매출액")}, "20240515000001"),
               2: ({}, None), 3: ({}, None), 4: ({}, None)}
    _write_reports_year(conn, "000001", 2024, reports, {"2024Q1"})
    conn.commit()
    _write_reports_year(conn, "000001", 2024, reports, {"2024Q1"})
    conn.commit()
    rrows = conn.execute(
        "SELECT * FROM financials_revision WHERE stock_code='000001' "
        "AND quarter='2024Q1' AND account_key='revenue'"
    ).fetchall()
    assert len(rrows) == 1
    conn.close()


# ---------------------------------------------------------------------------
# 2) 읽기 경로(_fin asof): 정정 전엔 원본, 정정 후엔 정정값 재현
# ---------------------------------------------------------------------------
def test_fin_asof_original_before_restated_after(tmp_path):
    conn = _conn(tmp_path)
    _seed_rev(conn, "000001", "2024Q1", "revenue", "2024-05-15", "A", 100.0)
    _seed_rev(conn, "000001", "2024Q1", "revenue", "2024-11-01", "B", 80.0)
    conn.commit()
    # 정정 전 시점: 원본값
    assert _fin(conn, "000001", "2024Q1", "revenue", asof="2024-06-01") == 100.0
    # 정정 후 시점: 정정값
    assert _fin(conn, "000001", "2024Q1", "revenue", asof="2024-12-01") == 80.0
    # 원본 공시 이전 시점: 아직 알 수 없음 → None(look-ahead 방지)
    assert _fin(conn, "000001", "2024Q1", "revenue", asof="2024-03-01") is None
    conn.close()


def test_fin_without_asof_reads_financials_latest(tmp_path):
    """asof 미지정(기존 compute_metrics 경로)은 financials 최신값만 읽는다(회귀 방지)."""
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
        "VALUES ('000001','2024Q1','2024-11-01','revenue',80.0)"
    )
    # revision에는 정정 전 원본도 있지만 asof 없이 부르면 무시된다
    _seed_rev(conn, "000001", "2024Q1", "revenue", "2024-05-15", "A", 100.0)
    conn.commit()
    assert _fin(conn, "000001", "2024Q1", "revenue") == 80.0
    conn.close()


def test_fin_asof_falls_back_to_financials_when_no_revision(tmp_path):
    """revision에 이력이 전혀 없는(도입 전 적재분) (종목,분기,계정)은 기존 financials로 폴백."""
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
        "VALUES ('000001','2024Q1','2024-05-15','revenue',55.0)"
    )
    conn.commit()
    assert _fin(conn, "000001", "2024Q1", "revenue", asof="2024-06-01") == 55.0
    conn.close()


# ---------------------------------------------------------------------------
# 3) effective_quarter_at: revision의 disclosed_date로 유효분기 판정
# ---------------------------------------------------------------------------
def test_effective_quarter_at_uses_revision_disclosed(tmp_path):
    conn = _conn(tmp_path)
    _seed_rev(conn, "000001", "2024Q1", "revenue", "2024-05-15", "A", 100.0)
    _seed_rev(conn, "000001", "2024Q1", "revenue", "2024-11-01", "B", 80.0)
    conn.commit()
    # 원본 공시 전: 유효분기 없음
    assert effective_quarter_at(conn, "000001", "2024-03-01") is None
    # 원본 공시 후: 2024Q1 유효
    assert effective_quarter_at(conn, "000001", "2024-06-01") == "2024Q1"
    conn.close()


def test_effective_quarter_at_falls_back_to_financials(tmp_path):
    """revision 없는 도입 전 데이터도 financials.disclosed_date로 유효분기 판정(폴백)."""
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
        "VALUES ('000001','2024Q1','2024-05-15','revenue',55.0)"
    )
    conn.commit()
    assert effective_quarter_at(conn, "000001", "2024-06-01") == "2024Q1"
    assert effective_quarter_at(conn, "000001", "2024-03-01") is None
    conn.close()


# ---------------------------------------------------------------------------
# 4) metrics_at 전 구간 재현: 정정 전 asof=원본 매출, 정정 후 asof=정정 매출
# ---------------------------------------------------------------------------
def test_metrics_at_reproduces_original_then_restated_revenue(tmp_path):
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO company(stock_code, name, market, sector) "
        "VALUES ('000001','가나전자','KOSPI','반도체')"
    )
    # 2024Q1 매출 정정: 원본 100e12(2024-05-15) → 정정 80e12(2024-11-01)
    _seed_rev(conn, "000001", "2024Q1", "revenue", "2024-05-15", "A", 100e12)
    _seed_rev(conn, "000001", "2024Q1", "revenue", "2024-11-01", "B", 80e12)
    # 두 asof 각각에 대해 그 시점 이하 종가가 있어야 metrics_at이 행을 만든다
    conn.execute("INSERT INTO prices(stock_code, date, close, market_cap) VALUES ('000001','2024-05-20',10000.0,1e14)")
    conn.execute("INSERT INTO prices(stock_code, date, close, market_cap) VALUES ('000001','2024-11-15',10000.0,1e14)")
    conn.commit()

    before = metrics_at(conn, "2024-06-01")
    assert len(before) == 1 and before[0]["quarter"] == "2024Q1"
    assert before[0]["revenue"] == 100e12  # 정정 전: 원본 매출

    after = metrics_at(conn, "2024-12-01")
    assert len(after) == 1 and after[0]["quarter"] == "2024Q1"
    assert after[0]["revenue"] == 80e12   # 정정 후: 정정 매출
    conn.close()
