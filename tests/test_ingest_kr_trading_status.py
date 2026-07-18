"""KRX 관리종목/매매거래정지 스냅샷 누적 수집 단위 테스트 (TDD).

과거 이력은 무료로 구할 수 없다(KRX/KIND 모두 조회일 파라미터를 무시하고 '오늘 현재'
스냅샷만 반환) → 대신 매 실행마다 '오늘 현재' 스냅샷을 받아 직전 실행 스냅샷과 diff 해서
앞으로의(미래) 지정~해제 구간을 누적으로 쌓는다. 스냅샷 자체는 시점조회가 안 되지만,
우리가 주기적으로 저장+diff 하면 우리 쪽에서 구간 이력을 만들 수 있다.

- KRX getJsonData(MDCSTAT21401=관리종목 / MDCSTAT21201=매매거래정지) 응답을 mock(DI)으로
  주입해 네트워크·로그인 없이 파싱·diff·upsert 를 검증한다(라이브 KRX 로그인은 절대 호출 안 함).
- 구간(start_date~end_date) 기반 스키마(us_delisting 선례): 한 종목이 지정→해제→재지정을
  반복할 수 있으니 여러 행을 허용한다. '직전 스냅샷'은 별도 저장 없이 DB의 열린 구간
  (end_date IS NULL)에서 유도한다.
- 이 데이터는 backtest look-ahead 경로에 연결하지 않는다(과거 이력이 없어 생존편향 악화).
  라이브(현재 시점) 필터링 전용 is_currently_administrative_or_halted 만 노출한다.
DB 접근은 임시 SQLite 에 시딩해 사용자 DB 와 완전히 격리한다.
"""
from __future__ import annotations

import sqlite3

from src.db import init_db
from src.ingest import kr_trading_status as kts


# KRX getJsonData 실제 응답 형식: {"output": [ {...}, ... ]}
# 관리종목 현황(MDCSTAT21401): ISU_CD/ISU_SRT_CD/ISU_NM/MKT_NM/FST_DESIGN_DD/LIST_BZ_RSN_NM
_ADMIN_PAYLOAD = {
    "output": [
        {"ISU_CD": "KR7093230006", "ISU_SRT_CD": "093230", "ISU_NM": "이아이디",
         "MKT_NM": "KOSDAQ", "FST_DESIGN_DD": "2023/03/24", "LIST_BZ_RSN_NM": "감사의견 거절"},
        {"ISU_CD": "KR7900140006", "ISU_SRT_CD": "900140", "ISU_NM": "엘브이엠씨",
         "MKT_NM": "KOSPI", "FST_DESIGN_DD": "2024/01/02", "LIST_BZ_RSN_NM": "자본잠식"},
    ]
}
# 매매거래정지 현황(MDCSTAT21201): ISU_CD/ISU_SRT_CD/ISU_NM/HALT_DESNRELS_DDTM/HALT_RSN_NM/LST_TRD_DD
_HALT_PAYLOAD = {
    "output": [
        {"ISU_CD": "KR7099220004", "ISU_SRT_CD": "099220", "ISU_NM": "SDN",
         "HALT_DESNRELS_DDTM": "2024-05-20 09:00:00", "HALT_RSN_NM": "불성실공시지정",
         "LST_TRD_DD": "2024/05/17"},
    ]
}


def _conn(db: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


# --------------------------------------------------------------------------
# 파싱: KRX output → 정규화 행
# --------------------------------------------------------------------------
def test_parse_admin_issue_extracts_fields():
    rows = kts.parse_admin_issue(_ADMIN_PAYLOAD)
    assert len(rows) == 2
    eid = next(r for r in rows if r["stock_code"] == "093230")
    assert eid["company_name"] == "이아이디"
    assert eid["market"] == "KOSDAQ"
    assert eid["reason"] == "감사의견 거절"
    assert eid["krx_designated_date"] == "2023-03-24"


def test_parse_trading_halt_extracts_fields():
    rows = kts.parse_trading_halt(_HALT_PAYLOAD)
    assert len(rows) == 1
    sdn = rows[0]
    assert sdn["stock_code"] == "099220"
    assert sdn["company_name"] == "SDN"
    assert sdn["reason"] == "불성실공시지정"
    # 일시("YYYY-MM-DD HH:MM:SS")에서 날짜부만 정규화
    assert sdn["krx_designated_date"] == "2024-05-20"


def test_parse_handles_alternate_output_keys_and_bare_list():
    # KRX 는 output / OutBlock_1 / bare list 를 섞어 반환할 수 있다.
    alt = {"OutBlock_1": _ADMIN_PAYLOAD["output"]}
    assert len(kts.parse_admin_issue(alt)) == 2
    assert len(kts.parse_admin_issue(_ADMIN_PAYLOAD["output"])) == 2
    assert kts.parse_admin_issue({}) == []


def test_normalize_stock_code_from_isin_and_short():
    assert kts._normalize_stock_code("KR7005930003") == "005930"  # 12자 ISIN → 6자리
    assert kts._normalize_stock_code("005930") == "005930"        # 이미 단축코드
    assert kts._normalize_stock_code("  093230 ") == "093230"     # 공백 제거
    assert kts._normalize_stock_code("") == ""


def test_normalize_date_variants():
    assert kts._normalize_date("2023/03/24") == "2023-03-24"
    assert kts._normalize_date("20230324") == "2023-03-24"
    assert kts._normalize_date("2023-03-24") == "2023-03-24"
    assert kts._normalize_date("2024-05-20 09:00:00") == "2024-05-20"
    assert kts._normalize_date("") is None
    assert kts._normalize_date(None) is None


# --------------------------------------------------------------------------
# diff: 열린구간(직전 스냅샷) vs 현재 스냅샷 → 신규지정/해제/유지
# --------------------------------------------------------------------------
def test_diff_snapshot_classifies_new_released_continuing():
    open_codes = {"A", "B"}
    current = {"B", "C"}
    new, released, continuing = kts.diff_snapshot(open_codes, current)
    assert new == {"C"}          # 새로 나타남 = 지정 시작
    assert released == {"A"}     # 사라짐 = 해제
    assert continuing == {"B"}   # 계속 있음 = 구간 유지


# --------------------------------------------------------------------------
# 스키마: 한 종목이 지정→해제→재지정 반복 = 여러 구간 행 허용
# --------------------------------------------------------------------------
def test_schema_allows_multiple_intervals_per_ticker(tmp_path):
    db = str(tmp_path / "s.db")
    init_db(db)
    conn = _conn(db)
    conn.execute(
        "INSERT INTO kr_trading_status(stock_code, status_type, start_date, end_date) "
        "VALUES (?,?,?,?)", ("093230", "admin", "2026-01-01", "2026-02-01"))
    conn.execute(
        "INSERT INTO kr_trading_status(stock_code, status_type, start_date, end_date) "
        "VALUES (?,?,?,?)", ("093230", "admin", "2026-05-01", None))
    conn.commit()
    n = conn.execute(
        "SELECT COUNT(*) FROM kr_trading_status WHERE stock_code='093230'").fetchone()[0]
    assert n == 2
    conn.close()


# --------------------------------------------------------------------------
# ingest: 1차 실행 = 열린 구간 개시(start_date=오늘, end_date=NULL)
# --------------------------------------------------------------------------
def test_ingest_first_run_opens_intervals(tmp_path):
    db = str(tmp_path / "i.db")

    def admin():
        return kts.parse_admin_issue(_ADMIN_PAYLOAD)

    def halt():
        return kts.parse_trading_halt(_HALT_PAYLOAD)

    r = kts.ingest_trading_status(db_path=db, fetch_admin_fn=admin, fetch_halt_fn=halt,
                                  today="2026-07-19")
    assert r["as_of"] == "2026-07-19"
    assert r["admin"]["opened"] == 2
    assert r["admin"]["released"] == 0
    assert r["admin"]["open_total"] == 2
    assert r["halt"]["opened"] == 1

    conn = _conn(db)
    rows = conn.execute(
        "SELECT stock_code, start_date, end_date, reason FROM kr_trading_status "
        "WHERE status_type='admin' ORDER BY stock_code").fetchall()
    assert [row["start_date"] for row in rows] == ["2026-07-19", "2026-07-19"]
    assert all(row["end_date"] is None for row in rows)  # 전부 열린 구간
    assert {row["reason"] for row in rows} == {"감사의견 거절", "자본잠식"}
    conn.close()


# --------------------------------------------------------------------------
# 멱등성: 같은 날 두 번 실행해도 신규/해제 0, 중복 행 없음
# --------------------------------------------------------------------------
def test_ingest_idempotent_same_day(tmp_path):
    db = str(tmp_path / "idem.db")

    def admin():
        return kts.parse_admin_issue(_ADMIN_PAYLOAD)

    def halt():
        return []

    kts.ingest_trading_status(db_path=db, fetch_admin_fn=admin, fetch_halt_fn=halt, today="2026-07-19")
    r2 = kts.ingest_trading_status(db_path=db, fetch_admin_fn=admin, fetch_halt_fn=halt, today="2026-07-19")

    assert r2["admin"]["opened"] == 0
    assert r2["admin"]["released"] == 0
    conn = _conn(db)
    total = conn.execute("SELECT COUNT(*) FROM kr_trading_status WHERE status_type='admin'").fetchone()[0]
    assert total == 2  # 중복 없음
    conn.close()


# --------------------------------------------------------------------------
# 핵심: 다음 실행에서 어떤 종목이 사라지면 실제로 end_date 가 채워진다
# --------------------------------------------------------------------------
def test_ingest_release_fills_end_date_and_opens_new(tmp_path):
    db = str(tmp_path / "rel.db")

    # 1차(D1): 관리종목 = {093230, 900140}
    def week1():
        return kts.parse_admin_issue(_ADMIN_PAYLOAD)

    def none_halt():
        return []

    kts.ingest_trading_status(db_path=db, fetch_admin_fn=week1, fetch_halt_fn=none_halt, today="2026-07-19")

    # 2차(D2): 093230 사라짐(해제), 신규 111111 등장
    def week2():
        return [
            {"stock_code": "900140", "company_name": "엘브이엠씨", "market": "KOSPI",
             "reason": "자본잠식", "krx_designated_date": "2024-01-02"},
            {"stock_code": "111111", "company_name": "신규관리", "market": "KOSDAQ",
             "reason": "매출액 미달", "krx_designated_date": "2026-07-01"},
        ]

    r = kts.ingest_trading_status(db_path=db, fetch_admin_fn=week2, fetch_halt_fn=none_halt, today="2026-07-26")
    assert r["admin"]["released"] == 1
    assert r["admin"]["opened"] == 1

    conn = _conn(db)
    # 093230: 해제되어 end_date=D2 채워짐
    a = conn.execute(
        "SELECT start_date, end_date FROM kr_trading_status "
        "WHERE stock_code='093230' AND status_type='admin'").fetchone()
    assert a["start_date"] == "2026-07-19"
    assert a["end_date"] == "2026-07-26"
    # 900140: 계속 열린 구간
    b = conn.execute(
        "SELECT end_date FROM kr_trading_status WHERE stock_code='900140' AND status_type='admin'").fetchone()
    assert b["end_date"] is None
    # 111111: 새 열린 구간 start=D2
    c = conn.execute(
        "SELECT start_date, end_date FROM kr_trading_status WHERE stock_code='111111' AND status_type='admin'").fetchone()
    assert c["start_date"] == "2026-07-26"
    assert c["end_date"] is None
    conn.close()


def test_ingest_redesignation_creates_new_interval(tmp_path):
    db = str(tmp_path / "redes.db")

    def none_halt():
        return []

    only_a = [{"stock_code": "093230", "company_name": "이아이디", "market": "KOSDAQ",
               "reason": "감사의견 거절", "krx_designated_date": "2023-03-24"}]
    empty = []

    # D1: 지정
    kts.ingest_trading_status(db_path=db, fetch_admin_fn=lambda: only_a, fetch_halt_fn=none_halt, today="2026-07-01")
    # D2: 해제(사라짐)
    kts.ingest_trading_status(db_path=db, fetch_admin_fn=lambda: empty, fetch_halt_fn=none_halt, today="2026-07-08")
    # D3: 재지정(다시 등장)
    kts.ingest_trading_status(db_path=db, fetch_admin_fn=lambda: only_a, fetch_halt_fn=none_halt, today="2026-07-15")

    conn = _conn(db)
    rows = conn.execute(
        "SELECT start_date, end_date FROM kr_trading_status "
        "WHERE stock_code='093230' AND status_type='admin' ORDER BY start_date").fetchall()
    assert len(rows) == 2  # (D1~D2 닫힌 구간) + (D3~열린 구간)
    assert (rows[0]["start_date"], rows[0]["end_date"]) == ("2026-07-01", "2026-07-08")
    assert (rows[1]["start_date"], rows[1]["end_date"]) == ("2026-07-15", None)
    conn.close()


# --------------------------------------------------------------------------
# 라이브 조회 전용 함수 (backtest 미연결)
# --------------------------------------------------------------------------
def test_is_currently_administrative_or_halted(tmp_path):
    db = str(tmp_path / "live.db")

    def admin():
        return [{"stock_code": "093230", "company_name": "이아이디", "market": "KOSDAQ",
                 "reason": "감사의견 거절", "krx_designated_date": "2023-03-24"}]

    def halt():
        return [{"stock_code": "099220", "company_name": "SDN", "market": None,
                 "reason": "불성실공시지정", "krx_designated_date": "2024-05-20"}]

    kts.ingest_trading_status(db_path=db, fetch_admin_fn=admin, fetch_halt_fn=halt, today="2026-07-19")

    conn = _conn(db)
    assert kts.is_currently_administrative_or_halted(conn, "093230") is True   # 관리종목(열린 구간)
    assert kts.is_currently_administrative_or_halted(conn, "099220") is True   # 거래정지(열린 구간)
    assert kts.is_currently_administrative_or_halted(conn, "005930") is False  # 정상 종목(이력 없음)

    # 해제되면 False 로 바뀐다
    kts.ingest_trading_status(db_path=db, fetch_admin_fn=lambda: [], fetch_halt_fn=lambda: [], today="2026-07-26")
    assert kts.is_currently_administrative_or_halted(conn, "093230") is False
    assert kts.is_currently_administrative_or_halted(conn, "099220") is False
    conn.close()


def test_ingest_uses_di_and_never_calls_live_krx(tmp_path):
    # fetch_*_fn 주입 시 실제 KRX 로그인/HTTP 를 절대 호출하지 않는다(네트워크 없이 동작).
    db = str(tmp_path / "di.db")
    called = {"admin": 0, "halt": 0}

    def admin():
        called["admin"] += 1
        return kts.parse_admin_issue(_ADMIN_PAYLOAD)

    def halt():
        called["halt"] += 1
        return kts.parse_trading_halt(_HALT_PAYLOAD)

    kts.ingest_trading_status(db_path=db, fetch_admin_fn=admin, fetch_halt_fn=halt, today="2026-07-19")
    assert called == {"admin": 1, "halt": 1}
