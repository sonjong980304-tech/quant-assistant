"""DART 공시목록 기반 관리종목/매매거래정지 '진짜 과거 이력' 수집·분류 단위 테스트 (TDD).

kr_trading_status(KRX '오늘 스냅샷'만 → 미래 전용)와 달리, DART OpenDART list.json 은 회사별
과거 공시 이력을 실제로 소급 조회할 수 있다(bgn_de=20150101 정상). 이 이력에서 관리종목 지정/
해제·매매거래정지 시작/해제 구간을 복원한다.

핵심 설계(실측 8+4개 종목 조사 기반):
- report_nm 은 '관리종목 지정/해제'를 문자 그대로 담지 않는 경우가 많다. 그래서 순수 분류 함수
  classify_disclosure 가 report_nm 만으로 판별 가능한 것만 이벤트로 확정하고, 방향이 모호한 것
  (KOSPI '매매거래정지및정지해제' 결합형, '주권매매거래정지기간변경')은 REVIEW 로 보류한다.
- build_status_intervals 가 이벤트를 시간순으로 짝지어 (지정~해제)/(정지~해제) 구간을 만든다.
- 네트워크(DART) 는 DI(fetch_fn 주입)로 분리해 단위 테스트한다(us_delisting/kr_stock_changes 관례).
"""
from __future__ import annotations

import sqlite3

from src.ingest import kr_admin_status_history as kash


def _conn(db: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


# --------------------------------------------------------------------------
# 순수 분류: classify_disclosure(report_nm) → (event, status_type, reason)
# --------------------------------------------------------------------------
def test_classify_halt_start_from_pure_halt_title():
    # KOSDAQ '주권매매거래정지(사유)' 는 순수 정지 개시 (해제/기간변경 아님)
    c = kash.classify_disclosure("주권매매거래정지 (투자자 보호)")
    assert c.event == kash.EVENT_HALT_START
    assert c.status_type == kash.STATUS_HALT


def test_classify_halt_end_from_release_title():
    c = kash.classify_disclosure("주권매매거래정지해제 (감자 주권 변경상장)")
    assert c.event == kash.EVENT_HALT_END
    assert c.status_type == kash.STATUS_HALT


def test_classify_period_change_is_review():
    # '기간변경' 은 정지가 '진행 중'이란 뜻 — 시작/해제 경계가 아니므로 사람 검토 보류
    c = kash.classify_disclosure("주권매매거래정지기간변경 (상장폐지 사유 발생)")
    assert c.event == kash.EVENT_REVIEW
    assert c.reason == "halt_period_change"


def test_classify_kospi_combined_halt_is_review():
    # KOSPI '매매거래정지및정지해제' 는 한 서식이 정지/해제 양쪽을 덮어 방향이 모호 → 보류
    c = kash.classify_disclosure("매매거래정지및정지해제(중요내용공시)")
    assert c.event == kash.EVENT_REVIEW
    assert c.reason == "halt_combined"


def test_classify_admin_designate_literal():
    # 실측: 내부결산 시점 회사 자율공시에 '관리종목지정' 이 문자 그대로 들어온다
    c = kash.classify_disclosure("내부결산시점관리종목지정ㆍ형식적상장폐지ㆍ상장적격성실질심사사유발생")
    assert c.event == kash.EVENT_ADMIN_DESIGNATE
    assert c.status_type == kash.STATUS_ADMIN


def test_classify_admin_release_from_cause_resolution():
    c = kash.classify_disclosure("기타시장안내 (관리종목 지정 사유 해소)")
    assert c.event == kash.EVENT_ADMIN_RELEASE
    assert c.status_type == kash.STATUS_ADMIN


def test_classify_admin_concern_is_ignored():
    # '관리종목 지정 우려' 는 경고일 뿐 실제 지정이 아니다 → 이벤트 아님(IGNORE)
    c = kash.classify_disclosure("투자유의안내(관리종목 지정 우려)")
    assert c.event == kash.EVENT_IGNORE


def test_classify_does_not_confuse_unfaithful_disclosure_designation():
    # '불성실공시법인지정' 은 '지정'을 담지만 관리종목이 아니다 → 절대 admin 으로 오분류 금지
    c = kash.classify_disclosure("불성실공시법인지정")
    assert c.event == kash.EVENT_IGNORE
    assert c.status_type != kash.STATUS_ADMIN


def test_classify_unrelated_disclosures_ignored():
    for nm in ("조회공시요구(현저한시황변동)", "소속부변경", "권리락", "무상증자결정"):
        assert kash.classify_disclosure(nm).event == kash.EVENT_IGNORE


def test_classify_strips_correction_prefix():
    # '[기재정정]' 접두어(정정 재공시)는 벗겨내고 본 제목으로 분류한다
    c = kash.classify_disclosure("[기재정정]주권매매거래정지기간변경 (상장적격성 실질심사 대상 결정)")
    assert c.event == kash.EVENT_REVIEW
    assert c.reason == "halt_period_change"


def test_classify_halt_mention_in_other_disclosure_is_review_not_start():
    # '기타시장안내(...매매거래정지 지속...)' 은 제목이 정지서식이 아니다 → HALT_START 로 오분류 금지
    c = kash.classify_disclosure("기타시장안내 (감사의견 관련 형식적 상장폐지 사유해소 및 매매거래정지 지속안내)")
    assert c.event == kash.EVENT_REVIEW
    assert c.reason == "halt_mention"


# --------------------------------------------------------------------------
# 순수 구간화: build_status_intervals(disclosures) → {admin, halt, review}
# --------------------------------------------------------------------------
def _disc(rcept_dt: str, report_nm: str) -> dict:
    return {"rcept_dt": rcept_dt, "report_nm": report_nm}


def test_build_pairs_halt_start_and_end():
    rows = [
        _disc("20160318", "주권매매거래정지(투자자 보호)"),
        _disc("20161208", "주권매매거래정지해제(기업심사위원회 심의결과 상장폐지 기준 미해당)"),
    ]
    out = kash.build_status_intervals(rows)
    assert len(out["halt"]) == 1
    iv = out["halt"][0]
    assert iv["start_date"] == "2016-03-18"
    assert iv["end_date"] == "2016-12-08"
    assert iv["status_type"] == kash.STATUS_HALT


def test_build_open_interval_when_no_release():
    rows = [_disc("20250401", "주권매매거래정지 (회생절차 개시신청)")]
    out = kash.build_status_intervals(rows)
    assert len(out["halt"]) == 1
    assert out["halt"][0]["start_date"] == "2025-04-01"
    assert out["halt"][0]["end_date"] is None  # 미해제 = 진행 중


def test_build_multiple_halt_episodes():
    rows = [
        _disc("20160318", "주권매매거래정지(투자자 보호)"),
        _disc("20161208", "주권매매거래정지해제(상장폐지 기준 미해당)"),
        _disc("20180911", "주권매매거래정지(불성실공시법인 지정)"),
        _disc("20181010", "주권매매거래정지해제(자본감소 변경상장)"),
    ]
    out = kash.build_status_intervals(rows)
    assert len(out["halt"]) == 2
    assert out["halt"][0]["start_date"] == "2016-03-18"
    assert out["halt"][0]["end_date"] == "2016-12-08"
    assert out["halt"][1]["start_date"] == "2018-09-11"
    assert out["halt"][1]["end_date"] == "2018-10-10"


def test_build_admin_designate_interval():
    rows = [_disc("20240229", "내부결산시점관리종목지정ㆍ형식적상장폐지ㆍ상장적격성실질심사사유발생")]
    out = kash.build_status_intervals(rows)
    assert len(out["admin"]) == 1
    assert out["admin"][0]["start_date"] == "2024-02-29"
    assert out["admin"][0]["end_date"] is None


def test_build_orphan_release_goes_to_review():
    # 지정을 못 본 채 해제만 있으면(조회창 이전에 지정된 것) 구간을 만들지 말고 보류로 보고
    rows = [_disc("20200313", "기타시장안내(관리종목 지정 사유 해소)")]
    out = kash.build_status_intervals(rows)
    assert out["admin"] == []
    reasons = [r["reason"] for r in out["review"]]
    assert "orphan_release" in reasons


def test_build_collects_ambiguous_as_review():
    rows = [
        _disc("20150817", "매매거래정지및정지해제(중요내용공시)"),
        _disc("20160408", "주권매매거래정지기간변경(상장적격성 실질심사 대상 결정)"),
    ]
    out = kash.build_status_intervals(rows)
    assert out["halt"] == []
    assert len(out["review"]) == 2
    reasons = {r["reason"] for r in out["review"]}
    assert reasons == {"halt_combined", "halt_period_change"}


def test_build_separates_admin_and_halt_and_sorts():
    # 입력 순서가 뒤섞여도 rcept_dt 로 정렬해 올바르게 짝짓는다
    rows = [
        _disc("20161208", "주권매매거래정지해제(상장폐지 기준 미해당)"),
        _disc("20240229", "내부결산시점관리종목지정ㆍ형식적상장폐지ㆍ상장적격성실질심사사유발생"),
        _disc("20160318", "주권매매거래정지(투자자 보호)"),
    ]
    out = kash.build_status_intervals(rows)
    assert len(out["halt"]) == 1
    assert out["halt"][0]["start_date"] == "2016-03-18"
    assert out["halt"][0]["end_date"] == "2016-12-08"
    assert len(out["admin"]) == 1
    assert out["admin"][0]["start_date"] == "2024-02-29"


# --------------------------------------------------------------------------
# 페이지네이션: fetch_disclosures(DI fetch_fn 주입, 네트워크 없이)
# --------------------------------------------------------------------------
def test_fetch_disclosures_paginates_until_last_page():
    pages = {
        1: {"status": "000", "total_page": 2, "list": [{"report_nm": "A", "rcept_dt": "20240101"}]},
        2: {"status": "000", "total_page": 2, "list": [{"report_nm": "B", "rcept_dt": "20240202"}]},
    }
    calls = []

    def fake_fetch(params):
        calls.append(params["page_no"])
        return pages[params["page_no"]]

    rows = kash.fetch_disclosures("KEY", "00125974", bgn_de="20150101", end_de="20241231", fetch_fn=fake_fetch)
    assert [r["report_nm"] for r in rows] == ["A", "B"]
    assert calls == [1, 2]  # total_page=2 까지만 순회


def test_fetch_disclosures_stops_on_nonzero_status():
    def fake_fetch(params):
        return {"status": "013", "message": "데이터 없음"}  # 조회 결과 없음

    rows = kash.fetch_disclosures("KEY", "99999999", fetch_fn=fake_fetch)
    assert rows == []


# --------------------------------------------------------------------------
# 적재: ingest_admin_status_history (DI fetch_fn(code), 멱등, 구간 upsert)
# --------------------------------------------------------------------------
def _fake_disclosures(code: str) -> list[dict]:
    return [
        _disc("20160318", "주권매매거래정지(투자자 보호)"),
        _disc("20161208", "주권매매거래정지해제(상장폐지 기준 미해당)"),
        _disc("20240229", "내부결산시점관리종목지정ㆍ형식적상장폐지ㆍ상장적격성실질심사사유발생"),
        _disc("20150817", "매매거래정지및정지해제(중요내용공시)"),  # 보류(review) 대상
    ]


def test_ingest_stores_intervals_from_fetch_fn(tmp_path):
    db = str(tmp_path / "a.db")
    r = kash.ingest_admin_status_history(db_path=db, codes=["023440"], fetch_fn=_fake_disclosures)
    assert r["intervals_stored"] == 2  # halt 1구간 + admin 1구간
    assert r["review_count"] == 1      # 결합형 1건

    conn = _conn(db)
    rows = conn.execute(
        "SELECT status_type, start_date, end_date FROM kr_admin_status_history "
        "WHERE stock_code='023440' ORDER BY status_type, start_date"
    ).fetchall()
    got = {(x["status_type"], x["start_date"], x["end_date"]) for x in rows}
    assert ("halt", "2016-03-18", "2016-12-08") in got
    assert ("admin", "2024-02-29", None) in got
    conn.close()


def test_ingest_is_idempotent_same_run(tmp_path):
    db = str(tmp_path / "idem.db")
    kash.ingest_admin_status_history(db_path=db, codes=["023440"], fetch_fn=_fake_disclosures)
    kash.ingest_admin_status_history(db_path=db, codes=["023440"], fetch_fn=_fake_disclosures)
    conn = _conn(db)
    total = conn.execute(
        "SELECT COUNT(*) FROM kr_admin_status_history WHERE stock_code='023440'"
    ).fetchone()[0]
    assert total == 2  # UNIQUE(stock_code,status_type,start_date) 로 재실행해도 중복 없음
    conn.close()


def test_ingest_fills_end_date_on_later_release(tmp_path):
    db = str(tmp_path / "reopen.db")

    # 1차: 정지 개시만 관측(열린 구간)
    def only_start(code):
        return [_disc("20160318", "주권매매거래정지(투자자 보호)")]

    kash.ingest_admin_status_history(db_path=db, codes=["023440"], fetch_fn=only_start)
    conn = _conn(db)
    assert conn.execute(
        "SELECT end_date FROM kr_admin_status_history WHERE stock_code='023440' AND status_type='halt'"
    ).fetchone()["end_date"] is None
    conn.close()

    # 2차: 나중에 해제가 관측되면 같은 구간의 end_date 를 채운다(멱등 갱신)
    def start_and_end(code):
        return [
            _disc("20160318", "주권매매거래정지(투자자 보호)"),
            _disc("20161208", "주권매매거래정지해제(상장폐지 기준 미해당)"),
        ]

    kash.ingest_admin_status_history(db_path=db, codes=["023440"], fetch_fn=start_and_end)
    conn = _conn(db)
    rows = conn.execute(
        "SELECT end_date FROM kr_admin_status_history WHERE stock_code='023440' AND status_type='halt'"
    ).fetchall()
    assert len(rows) == 1  # 새 행이 아니라 기존 구간 갱신
    assert rows[0]["end_date"] == "2016-12-08"
    conn.close()


def test_ingest_isolates_failed_ticker_and_continues(tmp_path, monkeypatch):
    db = str(tmp_path / "fail.db")
    alerts: list[str] = []
    monkeypatch.setattr(kash, "send_slack_alert", lambda msg: alerts.append(msg) or True)

    def fetch(code: str):
        if code == "BAD":
            raise RuntimeError("DART 조회 실패(mock)")
        return _fake_disclosures(code)

    r = kash.ingest_admin_status_history(db_path=db, codes=["BAD", "023440"], fetch_fn=fetch)
    assert r["failed"] == ["BAD"]
    assert r["tickers"] == 1          # 성공 종목만 카운트
    assert len(alerts) == 1
    conn = _conn(db)
    ok = conn.execute(
        "SELECT COUNT(*) FROM kr_admin_status_history WHERE stock_code='023440'"
    ).fetchone()[0]
    assert ok == 2
    conn.close()


def test_ingest_uses_di_and_never_calls_dart(tmp_path):
    db = str(tmp_path / "di.db")
    called = {"n": 0}

    def fetch(code: str):
        called["n"] += 1
        return []

    kash.ingest_admin_status_history(db_path=db, codes=["023440", "016790"], fetch_fn=fetch)
    assert called["n"] == 2


def test_get_admin_status_history_reads_back_intervals(tmp_path):
    db = str(tmp_path / "read.db")
    kash.ingest_admin_status_history(db_path=db, codes=["023440"], fetch_fn=_fake_disclosures)
    conn = _conn(db)
    hist = kash.get_admin_status_history(conn, "023440")
    conn.close()
    assert any(iv["start_date"] == "2016-03-18" and iv["end_date"] == "2016-12-08" for iv in hist["halt"])
    assert any(iv["start_date"] == "2024-02-29" for iv in hist["admin"])
