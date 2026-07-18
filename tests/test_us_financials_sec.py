"""SEC companyfacts XBRL 수집·정규화 단위 테스트 (TDD, AC4/AC5/AC7/AC13).

.omc/specs/brainstorming-sec-edgar-us-financials-backfill.md:
- AC4: 초기 백필은 companyfacts.zip 을 받아 추적 대상 종목만 걸러 새 테이블에 적재.
- AC5: 주간 갱신은 종목별 companyfacts API 로 최신 분기만 upsert(초당 10건 이하, User-Agent 필수).
- AC7: SEC 실제 filed(제출일)를 그대로 저장(look-ahead 방지의 기준 데이터).
- AC13: XBRL 파싱/정규화 로직을 TDD 로 구현.

네트워크/파일은 DI(fetch_facts_fn 주입, 로컬 zip 파일)로 분리한다.
"""
from __future__ import annotations

import io
import json
import sqlite3
import zipfile

from src.db import connect, init_db
from src.ingest import us_financials_sec as sec


# 샘플 companyfacts JSON — AAPL 축약본(us-gaap Revenues/NetIncomeLoss + dei 1개).
def _sample_facts(cik=320193):
    return {
        "cik": cik,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "label": "Revenues",
                    "units": {
                        "USD": [
                            # 연간(10-K, duration ~1년)
                            {"start": "2008-09-28", "end": "2009-09-26", "val": 42905000000,
                             "accn": "0001193125-09-214859", "fy": 2009, "fp": "FY",
                             "form": "10-K", "filed": "2009-10-27", "frame": "CY2009"},
                            # 분기(10-Q, duration ~3개월) — 더 최근 제출
                            {"start": "2010-06-27", "end": "2010-09-25", "val": 20343000000,
                             "accn": "0001193125-10-238044", "fy": 2010, "fp": "Q4",
                             "form": "10-Q", "filed": "2010-10-27"},
                        ]
                    },
                },
                "NetIncomeLoss": {
                    "label": "Net Income (Loss)",
                    "units": {
                        "USD": [
                            {"start": "2010-06-27", "end": "2010-09-25", "val": 4308000000,
                             "accn": "0001193125-10-238044", "fy": 2010, "fp": "Q4",
                             "form": "10-Q", "filed": "2010-10-27"},
                        ]
                    },
                },
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "label": "Shares Outstanding",
                    "units": {
                        # instant 팩트(start 없음) → period_start 는 None 이어야 함
                        "shares": [
                            {"end": "2010-10-15", "val": 909938383, "accn": "0001193125-10-238044",
                             "fy": 2010, "fp": "Q4", "form": "10-Q", "filed": "2010-10-27"},
                        ]
                    },
                },
            },
        },
    }


# --------------------------------------------------------------------------
# normalize_companyfacts — 원시 JSON → EAV 행
# --------------------------------------------------------------------------
def test_normalize_companyfacts_extracts_all_facts():
    rows = sec.normalize_companyfacts("AAPL", "0000320193", _sample_facts())
    # us-gaap Revenues(2) + NetIncomeLoss(1) + dei shares(1) = 4행
    assert len(rows) == 4


def test_normalize_companyfacts_maps_fields_correctly():
    rows = sec.normalize_companyfacts("AAPL", "0000320193", _sample_facts())
    rev_annual = next(r for r in rows if r["tag"] == "Revenues" and r["form"] == "10-K")
    assert rev_annual["stock_code"] == "AAPL"
    assert rev_annual["cik"] == "0000320193"
    assert rev_annual["taxonomy"] == "us-gaap"
    assert rev_annual["unit"] == "USD"
    assert rev_annual["value"] == 42905000000.0
    assert rev_annual["period_start"] == "2008-09-28"
    assert rev_annual["period_end"] == "2009-09-26"
    assert rev_annual["fy"] == 2009
    assert rev_annual["fp"] == "FY"
    assert rev_annual["filed"] == "2009-10-27"      # AC7: 실제 제출일 그대로
    assert rev_annual["frame"] == "CY2009"
    assert rev_annual["accn"] == "0001193125-09-214859"


def test_normalize_companyfacts_instant_fact_has_null_period_start():
    rows = sec.normalize_companyfacts("AAPL", "0000320193", _sample_facts())
    shares = next(r for r in rows if r["tag"] == "EntityCommonStockSharesOutstanding")
    assert shares["period_start"] is None
    assert shares["period_end"] == "2010-10-15"
    assert shares["taxonomy"] == "dei"


def test_normalize_companyfacts_empty_facts_returns_empty():
    assert sec.normalize_companyfacts("AAPL", "0000320193", {"facts": {}}) == []


# --------------------------------------------------------------------------
# latest filing filter — 주간 갱신용 "최신 분기만" (AC5)
# --------------------------------------------------------------------------
def test_latest_filing_rows_keeps_only_max_filed():
    rows = sec.normalize_companyfacts("AAPL", "0000320193", _sample_facts())
    latest = sec._latest_filing_rows(rows)
    # 최신 filed=2010-10-27 인 팩트만(2009-10-27 연간 매출은 제외)
    assert all(r["filed"] == "2010-10-27" for r in latest)
    assert len(latest) == 3  # Revenues(10-Q) + NetIncomeLoss + shares


# --------------------------------------------------------------------------
# ingest_companyfacts_api — 종목별 주간 갱신 (AC5)
# --------------------------------------------------------------------------
def _seed_company(db_path, code="AAPL", cik="0000320193"):
    init_db(db_path)
    conn = connect(db_path)
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, cik, updated_at) VALUES (?,?,?,?,?)",
        (code, f"{code} Inc", "NASDAQ", cik, "2026-07-19"),
    )
    conn.commit()
    conn.close()


def test_ingest_companyfacts_api_upserts_latest_quarter_only(tmp_path):
    db = str(tmp_path / "sec_api.db")
    _seed_company(db)
    fetch = lambda cik: _sample_facts()  # noqa: E731
    report = sec.ingest_companyfacts_api(
        db_path=db, fetch_facts_fn=fetch, sleep_fn=lambda _s: None
    )
    conn = sqlite3.connect(db)
    filed_dates = {r[0] for r in conn.execute("SELECT DISTINCT filed FROM us_financials_sec")}
    conn.close()
    assert filed_dates == {"2010-10-27"}  # 최신 분기(제출)만 적재
    assert report["succeeded"] == 1


def test_ingest_companyfacts_api_full_history_when_latest_only_false(tmp_path):
    db = str(tmp_path / "sec_api2.db")
    _seed_company(db)
    fetch = lambda cik: _sample_facts()  # noqa: E731
    sec.ingest_companyfacts_api(
        db_path=db, fetch_facts_fn=fetch, sleep_fn=lambda _s: None, latest_only=False
    )
    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM us_financials_sec").fetchone()[0]
    conn.close()
    assert n == 4  # 전체 히스토리(연간+분기 전부)


def test_ingest_companyfacts_api_skips_company_without_cik(tmp_path):
    db = str(tmp_path / "sec_api3.db")
    init_db(db)
    conn = connect(db)
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, cik, updated_at) VALUES (?,?,?,?,?)",
        ("NOCIK", "No Cik", "NYSE", None, "2026-07-19"),
    )
    conn.commit()
    conn.close()
    report = sec.ingest_companyfacts_api(
        db_path=db, fetch_facts_fn=lambda c: _sample_facts(), sleep_fn=lambda _s: None
    )
    assert report["succeeded"] == 0
    assert report["skipped_no_cik"] == 1


def test_ingest_companyfacts_api_isolates_failing_ticker(tmp_path):
    db = str(tmp_path / "sec_api4.db")
    _seed_company(db, code="AAPL", cik="0000320193")
    _seed_company_extra = connect(db)
    _seed_company_extra.execute(
        "INSERT INTO us_company(stock_code, name, exchange, cik, updated_at) VALUES (?,?,?,?,?)",
        ("BOOM", "Boom Inc", "NYSE", "0000000002", "2026-07-19"),
    )
    _seed_company_extra.commit()
    _seed_company_extra.close()

    def fetch(cik):
        if cik == "0000000002":
            raise RuntimeError("네트워크 오류")
        return _sample_facts()

    report = sec.ingest_companyfacts_api(db_path=db, fetch_facts_fn=fetch, sleep_fn=lambda _s: None)
    assert report["succeeded"] == 1
    assert "BOOM" in report["failed"]


def test_ingest_companyfacts_api_respects_rate_limit_constant():
    # SEC 초당 10건 제한(AC5): 최소 호출 간격이 0.1초 이상이어야 함(≤10/sec).
    assert sec.SEC_MIN_INTERVAL_SEC >= 0.1
    # User-Agent 헤더 상수 존재(AC5 필수)
    assert "@" in sec.SEC_USER_AGENT  # 연락처 이메일 포함(SEC 권고)


# --------------------------------------------------------------------------
# backfill_from_zip — 초기 대량 백필 (AC4)
# --------------------------------------------------------------------------
def _make_companyfacts_zip(tmp_path, entries: dict[str, dict]) -> str:
    """{'CIK0000320193.json': facts_dict, ...} 를 담은 zip 파일을 만들어 경로 반환."""
    zpath = tmp_path / "companyfacts.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, facts in entries.items():
            zf.writestr(name, json.dumps(facts))
    zpath.write_bytes(buf.getvalue())
    return str(zpath)


def test_backfill_from_zip_loads_only_tracked_tickers(tmp_path):
    db = str(tmp_path / "sec_zip.db")
    _seed_company(db, code="AAPL", cik="0000320193")  # 추적 대상 1종목
    # zip 에는 추적대상(AAPL)과 비추적(9999999) 두 회사가 들어있다
    zpath = _make_companyfacts_zip(tmp_path, {
        "CIK0000320193.json": _sample_facts(320193),
        "CIK0009999999.json": _sample_facts(9999999),
    })
    report = sec.backfill_from_zip(db_path=db, zip_path=zpath)
    conn = sqlite3.connect(db)
    ciks = {r[0] for r in conn.execute("SELECT DISTINCT cik FROM us_financials_sec")}
    conn.close()
    assert ciks == {"0000320193"}  # 추적 대상만 적재(비추적 회사는 무시)
    assert report["tickers_loaded"] == 1
    assert report["facts_loaded"] == 4  # AAPL 전체 히스토리


def test_backfill_from_zip_reports_missing_in_zip(tmp_path):
    db = str(tmp_path / "sec_zip2.db")
    _seed_company(db, code="AAPL", cik="0000320193")
    _seed_company(db, code="MSFT", cik="0000789019")  # zip 에 없는 종목
    zpath = _make_companyfacts_zip(tmp_path, {"CIK0000320193.json": _sample_facts(320193)})
    report = sec.backfill_from_zip(db_path=db, zip_path=zpath)
    assert "MSFT" in report["missing_in_zip"]
    assert report["tickers_loaded"] == 1


def test_backfill_from_zip_idempotent(tmp_path):
    db = str(tmp_path / "sec_zip3.db")
    _seed_company(db, code="AAPL", cik="0000320193")
    zpath = _make_companyfacts_zip(tmp_path, {"CIK0000320193.json": _sample_facts(320193)})
    sec.backfill_from_zip(db_path=db, zip_path=zpath)
    sec.backfill_from_zip(db_path=db, zip_path=zpath)  # 두 번째 실행
    conn = sqlite3.connect(db)
    n = conn.execute("SELECT COUNT(*) FROM us_financials_sec").fetchone()[0]
    conn.close()
    assert n == 4  # UNIQUE 제약으로 중복 적재되지 않음


# --------------------------------------------------------------------------
# 검증 리포트 — quarterly 커버리지(AC10) · 디스크 용량(AC11)
# --------------------------------------------------------------------------
def test_quarterly_coverage_reports_ratio(tmp_path):
    """추적 종목 중 quarterly(단일분기 duration) 팩트가 1개 이상인 종목 비율(AC10)."""
    db = str(tmp_path / "cov.db")
    _seed_company(db, code="AAPL", cik="0000320193")
    _seed_company(db, code="MSFT", cik="0000789019")  # 데이터 없는 종목
    conn = connect(db)
    # AAPL 에만 단일분기(duration ~91일) 팩트 1개 적재
    conn.execute(
        "INSERT INTO us_financials_sec(stock_code, cik, tag, taxonomy, unit, value, "
        "period_start, period_end, form, filed, source) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("AAPL", "0000320193", "NetIncomeLoss", "us-gaap", "USD", 10.0,
         "2024-10-01", "2024-12-31", "10-Q", "2025-02-14", "sec_companyfacts_zip"),
    )
    conn.commit()
    conn.close()
    conn2 = sqlite3.connect(db)
    report = sec.quarterly_coverage(conn2)
    conn2.close()
    assert report["total_tracked"] == 2
    assert report["with_quarterly"] == 1
    assert report["coverage_rate"] == 0.5


def test_table_disk_bytes_returns_positive(tmp_path):
    """새 테이블 실제 디스크 사용량(바이트)을 측정해 보고한다(AC11)."""
    db = str(tmp_path / "disk.db")
    _seed_company(db, code="AAPL", cik="0000320193")
    conn = connect(db)
    for i in range(50):
        conn.execute(
            "INSERT INTO us_financials_sec(stock_code, cik, tag, taxonomy, unit, value, "
            "period_start, period_end, form, filed, source) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("AAPL", "0000320193", f"Tag{i}", "us-gaap", "USD", float(i),
             "2024-10-01", "2024-12-31", "10-Q", "2025-02-14", "sec_companyfacts_zip"),
        )
    conn.commit()
    conn.close()
    conn2 = sqlite3.connect(db)
    size = sec.table_disk_bytes(conn2, "us_financials_sec")
    conn2.close()
    assert isinstance(size, int)
    assert size > 0
