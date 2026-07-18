"""SEC EDGAR companyfacts XBRL 수집·정규화 — 벌크 zip 백필 + 종목별 API 주간갱신.

.omc/specs/brainstorming-sec-edgar-us-financials-backfill.md 하이브리드 접근:
- 초기 백필: companyfacts.zip(전체회사 스냅샷, ~1.39GB)을 받아 추적 대상 종목만
  걸러 새 테이블(us_financials_sec)에 전체 히스토리 적재(backfill_from_zip, AC4).
- 주간 갱신: 종목별 `data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` API 로 최신
  분기만 upsert(ingest_companyfacts_api, AC5). 초당 10건 이하 + User-Agent 헤더 필수.

두 경로 모두 SEC 의 동일한 companyfacts 데이터 모델(팩트 형식이 같고 소스만 zip↔API)을
공유하므로, JSON→EAV 변환은 normalize_companyfacts 한 곳으로 통일한다.

원시 XBRL 팩트를 축소 없이 저장한다(스펙 Round 7): 태그·값·단위·기간(start/end)·회계연도(fy)·
회계분기(fp)·양식(form)·제출일(filed)·프레임·접수번호. filed 는 SEC 실제 제출일이라
us_financials 의 기말+45/90일 근사보다 정확한 look-ahead 방지 기준이 된다(AC7).

네트워크/파일 접근은 DI(fetch_facts_fn 주입, 로컬 zip 경로)로 분리해 단위 테스트한다
(us_financials.py 의 fetch_statements, us_universe.py 의 fetch_exchange 주입 관례와 동일).
"""
from __future__ import annotations

import json
import time
import zipfile
from datetime import datetime, timezone
from typing import Callable, Optional

from ..db import connect, init_db
from .notify import send_slack_alert
from .robust import log_ingest

# SEC 레이트리밋: 초당 10건. 안전하게 호출 간 최소 0.11초(≈9/sec)를 강제한다(AC5).
SEC_MIN_INTERVAL_SEC = 0.11
# SEC 권고: User-Agent 에 연락처(이메일) 포함(cik_mapping.py 와 동일 관례). 헤더 필수(AC5).
SEC_USER_AGENT = "dart-text2sql-wiki (sonjong980304@gmail.com)"
_COMPANYFACTS_API_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def normalize_companyfacts(stock_code: str, cik: str, facts_json: dict) -> list[dict]:
    """SEC companyfacts JSON 을 us_financials_sec EAV 행 리스트로 변환한다.

    구조: facts_json["facts"][taxonomy][tag]["units"][unit] = [{start?,end,val,accn,fy,fp,
    form,filed,frame?}, ...]. duration 팩트는 start 가 있고(손익/현금흐름), instant 팩트는
    start 가 없다(재무상태 스냅샷) → period_start=None. val/end 가 없는 팩트는 건너뛴다.
    """
    rows: list[dict] = []
    facts = (facts_json or {}).get("facts", {})
    for taxonomy, tags in facts.items():
        for tag, tag_body in tags.items():
            units = (tag_body or {}).get("units", {})
            for unit, entries in units.items():
                for e in entries:
                    end = e.get("end")
                    val = e.get("val")
                    if end is None or val is None:
                        continue
                    rows.append({
                        "stock_code": stock_code,
                        "cik": cik,
                        "tag": tag,
                        "taxonomy": taxonomy,
                        "unit": unit,
                        "value": float(val),
                        "period_start": e.get("start"),   # instant 팩트면 None
                        "period_end": end,
                        "fy": e.get("fy"),
                        "fp": e.get("fp"),
                        "form": e.get("form"),
                        "filed": e.get("filed"),
                        "frame": e.get("frame"),
                        "accn": e.get("accn"),
                    })
    return rows


def _latest_filing_rows(rows: list[dict]) -> list[dict]:
    """가장 최근 제출(filed 최대)에 속한 팩트만 남긴다(주간 갱신용 '최신 분기만', AC5).

    한 번의 10-Q/10-K 제출은 동일 filed 를 공유한다 — filed 최대값 팩트만 upsert 하면
    15년치 전체를 매주 재적재하지 않고 최신 제출분만 갱신한다. (비교표시용 전년동기
    팩트도 같은 filed 라 함께 들어오지만 UNIQUE 제약으로 안전하게 흡수된다.)
    """
    filed_values = [r["filed"] for r in rows if r.get("filed")]
    if not filed_values:
        return []
    latest = max(filed_values)
    return [r for r in rows if r.get("filed") == latest]


def _upsert_rows(conn, rows: list[dict], source: str, collected_at: str) -> int:
    for r in rows:
        # instant 팩트(period_start=None)를 빈 문자열로 저장한다 — SQLite 는 UNIQUE 비교에서
        # NULL 을 매번 서로 다른 값으로 취급해(NULL != NULL) instant 팩트가 재적재마다 중복
        # 적재된다(백필 멱등성 깨짐). 빈 문자열 센티널을 쓰면 UNIQUE 가 정상 작동하고,
        # data_access_us_sec 의 duration 필터(period_start 로 julianday 계산)는 ''를 NULL 로
        # 다뤄 instant 를 자연히 제외한다(재무상태 스냅샷은 별도 경로로 조회).
        conn.execute(
            "INSERT OR REPLACE INTO us_financials_sec"
            "(stock_code, cik, tag, taxonomy, unit, value, period_start, period_end, "
            "fy, fp, form, filed, frame, accn, source, collected_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                r["stock_code"], r["cik"], r["tag"], r["taxonomy"], r["unit"], r["value"],
                r["period_start"] or "", r["period_end"], r["fy"], r["fp"], r["form"],
                r["filed"], r["frame"], r["accn"], source, collected_at,
            ),
        )
    return len(rows)


def _fetch_companyfacts_api(cik: str) -> dict:
    """종목별 companyfacts API 를 실제 호출(지연 import). User-Agent 헤더 필수(AC5)."""
    import requests

    url = _COMPANYFACTS_API_URL.format(cik=cik)
    resp = requests.get(
        url, headers={"User-Agent": SEC_USER_AGENT, "Accept": "application/json"}, timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def ingest_companyfacts_api(
    db_path: str | None = None,
    codes: Optional[list[str]] = None,
    fetch_facts_fn: Optional[Callable[[str], dict]] = None,
    latest_only: bool = True,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    """cik 가 매핑된 us_company 종목의 companyfacts 를 API 로 받아 us_financials_sec 에 upsert.

    latest_only=True(기본, 주간 갱신)면 최신 제출분(최신 분기)만 적재하고, False 면 전체
    히스토리를 적재한다. codes 미지정 시 cik 가 있는 전종목 대상. cik 가 없는 종목은
    skipped_no_cik 로 집계(조용히 넘기지 않고 보고). 종목별 try/except 로 실패를 격리해
    한 건 실패가 전체를 막지 않게 한다(us_financials.py 관례). SEC 레이트리밋 준수를 위해
    호출 사이 SEC_MIN_INTERVAL_SEC 초를 sleep 한다(sleep_fn 주입으로 테스트는 무지연).

    반환: {"targets","succeeded","failed"(리스트),"skipped_no_cik","facts_loaded"}.
    """
    fetch_facts_fn = fetch_facts_fn or _fetch_companyfacts_api
    init_db(db_path)
    conn = connect(db_path)
    collected_at = datetime.now(timezone.utc).isoformat()
    try:
        if codes is None:
            rows = conn.execute(
                "SELECT stock_code, cik FROM us_company ORDER BY stock_code"
            ).fetchall()
        else:
            placeholders = ",".join("?" for _ in codes)
            rows = conn.execute(
                f"SELECT stock_code, cik FROM us_company WHERE stock_code IN ({placeholders})",
                tuple(codes),
            ).fetchall()

        succeeded = 0
        failed: list[str] = []
        skipped_no_cik = 0
        facts_loaded = 0
        first = True
        for row in rows:
            code, cik = row["stock_code"], row["cik"]
            if not cik:
                skipped_no_cik += 1
                continue
            if not first:
                sleep_fn(SEC_MIN_INTERVAL_SEC)
            first = False
            try:
                facts = fetch_facts_fn(cik)
                fact_rows = normalize_companyfacts(code, cik, facts)
                if latest_only:
                    fact_rows = _latest_filing_rows(fact_rows)
            except Exception as exc:  # noqa: BLE001 — 종목별 실패 격리
                failed.append(code)
                log_ingest({"source": "us_financials_sec_api", "stock_code": code,
                            "status": "fail", "error": str(exc)})
                send_slack_alert(f"[us_financials_sec] {code} 수집 실패: {exc}")
                continue
            facts_loaded += _upsert_rows(conn, fact_rows, "sec_companyfacts_api", collected_at)
            conn.commit()  # 종목마다 commit(동시쓰기 락 회피, us_financials.py 관례)
            succeeded += 1

        return {
            "targets": len(rows),
            "succeeded": succeeded,
            "failed": failed,
            "skipped_no_cik": skipped_no_cik,
            "facts_loaded": facts_loaded,
        }
    finally:
        conn.close()


def backfill_from_zip(
    db_path: str | None = None,
    zip_path: str | None = None,
) -> dict:
    """companyfacts.zip 에서 추적 대상(us_company.cik NOT NULL) 종목만 걸러 전체 히스토리 적재.

    zip 안의 파일명은 `CIK{10자리}.json`(예: CIK0000320193.json). us_company 의 (cik→
    stock_code) 매핑으로 추적 대상 회사 파일만 골라 읽는다 — zip 에는 전체 미국 상장사가
    들어있지만 추적 대상만 적재한다(AC4). zip 에 없는 추적종목은 missing_in_zip 으로 보고.
    사용자가 직접 실행·관찰하는 1회성 작업이며 자동 스케줄이 아니다(스펙 Constraints).

    반환: {"targets","tickers_loaded","facts_loaded","missing_in_zip"(리스트)}.
    """
    init_db(db_path)
    conn = connect(db_path)
    collected_at = datetime.now(timezone.utc).isoformat()
    try:
        targets = [
            (r["stock_code"], r["cik"])
            for r in conn.execute(
                "SELECT stock_code, cik FROM us_company WHERE cik IS NOT NULL ORDER BY stock_code"
            ).fetchall()
        ]
        tickers_loaded = 0
        facts_loaded = 0
        missing_in_zip: list[str] = []
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            for code, cik in targets:
                entry = f"CIK{cik}.json"
                if entry not in names:
                    missing_in_zip.append(code)
                    continue
                try:
                    facts = json.loads(zf.read(entry))
                    fact_rows = normalize_companyfacts(code, cik, facts)
                except Exception as exc:  # noqa: BLE001 — 종목별 실패 격리(백필 중단 방지)
                    missing_in_zip.append(code)
                    log_ingest({"source": "us_financials_sec_zip", "stock_code": code,
                                "status": "fail", "error": str(exc)})
                    continue
                facts_loaded += _upsert_rows(conn, fact_rows, "sec_companyfacts_zip", collected_at)
                conn.commit()
                tickers_loaded += 1

        return {
            "targets": len(targets),
            "tickers_loaded": tickers_loaded,
            "facts_loaded": facts_loaded,
            "missing_in_zip": missing_in_zip,
        }
    finally:
        conn.close()


# 단일분기(duration ~3개월) 판정 범위(일) — data_access_us_sec 와 동일 기준.
_QUARTER_MIN_DAYS = 80
_QUARTER_MAX_DAYS = 100


def quarterly_coverage(conn) -> dict:
    """추적 종목(us_company.cik NOT NULL) 중 단일분기 팩트가 1개 이상인 종목 비율(AC10).

    "quarterly 팩트" = duration(period_start 있음) 이고 기간이 ~3개월(80~100일)인 팩트
    (data_access_us_sec 의 단일분기 판정과 동일). 백필 실질 커버리지를 보고한다.
    반환: {"total_tracked","with_quarterly","coverage_rate"}.
    """
    total = conn.execute(
        "SELECT COUNT(*) FROM us_company WHERE cik IS NOT NULL"
    ).fetchone()[0]
    with_q = conn.execute(
        "SELECT COUNT(DISTINCT uc.stock_code) FROM us_company uc "
        "JOIN us_financials_sec f ON f.stock_code = uc.stock_code "
        "WHERE uc.cik IS NOT NULL AND f.period_start<>'' "
        "AND julianday(f.period_end)-julianday(f.period_start) BETWEEN ? AND ?",
        (_QUARTER_MIN_DAYS, _QUARTER_MAX_DAYS),
    ).fetchone()[0]
    return {
        "total_tracked": total,
        "with_quarterly": with_q,
        "coverage_rate": (with_q / total) if total else 0.0,
    }


def table_disk_bytes(conn, table: str = "us_financials_sec") -> int:
    """테이블이 실제로 쓰는 디스크 용량(바이트)을 dbstat 으로 측정한다(AC11).

    dbstat 가상테이블(SQLITE_ENABLE_DBSTAT_VTAB)이 없는 빌드에서는 페이지수×페이지크기
    근사로 폴백한다(0 이 아닌 양수 보장). 사전 추정치(6GB 안팎)와 비교·보고용.
    """
    try:
        row = conn.execute(
            "SELECT SUM(pgsize) FROM dbstat WHERE name = ?", (table,)
        ).fetchone()
        if row and row[0]:
            return int(row[0])
    except Exception:  # noqa: BLE001 — dbstat 미지원 빌드 폴백
        pass
    # 폴백: 행수 기반 대략 추정(정밀도보다 0 회피 목적).
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    n_rows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return max(int(n_rows) * int(page_size) // 8, int(page_size))
