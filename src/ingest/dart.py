"""OpenDART API 재무 적재.

키(DART_API_KEY) 발급 후 동작한다. 키가 없으면 호출하지 않고
generate_dummy()로 안내한다 (cli에서 분기 처리).

수집 항목: 재무상태표/손익계산서 주요 계정 + 발행주식수.
- 손익(매출/영업이익/순이익)은 분기 단독(당기 3개월) 값을 우선 사용.
- 재무상태(자산/부채/자본)는 시점 값.
각 행에 공시 기준분기/공시일을 기록하고, 적재 후 실제 공시된 최신 분기를
ingest_meta('latest_disclosed_quarter')로 저장한다.
"""
from __future__ import annotations

import io
import time
import zipfile
from datetime import date
from xml.etree import ElementTree as ET

import requests

from ..config import CONFIG
from ..db import connect, get_meta, init_db, set_meta
from ..version import estimate_disclosed_date, recent_quarters
from .companies import COMPANIES
from .ksic import market_from_corp_cls
from .normalize import normalize_account
from .notify import send_slack_alert
from .robust import DartQuotaError, log_ingest, request_with_retry

BASE = "https://opendart.fss.or.kr/api"

# 분기 라벨 → DART 보고서 코드 (1분기/반기/3분기/사업보고서)
REPRT_CODE = {1: "11013", 2: "11012", 3: "11014", 4: "11011"}

# 손익(분기 단독으로 공시) vs 재무상태(시점 잔액)
IS_KEYS = {"revenue", "operating_profit", "net_income"}
BS_KEYS = {"total_assets", "total_liabilities", "total_equity"}

# 광범위 수집: FLOW(손익+현금흐름 = 분기단독/Q4차분) vs STOCK(재무상태 = 시점값, 차분 안 함)
FLOW_KEYS = {
    "revenue", "cost_of_sales", "gross_profit", "sga", "operating_profit",
    "net_income", "interest_expense", "operating_cashflow", "depreciation", "dividend",
}


def get_corp_codes(api_key: str) -> dict[str, str]:
    """종목코드(6자리) → DART corp_code(8자리) 매핑."""
    r = request_with_retry(
        f"{BASE}/corpCode.xml", params={"crtfc_key": api_key}, expect_json=False
    )
    if r is None:
        return {}
    try:
        zf = zipfile.ZipFile(io.BytesIO(r.content))
    except zipfile.BadZipFile:
        # status 020(사용한도 초과) 등은 zip이 아닌 에러 XML을 반환한다 → 빈 매핑으로 격리
        return {}
    xml = zf.read(zf.namelist()[0])
    root = ET.fromstring(xml)
    mapping: dict[str, str] = {}
    for el in root.iter("list"):
        stock = (el.findtext("stock_code") or "").strip()
        corp = (el.findtext("corp_code") or "").strip()
        if stock and corp:
            mapping[stock] = corp
    return mapping


def fetch_company_profile(api_key: str, corp_code: str) -> dict:
    """회사개황(company.json) 호출 → {"corp_cls":..., "induty_code":...}.

    request_with_retry를 사용하므로 일일 한도 초과(020)면 DartQuotaError가 그대로 전파된다.
    status가 정상이 아니거나 응답 파싱 실패 시 빈 dict 반환.
    """
    r = request_with_retry(
        f"{BASE}/company.json",
        params={"crtfc_key": api_key, "corp_code": corp_code},
    )
    if r is None:
        return {}
    try:
        d = r.json()
    except Exception:  # noqa: BLE001 — JSON 파싱 실패는 빈 dict로 격리
        return {}
    if d.get("status") != "000":
        return {}
    return {
        "corp_cls": d.get("corp_cls", ""),
        "induty_code": d.get("induty_code", ""),
    }


def fetch_single_account(api_key: str, corp_code: str, year: int, reprt: str) -> list[dict]:
    """단일회사 주요계정 (fnlttSinglAcnt)."""
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": reprt,
    }
    r = requests.get(f"{BASE}/fnlttSinglAcnt.json", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "000":
        return []
    return data.get("list", [])


def _to_amount(s: str | None) -> float | None:
    if not s:
        return None
    s = s.replace(",", "").strip()
    if s in ("", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_all_accounts(api_key: str, corp_code: str, year, reprt: str, fs_div: str = "CFS") -> list[dict]:
    """전체 재무제표(fnlttSinglAcntAll). fs_div: CFS(연결) | OFS(별도)."""
    params = {
        "crtfc_key": api_key, "corp_code": corp_code, "bsns_year": str(year),
        "reprt_code": reprt, "fs_div": fs_div,
    }
    r = request_with_retry(f"{BASE}/fnlttSinglAcntAll.json", params=params)
    if r is None:
        return []
    try:
        d = r.json()
    except Exception:
        return []
    return d.get("list", []) if d.get("status") == "000" else []


def _parse_all(rows: list[dict]) -> tuple[dict, str | None]:
    """전체 재무제표 응답 파싱(이미 fs_div로 필터됨). account_key별 첫 매칭."""
    data: dict = {}
    rcept = None
    for r in rows:
        if rcept is None and r.get("rcept_no"):
            rcept = str(r["rcept_no"])[:8]
        key = normalize_account(r.get("account_nm", ""), r.get("account_id"))
        if not key or key in data:
            continue
        amt = _to_amount(r.get("thstrm_amount"))
        if amt is None:
            continue
        data[key] = (amt, r.get("account_nm"))
    return data, rcept


def _save_raw(conn, code: str, year: int, reprt: str, fs_div: str, rows: list) -> None:
    """전체재무제표 원본 응답을 zlib 압축해 raw_reports에 보관.

    새 계정이 필요해지면 재수집(DART 재호출) 없이 이 원본을 재파싱해 financials를 재생성한다.
    """
    import json
    import zlib

    from ..version import now_iso

    payload = zlib.compress(json.dumps(rows, ensure_ascii=False).encode("utf-8"))
    conn.execute(
        "INSERT OR REPLACE INTO raw_reports"
        "(stock_code, bsns_year, reprt_code, fs_div, payload, fetched_at) VALUES(?,?,?,?,?,?)",
        (code, year, reprt, fs_div, payload, now_iso()),
    )


def _upsert_company(conn, code: str, name: str, market: str, sector: str) -> None:
    """company 행을 채워넣되, 새 market/sector가 빈 값("")이면 기존 값을 지우지 않는다.

    유니버스 소스(예: DART corpCode.xml 기반 _dart_listed_universe)는 market/sector를
    모르면 ""로 채워 넘긴다. 예전 INSERT OR REPLACE는 이 빈 값으로 행 전체를 갈아치워
    KRX 섹터 백필(scripts/backfill_sector_krx.py) 결과를 통째로 지워버렸다(매일 새벽
    2시 ingest_dart_full 재실행마다 재현). "모르는 칸은 건드리지 않는다" 원칙으로 수정.
    """
    conn.execute(
        """INSERT INTO company(stock_code, name, market, sector) VALUES(?,?,?,?)
           ON CONFLICT(stock_code) DO UPDATE SET
             name=excluded.name,
             market=CASE WHEN excluded.market != '' THEN excluded.market ELSE company.market END,
             sector=CASE WHEN excluded.sector != '' THEN excluded.sector ELSE company.sector END""",
        (code, name, market, sector),
    )


def ingest_dart(
    db_path: str | None = None,
    years: int = 3,
    today: date | None = None,
    sleep: float = 0.4,
) -> dict:
    """COMPANIES 유니버스에 대해 최근 `years`년 분기 재무를 적재."""
    if not CONFIG.has_dart_key:
        raise RuntimeError("DART_API_KEY 없음 — generate_dummy()를 사용하세요.")

    today = today or date.today()
    api_key = CONFIG.dart_api_key
    init_db(db_path)
    conn = connect(db_path)
    try:
        corp_map = get_corp_codes(api_key)

        # company 적재
        for code, name, market, sector in COMPANIES:
            _upsert_company(conn, code, name, market, sector)
        conn.commit()

        from ..version import estimate_available_quarter

        latest_q = estimate_available_quarter(today)
        wanted = set(recent_quarters(latest_q, years * 4))
        years_set = sorted({int(q[:4]) for q in wanted})
        actual_latest = None
        n_rows = 0

        for code, name, *_ in COMPANIES:
            corp = corp_map.get(code)
            if not corp:
                continue
            for year in years_set:
                # 한 연도의 4개 보고서(1Q/반기/3Q/사업)를 전체재무제표로 받아둔다 (연결 우선, 없으면 별도)
                reports = {}  # qn -> (data, rcept)
                for qn in (1, 2, 3, 4):
                    rows = fetch_all_accounts(api_key, corp, year, REPRT_CODE[qn], "CFS")
                    time.sleep(sleep)
                    data, rcept = _parse_all(rows)
                    if not data:
                        rows = fetch_all_accounts(api_key, corp, year, REPRT_CODE[qn], "OFS")
                        time.sleep(sleep)
                        data, rcept = _parse_all(rows)
                    reports[qn] = (data, rcept)

                for qn in (1, 2, 3, 4):
                    quarter = f"{year}Q{qn}"
                    if quarter not in wanted:
                        continue
                    data, rcept = reports[qn]
                    if not data:
                        continue
                    disclosed = (
                        f"{rcept[:4]}-{rcept[4:6]}-{rcept[6:8]}"
                        if rcept and len(rcept) == 8
                        else estimate_disclosed_date(quarter)
                    )
                    for key, (amt, nm) in data.items():
                        # FLOW(손익·현금흐름)는 분기 단독으로 공시되나, 사업보고서(Q4)는 연간 누적이므로
                        # Q4 단독 = 연간 - (Q1+Q2+Q3). STOCK(재무상태)은 시점 잔액 그대로.
                        if key in FLOW_KEYS and qn == 4:
                            prior = sum(
                                reports[i][0][key][0]
                                for i in (1, 2, 3)
                                if key in reports[i][0]
                            )
                            value = amt - prior
                        else:
                            value = amt
                        conn.execute(
                            """INSERT OR REPLACE INTO financials
                                 (stock_code, quarter, disclosed_date, account_key, account_name, amount)
                               VALUES (?,?,?,?,?,?)""",
                            (code, quarter, disclosed, key, nm, round(value)),
                        )
                        n_rows += 1
                    actual_latest = max(actual_latest or quarter, quarter, key=_q_sort_key)
                conn.commit()

        if actual_latest:
            set_meta(conn, "latest_disclosed_quarter", actual_latest)
        return {"financials_rows": n_rows, "latest_disclosed_quarter": actual_latest}
    finally:
        conn.close()


def _q_sort_key(q: str) -> tuple[int, int]:
    return int(q[:4]), int(q[5])


def _write_reports_year(conn, code, year, reports, wanted, *, ingest_shares_fn=None):
    """연도별 reports({qn:(data,rcept)}) → financials 기록.

    라이브 수집(_ingest_one_company)과 원문 재파싱이 공유한다. Q4 흐름항목(FLOW_KEYS)은
    연간−(Q1+Q2+Q3) 차분, 재무상태(STOCK)는 시점값 그대로. ingest_shares_fn(qn, quarter,
    disclosed)이 주어지면 상장주식수도 함께 적재(재파싱 시엔 None → 스킵).
    반환: (이번 연도 적재 행수, 이번 연도 최신 분기 or None).
    """
    n_rows = 0
    latest: str | None = None
    for qn in (1, 2, 3, 4):
        quarter = f"{year}Q{qn}"
        if quarter not in wanted:
            continue
        data, rcept = reports[qn]
        if not data:
            continue
        disclosed = (
            f"{rcept[:4]}-{rcept[4:6]}-{rcept[6:8]}"
            if rcept and len(rcept) == 8
            else estimate_disclosed_date(quarter)
        )
        for key, (amt, nm) in data.items():
            if key in FLOW_KEYS and qn == 4:
                prior = sum(
                    reports[i][0][key][0]
                    for i in (1, 2, 3)
                    if key in reports[i][0]
                )
                value = amt - prior
            else:
                value = amt
            conn.execute(
                """INSERT OR REPLACE INTO financials
                     (stock_code, quarter, disclosed_date, account_key, account_name, amount)
                   VALUES (?,?,?,?,?,?)""",
                (code, quarter, disclosed, key, nm, round(value)),
            )
            n_rows += 1
        if ingest_shares_fn is not None:
            n_rows += ingest_shares_fn(qn, quarter, disclosed)
        latest = max(latest or quarter, quarter, key=_q_sort_key)
    return n_rows, latest


def _ingest_one_company(conn, api_key, code, corp, years_set, wanted, sleep) -> tuple[int, str | None]:
    """한 종목의 재무를 적재. 반환: (적재 행수, 이 종목의 최신 분기).

    ingest_dart의 종목별 로직과 동일(연결 우선/별도 폴백, Q4 FLOW 차분).

    헛호출 절감: 최근 3년의 1분기·사업보고서를 먼저 프로브해 데이터가 없으면(상폐/껍데기)
    종목당 ~80회 호출 대신 소수 호출로 즉시 스킵한다.
    """
    # --- 활성 종목 프로브 ---
    # 최근 3년의 1분기보고서(11013) + 사업보고서(11011)를 확인한다.
    # (기존엔 최근 2년의 1분기보고서만 봐서, 분기보고서를 안 내거나 최근 상폐된
    #  정상 기업의 과거 데이터까지 통째로 누락시키는 문제가 있었음 — 예: 연우.)
    if years_set:
        probe_years = years_set[-3:]  # 최근 3년
        active = False
        for py in reversed(probe_years):  # 최신 연도부터
            # 사업보고서(11011)만 프로브 — 연 1회 의무공시라 활성 판별에 충분하고,
            # 1분기보고서까지 보면 비활성 종목당 호출이 12회로 늘어 한도/속도를 낭비한다(6회로 축소).
            for fs in ("CFS", "OFS"):
                d, _ = _parse_all(fetch_all_accounts(api_key, corp, py, "11011", fs))
                time.sleep(sleep)
                if d:
                    active = True
                    break
            if active:
                break
        if not active:
            return 0, None  # 최근 3년 사업보고서에 데이터 없음 → 진짜 비활성/껍데기

    n_rows = 0
    company_latest: str | None = None
    for year in years_set:
        reports: dict = {}  # qn -> (data, rcept)
        for qn in (1, 2, 3, 4):
            fs_used = "CFS"
            rows = fetch_all_accounts(api_key, corp, year, REPRT_CODE[qn], "CFS")
            time.sleep(sleep)
            data, rcept = _parse_all(rows)
            if not data:
                fs_used = "OFS"
                rows = fetch_all_accounts(api_key, corp, year, REPRT_CODE[qn], "OFS")
                time.sleep(sleep)
                data, rcept = _parse_all(rows)
            if rows:  # 원본 응답 보관(재파싱용)
                _save_raw(conn, code, year, REPRT_CODE[qn], fs_used, rows)
            reports[qn] = (data, rcept)

        # 같은 보고서의 분기별 상장주식수도 함께 적재(라이브 수집 경로에서만).
        # ⚠️ 분기당 stockTotqySttus 호출이 1회 추가되므로 일일 한도 소모가 늘어난다.
        def _shares(qn, quarter, disclosed):
            return _ingest_shares(conn, api_key, code, corp, year, qn, quarter, disclosed, sleep)

        yr_rows, yr_latest = _write_reports_year(
            conn, code, year, reports, wanted, ingest_shares_fn=_shares
        )
        n_rows += yr_rows
        if yr_latest:
            company_latest = max(company_latest or yr_latest, yr_latest, key=_q_sort_key)
    return n_rows, company_latest


_SHARES_JUMP_RATIO = 100  # 직전 분기 대비 이 배수를 초과/미만이면 단위 오류로 간주(서린바이오 1000배 사고)


def _shares_jump_anomalous(prev: float | None, new: float) -> bool:
    """직전 공시 상장주식수(prev) 대비 new가 비정상적 배수로 뛰거나 급락했는지 판정.

    prev가 없으면(첫 데이터 포인트) 비교 기준이 없어 False(fetch_shares의 절대기준
    1e10만 적용됨). 액면분할 등 정상적 사유로도 수배~수십배 변동은 가능하지만,
    _SHARES_JUMP_RATIO(100배)를 넘는 변동은 실질적으로 전량 단위 오기입이다.
    """
    if not prev:
        return False
    return new > prev * _SHARES_JUMP_RATIO or new < prev / _SHARES_JUMP_RATIO


def _latest_shares_outstanding(conn, code: str) -> float | None:
    """financials에 이미 적재된 code의 가장 최근(quarter 기준) shares_outstanding."""
    row = conn.execute(
        "SELECT amount FROM financials WHERE stock_code = ? AND account_key = 'shares_outstanding' "
        "ORDER BY quarter DESC LIMIT 1",
        (code,),
    ).fetchone()
    return row["amount"] if row else None


def _ingest_shares(conn, api_key, code, corp, year, qn, quarter, disclosed, sleep) -> int:
    """해당 (연도, 분기) 보고서의 상장주식수를 financials에 적재. 적재 행수(0/1) 반환.

    account_key='shares_outstanding', account_name='상장주식수'로 재무 행과 동일 quarter에 UPSERT.
    DartQuotaError(일일 한도 초과)는 상위로 전파해 백필을 중단시킨다(재무와 동일 흐름).
    그 외 실패(네트워크/파싱/없음)는 try/except로 격리 — 주식수 실패가 재무 적재를 막지 않는다.
    직전 분기 대비 비정상적 배수 변동(_shares_jump_anomalous)이면 단위 오기입으로 보고 skip한다
    (서린바이오 038070 2025Q3 1000배 사고 — 절대기준 1e10만으로는 못 잡았음).
    """
    try:
        shares = fetch_shares(api_key, corp, year, REPRT_CODE[qn])
        time.sleep(sleep)
    except DartQuotaError:
        raise  # 한도 초과는 그대로 전파(상위 백필 중단)
    except Exception as exc:  # noqa: BLE001 — 주식수 호출 실패 격리(재무 적재 보호)
        log_ingest({"stock_code": code, "quarter": quarter, "status": "shares_error", "error": str(exc)})
        return 0
    if not shares:
        return 0
    prev = _latest_shares_outstanding(conn, code)
    if _shares_jump_anomalous(prev, shares):
        log_ingest({
            "stock_code": code, "quarter": quarter, "status": "shares_anomaly",
            "error": f"prev={prev} new={shares}",
        })
        send_slack_alert(f"[dart] {code} {quarter} 상장주식수 이상치 감지(prev={prev}, new={shares}) — skip")
        return 0
    conn.execute(
        """INSERT OR REPLACE INTO financials
             (stock_code, quarter, disclosed_date, account_key, account_name, amount)
           VALUES (?,?,?,?,?,?)""",
        (code, quarter, disclosed, "shares_outstanding", "상장주식수", round(shares)),
    )
    return 1


def _update_company_profile(conn, api_key, code, corp, sleep) -> None:
    """회사개황을 1회 호출해 company.market을 UPDATE.

    DartQuotaError(일일 한도 초과)는 그대로 전파해 바깥 루프가 백필을 중단하게 한다.
    그 외 모든 실패(네트워크/파싱/빈 응답)는 격리하여 이미 적재된 재무를 막지 않는다.

    sector는 더 이상 여기서 건드리지 않는다 — 유일한 최종 출처는
    scripts/backfill_sector_krx.py(KRX 정보데이터시스템)로 통일됐다. DART 회사개황(KSIC)
    기반 세분류(80개+)는 KRX 자체 대분류(20~30개)로 대체됐다(사용자 결정, 2026-07-14).
    """
    profile = fetch_company_profile(api_key, corp)  # 한도 초과면 여기서 DartQuotaError 전파
    time.sleep(sleep)
    if not profile:
        return
    market = market_from_corp_cls(profile.get("corp_cls"))
    try:
        conn.execute(
            "UPDATE company SET market = ? WHERE stock_code = ?",
            (market, code),
        )
        conn.commit()
    except Exception as exc:  # noqa: BLE001 — UPDATE 실패가 재무 적재를 막지 않도록 격리
        log_ingest({"stock_code": code, "status": "profile_update_error", "error": str(exc)})


def ingest_dart_full(
    db_path: str | None = None,
    years: int | None = None,
    today: date | None = None,
    sleep: float | None = None,
    resume: bool = True,
) -> dict:
    """full 유니버스(코스피+코스닥 전체) 재무 적재 — 수집 안정성 적용.

    - get_universe("full") 대상.
    - 종목별 try/except 격리(1종목 오류로 전체 중단 금지).
    - 체크포인트: ingest_meta('backfill_checkpoint')에 마지막 완료 stock_code 저장,
      resume=True면 그 다음 종목부터 재개.
    - 종목별 성공/실패 로그 + 수집 리포트 dict 반환.
    """
    if not CONFIG.has_dart_key:
        raise RuntimeError("DART_API_KEY 없음 — generate_dummy()를 사용하세요.")

    from .universe import get_universe

    today = today or date.today()
    years = years or CONFIG.data_years
    sleep = CONFIG.api_call_delay if sleep is None else sleep
    api_key = CONFIG.dart_api_key
    init_db(db_path)
    conn = connect(db_path)
    try:
        from ..version import estimate_available_quarter

        universe = get_universe("full")
        try:
            corp_map = get_corp_codes(api_key)
        except Exception as exc:  # noqa: BLE001 — corpCode.xml 다운로드 실패(주로 일일한도 초과 시 비-zip 응답)
            log_ingest({"status": "ABORT", "reason": f"corpCode 로드 실패(한도 초과 추정): {exc}"})
            return {
                "total": len(universe), "skipped": 0, "ok": 0, "failed": 0,
                "no_corp_code": 0, "financials_rows": 0, "missing": [],
                "latest_disclosed_quarter": None, "quota_exceeded": True,
            }

        latest_q = estimate_available_quarter(today)
        wanted = set(recent_quarters(latest_q, years * 4))
        years_set = sorted({int(q[:4]) for q in wanted})

        # resume: 이미 financials 데이터가 있는 종목은 건너뛴다.
        # (체크포인트 인덱스 방식은 한도초과(020)로 실패한 종목을 영구 누락시키므로 폐기)
        done_codes: set[str] = set()
        inactive_codes: set[str] = set()
        if resume:
            done_codes = {r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM financials")}
            # 이전 실행에서 '데이터 없음(비활성)'으로 판정된 종목 — 재순회 시 프로브조차 생략.
            _iv = get_meta(conn, "inactive_codes")
            if _iv:
                inactive_codes = set(_iv.split(","))

        total = len(universe)
        n_rows = 0
        ok, failed, no_corp, missing, skipped = 0, 0, 0, [], 0
        quota_hit = False
        actual_latest: str | None = None

        def _save_inactive() -> None:
            set_meta(conn, "inactive_codes", ",".join(sorted(inactive_codes)))

        for idx, (code, name, market, sector) in enumerate(universe):
            if resume and (code in done_codes or code in inactive_codes):
                skipped += 1
                continue
            # company 적재(실패해도 격리)
            try:
                _upsert_company(conn, code, name, market, sector)
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                log_ingest({"stock_code": code, "name": name, "status": "company_error", "error": str(exc)})

            corp = corp_map.get(code)
            if not corp:
                no_corp += 1
                missing.append(code)
                inactive_codes.add(code)  # corp_code 없음 → 비활성 마킹
                log_ingest({"stock_code": code, "name": name, "status": "no_corp_code"})
                continue

            try:
                rows, company_latest = _ingest_one_company(
                    conn, api_key, code, corp, years_set, wanted, sleep
                )
                conn.commit()
                n_rows += rows
                if rows == 0:
                    missing.append(code)
                    inactive_codes.add(code)  # 데이터 없음 → 비활성 마킹(다음 실행 시 재프로브 생략)
                else:
                    # 재무가 적재된 활성 종목만 회사개황 1회 호출 → market/sector를 표준산업분류로 채운다.
                    # (비활성/스킵 종목은 호출하지 않아 한도 절약.)
                    # DartQuotaError는 바깥 except로 전파해 백필을 즉시 중단하고,
                    # 그 외 실패는 격리해 이미 적재된 재무를 막지 않는다.
                    _update_company_profile(conn, api_key, code, corp, sleep)
                if company_latest:
                    actual_latest = max(actual_latest or company_latest, company_latest, key=_q_sort_key)
                ok += 1
                log_ingest({"stock_code": code, "name": name, "status": "ok", "rows": rows})
            except DartQuotaError:
                # 일일 한도 초과: 더 진행해도 전부 실패 → 즉시 중단(미처리 종목은 다음 실행에서 재시도)
                quota_hit = True
                _save_inactive()
                log_ingest({"stock_code": code, "name": name, "status": "QUOTA_EXCEEDED"})
                break
            except Exception as exc:  # noqa: BLE001 — 종목 실패 격리
                failed += 1
                missing.append(code)
                log_ingest({"stock_code": code, "name": name, "status": "error", "error": str(exc)})

            if idx % 100 == 0:  # 중단 대비 비활성 목록 주기적 저장
                _save_inactive()

        _save_inactive()
        if actual_latest:
            set_meta(conn, "latest_disclosed_quarter", actual_latest)

        return {
            "total": total,
            "skipped": skipped,
            "ok": ok,
            "failed": failed,
            "no_corp_code": no_corp,
            "financials_rows": n_rows,
            "missing": missing,
            "latest_disclosed_quarter": actual_latest,
            "quota_exceeded": quota_hit,
        }
    finally:
        conn.close()


def fetch_shares(api_key: str, corp_code: str, year: int, reprt: str = "11011") -> float | None:
    """주식총수현황(stockTotqySttus)에서 보통주 상장주식수.

    stockTotqySttus API는 reprt_code 파라미터를 받으므로 사업보고서뿐 아니라
    분기/반기보고서(11013/11012/11014)의 주식총수도 받을 수 있다 → 분기별 상장주식수.
    기본값 reprt="11011"(사업보고서)로 두어 연도만 주던 기존 호출과 호환된다.
    상장주식수 = 유통주식수(distb_stock_co) + 자기주식수(tesstk_co).
    주의: now_to_isu_stock_totqy(발행총수)는 증자·감자 누적이라 과대하므로 쓰지 않는다.

    request_with_retry를 사용하므로 일일 한도 초과(020)면 DartQuotaError가 그대로 전파된다.
    응답이 없거나 status가 정상이 아니거나 보통주 행이 없으면 None.
    """
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": reprt,
    }
    r = request_with_retry(f"{BASE}/stockTotqySttus.json", params=params)
    if r is None:
        return None
    try:
        d = r.json()
    except Exception:  # noqa: BLE001 — JSON 파싱 실패는 None으로 격리
        return None
    if d.get("status") != "000":
        return None
    # se(주식 종류) 표기는 회사마다 제각각이다:
    #   "보통주" / "의결권이 있는주식(보통주)" / "의결권 있는 주식" / "의결권주식" / "의결권 有"(한자) 등.
    # 보통주 = '보통주' 또는 '의결권' 포함하되 우선주·무의결권(無/없/우선)은 아닌 행.
    # 보통주 행이 아예 없고(예: 합계만 제공) 우선주도 없으면 '합계'를 보통주로 본다.
    common = None
    total_fallback = None
    has_preferred = False
    for row in d.get("list", []):
        se = str(row.get("se", "")).strip()
        distb = _to_amount(row.get("distb_stock_co")) or 0
        tes = _to_amount(row.get("tesstk_co")) or 0
        amt = distb + tes
        if se in ("비고", ""):
            continue
        if se == "합계":
            total_fallback = amt
            continue
        if "우선주" in se or "무의결권" in se or "없" in se or "無" in se or "우선" in se:
            has_preferred = True
            continue
        if ("보통주" in se or "의결권" in se) and amt > 0:
            common = amt
            break
    listed = common if common is not None else (total_fallback if not has_preferred else None)
    if listed and listed > 0:
        # DART 원본 단위 오류(천주·백만주 오기입) 방어. 한국 최대 상장주식수도
        # 삼성전자 약 60억 주 수준이라, 100억 주(1e10) 초과는 사실상 전량 오기입이다.
        if listed > 1e10:
            return None
        return listed
    return None


if __name__ == "__main__":
    print(ingest_dart())
