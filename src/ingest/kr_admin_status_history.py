"""DART 공시목록(list.json) 기반 관리종목/매매거래정지 '진짜 과거 이력' 복원.

kr_trading_status(KRX '오늘 스냅샷'만 → 미래 구간만 축적)의 한계를 DART OpenDART list.json 으로
보완한다: list.json 은 회사별 과거 공시목록을 실제로 소급 조회할 수 있어(bgn_de=20150101 정상),
지나간 관리종목 지정/해제·매매거래정지 시작/해제 이력을 과거까지 복원할 수 있다.

설계(실측 12개 종목 조사 기반):
- 순수 분류 classify_disclosure(report_nm) 가 '판별 가능한 것만' 이벤트로 확정한다. 실측 결과
  DART 공시 제목은 '관리종목 지정/해제'를 문자 그대로 담지 않는 경우가 많고, 매매거래정지도
  KOSPI 는 '매매거래정지및정지해제' 한 서식이 정지/해제 양쪽을 덮어 방향이 모호하다. 그래서
  방향이 확실한 것만 이벤트로 잡고(KOSDAQ '주권매매거래정지'=시작 / '주권매매거래정지해제'=해제,
  '관리종목지정'=지정 / '관리종목 지정 사유 해소'=해제), 모호한 것은 REVIEW 로 보류한다.
- 순수 구간화 build_status_intervals 가 이벤트를 rcept_dt(접수일) 순으로 짝지어 (지정~해제)/
  (정지~해제) 구간을 만든다. 짝을 못 맞춘 것(지정 없는 해제 등)과 REVIEW 이벤트는 review 목록으로
  보고해 사람이 검토하게 한다(추측성 오분류로 틀린 데이터를 만드는 것보다 낫다).
- 네트워크(DART)는 DI(fetch_fn 주입)로 분리해 단위 테스트한다(us_delisting/kr_stock_changes 관례).

⚠️ 이번 스코프는 '데이터 수집+파싱 로직'까지다. 이 이력을 backtest(data_access asof)에 연결하지
않는다 — 사용자가 데이터 품질을 검증한 뒤 별도로 연결 여부를 결정한다.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Callable, NamedTuple, Optional
import re

from ..db import connect, init_db
from .notify import send_slack_alert
from .robust import log_ingest, request_with_retry

BASE = "https://opendart.fss.or.kr/api"

STATUS_ADMIN = "admin"
STATUS_HALT = "halt"

EVENT_ADMIN_DESIGNATE = "admin_designate"
EVENT_ADMIN_RELEASE = "admin_release"
EVENT_HALT_START = "halt_start"
EVENT_HALT_END = "halt_end"
EVENT_REVIEW = "review"        # 관련은 있으나 방향/의미 모호 → 사람 검토 보류
EVENT_IGNORE = "ignore"        # 관리종목/거래정지와 무관


class Classification(NamedTuple):
    """공시 1건의 분류 결과. event 는 EVENT_* 중 하나.

    status_type: 관련 상태 종류('admin'|'halt', 무관하면 None).
    reason: REVIEW/IGNORE 세부 사유(예 'halt_combined','halt_period_change','admin_concern').
    """
    event: str
    status_type: Optional[str]
    reason: Optional[str]


# ---------------------------------------------------------------------------
# 정규화
# ---------------------------------------------------------------------------
_BRACKET_PREFIX = re.compile(r"^\s*\[[^\]]*\]\s*")


def _normalize_report_nm(raw: object) -> str:
    """report_nm 을 매칭용으로 정규화한다: 공백 축약 + 선두 '[기재정정]' 등 대괄호 접두어 제거.

    DART 는 정정 재공시를 '[기재정정]원제목' 형태로 준다 — 접두어를 벗겨 본 제목으로 분류해야
    정정본과 원본이 같은 이벤트로 판별된다.
    """
    s = re.sub(r"\s+", " ", str(raw or "")).strip()
    return _BRACKET_PREFIX.sub("", s).strip()


def _normalize_date(raw: object) -> Optional[str]:
    """DART rcept_dt('YYYYMMDD') 등을 'YYYY-MM-DD' 로 정규화한다(없으면 None)."""
    s = str(raw or "").strip()
    if not s:
        return None
    s = s.split(" ")[0].replace("/", "-")
    if len(s) == 8 and s.isdigit():  # 'YYYYMMDD'
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s[:10]


# ---------------------------------------------------------------------------
# 순수 분류: report_nm → Classification
# ---------------------------------------------------------------------------
def classify_disclosure(report_nm: str) -> Classification:
    """DART 공시 제목(report_nm) 하나를 관리종목/거래정지 이벤트로 분류한다(순수 함수).

    매매거래정지(halt): 제목이 정지 서식으로 '시작'해야 halt 이벤트로 본다 — '기타시장안내(...
    매매거래정지 지속...)'처럼 다른 공시가 괄호로 정지를 '언급'만 한 경우를 시작으로 오분류하지
    않기 위함. 그중 '기간변경'(정지 진행 중, 경계 아님)과 KOSPI '및정지해제' 결합형(방향 모호)은
    REVIEW 로 보류하고, 순수 '주권매매거래정지해제'만 해제, 나머지 순수 정지 제목은 시작으로 본다.

    관리종목(admin): 제목에 '관리종목'이 들어간 경우만 대상('불성실공시법인지정' 등 다른 '지정'과
    혼동 금지). '우려'(경고일 뿐 실제 지정 아님)는 무시, '해제/해소'는 해제, '지정'은 지정으로 본다.
    """
    n = _normalize_report_nm(report_nm)

    # --- 매매거래정지: 제목이 정지 서식으로 시작할 때만 ---
    if n.startswith("매매거래정지") or n.startswith("주권매매거래정지"):
        if "기간변경" in n:
            return Classification(EVENT_REVIEW, STATUS_HALT, "halt_period_change")
        if "및정지해제" in n:  # KOSPI 결합형: 정지/해제 한 서식 → 방향 모호
            return Classification(EVENT_REVIEW, STATUS_HALT, "halt_combined")
        if "정지해제" in n:
            return Classification(EVENT_HALT_END, STATUS_HALT, None)
        return Classification(EVENT_HALT_START, STATUS_HALT, None)

    # --- 관리종목 ---
    if "관리종목" in n:
        if "우려" in n:
            return Classification(EVENT_IGNORE, None, "admin_concern")
        if "해제" in n or "해소" in n:
            return Classification(EVENT_ADMIN_RELEASE, STATUS_ADMIN, None)
        if "지정" in n:
            return Classification(EVENT_ADMIN_DESIGNATE, STATUS_ADMIN, None)
        return Classification(EVENT_REVIEW, STATUS_ADMIN, "admin_unclear")

    # --- 정지 서식이 아닌 곳에서 '매매거래정지'가 언급만 됨 → 사람 검토 보류 ---
    if "매매거래정지" in n:
        return Classification(EVENT_REVIEW, STATUS_HALT, "halt_mention")

    return Classification(EVENT_IGNORE, None, None)


# ---------------------------------------------------------------------------
# 순수 구간화: 공시 목록 → {admin:[구간], halt:[구간], review:[보류]}
# ---------------------------------------------------------------------------
_OPENING_EVENTS = {EVENT_ADMIN_DESIGNATE, EVENT_HALT_START}
_CLOSING_EVENTS = {EVENT_ADMIN_RELEASE, EVENT_HALT_END}


def build_status_intervals(disclosures: list[dict]) -> dict:
    """DART 공시 목록(list of {report_nm, rcept_dt})을 관리종목/거래정지 구간으로 변환한다(순수).

    각 공시를 classify_disclosure 로 분류한 뒤 rcept_dt 순으로 상태기계를 돌린다:
    - 지정/정지 시작(opening): 열린 구간이 없으면 새로 연다. 이미 열려 있으면(해제를 못 본 채 재개시)
      기존 구간을 유지하고 그 시작 공시를 review('duplicate_start')로 보고한다.
    - 해제/정지해제(closing): 열린 구간이 있으면 그 end_date 를 채워 닫는다. 열린 구간이 없으면
      (조회창 이전에 지정된 것) 구간을 만들지 않고 review('orphan_release')로 보고한다.
    - REVIEW 이벤트(결합형/기간변경/언급/모호)는 구간을 만들지 않고 그대로 review 로 보고한다.
    - 모두 처리한 뒤 남은 열린 구간은 end_date=None(진행 중)으로 반환한다.

    반환: {"admin": [구간...], "halt": [구간...], "review": [보류항목...]}.
    구간 dict: {status_type, start_date, end_date, start_report_nm, end_report_nm}.
    보류 dict: {status_type, rcept_dt, report_nm, reason}.
    """
    events = []
    for d in disclosures:
        cls = classify_disclosure(d.get("report_nm"))
        if cls.event == EVENT_IGNORE:
            continue
        events.append((_normalize_date(d.get("rcept_dt")), d.get("report_nm"), cls))
    # rcept_dt(문자열 'YYYY-MM-DD') 오름차순. None 날짜는 뒤로.
    events.sort(key=lambda e: (e[0] is None, e[0] or ""))

    out: dict = {"admin": [], "halt": [], "review": []}
    open_iv: dict = {STATUS_ADMIN: None, STATUS_HALT: None}

    def _review(status_type, dt, nm, reason):
        out["review"].append({
            "status_type": status_type, "rcept_dt": dt, "report_nm": nm, "reason": reason,
        })

    for dt, nm, cls in events:
        if cls.event == EVENT_REVIEW:
            _review(cls.status_type, dt, nm, cls.reason)
            continue
        st = cls.status_type
        if cls.event in _OPENING_EVENTS:
            if open_iv[st] is None:
                open_iv[st] = {
                    "status_type": st, "start_date": dt, "end_date": None,
                    "start_report_nm": nm, "end_report_nm": None,
                }
            else:
                _review(st, dt, nm, "duplicate_start")
        elif cls.event in _CLOSING_EVENTS:
            if open_iv[st] is not None:
                open_iv[st]["end_date"] = dt
                open_iv[st]["end_report_nm"] = nm
                out[st].append(open_iv[st])
                open_iv[st] = None
            else:
                _review(st, dt, nm, "orphan_release")

    for st in (STATUS_ADMIN, STATUS_HALT):
        if open_iv[st] is not None:
            out[st].append(open_iv[st])
    return out


# ---------------------------------------------------------------------------
# DART list.json 조회 (I/O, 페이지네이션, DI)
# ---------------------------------------------------------------------------
def fetch_disclosures(
    api_key: str,
    corp_code: str,
    bgn_de: str = "20150101",
    end_de: str | None = None,
    fetch_fn: Optional[Callable[[dict], dict]] = None,
    page_count: int = 100,
    max_pages: int = 100,
) -> list[dict]:
    """DART list.json 을 page_no=1,2,... 로 순회하며 회사 공시목록 전체를 모은다.

    fetch_fn(params) -> dict(list.json 응답 한 페이지). 기본은 실제 HTTP(request_with_retry). status
    가 '000'이 아니면(013=데이터없음 포함) 그 페이지에서 멈춘다. total_page 까지(또는 max_pages
    방어상한까지) 순회한다. 반환: 원시 공시 dict 리스트(report_nm, rcept_dt, rm 등).
    """
    fetch_fn = fetch_fn or _default_page_fetch
    end_de = end_de or date.today().strftime("%Y%m%d")
    rows: list[dict] = []
    for page in range(1, max_pages + 1):
        params = {
            "crtfc_key": api_key, "corp_code": corp_code,
            "bgn_de": bgn_de, "end_de": end_de,
            "page_count": page_count, "page_no": page,
        }
        d = fetch_fn(params) or {}
        if d.get("status") != "000":
            break
        rows.extend(d.get("list", []) or [])
        if page >= int(d.get("total_page", 1) or 1):
            break
    return rows


def _default_page_fetch(params: dict) -> dict:
    """실제 DART list.json 한 페이지 호출. 일일 한도 초과(020)는 DartQuotaError 로 전파된다."""
    r = request_with_retry(f"{BASE}/list.json", params=params)
    if r is None:
        return {}
    try:
        return r.json()
    except Exception:  # noqa: BLE001 — JSON 파싱 실패는 빈 dict 로 격리
        return {}


# ---------------------------------------------------------------------------
# 적재: 공시 → 구간 upsert (DI fetch_fn(code), 종목별 실패 격리, 멱등)
# ---------------------------------------------------------------------------
def ingest_admin_status_history(
    db_path: str | None = None,
    codes: list[str] | None = None,
    fetch_fn: Optional[Callable[[str], list[dict]]] = None,
    commit_every: int = 20,
    updated_at: str | None = None,
    source: str = "dart_list",
) -> dict:
    """종목별 DART 공시목록에서 관리종목/거래정지 구간을 복원해 kr_admin_status_history 에 upsert.

    - codes: 순회할 종목코드 리스트(기본=company 테이블 전체).
    - fetch_fn(code) -> 원시 공시 dict 리스트(DI, 기본=실제 DART 조회). 테스트는 mock 을 주입해
      네트워크 없이 검증한다(라이브 DART 호출은 절대 하지 않음).
    - 종목별 실패는 격리(스킵+Slack)하고 다음 종목으로 계속한다(전종목 백필 보호, kr_stock_changes
      관례). commit_every 개마다 주기 커밋.
    - UNIQUE(stock_code, status_type, start_date) + INSERT OR REPLACE 로 재실행 멱등. 열린 구간이
      나중에 해제 공시로 닫히면 그 구간의 end_date 만 갱신된다(같은 start_date → 같은 UNIQUE 키).
    - review(모호·미짝 공시)는 종목별로 log_ingest 로 보고하고 총 건수를 결과에 담는다.

    반환: {"tickers","intervals_stored","review_count","failed","total_codes"}.
    """
    fetch_fn = fetch_fn or _default_fetch
    updated_at = updated_at or datetime.now(timezone.utc).isoformat()

    init_db(db_path)
    conn = connect(db_path)
    try:
        if codes is None:
            codes = [r["stock_code"] for r in conn.execute("SELECT stock_code FROM company").fetchall()]

        stored = 0
        review_total = 0
        done = 0
        failed: list[str] = []
        for code in codes:
            try:
                disclosures = fetch_fn(code)
            except Exception as exc:  # noqa: BLE001 — 종목별 실패 격리(다음 종목으로 계속)
                failed.append(code)
                log_ingest({"source": "kr_admin_status_history", "stock_code": code,
                            "status": "fail", "error": str(exc)})
                send_slack_alert(f"[kr_admin_status_history] {code} 수집 실패: {exc}")
                continue

            intervals = build_status_intervals(disclosures or [])
            for st in (STATUS_ADMIN, STATUS_HALT):
                for iv in intervals[st]:
                    conn.execute(
                        "INSERT OR REPLACE INTO kr_admin_status_history"
                        "(id, stock_code, status_type, start_date, end_date, "
                        "start_report_nm, end_report_nm, source, updated_at) VALUES "
                        "((SELECT id FROM kr_admin_status_history WHERE stock_code=? AND status_type=? "
                        "AND start_date=?), ?,?,?,?,?,?,?,?)",
                        (code, st, iv["start_date"],
                         code, st, iv["start_date"], iv["end_date"],
                         iv["start_report_nm"], iv["end_report_nm"], source, updated_at),
                    )
                    stored += 1

            n_review = len(intervals["review"])
            review_total += n_review
            if n_review:
                log_ingest({"source": "kr_admin_status_history", "stock_code": code,
                            "status": "review", "review_count": n_review,
                            "items": intervals["review"]}, to_file=False)
            done += 1
            if done % commit_every == 0:
                conn.commit()
        conn.commit()
        return {
            "tickers": done,
            "intervals_stored": stored,
            "review_count": review_total,
            "failed": failed,
            "total_codes": len(codes),
        }
    finally:
        conn.close()


def _default_fetch(code: str) -> list[dict]:
    """실제 DART: 종목코드→corp_code 매핑 후 공시목록 전체를 받아 반환한다.

    get_corp_codes 는 매 종목마다 부르면 낭비이므로 모듈 수준에서 1회 캐시한다(전종목 백필 시
    corpCode.xml 재다운로드 방지). corp_code 를 못 찾으면 빈 리스트(적재 스킵)."""
    from ..config import CONFIG
    api_key = CONFIG.dart_api_key
    corp = _corp_code_for(api_key, code)
    if not corp:
        return []
    return fetch_disclosures(api_key, corp)


_CORP_MAP_CACHE: dict[str, str] | None = None


def _corp_code_for(api_key: str, code: str) -> Optional[str]:
    """종목코드→corp_code(8자리). corpCode.xml 매핑을 모듈 수준 1회 캐시로 재사용."""
    global _CORP_MAP_CACHE
    if _CORP_MAP_CACHE is None:
        from .dart import get_corp_codes
        _CORP_MAP_CACHE = get_corp_codes(api_key)
    return _CORP_MAP_CACHE.get(code)


# ---------------------------------------------------------------------------
# 조회: 저장된 구간 이력 읽기
# ---------------------------------------------------------------------------
def get_admin_status_history(conn, stock_code: str) -> dict:
    """kr_admin_status_history 에서 한 종목의 관리종목/거래정지 구간 이력을 읽어 반환한다.

    반환: {"admin": [구간...], "halt": [구간...]}. 각 구간 dict: start_date/end_date/
    start_report_nm/end_report_nm(시작일 오름차순).
    """
    out: dict = {STATUS_ADMIN: [], STATUS_HALT: []}
    rows = conn.execute(
        "SELECT status_type, start_date, end_date, start_report_nm, end_report_nm "
        "FROM kr_admin_status_history WHERE stock_code=? ORDER BY status_type, start_date",
        (stock_code,),
    ).fetchall()
    for r in rows:
        st = r["status_type"] if not isinstance(r, tuple) else r[0]
        item = {
            "start_date": r["start_date"], "end_date": r["end_date"],
            "start_report_nm": r["start_report_nm"], "end_report_nm": r["end_report_nm"],
        }
        if st in out:
            out[st].append(item)
    return out


if __name__ == "__main__":
    print(ingest_admin_status_history())
