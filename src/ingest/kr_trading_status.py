"""KRX 관리종목/매매거래정지 '오늘 현재' 스냅샷 누적 수집.

과거 이력은 무료로 구할 수 없다: KRX(data.krx.co.kr)/KIND 둘 다 조회일 파라미터를 무시하고
'오늘 현재' 스냅샷만 반환함을 실측 확인했다. 그래서 과거 백테스트 보정은 불가능하다 → 대신
지금부터 매 실행(매일 1회)마다 '오늘 현재' 스냅샷을 저장하고 직전 실행 스냅샷과 diff 해서
앞으로의(미래) 지정~해제 구간을 우리 쪽에서 누적으로 쌓는다. 스냅샷 자체는 시점조회가 안
되지만, 우리가 주기적으로 저장+diff 하면 우리 쪽에서 구간 이력을 만들 수 있다. 이 데이터는
과거 백테스트엔 못 쓰지만 라이브(현재 시점) 필터링엔 바로 유효하다.

메커니즘(us_delisting 의 구간+멱등 사상 차용):
- '직전 스냅샷'은 별도 저장 없이 kr_trading_status 의 열린 구간(end_date IS NULL)에서 유도한다.
- 현재 스냅샷에 새로 나타난 종목 = 지정 개시 → 새 행(start_date=관측일, end_date=NULL).
- 사라진 종목 = 해제 → 그 열린 구간에 end_date=관측일 기입.
- 계속 있는 종목 = 구간 유지(사유/시각만 갱신). 같은 날 두 번 돌려도 신규/해제 0(멱등).
- 지정→해제→재지정 반복은 각각 다른 start_date 를 가진 별개 행으로 쌓인다.

⚠️ 이 데이터를 backtest look-ahead 경로(src/backtest/data_access.py 의 metrics_at/_is_alive 등)에
연결하지 마라 — 과거 이력이 없어 연결하면 look-ahead/생존편향을 오히려 악화시킨다. 라이브
'현재 시점' 필터링 전용 is_currently_administrative_or_halted 만 노출한다(이 프로젝트엔 라이브
매매 실행기가 없으므로 함수·데이터 축적 인프라만 만들고 실제 호출부는 이번 스코프 아님).

네트워크(KRX 로그인/HTTP)는 DI(fetch_admin_fn/fetch_halt_fn 주입)로 분리해 단위 테스트한다
(기존 us_delisting/us_financials_sec 관례와 동일). 라이브 KRX 로그인은 스크립트를 사용자가
직접 실행할 때만 일어난다. KRX 자격증명(KRX_ID/KRX_PW)은 pykrx build_krx_session 이 os.environ
에서만 읽으며(src.config 가 import 시 .env 로드), 값은 로그/출력에 남기지 않는다.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Callable, Optional

from ..db import connect, init_db
from .notify import send_slack_alert
from .robust import log_ingest

# KRX 통계 JSON 데이터 엔드포인트(pykrx krxio.py 와 동일). bld 코드로 화면을 지정한다.
KRX_JSON_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
# 관리종목 현황 / 매매거래정지종목 현황 (path_bld_information.json 실측 확인).
BLD_ADMIN_ISSUE = "dbms/MDC/STAT/issue/MDCSTAT21401"
BLD_TRADING_HALT = "dbms/MDC/STAT/issue/MDCSTAT21201"

STATUS_ADMIN = "admin"
STATUS_HALT = "halt"


# ---------------------------------------------------------------------------
# 정규화 헬퍼
# ---------------------------------------------------------------------------
def _normalize_stock_code(raw: object) -> str:
    """KRX 종목코드를 6자리 단축코드로 정규화한다.

    KRX 는 12자 표준코드(ISIN, 예 'KR7005930003')와 6자리 단축코드('005930')를 섞어 준다.
    12자 ISIN 이면 중간 6자리(문자[3:9])를 뽑고, 이미 단축코드면 그대로 쓴다.
    """
    s = str(raw or "").strip()
    if len(s) == 12 and s[:2].isalpha():
        return s[3:9]
    return s


def _normalize_date(raw: object) -> Optional[str]:
    """KRX 날짜/일시 문자열을 'YYYY-MM-DD' 로 정규화한다(없으면 None).

    허용 형식: 'YYYY/MM/DD', 'YYYY-MM-DD', 'YYYYMMDD', 'YYYY-MM-DD HH:MM:SS'(날짜부만).
    """
    s = str(raw or "").strip()
    if not s:
        return None
    s = s.split(" ")[0].replace("/", "-")
    if len(s) == 8 and s.isdigit():  # 'YYYYMMDD'
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s[:10]


def _iter_output(payload: object) -> list[dict]:
    """KRX getJsonData 응답에서 행 리스트를 꺼낸다(output/OutBlock_1/bare list 모두 허용)."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("output", "OutBlock_1", "block1"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
    return []


def _parse_rows(payload: object, reason_key: str, date_key: str) -> list[dict]:
    """관리종목/거래정지 공통 파서 → 정규화 행 리스트.

    반환 각 행: {stock_code, company_name, market, reason, krx_designated_date}.
    ISU_SRT_CD(단축코드) 우선, 없으면 ISU_CD. 종목코드가 비면 건너뛴다.
    """
    rows: list[dict] = []
    for e in _iter_output(payload):
        code = _normalize_stock_code(e.get("ISU_SRT_CD") or e.get("ISU_CD"))
        if not code:
            continue
        rows.append({
            "stock_code": code,
            "company_name": (str(e.get("ISU_NM") or "").strip() or None),
            "market": (str(e.get("MKT_NM") or "").strip() or None),
            "reason": (str(e.get(reason_key) or "").strip() or None),
            "krx_designated_date": _normalize_date(e.get(date_key)),
        })
    return rows


def parse_admin_issue(payload: object) -> list[dict]:
    """관리종목 현황(MDCSTAT21401) 응답을 정규화 행으로 변환한다.

    필드: ISU_CD/ISU_SRT_CD(종목코드), ISU_NM(종목명), MKT_NM(시장), FST_DESIGN_DD(최초지정일),
    LIST_BZ_RSN_NM(사유).
    """
    return _parse_rows(payload, reason_key="LIST_BZ_RSN_NM", date_key="FST_DESIGN_DD")


def parse_trading_halt(payload: object) -> list[dict]:
    """매매거래정지 현황(MDCSTAT21201) 응답을 정규화 행으로 변환한다.

    필드: ISU_CD/ISU_SRT_CD(종목코드), ISU_NM(종목명), HALT_DESNRELS_DDTM(지정/해제 일시),
    HALT_RSN_NM(사유). MKT_NM 은 없을 수 있다.
    """
    return _parse_rows(payload, reason_key="HALT_RSN_NM", date_key="HALT_DESNRELS_DDTM")


# ---------------------------------------------------------------------------
# diff: 직전 스냅샷(열린 구간) vs 현재 스냅샷
# ---------------------------------------------------------------------------
def diff_snapshot(open_codes: set[str], current_codes: set[str]) -> tuple[set[str], set[str], set[str]]:
    """(신규지정, 해제, 유지) 종목코드 집합을 계산한다.

    신규지정 = 현재엔 있는데 직전 열린 구간엔 없던 종목(지정 시작).
    해제      = 직전 열린 구간엔 있었는데 현재엔 없는 종목(지정 해제).
    유지      = 양쪽 모두에 있는 종목(구간 유지).
    """
    newly = current_codes - open_codes
    released = open_codes - current_codes
    continuing = current_codes & open_codes
    return newly, released, continuing


# ---------------------------------------------------------------------------
# 라이브 조회 전용(backtest 미연결)
# ---------------------------------------------------------------------------
def is_currently_administrative_or_halted(conn, stock_code: str) -> bool:
    """최신 스냅샷 기준 이 종목이 현재 관리종목이거나 매매거래정지 상태인지(bool).

    열린 구간(end_date IS NULL)이 하나라도 있으면 True. 이력이 아예 없으면 False.

    ⚠️ 라이브(현재 시점) 필터링 전용이다. backtest look-ahead 경로(_is_alive/metrics_at 등)에
    연결하지 마라 — 과거 이력이 없어 연결하면 생존편향을 악화시킨다.
    """
    row = conn.execute(
        "SELECT 1 FROM kr_trading_status WHERE stock_code=? AND end_date IS NULL LIMIT 1",
        (stock_code,),
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# 실제 KRX 호출(라이브 전용, 지연 import — pykrx import 시 자동 로그인 회피)
# ---------------------------------------------------------------------------
def _fetch_krx_json(bld: str, session_factory: Optional[Callable] = None) -> dict:
    """KRX getJsonData 를 bld 로 1회 POST 한다(로그인 세션의 쿠키 포함).

    session_factory()->KRXSession 은 테스트 주입용(기본=pykrx build_krx_session). 로그인 실패
    (자격증명 없음/거부)면 RuntimeError. mktId=ALL 로 전체 시장을 받는다(조회일은 무시되고
    '오늘 현재' 스냅샷이 온다 — 이 잡의 전제).
    """
    if session_factory is None:
        from pykrx.website.comm.auth import build_krx_session  # 지연 import(자동 로그인 회피)

        session_factory = build_krx_session
    krxs = session_factory()
    if krxs is None:
        raise RuntimeError("KRX 로그인 실패(KRX_ID/KRX_PW 확인)")
    params = {
        "bld": bld,
        "mktId": "ALL",
        "trdDd": date.today().strftime("%Y%m%d"),
        "money": "1",
        "csvxls_isNo": "false",
    }
    resp = krxs.post(KRX_JSON_URL, data=params)
    resp.raise_for_status()
    return resp.json()


def _default_fetch_admin() -> list[dict]:
    """실제 KRX 관리종목 현황 호출 → 정규화 행."""
    return parse_admin_issue(_fetch_krx_json(BLD_ADMIN_ISSUE))


def _default_fetch_halt() -> list[dict]:
    """실제 KRX 매매거래정지 현황 호출 → 정규화 행."""
    return parse_trading_halt(_fetch_krx_json(BLD_TRADING_HALT))


# ---------------------------------------------------------------------------
# ingest: 스냅샷 diff → 구간 upsert
# ---------------------------------------------------------------------------
def _apply_snapshot(conn, status_type: str, current_rows: list[dict], today: str, updated_at: str) -> dict:
    """한 status_type 의 현재 스냅샷을 열린 구간과 diff 해서 테이블에 반영한다.

    반환: {opened, released, continuing, open_total}.
    """
    # 현재 스냅샷(종목코드→행). 같은 코드 중복 시 마지막 행을 쓴다(KRX 목록은 종목당 1행이 정상).
    current_by_code = {r["stock_code"]: r for r in current_rows}
    current_codes = set(current_by_code)

    open_codes = {
        row["stock_code"]
        for row in conn.execute(
            "SELECT stock_code FROM kr_trading_status WHERE status_type=? AND end_date IS NULL",
            (status_type,),
        ).fetchall()
    }

    newly, released, continuing = diff_snapshot(open_codes, current_codes)

    # 신규지정: 새 열린 구간 개시. UNIQUE(stock_code,status_type,start_date) → 같은 날 재실행 멱등
    # (INSERT OR IGNORE: 이미 그 날짜 구간이 있으면 건너뜀).
    for code in sorted(newly):
        r = current_by_code[code]
        conn.execute(
            "INSERT OR IGNORE INTO kr_trading_status"
            "(stock_code, status_type, company_name, market, reason, start_date, end_date, "
            "krx_designated_date, updated_at) VALUES (?,?,?,?,?,?,NULL,?,?)",
            (code, status_type, r["company_name"], r["market"], r["reason"], today,
             r["krx_designated_date"], updated_at),
        )

    # 해제: 사라진 종목의 열린 구간에 end_date=관측일 기입.
    for code in sorted(released):
        conn.execute(
            "UPDATE kr_trading_status SET end_date=?, updated_at=? "
            "WHERE stock_code=? AND status_type=? AND end_date IS NULL",
            (today, updated_at, code, status_type),
        )

    # 유지: 사유/시장/원본지정일 최신화(열린 구간만). 값이 바뀌어도 구간은 유지.
    for code in sorted(continuing):
        r = current_by_code[code]
        conn.execute(
            "UPDATE kr_trading_status SET company_name=?, market=?, reason=?, "
            "krx_designated_date=?, updated_at=? "
            "WHERE stock_code=? AND status_type=? AND end_date IS NULL",
            (r["company_name"], r["market"], r["reason"], r["krx_designated_date"],
             updated_at, code, status_type),
        )

    open_total = conn.execute(
        "SELECT COUNT(*) FROM kr_trading_status WHERE status_type=? AND end_date IS NULL",
        (status_type,),
    ).fetchone()[0]
    return {
        "opened": len(newly),
        "released": len(released),
        "continuing": len(continuing),
        "open_total": open_total,
    }


def ingest_trading_status(
    db_path: str | None = None,
    fetch_admin_fn: Optional[Callable[[], list[dict]]] = None,
    fetch_halt_fn: Optional[Callable[[], list[dict]]] = None,
    today: str | None = None,
) -> dict:
    """KRX 관리종목/거래정지 '오늘 현재' 스냅샷을 받아 kr_trading_status 에 구간 반영한다.

    - fetch_admin_fn/fetch_halt_fn: () -> 정규화 행 리스트(DI, 기본=실제 KRX 호출). 테스트는
      mock 을 주입해 네트워크·로그인 없이 검증한다(라이브 KRX 로그인은 절대 호출 안 함).
    - today: 관측일 override(테스트용, 기본=오늘 YYYY-MM-DD). 매 실행의 start_date/end_date 기준.
    - 두 번 연속 같은 today 로 실행해도 신규/해제 0(멱등). 사라진 종목은 다음 실행에서 end_date 채워짐.

    반환: {"as_of": today, "admin": {opened,released,continuing,open_total}, "halt": {...}}.
    """
    fetch_admin_fn = fetch_admin_fn or _default_fetch_admin
    fetch_halt_fn = fetch_halt_fn or _default_fetch_halt
    today = today or date.today().isoformat()

    init_db(db_path)
    admin_rows = fetch_admin_fn()
    halt_rows = fetch_halt_fn()

    conn = connect(db_path)
    updated_at = datetime.now(timezone.utc).isoformat()
    try:
        admin_stats = _apply_snapshot(conn, STATUS_ADMIN, admin_rows, today, updated_at)
        halt_stats = _apply_snapshot(conn, STATUS_HALT, halt_rows, today, updated_at)
        conn.commit()
        return {"as_of": today, "admin": admin_stats, "halt": halt_stats}
    except Exception as exc:  # noqa: BLE001 — 수집 실패는 격리·보고(us_delisting 관례)
        log_ingest({"source": "kr_trading_status", "status": "fail", "error": str(exc)})
        send_slack_alert(f"[kr_trading_status] 수집 실패: {exc}")
        raise
    finally:
        conn.close()
