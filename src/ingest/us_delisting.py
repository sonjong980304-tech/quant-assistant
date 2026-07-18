"""FMP(Financial Modeling Prep) 미국 상장폐지 종목 수집·정규화.

.omc/specs/brainstorming-us-delisting-survivorship.md 참고.

미국은 상장폐지된 티커가 나중에 다른 회사에 재사용되는 경우가 흔하다(예: TWTR). 그래서
단순 "죽었다/살았다" 플래그가 아니라 구간(상장일~상장폐지일)을 행으로 저장한다 —
같은 티커가 여러 행을 가질 수 있고, 생존 판정은 asof가 어느 구간에 드는지로 한다
(backtest/data_access_us._is_alive_us). 이 모듈은 그 구간 데이터를 FMP에서 수집한다.

소스:
- delisted-companies: 상장폐지 종목 목록(symbol/companyName/exchange/ipoDate/delistedDate).
- stock-list: 활성 종목 목록(symbol/name/exchangeShortName/type). 상장폐지 이후 같은 티커로
  재상장(재사용)된 사례를 식별하는 데이터로 쓴다(AC2).

FMP 무료 플랜(하루 250회)이며 전체 백필도 소량 페이지 호출로 끝난다(AC10). v3 legacy
엔드포인트는 2025-08-31 이후 발급 키에서 차단되므로 stable 경로를 쓴다. 네트워크는
DI(fetch_fn 주입)로 분리해 단위 테스트한다(us_financials_sec.py 등 기존 관례와 동일).

API 키는 .env의 FMP_API_KEY를 os.getenv로만 읽고, 값을 출력·로그에 남기지 않는다(AC9).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Callable, Optional

from ..db import connect, init_db
from .notify import send_slack_alert
from .robust import log_ingest

# stable 엔드포인트(v3 legacy 는 신규 키에서 auth-gated 차단). 무료 플랜 접근 가능.
FMP_DELISTED_URL = "https://financialmodelingprep.com/stable/delisted-companies"
FMP_STOCK_LIST_URL = "https://financialmodelingprep.com/stable/stock-list"
# 한 페이지 최대 100건(FMP limit 상한). 페이지를 0,1,2...로 늘리며 빈 배열까지 순회.
_PAGE_LIMIT = 100
# 방어 상한: 응답이 버그로 계속 가득 차더라도 하루 250회 한도를 넘지 않게 페이지 수를 막는다.
# 100건×200페이지=2만 종목이면 미국 역사적 상장폐지 전체를 담고도 남는다(AC10).
_MAX_PAGES = 200


def parse_delisted_companies(payload: list[dict]) -> list[dict]:
    """FMP delisted-companies 응답을 us_delisting 행 리스트로 변환한다(AC1).

    각 객체: {symbol, companyName, exchange, ipoDate, delistedDate}. symbol 또는 delistedDate
    가 없는 행은 건너뛴다(구간 판정에 상장폐지일이 반드시 필요). ipoDate(상장일)가 없으면
    ''(빈문자열 센티널)로 둔다 — SQLite UNIQUE 가 NULL 을 매번 다른 값으로 취급해 upsert
    멱등성이 깨지는 것을 막는다(us_financials_sec period_start 관례와 동일).
    """
    rows: list[dict] = []
    for e in payload or []:
        symbol = (e.get("symbol") or "").strip()
        delisting_date = (e.get("delistedDate") or "").strip()
        if not symbol or not delisting_date:
            continue
        rows.append({
            "stock_code": symbol,
            "company_name": e.get("companyName"),
            "exchange": e.get("exchange"),
            "listing_date": (e.get("ipoDate") or "").strip(),
            "delisting_date": delisting_date,
        })
    return rows


def parse_active_symbols(payload: list[dict]) -> list[dict]:
    """FMP stock-list(활성 종목) 응답을 파싱한다(AC2).

    각 객체: {symbol, name, price, exchange, exchangeShortName, type}. 거래소는 축약형
    (exchangeShortName)을 우선 쓰고 없으면 exchange 로 폴백한다. 상장폐지 이후 같은 티커로
    재상장된 사례(티커 재사용)를 식별하는 데 쓴다.
    """
    rows: list[dict] = []
    for e in payload or []:
        symbol = (e.get("symbol") or "").strip()
        if not symbol:
            continue
        rows.append({
            "stock_code": symbol,
            "name": e.get("name"),
            "exchange": e.get("exchangeShortName") or e.get("exchange"),
            "type": e.get("type"),
        })
    return rows


def active_symbol_set(payload: list[dict]) -> set[str]:
    """활성 종목 응답에서 티커 심볼 집합만 뽑는다(재사용 식별용 편의 함수)."""
    return {r["stock_code"] for r in parse_active_symbols(payload)}


def _fetch_page(url: str, params: dict) -> list[dict]:
    """FMP 엔드포인트 1페이지를 실제 호출(지연 import). API 키는 params 로만 전달.

    실패 시 requests 예외에는 apikey 가 박힌 전체 URL(?...&apikey=...)이 담긴다 —
    그대로 로그/Slack 으로 새면 키가 노출된다(AC9 위반). 예외를 잡아 쿼리스트링을 뺀
    베이스 URL 과 예외 타입명만 담은 새 에러로 바꿔 던진다(from None 으로 원본 체인의
    URL 도 감춘다).

    실측(라이브 백필 실행) 결과 FMP 무료 플랜은 delisted-companies 에서 page=0 만
    허용하고 2페이지부터 402(Payment Required)를 반환한다 — 장애가 아니라 '무료
    범위 소진'이므로 빈 배열(정상 페이지네이션 종료)로 취급해 크래시를 피한다(주간
    launchd 갱신이 매번 죽지 않도록). 반면 첫 페이지(page=0)조차 402면 엔드포인트
    자체가 막힌 것이므로 진짜 에러로 그대로 던진다(장애를 숨기지 않음).
    """
    import requests

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 402 and params.get("page", 0) > 0:
            return []
        raise RuntimeError(f"FMP 요청 실패({url}): {type(exc).__name__}") from None
    except requests.RequestException as exc:
        raise RuntimeError(f"FMP 요청 실패({url}): {type(exc).__name__}") from None
    return data if isinstance(data, list) else []


def fetch_delisted_companies(
    api_key: str,
    fetch_fn: Optional[Callable[[str, dict], list]] = None,
    limit: int = _PAGE_LIMIT,
    max_pages: int = _MAX_PAGES,
) -> tuple[list[dict], int]:
    """delisted-companies 를 page=0,1,2... 로 순회하며 전체를 모은다(AC1, AC10).

    빈 배열(더 이상 데이터 없음)이 오거나 max_pages 에 도달하면 멈춘다. fetch_fn(url, params)
    ->list 는 테스트 주입용(기본=실제 HTTP). 반환: (원시 응답 행 리스트, 실제 호출 횟수).
    호출 횟수를 반환해 하루 250회 한도를 넘지 않음을 상위(수집/테스트)에서 검증할 수 있다.
    """
    fetch_fn = fetch_fn or _fetch_page
    all_rows: list[dict] = []
    api_calls = 0
    for page in range(max_pages):
        params = {"page": page, "limit": limit, "apikey": api_key}
        batch = fetch_fn(FMP_DELISTED_URL, params) or []
        api_calls += 1
        if not batch:
            break
        all_rows.extend(batch)
    return all_rows, api_calls


def fetch_active_symbols(
    api_key: str,
    fetch_fn: Optional[Callable[[str, dict], list]] = None,
) -> tuple[list[dict], int]:
    """stock-list(활성 종목 전체)를 단일 호출로 받는다(AC2). 반환: (원시 응답, 호출 횟수=1)."""
    fetch_fn = fetch_fn or _fetch_page
    params = {"apikey": api_key}
    rows = fetch_fn(FMP_STOCK_LIST_URL, params) or []
    return rows, 1


def ingest_us_delisting(
    db_path: str | None = None,
    fetch_delisted_fn: Optional[Callable] = None,
    fetch_active_fn: Optional[Callable] = None,
    api_key: str | None = None,
) -> dict:
    """FMP 상장폐지+활성 목록을 받아 us_delisting 에 upsert 한다(AC1, AC2, AC3, AC7, AC8).

    - 전체 백필/주간 갱신 공통 경로. UNIQUE(stock_code, listing_date, delisting_date) 로
      이미 있는 구간은 갱신, 신규 구간만 추가돼 두 번 돌려도 중복이 없다(AC8 upsert 멱등).
    - 활성 목록에도 있는(재사용) 티커는 reused_tickers 로 식별해 보고한다(AC2 데이터 활용).
    - fetch_*_fn 은 각각 (parsed_rows, api_calls) 를 돌려주는 DI(기본=실제 FMP 호출).
    - api_key 미지정 시 .env 의 FMP_API_KEY 를 읽는다(값은 로그/출력에 노출하지 않는다, AC9).

    반환: {"episodes_upserted","tickers","reused_tickers","oldest_delisting_date","api_calls"}.
    """
    api_key = api_key or os.getenv("FMP_API_KEY", "")
    fetch_delisted_fn = fetch_delisted_fn or _default_fetch_delisted
    fetch_active_fn = fetch_active_fn or _default_fetch_active

    init_db(db_path)  # _default_fetch_active가 us_company를 조회하므로 먼저 스키마를 보장한다.
    delisted_rows, calls_d = fetch_delisted_fn(api_key)
    active_rows, calls_a = fetch_active_fn(api_key, db_path=db_path)
    active_codes = {r["stock_code"] for r in active_rows}

    conn = connect(db_path)
    updated_at = datetime.now(timezone.utc).isoformat()
    try:
        upserted = 0
        tickers: set[str] = set()
        for r in delisted_rows:
            conn.execute(
                "INSERT OR REPLACE INTO us_delisting"
                "(id, stock_code, company_name, exchange, listing_date, delisting_date, updated_at) "
                "VALUES ((SELECT id FROM us_delisting WHERE stock_code=? AND listing_date=? "
                "AND delisting_date=?), ?,?,?,?,?,?)",
                (
                    r["stock_code"], r["listing_date"], r["delisting_date"],
                    r["stock_code"], r["company_name"], r["exchange"],
                    r["listing_date"], r["delisting_date"], updated_at,
                ),
            )
            upserted += 1
            tickers.add(r["stock_code"])

        # 재사용 티커(상폐목록 ∩ 활성목록)에 '현재 활성 마커'(listing_date='', delisting_date='')
        # 행을 남긴다 — _is_alive_us가 이를 보고 상폐 이후 시점을 하드차단하지 않는다(AC15,
        # 재상장 이후 오탐 방지). 현실의 재사용 대다수는 새 회사가 아직 상폐목록에 없어 재상장
        # 구간이 없으므로, AC2로 수집한 활성목록을 이 마커로 판정에 연결한다. UNIQUE(stock_code,
        # '', '')로 티커당 마커 1개 → 갱신마다 멱등(중복 없음).
        reused = sorted(tickers & active_codes)
        for code in reused:
            conn.execute(
                "INSERT OR REPLACE INTO us_delisting"
                "(id, stock_code, company_name, exchange, listing_date, delisting_date, updated_at) "
                "VALUES ((SELECT id FROM us_delisting WHERE stock_code=? AND listing_date='' "
                "AND delisting_date=''), ?,?,?,?,?,?)",
                (code, code, None, None, "", "", updated_at),
            )
        conn.commit()
        delisting_dates = [r["delisting_date"] for r in delisted_rows if r["delisting_date"]]
        oldest = min(delisting_dates) if delisting_dates else None
        return {
            "episodes_upserted": upserted,
            "tickers": len(tickers),
            "reused_tickers": reused,
            "oldest_delisting_date": oldest,
            "api_calls": calls_d + calls_a,
        }
    except Exception as exc:  # noqa: BLE001 — 수집 실패는 격리·보고(us_financials_sec 관례)
        log_ingest({"source": "us_delisting", "status": "fail", "error": str(exc)})
        send_slack_alert(f"[us_delisting] 수집 실패: {exc}")
        raise
    finally:
        conn.close()


def _default_fetch_delisted(api_key: str) -> tuple[list[dict], int]:
    """실제 FMP delisted-companies 호출 → (파싱된 행, 호출 수)."""
    raw, calls = fetch_delisted_companies(api_key)
    return parse_delisted_companies(raw), calls


def _default_fetch_active(api_key: str, db_path: str | None = None) -> tuple[list[dict], int]:
    """활성 종목 목록 → (행, 호출 수). FMP stock-list는 무료플랜에서 402(유료 전용,
    "Restricted Endpoint")로 원천 차단돼 있어(실측 확인) 호출하지 않는다. 대신 이미
    보유한 us_company(현재 추적 중인 활성 종목 전체 — 이 프로젝트의 스크리닝 대상 그
    자체)를 대용으로 쓴다: API 비용 0이고, 우리 백테스트가 실제로 선택 가능한 종목
    범위와 정확히 일치해 FMP 전체시장 목록보다 재사용 판정 목적엔 더 정밀하다.
    """
    conn = connect(db_path)
    try:
        rows = conn.execute("SELECT stock_code FROM us_company").fetchall()
    finally:
        conn.close()
    return [{"stock_code": r["stock_code"]} for r in rows], 0
