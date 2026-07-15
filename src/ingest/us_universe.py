"""US 티커+섹터 유니버스 — NASDAQ Screener API 파서 + 오케스트레이터.

2026-07-12 소스 전환 배경: investing.com Stock Screener는 게스트 세션에서 2페이지
이상이 InvestingPro 유료 게이트로 막혀있음을 Playwright 브라우저 레벨(실제 '다음'
클릭이 유발하는 POST https://www.investing.com/pro/_/screener-v2/query 를 직접
캡처)로 확정했고, 무료 계정으로 로그인한 상태에서도 동일하게 totalItems:0이 나옴을
재확인했다(로그인 여부가 아니라 구독 여부가 기준). 대신 NASDAQ 공식 스크리너 API
(`api.nasdaq.com/api/screener/stocks`)는 인증/로그인 없이 curl로 즉시 전체 결과를
반환함을 실측했다: NASDAQ 4,110 + NYSE 2,718 + AMEX 295 = 7,123건, `tableonly=true`
파라미터를 주면 `limit`을 무시하고 거래소당 1회 호출로 전량을 반환한다(페이지네이션
불필요). sector도 12개 표준 카테고리(Technology/Health Care/Finance 등)로
investing.com 원문 taxonomy보다 다루기 쉽다.

거래소(NASDAQ/NYSE/AMEX)를 순회하며 각각 독립 호출한다 — 페이지 기반과 달리 거래소
간 순서/데이터 의존성이 없으므로, 한 거래소 실패는 격리하고 나머지는 계속 진행한다
(yfinance 개별종목 실패 격리와 동일 원칙). fetch_exchange 주입으로 네트워크와
분리해 순회/병합/upsert 로직만 단위 테스트한다.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Optional

from ..db import connect, init_db
from .notify import send_slack_alert
from .robust import log_ingest
from .us_common import parse_market_cap

_ALLOWED_EXCHANGES = {"NASDAQ", "NYSE", "NYSE Amex"}
_EXCHANGES = {"nasdaq": "NASDAQ", "nyse": "NYSE", "amex": "NYSE Amex"}
_SCREENER_URL = (
    "https://api.nasdaq.com/api/screener/stocks"
    "?tableonly=true&limit=1&offset=0&download=true&exchange={exchange}"
)
_REQUEST_DELAY_SEC = 2.0  # 요청 간 최소 2초 딜레이(스펙 AC11)


def parse_nasdaq_rows(raw_rows: list[dict], exchange_label: str) -> list[dict]:
    """NASDAQ 스크리너 API의 원시 rows를 {symbol,name,exchange,sector,market_cap} 행으로 변환.

    exchange_label은 호출한 거래소(어느 파라미터로 요청했는지)를 그대로 붙인다 —
    API 응답 자체에는 행별 exchange 필드가 없다(2026-07-12 실측 확인).
    market_cap은 문자열("42169397.00")을 그대로 넘긴다(parse_market_cap이 숫자
    문자열도 처리하므로 정규화 단계에서 한 번에 파싱한다).
    """
    return [
        {
            "symbol": r.get("symbol"),
            "name": r.get("name"),
            "exchange": exchange_label,
            "sector": r.get("sector"),
            "market_cap": r.get("marketCap"),
        }
        for r in raw_rows
    ]


def normalize_universe_rows(raw_rows: list[dict]) -> list[dict]:
    """여러 거래소 원시 행을 병합해 거래소 필터+market_cap 파싱+Symbol 중복제거."""
    seen: set[str] = set()
    result: list[dict] = []
    for row in raw_rows:
        symbol = (row.get("symbol") or "").strip()
        name = (row.get("name") or "").strip()
        exchange = (row.get("exchange") or "").strip()
        if not symbol or not name:
            continue
        if exchange not in _ALLOWED_EXCHANGES:
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        result.append({
            "symbol": symbol,
            "name": name,
            "exchange": exchange,
            "sector": (row.get("sector") or "").strip() or None,
            "market_cap": parse_market_cap(row.get("market_cap")),
        })
    return result


def _fetch_exchange(exchange_key: str) -> list[dict]:
    """NASDAQ 공식 스크리너 API를 호출해 원시 rows를 가져온다(인증 불필요).

    2026-07-12 curl 실측: 로그인/구독 없이 접근 가능, tableonly=true면 limit을
    무시하고 거래소당 전체(NASDAQ 4110/NYSE 2718/AMEX 295건)를 한 번에 반환한다.
    requests는 이 API에서 Cloudflare 403에 걸리지 않음을 확인했다(investing.com과
    달리 브라우저가 필요 없다).
    """
    import requests  # 지연 import

    url = _SCREENER_URL.format(exchange=exchange_key)
    resp = requests.get(
        url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=30
    )
    resp.raise_for_status()
    return resp.json()["data"]["rows"]


def ingest_us_universe(
    db_path: str | None = None,
    fetch_exchange: Optional[Callable[[str], list[dict]]] = None,
) -> dict:
    """3개 거래소(NASDAQ/NYSE/AMEX)를 순회하며 종목을 모아 us_company에 upsert.

    fetch_exchange(exchange_key) -> raw_rows(list[dict]) 를 주입 가능하게 해
    네트워크 없이 단위 테스트한다(us_prices.py의 fetch_history 주입 패턴과 동일).
    한 거래소 fetch 실패는 격리해 로그+Slack 알림 후 나머지 거래소는 계속 진행한다
    (페이지 기반과 달리 거래소 간 순서 의존성이 없으므로 중단이 아니라 격리가 맞다).
    """
    fetch_exchange = fetch_exchange or _fetch_exchange
    init_db(db_path)
    conn = connect(db_path)
    updated_at = datetime.now(timezone.utc).isoformat()
    try:
        raw_rows: list[dict] = []
        fetched_exchanges: list[str] = []
        failed_exchanges: list[str] = []
        for i, (key, label) in enumerate(_EXCHANGES.items()):
            if i > 0:
                time.sleep(_REQUEST_DELAY_SEC)
            try:
                raw = fetch_exchange(key)
                page_rows = parse_nasdaq_rows(raw, label)
            except Exception as exc:  # noqa: BLE001 — 거래소 실패를 격리하고 나머지는 계속
                failed_exchanges.append(key)
                log_ingest({"source": "us_universe", "exchange": key, "status": "fail", "error": str(exc)})
                send_slack_alert(f"[us_universe] {key} 거래소 수집 실패: {exc}")
                continue
            fetched_exchanges.append(key)
            raw_rows.extend(page_rows)

        companies = normalize_universe_rows(raw_rows)
        for c in companies:
            conn.execute(
                "INSERT OR REPLACE INTO us_company"
                "(stock_code, name, exchange, sector, market_cap, updated_at) "
                "VALUES (?,?,?,?,?,?)",
                (c["symbol"], c["name"], c["exchange"], c["sector"], c["market_cap"], updated_at),
            )
        conn.commit()
        return {"exchanges": fetched_exchanges, "failed_exchanges": failed_exchanges, "companies": len(companies)}
    finally:
        conn.close()
