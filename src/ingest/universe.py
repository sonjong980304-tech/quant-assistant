"""유니버스 조회 + 상장폐지 추정 적재 (생존편향 제거).

- get_universe(scope): "dev"면 하드코딩 COMPANIES, "full"이면 pykrx로 코스피+코스닥 전체.
- ingest_delisting(): 과거 시점 상장종목과 현재 상장종목의 차집합을 상폐로 추정해 적재.

pykrx는 네트워크 의존이라 지연 import하고, 실패 시 dev로 폴백한다.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional, Tuple

from ..config import CONFIG
from ..db import connect, init_db
from ..version import today_str
from .companies import COMPANIES
from .robust import call_with_retry

# (종목코드, 회사명, 시장, 업종)
Company = Tuple[str, str, str, str]

MARKETS = ("KOSPI", "KOSDAQ")


def _fetch_market_tickers(stock, market: str, on_date: Optional[str] = None) -> List[str]:
    """해당 시장의 상장 종목코드 리스트 (재시도 적용)."""
    def _call():
        if on_date:
            return stock.get_market_ticker_list(on_date, market=market)
        return stock.get_market_ticker_list(market=market)

    result = call_with_retry(_call, label=f"ticker_list({market},{on_date or 'now'})")
    return list(result) if result else []


def get_universe(scope: Optional[str] = None) -> List[Company]:
    """수집 대상 유니버스를 (code, name, market, sector) 리스트로 반환.

    scope: None이면 CONFIG.target_scope 사용. "dev"면 하드코딩 50개사,
    "full"이면 pykrx로 코스피+코스닥 전체 종목. 업종은 모르면 "".
    네트워크 실패 시 dev로 폴백 + 경고.
    """
    scope = (scope or CONFIG.target_scope or "dev").lower()
    if scope != "full":
        return list(COMPANIES)

    # pykrx 전종목 목록 API가 차단된 환경이므로 DART corpCode를 1차 소스로 사용한다.
    # (pykrx 전종목 호출은 빈 응답이거나 종목별 이름 조회에서 행 상태로 멈추는 문제가 있어
    #  pykrx는 DART 실패 시의 폴백으로만 둔다.)
    universe: List[Company] = _dart_listed_universe()
    if universe:
        return universe

    try:
        from pykrx import stock  # 지연 import (네트워크 의존)
    except Exception as exc:  # noqa: BLE001
        print(f"[universe] pykrx import 실패 — dev로 폴백: {exc}")
        return list(COMPANIES)

    for market in MARKETS:
        codes = _fetch_market_tickers(stock, market)
        for code in codes:
            try:
                name = stock.get_market_ticker_name(code)
            except Exception:  # noqa: BLE001 — 단일 종목 실패 격리
                name = ""
            universe.append((code, name or "", market, ""))

    if not universe:
        print("[universe] full 조회 실패 — dev로 폴백")
        return list(COMPANIES)
    return universe


def _dart_listed_universe() -> List[Company]:
    """DART corpCode.xml에서 상장사(6자리 stock_code 보유) 전체. market/sector는 미상("").

    pykrx 전종목 목록 API가 차단된 경우의 폴백. 코스피+코스닥+코넥스 포함.
    """
    if not CONFIG.has_dart_key:
        return []
    import io
    import zipfile
    from xml.etree import ElementTree as ET

    from .robust import request_with_retry

    r = request_with_retry(
        "https://opendart.fss.or.kr/api/corpCode.xml",
        params={"crtfc_key": CONFIG.dart_api_key}, expect_json=False,
    )
    if r is None:
        return []
    try:
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        root = ET.fromstring(zf.read(zf.namelist()[0]))
    except Exception:  # noqa: BLE001
        return []
    out: List[Company] = []
    for el in root.iter("list"):
        code = (el.findtext("stock_code") or "").strip()
        name = (el.findtext("corp_name") or "").strip()
        if len(code) == 6 and code.isdigit():
            out.append((code, name, "", ""))
    return out


def ingest_delisting(db_path: Optional[str] = None, years_back: int = 10,
                     on: Optional[date] = None) -> dict:
    """상장폐지 추정 적재 (생존편향 제거).

    pykrx에 직접 상폐 목록 API가 없으므로, `years_back`년 전 시점 상장종목과
    현재 상장종목의 차집합을 상폐로 추정해 delisting 테이블에 적재한다.
    delisting_date는 정확히 모르므로 공란("")으로 둔다.

    반환: 리포트 dict.
    """
    on = on or date.today()
    past = on - timedelta(days=365 * years_back)
    past_str = today_str(past)
    now_str = today_str(on)

    try:
        from pykrx import stock  # 지연 import
    except Exception as exc:  # noqa: BLE001
        print(f"[delisting] pykrx import 실패 — 건너뜀: {exc}")
        return {"delisted": 0, "error": "pykrx_import_failed"}

    past_set: set = set()
    now_set: set = set()
    for market in MARKETS:
        past_set.update(_fetch_market_tickers(stock, market, on_date=past_str))
        now_set.update(_fetch_market_tickers(stock, market, on_date=now_str))

    if not past_set:
        print("[delisting] 과거 시점 조회 실패/0건 — 건너뜀")
        return {"delisted": 0, "past_count": 0, "now_count": len(now_set)}

    delisted = sorted(past_set - now_set)
    init_db(db_path)
    conn = connect(db_path)
    try:
        n = 0
        for code in delisted:
            try:
                name = stock.get_market_ticker_name(code)
            except Exception:  # noqa: BLE001
                name = ""
            conn.execute(
                "INSERT OR REPLACE INTO delisting(stock_code, name, delisting_date) VALUES(?,?,?)",
                (code, name or "", ""),
            )
            n += 1
        conn.commit()
        return {
            "delisted": n,
            "past_count": len(past_set),
            "now_count": len(now_set),
            "past_date": past_str,
            "now_date": now_str,
        }
    finally:
        conn.close()
