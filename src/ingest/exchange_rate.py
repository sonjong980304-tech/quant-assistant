"""USD/KRW 환율 — 네이버 증권 실시간 조회 + 당일 캐시.

.omc/specs/brainstorming-us-nl-sql-integration.md 참고. 한미 종목 절대금액(시가총액/
매출 등) 한미 비교 질의에서만 예외적으로 쓴다 — 단일시장 질의는 환율을 절대 적용하지
않는다(원칙, Round3). 하루 1회만 크롤링해 재사용한다(price_live.py의 당일 캐시 패턴과
동일 — ingest_meta를 캐시 저장소로 재사용해 새 테이블을 만들지 않는다).
"""
from __future__ import annotations

import re
from datetime import date

from ..db import get_meta, set_meta

_NAVER_URL = (
    "https://m.stock.naver.com/front-api/marketIndex/prices"
    "?category=exchange&reutersCode=FX_USDKRW"
)

# 절대금액(원/달러 혼동 위험) 지표만 감지 — PER/ROE 등 통화 무관 비율은 대상 아님(Round3/8).
_ABSOLUTE_AMOUNT_HINT = re.compile(
    r"시가총액|시총|매출액|매출|영업이익|순이익|당기순이익|총자산|자산총계|부채총계|자본총계"
)


def needs_exchange_rate(question: str) -> bool:
    """질문이 절대금액(원/달러 혼동 위험) 지표를 묻는지 판별한다.

    PER/PBR/ROE 등 통화 무관 비율 지표는 대상이 아니다(단일시장 질의에도 걸리지만,
    환율 정보를 참고용으로 덧붙이는 것 자체는 KR 전용 질문에도 무해하다 — LLM이
    필요 없으면 무시하므로, "한미 비교인지"를 정교하게 판별하기보다 절대금액 여부만 본다).
    """
    return bool(_ABSOLUTE_AMOUNT_HINT.search(question or ""))


def parse_naver_rate_response(data: dict) -> float:
    """네이버 증권 marketIndex/prices 응답에서 최신 원/달러 종가를 뽑는다.

    result[0]이 가장 최근 거래일(내림차순 정렬), closePrice는 쉼표 포함 문자열(예: "1,503.40").
    """
    close = data["result"][0]["closePrice"]
    return float(close.replace(",", ""))


def _fetch_usdkrw_rate() -> float:
    """네이버 증권에서 최신 원/달러 환율을 가져온다(2026-07-12 실측 확인된 엔드포인트)."""
    import requests  # 지연 import — 이 프로젝트의 기존 네트워크 호출 관례와 동일

    resp = requests.get(_NAVER_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    resp.raise_for_status()
    return parse_naver_rate_response(resp.json())


def get_usdkrw_rate(conn, on: date | None = None, fetch_fn=None) -> float:
    """오늘자 원/달러 환율을 반환한다(당일 캐시, 없으면 크롤링 후 저장).

    fetch_fn()->float 주입 가능(테스트용, 기본은 _fetch_usdkrw_rate).
    price_live.py의 "_has_today" 당일 캐시 패턴과 동일하게, ingest_meta에
    usdkrw_rate/usdkrw_rate_date 두 키로 저장해 같은 날 재요청 시 재크롤링하지 않는다.
    """
    fetch_fn = fetch_fn or _fetch_usdkrw_rate
    on = on or date.today()
    today = on.strftime("%Y-%m-%d")
    if get_meta(conn, "usdkrw_rate_date") == today:
        cached = get_meta(conn, "usdkrw_rate")
        if cached:
            return float(cached)
    rate = fetch_fn()
    set_meta(conn, "usdkrw_rate", str(rate))
    set_meta(conn, "usdkrw_rate_date", today)
    return rate
