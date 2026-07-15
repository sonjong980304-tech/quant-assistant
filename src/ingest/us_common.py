"""미국 데이터 플레인 공용 유틸.

us_universe/us_prices/us_financials 세 모듈이 공유하는 파싱 로직을 모은다.
"""
from __future__ import annotations

_SUFFIX_MULTIPLIER = {
    "T": 1e12,
    "B": 1e9,
    "M": 1e6,
    "K": 1e3,
}


def parse_market_cap(raw: str | int | float | None) -> float | None:
    """investing.com 시가총액 값을 달러 float로 변환.

    "2.32T"/"958.12B"/"4.64M"/"$5.11T" 같은 축약 문자열은 단위 접미사를 곱해
    파싱한다. __NEXT_DATA__ JSON의 raw 필드처럼 이미 숫자인 경우는 그대로
    float로 통과시킨다(재파싱 불필요).
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)

    text = raw.strip().lstrip("$")
    if not text or text == "-":
        return None

    suffix = text[-1].upper()
    multiplier = _SUFFIX_MULTIPLIER.get(suffix)
    if multiplier is not None:
        return float(text[:-1]) * multiplier
    return float(text)
