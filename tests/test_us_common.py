"""us_common.py 공용 유틸 테스트.

.omc/specs/brainstorming-us-market-data-plane.md Round 10(market_cap은
REAL/달러로 파싱해 저장) 검증. investing.com 스크리너가 내려주는
"2.32T"/"958.12B"/"4.64M" 같은 단위 축약 문자열을 달러 단위 float로 변환한다.
"""
from __future__ import annotations

from src.ingest.us_common import parse_market_cap


def test_parse_market_cap_trillions_suffix():
    assert parse_market_cap("2.32T") == 2.32e12


def test_parse_market_cap_billions_suffix():
    assert parse_market_cap("958.12B") == 958.12e9


def test_parse_market_cap_millions_suffix():
    assert parse_market_cap("4.64M") == 4.64e6


def test_parse_market_cap_plain_number_no_suffix():
    assert parse_market_cap("12345.67") == 12345.67


def test_parse_market_cap_missing_value_returns_none():
    assert parse_market_cap("-") is None
    assert parse_market_cap("") is None
    assert parse_market_cap(None) is None


def test_parse_market_cap_strips_leading_dollar_sign():
    # 실제 investing.com 스크리너 페이지(2026-07-12 curl 검증)의 표시값 형식: "$5.11T"
    assert parse_market_cap("$5.11T") == 5.11e12


def test_parse_market_cap_numeric_input_passes_through_as_float():
    # investing.com __NEXT_DATA__ JSON의 raw 필드는 이미 숫자(예: 5110000000000)라
    # 문자열 파싱 없이 그대로 float로 반환해야 한다.
    assert parse_market_cap(5110000000000) == 5110000000000.0
    assert parse_market_cap(958.12e9) == 958.12e9
