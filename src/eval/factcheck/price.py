"""factcheck eval "오늘 종가" 실사실 대조 (US-4).

.omc/specs/brainstorming-factcheck-eval.md AC2 참고: 시가총액 상위 종목의 "오늘 종가"
질의 응답값이 네이버 fchart API 재조회값과 완전일치하는지 자동 판정한다(오차 허용 없음 —
같은 출처를 재조회하므로 다를 이유가 없다는 전제).

네이버 재조회는 새로 발명하지 않고 src.ingest.naver_prices.fetch_daily_prices 를 그대로
재사용한다(fchart XML 파싱까지 이미 검증된 기존 로직).
"""
from __future__ import annotations

from typing import Callable

from ...ingest.naver_prices import fetch_daily_prices
from .tolerance import exact_match

_UNMEASURABLE_NOTE = "측정불가"


def _price_question(item: dict) -> str:
    name = item.get("name", item["stock_code"])
    return f"{name}({item['stock_code']})의 오늘 종가는 얼마인가요?"


def run_price_check(items: list[dict], llm_fn: Callable[[str], object]) -> list[dict]:
    """각 item에 대해 시스템 응답값(오늘 종가)을 네이버 fchart 재조회값과 대조한다.

    llm_fn: 질문 문자열을 받아 시스템이 답한 종가 값(숫자, 이미 파싱된 값)을 반환하는 함수.
    네이버 재조회가 실패(예외 또는 빈 응답)하면 예외를 밖으로 전파하지 않고 해당 item을
    pass=None, note="측정불가"로 기록한다.

    반환: [{"stock_code":..., "expected":..., "actual":..., "pass":bool|None, "note":str}]
    """
    results: list[dict] = []
    for item in items:
        stock_code = item["stock_code"]
        actual = llm_fn(_price_question(item))

        try:
            rows = fetch_daily_prices(stock_code, count=1)
            if not rows:
                raise ValueError("빈 응답(필드 누락)")
            expected = rows[0]["close"]
        except Exception:  # noqa: BLE001 — 네이버 재조회 실패는 측정불가로 흡수(AC2)
            results.append(
                {
                    "stock_code": stock_code,
                    "expected": None,
                    "actual": actual,
                    "pass": None,
                    "note": _UNMEASURABLE_NOTE,
                }
            )
            continue

        results.append(
            {
                "stock_code": stock_code,
                "expected": expected,
                "actual": actual,
                "pass": exact_match(expected, actual),
                "note": "",
            }
        )
    return results
