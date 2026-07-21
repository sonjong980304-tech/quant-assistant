"""run_price_check() "오늘 종가" 실사실 대조 단위 테스트 (US-4, TDD).

.omc/specs/brainstorming-factcheck-eval.md AC2 참고: 시가총액 상위 종목의
"오늘 종가" 질의 응답값이 네이버 fchart API 재조회값과 완전일치(tolerance.exact_match)
하는지 자동 판정한다.

대상: src/eval/factcheck/price.py
    run_price_check(items: list[dict], llm_fn) -> list[dict]
    - items: [{"stock_code":..., "name":...}]
    - 각 item마다 (a) llm_fn(question) 으로 시스템 응답값을 얻고,
      (b) src.ingest.naver_prices.fetch_daily_prices 를 그 자리에서 재호출해 정답을 얻은 뒤,
      (c) tolerance.exact_match 로 비교한다.
    - 네이버 재호출이 실패(예외/빈 응답)하면 pass=None, note="측정불가"로 기록하고
      예외를 밖으로 전파하지 않는다.

네이버 호출은 src.ingest.naver_prices.fetch_daily_prices 를 price 모듈 네임스페이스에서
monkeypatch 하여 대체한다(실제 네트워크 호출 없음). llm_fn 도 테스트마다 원하는 값을
반환하는 간단한 함수로 대체한다.
"""
from __future__ import annotations

import src.eval.factcheck.price as price_mod
from src.eval.factcheck.price import run_price_check


def test_run_price_check_pass_when_llm_and_naver_agree(monkeypatch):
    def fake_fetch_daily_prices(stock_code, count=1):
        assert stock_code == "005930"
        return [{"date": "2026-07-20", "open": 70000, "high": 70500, "low": 69800, "close": 70200, "volume": 100}]

    monkeypatch.setattr(price_mod, "fetch_daily_prices", fake_fetch_daily_prices)

    def llm_fn(question: str):
        assert "005930" in question
        return 70200

    items = [{"stock_code": "005930", "name": "삼성전자"}]
    result = run_price_check(items, llm_fn)

    assert result == [
        {"stock_code": "005930", "expected": 70200, "actual": 70200, "pass": True, "note": ""}
    ]


def test_run_price_check_fail_when_llm_and_naver_disagree(monkeypatch):
    def fake_fetch_daily_prices(stock_code, count=1):
        return [{"date": "2026-07-20", "open": 70000, "high": 70500, "low": 69800, "close": 70200, "volume": 100}]

    monkeypatch.setattr(price_mod, "fetch_daily_prices", fake_fetch_daily_prices)

    def llm_fn(question: str):
        return 71000  # 시스템이 틀린 값을 응답

    items = [{"stock_code": "005930", "name": "삼성전자"}]
    result = run_price_check(items, llm_fn)

    assert result == [
        {"stock_code": "005930", "expected": 70200, "actual": 71000, "pass": False, "note": ""}
    ]


def test_run_price_check_marks_unmeasurable_when_naver_call_fails(monkeypatch):
    def fake_fetch_daily_prices(stock_code, count=1):
        raise ConnectionError("네이버 fchart 호출 실패")

    monkeypatch.setattr(price_mod, "fetch_daily_prices", fake_fetch_daily_prices)

    def llm_fn(question: str):
        return 70200

    items = [{"stock_code": "005930", "name": "삼성전자"}]
    # 예외가 밖으로 전파되지 않아야 한다.
    result = run_price_check(items, llm_fn)

    assert result == [
        {"stock_code": "005930", "expected": None, "actual": 70200, "pass": None, "note": "측정불가"}
    ]


def test_run_price_check_marks_unmeasurable_when_naver_returns_empty(monkeypatch):
    def fake_fetch_daily_prices(stock_code, count=1):
        return []

    monkeypatch.setattr(price_mod, "fetch_daily_prices", fake_fetch_daily_prices)

    def llm_fn(question: str):
        return 70200

    items = [{"stock_code": "005930", "name": "삼성전자"}]
    result = run_price_check(items, llm_fn)

    assert result[0]["pass"] is None
    assert result[0]["note"] == "측정불가"


def test_run_price_check_processes_multiple_items_independently(monkeypatch):
    fetch_calls: list[str] = []

    def fake_fetch_daily_prices(stock_code, count=1):
        fetch_calls.append(stock_code)
        if stock_code == "000660":
            raise ConnectionError("타임아웃")
        return [{"date": "2026-07-20", "open": 1, "high": 1, "low": 1, "close": 100, "volume": 1}]

    monkeypatch.setattr(price_mod, "fetch_daily_prices", fake_fetch_daily_prices)

    def llm_fn(question: str):
        return 100

    items = [
        {"stock_code": "005930", "name": "삼성전자"},
        {"stock_code": "000660", "name": "SK하이닉스"},
    ]
    result = run_price_check(items, llm_fn)

    assert fetch_calls == ["005930", "000660"]
    assert result[0]["pass"] is True
    assert result[1]["pass"] is None
    assert result[1]["note"] == "측정불가"
