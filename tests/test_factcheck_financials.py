"""재무제표 사실확인(factcheck) run_financials_check() 단위 테스트 (US-3, TDD).

.omc/specs/brainstorming-factcheck-eval.md AC1 참고: 시가총액 상위 종목의 재무 지표
질의 응답값이 OpenDART API 원문 재조회값과 ±1% 이내인지 자동 판정한다.

대상: src/eval/factcheck/financials.py
    run_financials_check(items, llm_fn, dart_api_key) -> list[dict]

실제 네트워크를 타지 않도록 DART 재조회 함수(_fetch_dart_original)와 llm_fn을
모두 unittest.mock으로 모킹한다. 세 가지 핵심 케이스만 검증한다:
  (i)   오차 1% 이내 -> pass=True
  (ii)  오차 1% 초과 -> pass=False
  (iii) DART 호출 예외(한도초과 등) -> pass=None, note="측정불가" (예외 전파 없음)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.eval.factcheck.financials import run_financials_check


class TestRunFinancialsCheck:
    def test_within_tolerance_passes(self):
        items = [{"stock_code": "005930", "name": "삼성전자", "metric": "operating_profit"}]
        llm_fn = MagicMock(return_value=100_500.0)  # dart원문 100000 대비 오차 0.5%

        with patch(
            "src.eval.factcheck.financials._fetch_dart_original", return_value=100_000.0
        ) as mock_dart:
            result = run_financials_check(items, llm_fn, dart_api_key="dummy-key")

        assert result == [
            {
                "stock_code": "005930",
                "expected": 100_000.0,
                "actual": 100_500.0,
                "pass": True,
                "note": "",
            }
        ]
        mock_dart.assert_called_once_with("dummy-key", "005930", "operating_profit")
        llm_fn.assert_called_once_with(items[0])

    def test_outside_tolerance_fails(self):
        items = [{"stock_code": "000660", "name": "SK하이닉스", "metric": "operating_profit"}]
        llm_fn = MagicMock(return_value=102_000.0)  # dart원문 100000 대비 오차 2%

        with patch(
            "src.eval.factcheck.financials._fetch_dart_original", return_value=100_000.0
        ):
            result = run_financials_check(items, llm_fn, dart_api_key="dummy-key")

        assert result == [
            {
                "stock_code": "000660",
                "expected": 100_000.0,
                "actual": 102_000.0,
                "pass": False,
                "note": "",
            }
        ]

    def test_dart_call_failure_marks_unmeasurable_without_raising(self):
        items = [{"stock_code": "005380", "name": "현대차", "metric": "operating_profit"}]
        llm_fn = MagicMock(return_value=50_000.0)

        with patch(
            "src.eval.factcheck.financials._fetch_dart_original",
            side_effect=RuntimeError("DART 일일 사용한도 초과(020)"),
        ):
            # 예외가 run_financials_check 밖으로 전파되지 않아야 한다.
            result = run_financials_check(items, llm_fn, dart_api_key="dummy-key")

        assert result == [
            {
                "stock_code": "005380",
                "expected": None,
                "actual": 50_000.0,
                "pass": None,
                "note": "측정불가",
            }
        ]

    def test_multiple_items_processed_independently(self):
        items = [
            {"stock_code": "005930", "name": "삼성전자", "metric": "operating_profit"},
            {"stock_code": "000660", "name": "SK하이닉스", "metric": "operating_profit"},
        ]
        llm_fn = MagicMock(side_effect=[100_500.0, 999_999.0])

        with patch(
            "src.eval.factcheck.financials._fetch_dart_original",
            side_effect=[100_000.0, RuntimeError("한도초과")],
        ):
            result = run_financials_check(items, llm_fn, dart_api_key="dummy-key")

        assert len(result) == 2
        assert result[0]["pass"] is True
        assert result[1]["pass"] is None
        assert result[1]["note"] == "측정불가"
