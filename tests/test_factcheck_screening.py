"""top-N 스크리닝 계산값 재검증(factcheck) run_screening_check() 단위 테스트 (US-6, TDD).

.omc/specs/brainstorming-factcheck-eval.md AC4 / Round4 참고: 스크리닝 컴포넌트는
의도 파싱(질문에서 어떤 지표·개수를 원하는지 알아내는 것 — 그건 기존 goldset의 몫)이
아니라 top-N '계산값' 자체를 DB 원본 재계산과 대조해 독립적으로 재검증한다(완전일치).

대상: src/eval/factcheck/screening.py
    run_screening_check(items, llm_fn, conn) -> list[dict]
    - items: [{"question":"PER 낮은 5개", "top_n":5, "metric":"per"}, ...]
    - 각 item마다 (a) llm_fn(item)으로 시스템 스크리닝 top-N 종목코드 리스트를 얻고,
      (b) _recompute_top_n(conn, item)(metrics_at 기반 DB 원본 재계산)으로 정답을 얻은 뒤,
      (c) tolerance.exact_match로 두 리스트(종목코드, 순서 포함)가 일치하는지 비교한다.
    - conn을 요구하는 이유: 자매 모듈 financials.py의 dart_api_key와 동일한 이유로,
      원본 재계산에 외부 리소스(DB 커넥션)가 필요하기 때문이다.

_recompute_top_n을 screening 모듈 네임스페이스에서 patch 하여 실제 DB 쿼리 없이
run_screening_check의 비교/판정 로직만 검증한다. llm_fn도 각 케이스가 원하는 값을
반환하는 MagicMock으로 대체한다(실제 LLM 호출 없음).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.eval.factcheck.screening import run_screening_check


class TestRunScreeningCheck:
    def test_matching_top_n_lists_pass(self):
        items = [{"question": "PER 낮은 5개", "top_n": 5, "metric": "per"}]
        llm_fn = MagicMock(return_value=["000003", "000009", "000007", "000001", "000010"])

        with patch(
            "src.eval.factcheck.screening._recompute_top_n",
            return_value=["000003", "000009", "000007", "000001", "000010"],
        ) as mock_recompute:
            result = run_screening_check(items, llm_fn, conn="dummy-conn")

        assert result == [{"question": "PER 낮은 5개", "pass": True, "note": ""}]
        llm_fn.assert_called_once_with(items[0])
        mock_recompute.assert_called_once_with("dummy-conn", items[0])

    def test_mismatching_top_n_lists_fail(self):
        items = [{"question": "ROE 높은 3개", "top_n": 3, "metric": "roe", "ascending": False}]
        llm_fn = MagicMock(return_value=["000003", "000009", "000006"])

        with patch(
            "src.eval.factcheck.screening._recompute_top_n",
            return_value=["000003", "000006", "000009"],  # 순서가 달라 불일치
        ):
            result = run_screening_check(items, llm_fn, conn="dummy-conn")

        assert len(result) == 1
        assert result[0]["question"] == "ROE 높은 3개"
        assert result[0]["pass"] is False
        assert result[0]["note"] != ""  # 불일치 사유를 note에 남긴다

    def test_multiple_items_processed_independently(self):
        items = [
            {"question": "PER 낮은 2개", "top_n": 2, "metric": "per"},
            {"question": "PBR 낮은 2개", "top_n": 2, "metric": "pbr"},
        ]
        llm_fn = MagicMock(side_effect=[["000003", "000009"], ["000003", "000009"]])

        with patch(
            "src.eval.factcheck.screening._recompute_top_n",
            side_effect=[
                ["000003", "000009"],  # 첫 item: 일치
                ["000009", "000003"],  # 두번째 item: 순서 달라 불일치
            ],
        ):
            result = run_screening_check(items, llm_fn, conn="dummy-conn")

        assert [r["pass"] for r in result] == [True, False]
        assert [r["question"] for r in result] == ["PER 낮은 2개", "PBR 낮은 2개"]
