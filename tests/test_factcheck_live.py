"""실제 시스템(run_hierarchical) 연동 어댑터 단위테스트 (US-8, TDD).

.omc/specs/brainstorming-factcheck-eval.md 참고: 5개 도메인 factcheck 모듈은 지금까지
llm_fn/vision_fn을 모킹했다. src/eval/factcheck/live.py의 어댑터는 그 자리에 실제
run_hierarchical 응답에서 구조화된 값(domain_results)을 뽑아 각 모듈이 기대하는 시그니처로
넘겨준다. run_hierarchical 자체 호출은 여기서 모킹하고(run_fn 주입), 어댑터의 '추출/정규화'
로직만 결정론적으로 검증한다(실제 LLM/DART/네이버 호출 없음).

핵심 케이스:
  - financial: domain_results.kr.financial.value 를 그대로 뽑는다. 없으면 nan(측정불가가
    아니라 comparison에서 fail 처리되도록 — within_pct_tolerance가 예외로 죽지 않게).
  - price: domain_results.kr.price[0].close 를 뽑는다. 없으면 nan.
  - screening: domain_results.kr.result(랭킹 리스트)에서 stock_code를 top_n개 뽑는다.
    리스트가 아니면(both_extremes 등) 빈 리스트.
  - chart: 최상위 chart_base64 + 도메인 랭킹 데이터(stock_code 리스트)를 함께 돌려준다.
  - vision 판정 파싱: 응답 텍스트에서 pass/fail(적절/부적절, yes/no)을 bool로 정규화.
"""
from __future__ import annotations

import math

from src.eval.factcheck import live


# --------------------------------------------------------------------------
# financial 어댑터
# --------------------------------------------------------------------------
class TestFinancialAdapter:
    def test_extracts_financial_value(self):
        def fake_run(question, conn, llm_fn=None, steps=None):
            return {"domain_results": {"kr": {"financial": {"value": 123456.0, "metric": "operating_profit"}}}}

        fn = live.make_financial_llm_fn(conn="C", llm_fn=lambda p: "", run_fn=fake_run)
        item = {"stock_code": "005930", "name": "삼성전자", "metric": "operating_profit"}
        assert fn(item) == 123456.0

    def test_missing_financial_returns_nan(self):
        def fake_run(question, conn, llm_fn=None, steps=None):
            return {"domain_results": {"kr": {"financial": None}}}

        fn = live.make_financial_llm_fn(conn="C", llm_fn=lambda p: "", run_fn=fake_run)
        item = {"stock_code": "005930", "name": "삼성전자", "metric": "operating_profit"}
        assert math.isnan(fn(item))

    def test_run_exception_returns_nan(self):
        def fake_run(question, conn, llm_fn=None, steps=None):
            raise RuntimeError("system down")

        fn = live.make_financial_llm_fn(conn="C", llm_fn=lambda p: "", run_fn=fake_run)
        item = {"stock_code": "005930", "name": "삼성전자", "metric": "operating_profit"}
        assert math.isnan(fn(item))


# --------------------------------------------------------------------------
# price 어댑터
# --------------------------------------------------------------------------
class TestPriceAdapter:
    def test_extracts_close_from_price_list(self):
        def fake_run(question, conn, llm_fn=None, steps=None):
            return {"domain_results": {"kr": {"price": [{"date": "2026-07-20", "close": 70200.0}]}}}

        fn = live.make_price_llm_fn(conn="C", llm_fn=lambda p: "", run_fn=fake_run)
        assert fn("삼성전자(005930)의 오늘 종가는?") == 70200.0

    def test_missing_price_returns_nan(self):
        def fake_run(question, conn, llm_fn=None, steps=None):
            return {"domain_results": {"kr": {"price": None}}}

        fn = live.make_price_llm_fn(conn="C", llm_fn=lambda p: "", run_fn=fake_run)
        assert math.isnan(fn("q"))


# --------------------------------------------------------------------------
# screening 어댑터
# --------------------------------------------------------------------------
class TestScreeningAdapter:
    def test_extracts_top_n_codes_in_order(self):
        def fake_run(question, conn, llm_fn=None, steps=None):
            return {
                "domain_results": {
                    "kr": {
                        "intent": "screening",
                        "result": [
                            {"stock_code": "000001"},
                            {"stock_code": "000002"},
                            {"stock_code": "000003"},
                        ],
                    }
                }
            }

        fn = live.make_screening_llm_fn(conn="C", llm_fn=lambda p: "", run_fn=fake_run)
        item = {"question": "PER 낮은 순 2개", "top_n": 2, "metric": "per"}
        assert fn(item) == ["000001", "000002"]

    def test_non_list_result_returns_empty(self):
        def fake_run(question, conn, llm_fn=None, steps=None):
            return {"domain_results": {"kr": {"intent": "screening", "result": {"highest": []}}}}

        fn = live.make_screening_llm_fn(conn="C", llm_fn=lambda p: "", run_fn=fake_run)
        item = {"question": "PBR 최댓값과 최솟값", "top_n": 1, "metric": "pbr"}
        assert fn(item) == []


# --------------------------------------------------------------------------
# chart 어댑터
# --------------------------------------------------------------------------
class TestChartAdapter:
    def test_returns_base64_and_actual_codes(self):
        def fake_run(question, conn, llm_fn=None, steps=None):
            return {
                "chart_base64": "PNGBASE64",
                "domain_results": {
                    "kr": {
                        "intent": "screening",
                        "result": [{"stock_code": "000001"}, {"stock_code": "000002"}],
                    }
                },
            }

        fn = live.make_chart_llm_fn(conn="C", llm_fn=lambda p: "", run_fn=fake_run)
        out = fn("시가총액 상위 2개 막대그래프")
        assert out["chart_base64"] == "PNGBASE64"
        assert out["actual_data"] == ["000001", "000002"]

    def test_missing_chart_base64_is_none(self):
        def fake_run(question, conn, llm_fn=None, steps=None):
            return {"domain_results": {"kr": {"result": [{"stock_code": "000001"}]}}}

        fn = live.make_chart_llm_fn(conn="C", llm_fn=lambda p: "", run_fn=fake_run)
        out = fn("q")
        assert out["chart_base64"] is None
        assert out["actual_data"] == ["000001"]


# --------------------------------------------------------------------------
# vision 판정 파싱 + vision_fn
# --------------------------------------------------------------------------
class TestVisionVerdict:
    def test_parse_pass(self):
        assert live._parse_vision_verdict("판정: 적절합니다. PASS") is True
        assert live._parse_vision_verdict("yes, 데이터를 잘 표현함") is True

    def test_parse_fail(self):
        assert live._parse_vision_verdict("부적절합니다. FAIL") is False
        assert live._parse_vision_verdict("no") is False

    def test_parse_unknown_returns_none(self):
        assert live._parse_vision_verdict("잘 모르겠습니다") is None
        assert live._parse_vision_verdict("") is None

    def test_vision_fn_raises_when_no_image(self):
        vfn = live.make_vision_fn(vision_client=object())
        # chart_base64가 없으면 판정 불가 → 예외(호출부 chart.py가 측정불가로 처리)
        try:
            vfn("q", None)
            raised = False
        except Exception:
            raised = True
        assert raised

    def test_vision_fn_parses_client_response(self):
        class FakeResult:
            def __init__(self, text):
                self.text = text
                self.ok = True

        class FakeClient:
            def complete_vision(self, prompt, image_base64, role="sql"):
                return FakeResult("적절합니다 PASS")

        vfn = live.make_vision_fn(vision_client=FakeClient())
        assert vfn("q", "BASE64") is True
