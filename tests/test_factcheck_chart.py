"""AC3: 차트 근거 데이터 일치 + vision 시각 품질 검증 단위 테스트 (US-5, TDD).

.omc/specs/brainstorming-factcheck-eval.md Round 3: 차트가 정확하다는 것은 (a) 차트가
쓴 근거 데이터가 기대값과 일치하는 것 + (b) 이미지 자체의 시각적 품질(vision 판정) 둘
다를 의미한다 — 두 기준 모두 충족해야 pass=True. 실제 이미지 생성/API 호출 없이
llm_fn(차트 생성)/vision_fn(vision 판정)을 모두 모킹해 판정 로직만 검증한다.

대상: src/eval/factcheck/chart.py
    run_chart_check(items, llm_fn, vision_fn) -> list[dict]
"""
from __future__ import annotations

from src.eval.factcheck.chart import run_chart_check


def test_data_match_and_vision_pass_yields_overall_pass():
    items = [{"question": "월별 매출 추이 그려줘", "expected_data": [100, 200, 300]}]

    def llm_fn(question):
        return {"chart_base64": "FAKEBASE64", "actual_data": [100, 200, 300]}

    def vision_fn(prompt, image_base64):
        return True

    result = run_chart_check(items, llm_fn, vision_fn)

    assert result == [
        {
            "question": "월별 매출 추이 그려줘",
            "data_match": True,
            "vision_verdict": True,
            "pass": True,
            "note": "",
        }
    ]


def test_data_mismatch_yields_overall_fail_even_if_vision_passes():
    items = [{"question": "월별 매출 추이 그려줘", "expected_data": [100, 200, 300]}]

    def llm_fn(question):
        return {"chart_base64": "FAKEBASE64", "actual_data": [999, 200, 300]}

    def vision_fn(prompt, image_base64):
        return True  # vision은 통과해도 데이터가 다르면 전체는 실패여야 함

    result = run_chart_check(items, llm_fn, vision_fn)

    assert result[0]["data_match"] is False
    assert result[0]["vision_verdict"] is True
    assert result[0]["pass"] is False


def test_vision_pass_false_yields_overall_fail_even_if_data_matches():
    items = [{"question": "월별 매출 추이 그려줘", "expected_data": [100, 200, 300]}]

    def llm_fn(question):
        return {"chart_base64": "FAKEBASE64", "actual_data": [100, 200, 300]}

    def vision_fn(prompt, image_base64):
        return False  # 데이터는 맞아도 이미지 시각 품질이 별로면 전체는 실패

    result = run_chart_check(items, llm_fn, vision_fn)

    assert result[0]["data_match"] is True
    assert result[0]["vision_verdict"] is False
    assert result[0]["pass"] is False


def test_vision_fn_exception_yields_pass_none_and_note():
    items = [{"question": "월별 매출 추이 그려줘", "expected_data": [100, 200, 300]}]

    def llm_fn(question):
        return {"chart_base64": "FAKEBASE64", "actual_data": [100, 200, 300]}

    def vision_fn(prompt, image_base64):
        raise RuntimeError("vision API 실패")

    result = run_chart_check(items, llm_fn, vision_fn)

    assert result[0]["data_match"] is True  # 데이터 일치 여부는 vision과 무관하게 그대로 기록
    assert result[0]["vision_verdict"] is None
    assert result[0]["pass"] is None
    assert result[0]["note"].startswith("측정불가")
    assert "vision API 실패" in result[0]["note"]  # 실제 예외 메시지가 note에 남아야 원인 진단 가능


def test_vision_fn_exception_with_no_chart_image_reports_missing_image_reason():
    """chart_base64가 없어(None) vision_fn이 ValueError를 던지는 경우, note에 그 사유가 남아야
    "이미지 생성 자체가 실패했는지" vs "vision API 호출이 실패했는지"를 구분할 수 있다."""
    items = [{"question": "월별 매출 추이 그려줘", "expected_data": [100, 200, 300]}]

    def llm_fn(question):
        return {"chart_base64": None, "actual_data": [100, 200, 300]}

    def vision_fn(prompt, image_base64):
        raise ValueError("차트 이미지(base64)가 없어 vision 판정 불가")

    result = run_chart_check(items, llm_fn, vision_fn)

    assert result[0]["note"].startswith("측정불가")
    assert "이미지" in result[0]["note"]
