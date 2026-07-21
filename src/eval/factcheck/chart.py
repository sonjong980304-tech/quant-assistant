"""AC3: 차트 근거 데이터 일치 + vision 시각 품질 검증 (US-5).

.omc/specs/brainstorming-factcheck-eval.md Round 3: 차트가 정확하다는 것은 (a) 차트가
사용한 근거 데이터가 기대값과 일치하는 것 + (b) 이미지 자체의 시각적 품질(vision 판정)
둘 다를 의미한다 — 두 기준 모두 충족해야 pass=True. vision_fn 자체가 예외를 던지면
(모델 응답 파싱 실패 등) 판정 불가로 처리한다 — 무조건 fail로 단정하지 않고
pass=None, note="측정불가"로 남긴다(Round 9/10: vision 모델은 gpt-5.4-mini 재사용,
비용 절감을 위해 gpt-5.5는 쓰지 않기로 확정했지만 신뢰성 이슈는 여전히 남을 수 있음).

llm_fn/vision_fn은 순수 함수로 주입받는다 — 실제 호출부(스크립트)가 llm_fn에는
"질문으로 차트를 생성해 base64+근거데이터를 돌려주는" 로직을, vision_fn에는
src.llm.LLM.complete_vision(role="sql")을 감싼 판정 로직을 넘긴다. 테스트에서는
둘 다 모킹해 실제 이미지/API 호출 없이 판정 로직만 검증한다.
"""
from __future__ import annotations

from typing import Callable

from src.eval.factcheck.tolerance import exact_match


def run_chart_check(
    items: list[dict],
    llm_fn: Callable[[str], dict],
    vision_fn: Callable[[str, str], bool],
) -> list[dict]:
    """차트 질의 목록에 대해 데이터 일치 + vision 시각 품질을 함께 채점한다.

    items: [{"question": str, "expected_data": Any}, ...]
    llm_fn(question) -> {"chart_base64": str, "actual_data": Any} (시스템이 실제로
        생성한 차트의 base64 PNG와, 그 차트가 근거로 삼은 데이터)
    vision_fn(question, chart_base64) -> bool (이미지가 데이터를 올바르게 표현했으면
        True). 예외를 던지면 측정불가로 처리한다.

    반환: [{"question", "data_match", "vision_verdict", "pass", "note"}, ...]
    """
    results: list[dict] = []
    for item in items:
        question = item.get("question")
        expected_data = item.get("expected_data")

        chart = llm_fn(question)
        chart_base64 = chart.get("chart_base64")
        actual_data = chart.get("actual_data")

        data_match = exact_match(expected_data, actual_data)

        try:
            vision_verdict = vision_fn(question, chart_base64)
        except Exception as exc:  # noqa: BLE001 — vision 실패는 fail이 아니라 측정불가
            # 원인 진단을 위해 실제 예외 메시지를 note에 남긴다(예: "차트 이미지가 없어
            # vision 판정 불가" vs "vision API 호출/파싱 실패"를 구분할 수 있게).
            results.append(
                {
                    "question": question,
                    "data_match": data_match,
                    "vision_verdict": None,
                    "pass": None,
                    "note": f"측정불가: {type(exc).__name__}: {exc}",
                }
            )
            continue

        results.append(
            {
                "question": question,
                "data_match": data_match,
                "vision_verdict": vision_verdict,
                "pass": bool(data_match) and bool(vision_verdict),
                "note": "",
            }
        )
    return results
