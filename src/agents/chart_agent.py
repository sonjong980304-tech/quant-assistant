"""차트 서브에이전트 — matplotlib 전체에서 LLM이 질문에 맞는 그래프를 자유롭게 골라 그린다.

기존에는 차트 생성 로직이 두 곳에 따로 있었다: supervisor.py의 _build_charts()는 LLM 없이
domain_results 모양(산점도/분위수막대/히스토그램/시계열 3케이스)을 파이썬 코드가 기계적으로
판정했고, conversation.py의 멀티턴 후속질문은 LLM이 코드를 쓰되 charting.py의 4개 헬퍼
함수(히스토그램/막대/산점도/라인) 중에서만 고르도록 제한돼 있었다. "스크리닝 결과(종목별
단일 수익률)처럼 이 4~7가지 모양 어디에도 안 맞는 데이터"는 조용히 차트 없이 텍스트만
나갔다(실사용 재현: LLM이 대신 텍스트 ASCII 막대를 즉석에서 지어내는 부작용까지 발생).

이 모듈은 두 경로가 공유하는 단일 진입점이다. charting.py의 4개 헬퍼는 삭제하지 않고
"참고용으로 써도 되는 옵션"으로 프롬프트에 계속 안내한다 — LLM이 그걸 import해서 써도 되고,
matplotlib을 직접 호출하는 완전히 다른 종류(파이차트/박스플롯 등)를 그려도 된다.

exec_fallback.py/conversation.py의 기존 패턴을 그대로 재사용한다(새로 발명하지 않음):
_extract_python_code(코드펜스 제거), execute_python(별도 프로세스 샌드박스), 1회 자가수정
재시도(실패 사유를 다음 프롬프트에 피드백).
"""
from __future__ import annotations

from typing import Callable

from src.agents.exec_fallback import _extract_python_code
from src.agents.exec_runtime import execute_python

_MAX_CHART_ATTEMPTS = 2


def _summarize_data_shape(data) -> str:
    """LLM 프롬프트용 구조 요약 — 실제 값은 넣지 않고 컬럼명·행수 등 구조만 설명한다.

    conversation.py의 동명 함수와 동일한 원칙(요약만 프롬프트에, 원본은 실행 컨텍스트에만) —
    circular import를 피하려고 이 모듈 안에 독립적으로 둔다(신규 로직 아님, 패턴 재사용).
    """
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            cols = sorted({k for row in data for k in row.keys()})
            return f"리스트[dict] {len(data)}개, 키: {cols}"
        return f"리스트 {len(data)}개"
    if isinstance(data, dict):
        return f"dict, 최상위 키: {sorted(data.keys())}"
    return f"{type(data).__name__} 값"


def _chart_prompt(question: str, data_summary: str, code_error: str | None = None) -> str:
    retry_note = (
        f"\n\n[직전 코드 실행 실패] 방금 작성한 코드가 다음 이유로 실패했습니다: {code_error}\n"
        "같은 실수를 반복하지 말고 원인을 고쳐 다시 작성하세요."
        if code_error else ""
    )
    return (
        "다음 데이터로 이 질문에 가장 적합한 그래프를 matplotlib으로 자유롭게 그리는 Python "
        f"코드를 작성하세요(구조: {data_summary}, 실제 데이터는 `data`라는 변수로 이미 주어집니다).\n"
        "막대그래프/선그래프/산점도/히스토그램/파이차트/박스플롯 등 matplotlib이 지원하는 어떤 "
        "종류든 질문과 데이터에 가장 잘 맞는 걸 고르세요. src.agents.charting 모듈의 기존 헬퍼 "
        "(render_histogram_chart_base64/render_bar_chart_base64/render_scatter_chart_base64/"
        "render_line_chart_base64)를 참고용으로 import해서 써도 되고, matplotlib을 직접 "
        "호출하는 코드를 새로 써도 됩니다.\n"
        "완성된 그래프 이미지는 base64로 인코딩한 PNG 문자열로 `chart_base64` 변수에, 제목은 "
        f"`chart_title` 변수에 각각 담으세요.{retry_note}\n\n질문: {question}\nPython 코드:"
    )


def build_chart_freeform(
    question: str,
    data,
    llm_fn: Callable[[str], str] | None,
    execute_python_fn: Callable | None = None,
) -> dict | None:
    """질문+데이터로 matplotlib 차트를 자유롭게 그려 {"chart_base64","chart_title"}를 반환한다.

    차트는 부가 기능이다 — llm_fn이 없거나, 코드 생성/실행/결과 중 어디서든 실패하면 예외를
    던지지 않고 조용히 None을 반환한다(본문 응답을 절대 깨뜨리지 않는다는 이 프로젝트 기존
    원칙, supervisor.py _build_charts 주석과 동일).
    """
    if llm_fn is None:
        return None

    execute_python_fn = execute_python_fn or execute_python
    summary = _summarize_data_shape(data)
    code_error: str | None = None

    for _ in range(_MAX_CHART_ATTEMPTS):
        code_raw = llm_fn(_chart_prompt(question, summary, code_error)) or ""
        code = _extract_python_code(code_raw)
        if not code:
            code_error = "LLM이 Python 코드를 생성하지 못했습니다."
            continue

        # 이 서브에이전트는 chart_base64/chart_title만 필요하지만, execute_python_fn 호출
        # 관례(conversation.py 등 기존 호출부와 동일하게 result_var를 항상 명시)를 따른다 —
        # 생성 코드가 result를 안 채워도 무해하게 무시된다.
        py_result = execute_python_fn(
            code, context={"data": data}, result_var="result",
            extra_vars=["chart_base64", "chart_title"],
        )
        if not py_result.get("ok"):
            code_error = f"Python 실행 실패: {py_result.get('error')}"
            continue

        extra = py_result.get("extra", {})
        chart_base64 = extra.get("chart_base64")
        if not chart_base64:
            code_error = "생성된 코드가 chart_base64를 채우지 않았습니다."
            continue

        return {"chart_base64": chart_base64, "chart_title": extra.get("chart_title")}

    return None
