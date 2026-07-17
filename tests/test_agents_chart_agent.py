"""차트 서브에이전트(build_chart_freeform) — LLM이 matplotlib 아무 차트나 자유롭게 골라
그리도록 코드를 짜게 하고, exec_runtime 샌드박스에서 실행해 chart_base64/chart_title을
꺼내는 순수 로직. 스크리닝(단발성)·멀티턴(후속질문) 두 경로가 이 하나를 공유한다.
"""
from src.agents.chart_agent import build_chart_freeform


def test_build_chart_freeform_returns_none_when_llm_fn_is_none():
    result = build_chart_freeform("코스피 상위 20개 종목 그래프로 그려줘", [{"a": 1}], llm_fn=None)
    assert result is None


def test_build_chart_freeform_returns_none_when_llm_produces_no_code():
    def fake_llm(prompt):
        return ""  # 빈 응답 → _extract_python_code도 빈 문자열

    def fake_exec(code, context, result_var=None, extra_vars=None):
        raise AssertionError("코드가 비었으면 실행 자체를 시도하면 안 된다")

    result = build_chart_freeform(
        "그래프 그려줘", [{"a": 1}], fake_llm, execute_python_fn=fake_exec,
    )
    assert result is None


def test_build_chart_freeform_returns_none_when_execution_fails():
    def fake_llm(prompt):
        return "```python\nchart_base64 = 'PNG'\n```"

    def fake_exec(code, context, result_var=None, extra_vars=None):
        return {"ok": False, "result": None, "error": "boom", "extra": {}}

    result = build_chart_freeform(
        "그래프 그려줘", [{"a": 1}], fake_llm, execute_python_fn=fake_exec,
    )
    assert result is None


def test_build_chart_freeform_returns_none_when_chart_base64_empty():
    def fake_llm(prompt):
        return "```python\nresult = data\n```"  # chart_base64를 안 채움

    def fake_exec(code, context, result_var=None, extra_vars=None):
        return {"ok": True, "result": context["data"], "error": None,
                "extra": {"chart_base64": None, "chart_title": None}}

    result = build_chart_freeform(
        "그래프 그려줘", [{"a": 1}], fake_llm, execute_python_fn=fake_exec,
    )
    assert result is None


def test_build_chart_freeform_success_returns_base64_and_title():
    captured = {}

    def fake_llm(prompt):
        captured["prompt"] = prompt
        return "```python\nimport matplotlib.pyplot as plt\nchart_base64='PNG'\nchart_title='제목'\n```"

    def fake_exec(code, context, result_var=None, extra_vars=None):
        captured["context"] = context
        captured["extra_vars"] = extra_vars
        return {"ok": True, "result": None, "error": None,
                "extra": {"chart_base64": "PNG", "chart_title": "제목"}}

    result = build_chart_freeform(
        "코스피 상위 20개 종목 수익률 그래프로 그려줘", [{"name": "삼성전자", "return_12m": 12.3}],
        fake_llm, execute_python_fn=fake_exec,
    )
    assert result == {"chart_base64": "PNG", "chart_title": "제목"}
    assert captured["context"] == {"data": [{"name": "삼성전자", "return_12m": 12.3}]}
    assert set(captured["extra_vars"]) == {"chart_base64", "chart_title"}
    # 프롬프트에는 실제 값이 아니라 구조 요약만 들어가야 한다(요약원칙 재사용, AC3와 동일 규약)
    assert "삼성전자" not in captured["prompt"]
    assert "리스트" in captured["prompt"] or "dict" in captured["prompt"]


def test_build_chart_freeform_prompt_allows_any_matplotlib_chart_not_just_four_helpers():
    """4개 헬퍼로 제한하지 않고 matplotlib 자유선택을 명시적으로 안내해야 한다."""
    captured = {}

    def fake_llm(prompt):
        captured["prompt"] = prompt
        return "```python\nchart_base64='PNG'\n```"

    def fake_exec(code, context, result_var=None, extra_vars=None):
        return {"ok": True, "result": None, "error": None,
                "extra": {"chart_base64": "PNG", "chart_title": None}}

    build_chart_freeform("파이차트로 그려줘", [{"a": 1}], fake_llm, execute_python_fn=fake_exec)
    assert "matplotlib" in captured["prompt"]
    assert "chart_base64" in captured["prompt"]


def test_build_chart_freeform_retries_once_after_execution_failure_then_succeeds():
    attempts = []

    def fake_llm(prompt):
        attempts.append(prompt)
        return "```python\nchart_base64='PNG'\n```"

    calls = {"n": 0}

    def fake_exec(code, context, result_var=None, extra_vars=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"ok": False, "result": None, "error": "NameError: plt", "extra": {}}
        return {"ok": True, "result": None, "error": None,
                "extra": {"chart_base64": "PNG", "chart_title": "T"}}

    result = build_chart_freeform(
        "그래프 그려줘", [{"a": 1}], fake_llm, execute_python_fn=fake_exec,
    )
    assert result == {"chart_base64": "PNG", "chart_title": "T"}
    assert calls["n"] == 2
    assert len(attempts) == 2
    assert "NameError: plt" in attempts[1]  # 실패 사유가 재시도 프롬프트에 피드백됨


def test_build_chart_freeform_injects_flat_stock_list_as_data_and_summarizes_it_as_list():
    """'종목별 단일값 리스트'(스크리닝 결과 모양)를 그대로 data로 주입하고, 프롬프트 요약도
    'dict 래퍼'가 아니라 '리스트[dict]'로 나와야 한다.

    실측 회귀: 폴백이 도메인키 래퍼({"kr": {...}})를 넘기면 요약이 'dict, 최상위 키: [kr]'로만
    나와 LLM이 실제 종목 리스트를 못 찾았다. 폴백은 flat 리스트를 넘겨야 하며, 그때 이 서브
    에이전트는 그 리스트를 data로 그대로 주입하고 리스트로 요약한다(supervisor._chartable_payload
    가 언랩해 넘겨준다는 계약을 이 레벨에서 문서화)."""
    stocks = [
        {"stock_code": f"00{i:04d}", "name": f"종목{i}", "return_12m": 0.3 - i * 0.02}
        for i in range(10)
    ]
    captured = {}

    def fake_llm(prompt):
        captured["prompt"] = prompt
        return "```python\nchart_base64='PNG'\nchart_title='상위10 수익률'\n```"

    def fake_exec(code, context, result_var=None, extra_vars=None):
        captured["context"] = context
        return {"ok": True, "result": None, "error": None,
                "extra": {"chart_base64": "PNG", "chart_title": "상위10 수익률"}}

    result = build_chart_freeform(
        "상위 10개 종목 수익률 그래프로 그려줘", stocks, fake_llm, execute_python_fn=fake_exec,
    )
    assert result == {"chart_base64": "PNG", "chart_title": "상위10 수익률"}
    # flat 리스트가 그대로 data로 주입돼야 한다(도메인키 래퍼가 아니라).
    assert captured["context"] == {"data": stocks}
    # 프롬프트 요약이 '리스트[dict]'로 나와 LLM이 종목 리스트임을 인지할 수 있어야 한다.
    assert "리스트[dict] 10개" in captured["prompt"]
    assert "return_12m" in captured["prompt"]


def test_build_chart_freeform_gives_up_after_max_attempts():
    def fake_llm(prompt):
        return "```python\nchart_base64='PNG'\n```"

    def fake_exec(code, context, result_var=None, extra_vars=None):
        return {"ok": False, "result": None, "error": "boom", "extra": {}}

    result = build_chart_freeform(
        "그래프 그려줘", [{"a": 1}], fake_llm, execute_python_fn=fake_exec,
    )
    assert result is None
