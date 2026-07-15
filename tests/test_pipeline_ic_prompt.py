"""PIPELINE_USER 프롬프트에 compute_ic(8번째 프리미티브)가 노출되는지 검증 (TDD).

run_backtest 노출 때와 동일한 계약: "프롬프트에 실제로 포함되는지"는 코드로
검증 가능하므로 순수 문자열 상수 변경이라도 RED 테스트로 먼저 잡는다.
"""
from __future__ import annotations

from src.legacy.graph import prompts


def test_pipeline_user_prompt_includes_compute_ic():
    assert "compute_ic" in prompts.PIPELINE_USER


def test_pipeline_user_prompt_still_formats_without_error_after_ic_addition():
    # 새 예시 안에 리터럴 {}가 있으면 .format()의 {{/}} 이스케이프 규칙을 지켜야 한다
    # (기존 run_backtest 예시가 이미 지키는 관례를 그대로 따름).
    formatted = prompts.PIPELINE_USER.format(schema="(s)", today="2026-07-12", question="q")
    assert "compute_ic" in formatted
