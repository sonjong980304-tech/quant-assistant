"""파마프렌치 팩터 온디맨드 조회 테스트.

.omc/specs/brainstorming-fama-french-factor-lookup.md 참고.
분류(LLM)→확인(y/n)→조회(pandas_datareader) 흐름을 각각 순수 로직으로 분리해
네트워크/실제 LLM 호출 없이 단위 테스트한다(DI 패턴, 기존 ThrottledFetcher 관례와 동일).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import CONFIG
from src.eval.goldset import GOLDSET
from src.factors.fama_french import (
    classify_factor_intent,
    fetch_factor_data,
    format_factor_result,
    handle_query,
)
from tests.conftest import FakeLLM


# --------------------------------------------------------------------------
# classify_factor_intent
# --------------------------------------------------------------------------
def test_classify_factor_intent_returns_none_for_non_factor_question():
    llm = FakeLLM('{"is_factor_question": false}')
    assert classify_factor_intent("PER이 낮은 회사 알려줘", llm=llm) is None


def test_classify_factor_intent_returns_intent_dict_for_factor_question():
    llm = FakeLLM(
        '{"is_factor_question": true, "dataset": "momentum", "frequency": "monthly", '
        '"latest_only": true, "start": null, "end": null}'
    )
    intent = classify_factor_intent("모멘텀 팩터 이번달 값 알려줘", llm=llm)
    assert intent == {
        "is_factor_question": True,
        "dataset": "momentum",
        "frequency": "monthly",
        "latest_only": True,
        "start": None,
        "end": None,
    }


def test_classify_factor_intent_returns_none_when_llm_unavailable():
    llm = FakeLLM('{"is_factor_question": true}', available=False)
    assert classify_factor_intent("모멘텀 팩터 값?", llm=llm) is None
    assert llm.calls == []  # 미가용이면 호출 자체를 안 함


def test_classify_factor_intent_returns_none_when_llm_call_fails():
    llm = FakeLLM("", ok=False)
    assert classify_factor_intent("모멘텀 팩터 값?", llm=llm) is None


# --------------------------------------------------------------------------
# fetch_factor_data
# --------------------------------------------------------------------------
def _sample_df():
    return pd.DataFrame(
        {
            "Mkt-RF": [1.2, -0.5],
            "SMB": [0.3, 0.1],
            "HML": [-0.2, 0.4],
            "RMW": [0.1, 0.2],
            "CMA": [0.05, -0.1],
            "RF": [0.01, 0.01],
        },
        index=pd.PeriodIndex(["2026-05", "2026-06"], freq="M"),
    )


def test_fetch_factor_data_calls_fetch_fn_with_mapped_dataset_name():
    calls = []

    def fake_fetch(name, start, end):
        calls.append((name, start, end))
        return _sample_df()

    fetch_factor_data("5factor", "monthly", fetch_fn=fake_fetch)
    assert calls == [("F-F_Research_Data_5_Factors_2x3", None, None)]


def test_fetch_factor_data_daily_momentum_maps_to_daily_dataset_name():
    calls = []

    def fake_fetch(name, start, end):
        calls.append(name)
        return _sample_df()

    fetch_factor_data("momentum", "daily", fetch_fn=fake_fetch)
    assert calls == ["F-F_Momentum_Factor_daily"]


def test_fetch_factor_data_latest_only_returns_single_last_row():
    rows = fetch_factor_data("5factor", "monthly", latest_only=True, fetch_fn=lambda n, s, e: _sample_df())
    assert len(rows) == 1
    assert rows[0]["period"] == "2026-06"
    assert rows[0]["SMB"] == 0.1


def test_fetch_factor_data_range_returns_all_fetched_rows():
    rows = fetch_factor_data(
        "5factor", "monthly", latest_only=False, start="2026-01-01", end="2026-06-30",
        fetch_fn=lambda n, s, e: _sample_df(),
    )
    assert len(rows) == 2
    assert [r["period"] for r in rows] == ["2026-05", "2026-06"]


def test_fetch_factor_data_no_cache_calls_fetch_fn_every_time():
    fetch_fn = lambda n, s, e: _sample_df()
    calls = {"n": 0}

    def counting_fetch(n, s, e):
        calls["n"] += 1
        return _sample_df()

    fetch_factor_data("5factor", "monthly", fetch_fn=counting_fetch)
    fetch_factor_data("5factor", "monthly", fetch_fn=counting_fetch)
    assert calls["n"] == 2


def test_fetch_factor_data_unsupported_combo_raises_value_error():
    try:
        fetch_factor_data("unknown", "monthly", fetch_fn=lambda n, s, e: _sample_df())
        assert False, "ValueError가 발생해야 함"
    except ValueError:
        pass


# --------------------------------------------------------------------------
# handle_query — 분류→확인→조회 통합 흐름
# --------------------------------------------------------------------------
def test_handle_query_returns_none_when_not_factor_question():
    classify = lambda q: None
    confirm_calls = []
    result = handle_query("PER 낮은 회사", ask_confirm=lambda p: confirm_calls.append(p), classify=classify)
    assert result is None
    assert confirm_calls == []  # 팩터 질문이 아니면 확인도 안 물어봄


def test_handle_query_asks_confirmation_when_factor_question():
    intent = {"is_factor_question": True, "dataset": "momentum", "frequency": "monthly",
              "latest_only": True, "start": None, "end": None}
    prompts = []

    def ask(p):
        prompts.append(p)
        return "n"

    handle_query("모멘텀 팩터 값?", ask_confirm=ask, classify=lambda q: intent)
    assert len(prompts) == 1
    assert "y/n" in prompts[0]


def test_handle_query_returns_none_when_user_declines():
    intent = {"is_factor_question": True, "dataset": "momentum", "frequency": "monthly",
              "latest_only": True, "start": None, "end": None}
    fetch_calls = []

    def fetch(*a, **kw):
        fetch_calls.append((a, kw))
        return [{"period": "2026-06", "Mom": 1.0}]

    result = handle_query("모멘텀 팩터 값?", ask_confirm=lambda p: "n", classify=lambda q: intent, fetch=fetch)
    assert result is None
    assert fetch_calls == []


def test_handle_query_fetches_and_formats_when_user_confirms():
    intent = {"is_factor_question": True, "dataset": "momentum", "frequency": "monthly",
              "latest_only": True, "start": None, "end": None}

    def fetch(dataset, frequency, start=None, end=None, latest_only=True):
        assert dataset == "momentum" and frequency == "monthly" and latest_only is True
        return [{"period": "2026-06", "Mom": 1.42}]

    result = handle_query("모멘텀 팩터 값?", ask_confirm=lambda p: "y", classify=lambda q: intent, fetch=fetch)
    assert "2026-06" in result
    assert "1.42" in result


def test_handle_query_returns_error_message_on_fetch_failure_without_raising():
    intent = {"is_factor_question": True, "dataset": "momentum", "frequency": "monthly",
              "latest_only": True, "start": None, "end": None}

    def failing_fetch(*a, **kw):
        raise ConnectionError("Ken French 접속 실패(mock)")

    result = handle_query("모멘텀 팩터 값?", ask_confirm=lambda p: "y", classify=lambda q: intent, fetch=failing_fetch)
    assert "실패" in result
    assert "Ken French" in result


# --------------------------------------------------------------------------
# format_factor_result
# --------------------------------------------------------------------------
@pytest.mark.skipif(not CONFIG.has_openai_key, reason="OpenAI 키 필요(README 관례: 키 없으면 스킵)")
def test_classify_factor_intent_no_false_positive_on_goldset_questions():
    """기존 goldset 50문항이 파마프렌치 경로로 오분류되지 않는지 실제 LLM으로 확인."""
    false_positives = [
        item["question"] for item in GOLDSET if classify_factor_intent(item["question"])
    ]
    assert false_positives == []


def test_format_factor_result_includes_dataset_and_all_rows():
    intent = {"dataset": "5factor", "frequency": "monthly"}
    rows = [
        {"period": "2026-05", "SMB": 0.3},
        {"period": "2026-06", "SMB": 0.1},
    ]
    text = format_factor_result(intent, rows)
    assert "5factor" in text and "monthly" in text
    assert "2026-05" in text and "2026-06" in text
