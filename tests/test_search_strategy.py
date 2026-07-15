"""search_strategy 프리미티브 테스트 (brainstorming-talib-reverse-backtest.md AC9-AC18).

- 탐색 대상은 candidates(criteria 조합 리스트)뿐 — n/rebalance/sectors/markets/market은
  호출 시 고정값으로 모든 후보에 동일 적용한다(Non-Goal: 이 파라미터들을 탐색범위로 넓히지 않음).
- 후보 21개 이상은 실행 전에 하드 거부한다(_MAX_REBALANCE_STEPS류 사전상한 검사 관례).
- 종목별 지표 계산 콜백(callbacks_fn)은 후보 반복 전체에서 1회만 생성해 공유한다(AC12,
  run_backtest_primitive처럼 매번 재호출하면 캐시가 무효화돼 최대 20배 느려짐).
"""
from __future__ import annotations

import pytest

from src.backtest.primitives import search_strategy

_FAKE_DATES = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]


def _fake_dates_fn(start_year, end_year, rebalance):
    return _FAKE_DATES


def _fake_max_date_fn(conn):
    return "2024-12-31"


def _fake_callbacks_fn_factory():
    calls = []

    def fake(conn):
        calls.append(conn)
        metrics_fn = lambda d: []          # noqa: E731 (테스트 전용 더미)
        price_fn = lambda d, c: 100.0      # noqa: E731
        return metrics_fn, price_fn

    fake.calls = calls
    return fake


def _fake_backtest_fn_factory(performance=None):
    calls = []

    def fake(dates, metrics_fn, price_fn, params, benchmark_fn=None, weights=None):
        calls.append({
            "dates": dates, "metrics_fn": metrics_fn, "price_fn": price_fn,
            "params": params, "benchmark_fn": benchmark_fn, "weights": weights,
        })
        return {"performance": performance or {"sharpe": 1.0, "mdd": -5.0}, "holdings": []}

    fake.calls = calls
    return fake


def _run(candidates, backtest_fn=None, callbacks_fn=None, **kwargs):
    return search_strategy(
        conn=object(),
        candidates=candidates,
        start_year=2024,
        end_year=2024,
        dates_fn=_fake_dates_fn,
        max_date_fn=_fake_max_date_fn,
        callbacks_fn=callbacks_fn or _fake_callbacks_fn_factory(),
        backtest_fn=backtest_fn or _fake_backtest_fn_factory(),
        **kwargs,
    )


# --------------------------------------------------------------------------
# 후보 상한 검사 (AC11) — 무거운 연산(콜백 생성/run_backtest) 전에 거부
# --------------------------------------------------------------------------
def test_search_strategy_rejects_more_than_20_candidates():
    candidates = [[{"key": "per", "direction": "low", "weight": 1.0}] for _ in range(21)]
    callbacks_fn = _fake_callbacks_fn_factory()
    backtest_fn = _fake_backtest_fn_factory()
    with pytest.raises(ValueError):
        _run(candidates, backtest_fn=backtest_fn, callbacks_fn=callbacks_fn)
    assert len(callbacks_fn.calls) == 0
    assert len(backtest_fn.calls) == 0


def test_search_strategy_accepts_exactly_20_candidates():
    candidates = [[{"key": "per", "direction": "low", "weight": 1.0}] for _ in range(20)]
    backtest_fn = _fake_backtest_fn_factory()
    results = _run(candidates, backtest_fn=backtest_fn)
    assert len(results) == 20
    assert len(backtest_fn.calls) == 20


def test_search_strategy_rejects_empty_candidates():
    callbacks_fn = _fake_callbacks_fn_factory()
    with pytest.raises(ValueError):
        _run([], callbacks_fn=callbacks_fn)
    assert len(callbacks_fn.calls) == 0


# --------------------------------------------------------------------------
# 콜백 캐시 공유 (AC12 — 이 스토리의 핵심)
# --------------------------------------------------------------------------
def test_search_strategy_builds_callbacks_exactly_once_regardless_of_candidate_count():
    candidates = [
        [{"key": "per", "direction": "low", "weight": 1.0}],
        [{"key": "roe", "direction": "high", "weight": 1.0}],
        [{"key": "pbr", "direction": "low", "weight": 1.0}],
    ]
    callbacks_fn = _fake_callbacks_fn_factory()
    backtest_fn = _fake_backtest_fn_factory()
    _run(candidates, backtest_fn=backtest_fn, callbacks_fn=callbacks_fn)
    assert len(callbacks_fn.calls) == 1          # 후보 3개인데도 콜백 생성은 1번만
    assert len(backtest_fn.calls) == 3           # run_backtest(엔진)는 후보마다 호출


def test_search_strategy_reuses_same_metrics_and_price_fn_objects_across_candidates():
    candidates = [
        [{"key": "per", "direction": "low", "weight": 1.0}],
        [{"key": "roe", "direction": "high", "weight": 1.0}],
    ]
    backtest_fn = _fake_backtest_fn_factory()
    _run(candidates, backtest_fn=backtest_fn)
    m0, p0 = backtest_fn.calls[0]["metrics_fn"], backtest_fn.calls[0]["price_fn"]
    m1, p1 = backtest_fn.calls[1]["metrics_fn"], backtest_fn.calls[1]["price_fn"]
    assert m0 is m1
    assert p0 is p1


def test_search_strategy_20_candidates_still_builds_callbacks_once():
    candidates = [[{"key": "per", "direction": "low", "weight": 1.0}] for _ in range(20)]
    callbacks_fn = _fake_callbacks_fn_factory()
    _run(candidates, callbacks_fn=callbacks_fn)
    assert len(callbacks_fn.calls) == 1


# --------------------------------------------------------------------------
# 탐색 대상 = candidates(criteria)뿐, 나머지는 고정값 (AC10)
# --------------------------------------------------------------------------
def test_search_strategy_applies_same_fixed_params_to_every_candidate():
    candidates = [
        [{"key": "per", "direction": "low", "weight": 1.0}],
        [{"key": "roe", "direction": "high", "weight": 1.0}],
    ]
    backtest_fn = _fake_backtest_fn_factory()
    _run(
        candidates, backtest_fn=backtest_fn,
        n=15, rebalance="monthly", sectors=["반도체"], markets=["KOSPI"], market="KR",
    )
    assert len(backtest_fn.calls) == 2
    for call in backtest_fn.calls:
        p = call["params"]
        assert p["n"] == 15
        assert p["rebalance"] == "monthly"
        assert p["sectors"] == ["반도체"]
        assert p["markets"] == ["KOSPI"]
        assert call["dates"] == _FAKE_DATES
    # criteria만 후보마다 달라야 함
    assert backtest_fn.calls[0]["params"]["criteria"] == candidates[0]
    assert backtest_fn.calls[1]["params"]["criteria"] == candidates[1]


def test_search_strategy_uses_us_callbacks_when_market_is_us():
    """market='US'면 build_callbacks_us 계열이 쓰여야 한다(run_backtest_primitive와 동일 관례).
    callbacks_fn을 주입하면 market과 무관하게 그대로 쓰이므로, 주입된 콜백이 실제로
    호출됐는지만 확인해 배선을 검증한다(US 콜백 자체의 정확성은 별도 스펙 범위)."""
    candidates = [[{"key": "per", "direction": "low", "weight": 1.0}]]
    callbacks_fn = _fake_callbacks_fn_factory()
    _run(candidates, callbacks_fn=callbacks_fn, market="US")
    assert len(callbacks_fn.calls) == 1


# --------------------------------------------------------------------------
# 결과 형식 (기본 구조 — 필터링/정렬은 US-7)
# --------------------------------------------------------------------------
def test_search_strategy_result_includes_criteria_and_performance():
    candidates = [[{"key": "per", "direction": "low", "weight": 1.0}]]
    backtest_fn = _fake_backtest_fn_factory(performance={"sharpe": 1.5, "mdd": -8.0})
    results = _run(candidates, backtest_fn=backtest_fn)
    assert results[0]["criteria"] == candidates[0]
    assert results[0]["performance"] == {"sharpe": 1.5, "mdd": -8.0}


def test_search_strategy_returns_one_result_per_candidate_in_order():
    candidates = [
        [{"key": "per", "direction": "low", "weight": 1.0}],
        [{"key": "roe", "direction": "high", "weight": 1.0}],
        [{"key": "pbr", "direction": "low", "weight": 1.0}],
    ]
    results = _run(candidates)
    assert [r["criteria"] for r in results] == candidates


# --------------------------------------------------------------------------
# 제약조건 필터링 + 정렬 (AC13, AC14)
# --------------------------------------------------------------------------
def _fake_backtest_fn_varying_factory(performances):
    calls = []
    state = {"i": 0}

    def fake(dates, metrics_fn, price_fn, params, benchmark_fn=None, weights=None):
        calls.append({"dates": dates, "params": params})
        perf = performances[state["i"]]
        state["i"] += 1
        return {"performance": perf, "holdings": []}

    fake.calls = calls
    return fake


_THREE_CANDIDATES = [
    [{"key": "per", "direction": "low", "weight": 1.0}],
    [{"key": "roe", "direction": "high", "weight": 1.0}],
    [{"key": "pbr", "direction": "low", "weight": 1.0}],
]


def test_search_strategy_filters_out_candidates_violating_single_constraint():
    performances = [
        {"sharpe": 1.0, "mdd": -5.0},   # 통과 (mdd >= -10)
        {"sharpe": 0.5, "mdd": -20.0},  # 탈락 (mdd < -10)
        {"sharpe": 2.0, "mdd": -8.0},   # 통과
    ]
    backtest_fn = _fake_backtest_fn_varying_factory(performances)
    results = _run(
        _THREE_CANDIDATES, backtest_fn=backtest_fn,
        constraints=[{"metric": "mdd", "op": ">=", "value": -10.0}],
    )
    assert len(results) == 2
    assert all(r["performance"]["mdd"] >= -10.0 for r in results)


def test_search_strategy_applies_and_across_multiple_constraints():
    performances = [
        {"sharpe": 1.5, "mdd": -5.0},    # 둘 다 통과
        {"sharpe": 0.5, "mdd": -5.0},    # sharpe 미달
        {"sharpe": 2.0, "mdd": -20.0},   # mdd 미달
    ]
    backtest_fn = _fake_backtest_fn_varying_factory(performances)
    results = _run(
        _THREE_CANDIDATES, backtest_fn=backtest_fn,
        constraints=[
            {"metric": "mdd", "op": ">=", "value": -10.0},
            {"metric": "sharpe", "op": ">=", "value": 1.0},
        ],
    )
    assert len(results) == 1
    assert results[0]["performance"] == {"sharpe": 1.5, "mdd": -5.0}


def test_search_strategy_returns_empty_list_when_no_candidate_satisfies_constraints():
    performances = [{"sharpe": 0.1}, {"sharpe": 0.2}, {"sharpe": 0.3}]
    backtest_fn = _fake_backtest_fn_varying_factory(performances)
    results = _run(
        _THREE_CANDIDATES, backtest_fn=backtest_fn,
        constraints=[{"metric": "sharpe", "op": ">=", "value": 5.0}],
    )
    assert results == []


def test_search_strategy_rejects_unsupported_constraint_operator():
    with pytest.raises(ValueError):
        _run(
            [_THREE_CANDIDATES[0]],
            constraints=[{"metric": "mdd", "op": "~=", "value": -10.0}],
        )


def test_search_strategy_rejects_unsupported_operator_before_running_any_backtest():
    """architect 검토(MINOR 2): 오타 연산자 하나 때문에 무거운 백테스트를 다 돌린 뒤에야
    에러가 나면 안 된다 — 프로젝트 불변원칙(무거운 연산은 실행 전 사전검사)과 일치시킨다."""
    backtest_fn = _fake_backtest_fn_factory()
    callbacks_fn = _fake_callbacks_fn_factory()
    with pytest.raises(ValueError):
        _run(
            _THREE_CANDIDATES, backtest_fn=backtest_fn, callbacks_fn=callbacks_fn,
            constraints=[{"metric": "mdd", "op": "~=", "value": -10.0}],
        )
    assert len(callbacks_fn.calls) == 0
    assert len(backtest_fn.calls) == 0


def test_search_strategy_sorts_by_default_rank_by_sharpe_descending():
    performances = [{"sharpe": 0.5}, {"sharpe": 2.0}, {"sharpe": 1.0}]
    backtest_fn = _fake_backtest_fn_varying_factory(performances)
    results = _run(_THREE_CANDIDATES, backtest_fn=backtest_fn)
    assert [r["performance"]["sharpe"] for r in results] == [2.0, 1.0, 0.5]


def test_search_strategy_sorts_by_custom_rank_by():
    performances = [
        {"sharpe": 0.5, "cagr": 20.0},
        {"sharpe": 2.0, "cagr": 5.0},
        {"sharpe": 1.0, "cagr": 15.0},
    ]
    backtest_fn = _fake_backtest_fn_varying_factory(performances)
    results = _run(_THREE_CANDIDATES, backtest_fn=backtest_fn, rank_by="cagr")
    assert [r["performance"]["cagr"] for r in results] == [20.0, 15.0, 5.0]


def test_search_strategy_no_constraints_still_sorts_all_candidates():
    performances = [{"sharpe": 0.5}, {"sharpe": 2.0}, {"sharpe": 1.0}]
    backtest_fn = _fake_backtest_fn_varying_factory(performances)
    results = _run(_THREE_CANDIDATES, backtest_fn=backtest_fn, constraints=None)
    assert len(results) == 3
    assert [r["performance"]["sharpe"] for r in results] == [2.0, 1.0, 0.5]
