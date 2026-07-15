"""src/backtest/performance.py 단위 테스트 (TDD).

son-checker 이슈 #23 BUG-2: beta 계산에서 np.cov(ddof=1 기본)와 ndarray.var(ddof=0 기본)의
추정량이 어긋나 베타가 과대계산되던 문제의 회귀 테스트. 이 파일이 생기기 전에는
performance()를 직접 겨냥한 테스트가 전혀 없었다(엔진 경유 간접 테스트뿐).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.backtest.performance import performance


def _ols_beta(rets: np.ndarray, brets: np.ndarray) -> float:
    """OLS 회귀계수(베타)의 독립적 정의: cov와 var이 같은 ddof를 쓰면 (n-1)이 상쇄되어
    이 형태(평균편차곱 합 / 평균편차제곱 합)와 항상 같아야 한다. ddof 선택에 의존하지
    않는 검증식이므로, performance()의 내부 구현이 무엇이든 이 식과 일치해야 정답이다.
    """
    r_dev = rets - rets.mean()
    b_dev = brets - brets.mean()
    return float((r_dev * b_dev).sum() / (b_dev ** 2).sum())


def test_beta_matches_ols_slope_definition_ddof_consistent():
    navs = [1.0, 1.1, 1.045, 1.0659, 1.097877]
    benchmark = [1.0, 1.08, 1.0476, 1.058076, 1.0792375]

    result = performance(navs, periods_per_year=4, benchmark=benchmark)

    rets = np.asarray(navs[1:]) / np.asarray(navs[:-1]) - 1
    brets = np.asarray(benchmark[1:]) / np.asarray(benchmark[:-1]) - 1
    expected_beta = _ols_beta(rets, brets)

    assert result["beta"] == pytest.approx(round(expected_beta, 3), abs=1e-6)


def test_beta_absent_when_benchmark_flat():
    navs = [1.0, 1.05, 1.1, 1.02]
    benchmark = [1.0, 1.0, 1.0, 1.0]

    result = performance(navs, periods_per_year=4, benchmark=benchmark)

    assert "beta" not in result


def test_performance_smoke_metrics_present():
    navs = [1.0, 1.1, 1.05, 1.2, 1.15]

    result = performance(navs, periods_per_year=4)

    assert result["total_return"] == pytest.approx(15.0, abs=0.01)
    assert result["periods"] == 4
    assert "cagr" in result
    assert "mdd" in result
    assert result["mdd"] <= 0
    assert "sharpe" in result
    assert "sortino" in result
    assert "win_rate" in result


def test_performance_insufficient_data_returns_error():
    result = performance([1.0], periods_per_year=4)

    assert "error" in result
