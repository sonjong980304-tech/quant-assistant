"""올웨더 몬테카를로 비중 최적화 테스트 (AC4/AC5 + 종목당 비중 상하한).

.omc/specs/brainstorming-all-weather-portfolio.md 참고.
quant_trader/portfolio/rebalancer.py의 몬테카를로 계산식(연율화, sharpe 공식)은 그대로 따르되,
2026-07-17 21.7년 실측 검증 결과 무제약 샤프비율 극대화가 한두 종목(QQQ+삼성전자)에 99% 가까이
쏠리는 코너 솔루션으로 수렴하는 것을 확인해, 종목당 최소10%~최대45% 비중 상하한을 추가했다
(사용자 결정 — quant_trader 원본과의 유일한 실질적 차이). 계산은 순수 로직(DB/네트워크 의존
없음)이라 작은 시뮬레이션 횟수로 빠르게 검증하고, N_SIMULATIONS 상수(100,000)만 별도로
고정한다(실제 배치는 이 상수를 쓴다).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.allweather.data import TICKERS
from src.allweather.montecarlo import (
    MAX_WEIGHT,
    MIN_WEIGHT,
    N_SIMULATIONS,
    run_monte_carlo,
    run_monte_carlo_mdd_constrained,
)


def _panel(n: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    data = {}
    for i, c in enumerate(["QQQ", "005930.KS", "TLT", "411060.KS"]):
        steps = rng.normal(0.0005 * (i + 1), 0.01, size=n)
        data[c] = 100 * np.exp(np.cumsum(steps))
    return pd.DataFrame(data, index=dates)


def test_n_simulations_constant_is_100k():
    # AC5: 시뮬레이션 횟수 상수는 quant_trader와 동일한 100,000.
    assert N_SIMULATIONS == 100_000


def test_run_monte_carlo_returns_normalized_weights_over_all_tickers():
    p = _panel()
    res = run_monte_carlo(p, risk_free_rate=0.045, n_simulations=2000, seed=42)
    assert set(res["weights"].keys()) == set(p.columns)
    assert abs(sum(res["weights"].values()) - 1.0) < 1e-3
    for w in res["weights"].values():
        assert 0.0 <= w <= 1.0


def test_sharpe_reflects_risk_free_rate():
    # sharpe = (annual_return - rf)/annual_vol 를 재구성해 rf가 실제 계산에 쓰였는지 확인(AC6/AC7 근거).
    p = _panel()
    res = run_monte_carlo(p, risk_free_rate=0.045, n_simulations=3000, seed=7)
    recon = res["sharpe"] * res["annual_vol"] + 0.045
    assert abs(recon - res["annual_return"]) < 1e-2


def test_higher_risk_free_rate_lowers_max_sharpe_for_same_seed():
    # 같은 시드(동일 후보 포트폴리오)면 무위험이자율이 높을수록 최대 샤프비율은 낮아진다.
    p = _panel()
    lo = run_monte_carlo(p, risk_free_rate=0.00, n_simulations=3000, seed=11)
    hi = run_monte_carlo(p, risk_free_rate=0.20, n_simulations=3000, seed=11)
    assert hi["sharpe"] < lo["sharpe"]


def test_weights_respect_min_and_max_bounds():
    # 21.7년 실측에서 무제약 방식이 한두 종목에 99% 가까이 쏠리는 것을 확인해 상하한을 추가했다.
    p = _panel()
    res = run_monte_carlo(p, risk_free_rate=0.045, n_simulations=3000, seed=5)
    for w in res["weights"].values():
        assert MIN_WEIGHT - 1e-9 <= w <= MAX_WEIGHT + 1e-9


def test_bounds_prevent_near_zero_concentration_even_when_one_asset_dominates():
    # 한 종목이 압도적으로 좋고 다른 종목이 마이너스여도(TLT가 2022년에 그랬듯) 상하한 때문에
    # 그 종목이 0%까지 밀리지는 않는다 — "안전자산은 성과가 나빠도 최소한은 들고 간다"는 취지.
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    rng = np.random.default_rng(1)
    data = {
        "QQQ": 100 * np.exp(np.cumsum(rng.normal(0.003, 0.01, size=200))),
        "005930.KS": 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, size=200))),
        "TLT": 100 * np.exp(np.cumsum(rng.normal(-0.001, 0.008, size=200))),
        "411060.KS": 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.012, size=200))),
    }
    p = pd.DataFrame(data, index=dates)
    res = run_monte_carlo(p, risk_free_rate=0.02, n_simulations=5000, seed=3)
    for w in res["weights"].values():
        assert w >= MIN_WEIGHT - 1e-9


def _panel6(n: int = 250, seed: int = 0) -> pd.DataFrame:
    """6종목(IEF/TIP 추가 후 실제 유니버스) 규모의 합성 가격 패널. TLT를 일부러 변동성이 커서
    낙폭이 깊게 나도록 만들어(mu 낮고 sigma 큼) MDD 제약이 실제로 작동할 여지를 만든다."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    specs = {
        "QQQ": (0.0009, 0.012), "005930.KS": (0.0004, 0.011),
        "TLT": (-0.0004, 0.02), "411060.KS": (0.0003, 0.009),
        "IEF": (0.0001, 0.006), "TIP": (0.0001, 0.005),
    }
    data = {c: 100 * np.exp(np.cumsum(rng.normal(mu, sigma, size=n))) for c, (mu, sigma) in specs.items()}
    return pd.DataFrame(data, index=dates)


# ── MDD 제약(-20% 이내) 샤프비율 최대 탐색 — 2026-07 사용자 결정 ────────────────────────
# 기존 run_monte_carlo는 평균/공분산만으로 샤프비율을 계산해 MDD 개념이 아예 없다(낙폭은
# 워크포워드 백테스트가 실현 곡선에서 사후 계산). "그 구간에서 실제로 이 비중을 들고 있었으면
# 낙폭이 얼마였을까"를 각 후보의 실제 가격 경로로 계산해, 제약을 만족하는 후보 중에서만
# 샤프비율을 최대화한다.

def test_mdd_constrained_result_satisfies_the_limit_when_feasible():
    p = _panel6()
    res = run_monte_carlo_mdd_constrained(
        p, risk_free_rate=0.02, max_drawdown=-0.20, n_simulations=3000, seed=1,
    )
    assert res["constraint_satisfied"] is True
    assert res["mdd"] >= -0.20 - 1e-9
    assert set(res["weights"].keys()) == set(p.columns)
    assert abs(sum(res["weights"].values()) - 1.0) < 1e-3


def test_mdd_constrained_reduces_to_unconstrained_when_limit_is_loose():
    # 제약이 사실상 무의미할 만큼 느슨하면(-0.99) 무제약 최적과 동일한 후보를 골라야 한다
    # (같은 시드 = 같은 후보 풀이므로 완전히 같은 값이 나와야 함).
    p = _panel6()
    unconstrained = run_monte_carlo(p, risk_free_rate=0.02, n_simulations=3000, seed=9)
    constrained = run_monte_carlo_mdd_constrained(
        p, risk_free_rate=0.02, max_drawdown=-0.99, n_simulations=3000, seed=9,
    )
    assert constrained["weights"] == unconstrained["weights"]
    assert constrained["sharpe"] == unconstrained["sharpe"]
    assert constrained["constraint_satisfied"] is True


def test_mdd_constrained_sharpe_never_exceeds_unconstrained_sharpe():
    # 제약 조건 안에서 찾은 최적은 무제약 최적보다 샤프비율이 더 좋을 수 없다(부분집합 안 최적화).
    p = _panel6()
    unconstrained = run_monte_carlo(p, risk_free_rate=0.02, n_simulations=4000, seed=3)
    constrained = run_monte_carlo_mdd_constrained(
        p, risk_free_rate=0.02, max_drawdown=-0.20, n_simulations=4000, seed=3,
    )
    assert constrained["sharpe"] <= unconstrained["sharpe"] + 1e-9


def test_mdd_constrained_falls_back_honestly_when_no_candidate_satisfies():
    # 현실적으로 달성 불가능한 낙폭(-0.01%)을 요구하면 만족하는 후보가 없다 — 크래시 대신
    # "가장 낙폭이 얕았던 후보"로 폴백하고 constraint_satisfied=False로 정직하게 표시한다.
    p = _panel6()
    res = run_monte_carlo_mdd_constrained(
        p, risk_free_rate=0.02, max_drawdown=-0.0001, n_simulations=2000, seed=2,
    )
    assert res["constraint_satisfied"] is False
    assert res["weights"] is not None
    assert abs(sum(res["weights"].values()) - 1.0) < 1e-3


def test_mdd_constrained_weights_respect_min_and_max_bounds():
    p = _panel6()
    res = run_monte_carlo_mdd_constrained(
        p, risk_free_rate=0.02, max_drawdown=-0.20, n_simulations=3000, seed=4,
    )
    for w in res["weights"].values():
        assert MIN_WEIGHT - 1e-9 <= w <= MAX_WEIGHT + 1e-9


def test_min_weight_times_asset_count_is_feasible():
    # 상하한 자체가 모순이면(예: 종목수*최소비중 > 1) 어떤 조합도 못 뽑는다 — 실제 운용 종목수
    # (data.TICKERS, IEF/TIP 추가로 4→6) 기준으로 항상 실현 가능해야 한다는 전제를 고정해둔다
    # (회귀 방지 — 종목수가 앞으로 또 바뀌어도 이 테스트가 하드코딩 없이 자동으로 따라간다).
    n = len(TICKERS)
    assert MIN_WEIGHT * n <= 1.0
    assert MAX_WEIGHT * n >= 1.0
