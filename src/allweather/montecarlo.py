"""몬테카를로 비중 최적화 — 종목당 최소10%~최대45% 상하한 (AC4/AC5 + 상하한 결정).

.omc/specs/brainstorming-all-weather-portfolio.md AC4/AC5 참고.

원래 이 함수는 quant_trader/portfolio/rebalancer.py::run_monte_carlo 의 계산 로직을 그대로
복제한 것이었다(무제약 샤프비율 극대화). 그런데 2026-07-17 실제 21.7년 데이터로 검증해보니,
무제약 방식이 특정 구간(예: 최근 10년 lookback)에서 QQQ+삼성전자에 99% 가까이 쏠리는 코너
솔루션으로 수렴하는 것을 확인했다 — TLT/금현물처럼 그 구간 수익률이 낮거나 마이너스인 자산은
0%에 가깝게 밀려나, "올웨더(전천후)"라는 취지와 어긋났다. 그래서 종목당 최소10%~최대45% 비중
상하한을 추가했다(사용자 결정, quant_trader 원본과의 실질적 차이는 이 상하한 하나뿐 — 연율화
(×252/√252)·샤프비율 공식(sharpe=(ret-rf)/vol)·argmax 방식은 원본과 동일).

quant_trader 원본과의 차이:
  1) RISK_FREE_RATE(원본 고정 0.045) → 인자 risk_free_rate 로 분리(AC6/AC7, walk-forward가
     리밸런싱 시점마다 ^IRX 값을 넣어준다).
  2) N_SIMULATIONS/seed 를 인자로 노출 — 기본값은 원본과 동일(100,000).
  3) [신규] 종목당 비중 상하한(MIN_WEIGHT~MAX_WEIGHT) — 원본엔 없던 제약. 거부샘플링(rejection
     sampling)으로 상하한을 만족하는 조합만 후보로 남긴다.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

# quant_trader rebalancer.py 와 동일한 시뮬레이션 횟수(AC5).
N_SIMULATIONS = 100_000

# 종목당 비중 상하한 — 무제약 샤프비율 극대화가 코너 솔루션(한두 종목 몰빵)으로 수렴하는 것을
# 막기 위한 신규 제약(2026-07-17 실측 검증 후 결정). 6종목(IEF/TIP 추가 후) 기준
# 10%*6=60%<=1.0, 45%*6=270%>=1.0 이라 여전히 실현 가능.
MIN_WEIGHT = 0.10
MAX_WEIGHT = 0.45

# MDD 제약 몬테카를로(run_monte_carlo_mdd_constrained)의 기본 낙폭 허용치 — "MDD -20% 이내에서
# 샤프비율 최대"(2026-07 사용자 결정). 0 이하 소수(예: -0.20)로 표기해 max_drawdown 값과
# 부호를 맞춘다(더 얕은 낙폭일수록 값이 0에 가까움 = 더 큼).
DEFAULT_MAX_DRAWDOWN = -0.20


def _sample_bounded_weights(n_assets: int, n_simulations: int, rng: np.random.Generator) -> np.ndarray:
    """상하한(MIN_WEIGHT~MAX_WEIGHT)을 만족하는 비중 조합을 n_simulations개 뽑는다.

    Dirichlet(1,...,1)은 단체(simplex) 위 균등분포라 원본의 uniform-정규화 방식보다 편향이 적다.
    거부샘플링: 상하한을 만족하는 것만 남기고, 부족하면 더 뽑는다(최대 10회 시도).
    상하한 자체가 실현 불가능한 조합이면(자산 수 대비 모순) 균등비중 1개로 폴백한다.
    """
    accepted: list[np.ndarray] = []
    total = 0
    for _ in range(10):
        if sum(len(a) for a in accepted) >= n_simulations:
            break
        batch = n_simulations * 3
        candidates = rng.dirichlet(np.ones(n_assets), size=batch)
        mask = (candidates.min(axis=1) >= MIN_WEIGHT) & (candidates.max(axis=1) <= MAX_WEIGHT)
        accepted.append(candidates[mask])
        total += batch

    pool = np.concatenate(accepted) if accepted else np.empty((0, n_assets))
    if len(pool) == 0:
        return np.full((1, n_assets), 1.0 / n_assets)
    return pool[:n_simulations]


def run_monte_carlo(
    prices: pd.DataFrame,
    risk_free_rate: float,
    n_simulations: int = N_SIMULATIONS,
    seed: int | None = None,
) -> dict:
    """상하한(MIN_WEIGHT~MAX_WEIGHT) 안에서 N회 몬테카를로로 최대 샤프비율 포트폴리오를 산출한다.

    반환:
      weights      : {ticker: 최적 비중} (모두 MIN_WEIGHT~MAX_WEIGHT 이내)
      annual_return: 예상 연간 수익률
      annual_vol   : 예상 연간 변동성
      sharpe       : 샤프비율
      corr         : 상관계수 행렬 (dict)
      period_start/period_end : 사용한 데이터 구간
    """
    returns = prices.pct_change().dropna()
    mean_ret = returns.mean().values
    cov_matrix = returns.cov().values
    n_assets = len(returns.columns)
    tickers = list(returns.columns)

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(int(datetime.now().strftime("%Y%m")))

    W = _sample_bounded_weights(n_assets, n_simulations, rng)

    # macOS Accelerate BLAS의 알려진 부작용으로 대량 matmul에서 divide-by-zero/overflow/invalid
    # RuntimeWarning이 허위로 뜰 수 있다(실측 확인: W/mean_ret에 NaN 없고 결과값도 정상 범위 —
    # SIMD 연산 중 미사용 메모리 레인을 스치면서 나는 경고일 뿐, 실제 계산 오류 아님).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        port_ret = (W @ mean_ret) * 252
        ann_cov = cov_matrix * 252
        port_vol = np.sqrt(np.einsum("ij,jk,ik->i", W, ann_cov, W))
    sharpe = np.where(port_vol > 0, (port_ret - risk_free_rate) / port_vol, 0.0)

    best_idx = int(sharpe.argmax())
    best_w = W[best_idx]

    weights = {t: round(float(w), 4) for t, w in zip(tickers, best_w)}
    corr = returns.corr().round(3).to_dict()

    return {
        "weights": weights,
        "annual_return": round(float(port_ret[best_idx]), 4),
        "annual_vol": round(float(port_vol[best_idx]), 4),
        "sharpe": round(float(sharpe[best_idx]), 4),
        "corr": corr,
        "period_start": str(prices.index[0].date()),
        "period_end": str(prices.index[-1].date()),
    }


def _historical_mdd_for_weights(returns: np.ndarray, W: np.ndarray) -> np.ndarray:
    """각 후보 비중(W, (n_candidates, n_assets))으로 그 구간을 실제로 들고 있었을 때의
    낙폭(MDD, 0 이하 소수)을 반환한다((n_candidates,)).

    run_monte_carlo의 평균/공분산 기반 샤프비율과 달리, 여기는 그 구간의 일별 실현수익률
    (returns, (n_days, n_assets))을 그대로 따라가는 실제 NAV 경로를 만들어 계산한다 —
    "이 비중으로 그 기간을 실제로 버텼으면 최대 몇 % 빠졌을까"를 재현하는 것이 목적이라
    평균/분산으로는 대체할 수 없다(워크포워드 백테스트의 max_drawdown과 동일 정의).
    """
    # run_monte_carlo와 동일한 macOS Accelerate BLAS 허위 경고 대상(대량 matmul) — 실제 계산
    # 오류가 아니다(위 run_monte_carlo 상단 주석 참고).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        port_daily = returns @ W.T  # (n_days, n_candidates)
        nav = np.cumprod(1.0 + port_daily, axis=0)
        running_max = np.maximum.accumulate(nav, axis=0)
        drawdown = nav / running_max - 1.0
    return drawdown.min(axis=0)


def run_monte_carlo_mdd_constrained(
    prices: pd.DataFrame,
    risk_free_rate: float,
    max_drawdown: float = DEFAULT_MAX_DRAWDOWN,
    n_simulations: int = N_SIMULATIONS,
    seed: int | None = None,
    initial_scan: int = 2000,
) -> dict:
    """그 구간 실제 가격 경로 기준 낙폭이 max_drawdown(예: -0.20) 이내인 후보 중에서
    샤프비율이 가장 높은 포트폴리오를 찾는다(2026-07 사용자 결정: "MDD -20% 이내에서
    샤프비율 최고"). 종목당 비중 상하한(MIN_WEIGHT~MAX_WEIGHT)은 run_monte_carlo와 동일.

    run_monte_carlo는 평균/공분산만으로 샤프비율을 계산해 낙폭 개념이 없다 — 실측(2026-07-22
    스냅샷)으로도 그 결과가 MDD -34%까지 벌어질 수 있음을 확인했다. "전천후"라는 취지에 맞게
    후보를 샤프비율 내림차순으로 스캔하며 그 구간 실제 가격 경로로 낙폭을 계산해, 제약을
    만족하는 첫 후보를 채택한다(100,000개 전부의 낙폭을 미리 계산하는 대신 상위권부터
    점증 확장 — 대개 상위 몇 개 안에서 찾아지므로 낭비를 피한다). 제약을 만족하는 후보가
    끝내 없으면(상하한 자체가 빡빡하거나 그 구간이 유난히 험난했던 경우) 지금까지 본 것 중
    낙폭이 가장 얕았던 후보로 폴백하고, constraint_satisfied=False로 정직하게 표시한다
    (SoT: 만족 못 했는데 만족한 것처럼 조용히 넘어가지 않는다).
    """
    returns_df = prices.pct_change().dropna()
    returns = returns_df.values
    mean_ret = returns_df.mean().values
    cov_matrix = returns_df.cov().values
    n_assets = len(returns_df.columns)
    tickers = list(returns_df.columns)

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(int(datetime.now().strftime("%Y%m")))

    W = _sample_bounded_weights(n_assets, n_simulations, rng)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        port_ret = (W @ mean_ret) * 252
        ann_cov = cov_matrix * 252
        port_vol = np.sqrt(np.einsum("ij,jk,ik->i", W, ann_cov, W))
    sharpe = np.where(port_vol > 0, (port_ret - risk_free_rate) / port_vol, 0.0)

    order = np.argsort(-sharpe)  # 샤프비율 내림차순

    best_idx: int | None = None
    best_mdd_seen = -np.inf
    best_mdd_idx = int(order[0])
    checked = 0
    chunk = max(1, min(initial_scan, len(order)))
    while checked < len(order):
        idx_chunk = order[checked:checked + chunk]
        mdd_chunk = _historical_mdd_for_weights(returns, W[idx_chunk])

        local_best_pos = int(np.argmax(mdd_chunk))  # MDD는 0 이하 → argmax=가장 얕음
        if mdd_chunk[local_best_pos] > best_mdd_seen:
            best_mdd_seen = float(mdd_chunk[local_best_pos])
            best_mdd_idx = int(idx_chunk[local_best_pos])

        feasible_pos = np.where(mdd_chunk >= max_drawdown)[0]
        if len(feasible_pos):
            best_idx = int(idx_chunk[feasible_pos[0]])  # 청크 안에서도 샤프 내림차순이라 첫 feasible이 최적
            break

        checked += chunk
        chunk = min(chunk * 2, len(order) - checked)

    constraint_satisfied = best_idx is not None
    if best_idx is None:
        best_idx = best_mdd_idx

    best_w = W[best_idx]
    best_mdd = float(_historical_mdd_for_weights(returns, best_w[None, :])[0])

    weights = {t: round(float(w), 4) for t, w in zip(tickers, best_w)}
    corr = returns_df.corr().round(3).to_dict()

    return {
        "weights": weights,
        "annual_return": round(float(port_ret[best_idx]), 4),
        "annual_vol": round(float(port_vol[best_idx]), 4),
        "sharpe": round(float(sharpe[best_idx]), 4),
        "mdd": round(best_mdd, 6),
        "constraint_satisfied": constraint_satisfied,
        "corr": corr,
        "period_start": str(prices.index[0].date()),
        "period_end": str(prices.index[-1].date()),
    }
