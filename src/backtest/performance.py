"""백테스트 성과지표.

NAV(순자산가치) 시계열로부터:
- 누적/연환산 수익률(CAGR), MDD, 변동성
- 샤프지수, 소르티노지수, 승률
- (벤치마크 제공 시) 초과수익, 베타
회전율은 리밸런싱 시뮬레이터가 별도로 계산해 합산한다.
"""
from __future__ import annotations

import numpy as np


def performance(navs: list[float], periods_per_year: float, benchmark: list[float] | None = None,
                rf: float = 0.0) -> dict:
    nav = np.asarray(navs, dtype=float)
    if len(nav) < 2:
        return {"error": "데이터 부족(시점 2개 미만)"}
    rets = nav[1:] / nav[:-1] - 1
    n_periods = len(rets)
    years = n_periods / periods_per_year if periods_per_year else 0

    total_return = nav[-1] / nav[0] - 1
    cagr = (nav[-1] / nav[0]) ** (1 / years) - 1 if years > 0 else 0.0

    peak = np.maximum.accumulate(nav)
    drawdown = (nav - peak) / peak
    mdd = float(drawdown.min())

    vol = float(rets.std(ddof=1) * np.sqrt(periods_per_year)) if n_periods > 1 else 0.0
    sharpe = (cagr - rf) / vol if vol > 0 else 0.0

    downside = rets[rets < 0]
    dvol = float(downside.std(ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else 0.0
    sortino = (cagr - rf) / dvol if dvol > 0 else 0.0

    win_rate = float((rets > 0).mean() * 100)

    out = {
        "total_return": round(total_return * 100, 2),
        "cagr": round(cagr * 100, 2),
        "mdd": round(mdd * 100, 2),
        "volatility": round(vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(win_rate, 1),
        "periods": n_periods,
    }

    if benchmark is not None and len(benchmark) == len(nav):
        b = np.asarray(benchmark, dtype=float)
        brets = b[1:] / b[:-1] - 1
        b_total = b[-1] / b[0] - 1
        out["benchmark_return"] = round(b_total * 100, 2)
        out["excess_return"] = round((total_return - b_total) * 100, 2)
        if brets.std() > 1e-6:
            # np.cov는 기본 ddof=1(N-1)인데 ndarray.var()는 기본 ddof=0(N)이라 분자·분모의
            # 추정량이 어긋나 베타가 항상 N/(N-1)배 과대계산됐다(son-checker 이슈 #23 BUG-2).
            # 분모를 공분산과 같은 ddof=1로 맞춘다.
            beta = float(np.cov(rets, brets)[0, 1] / brets.var(ddof=1))
            out["beta"] = round(beta, 3)
    return out
