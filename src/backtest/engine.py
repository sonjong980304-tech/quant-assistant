"""리밸런싱 백테스트 시뮬레이터.

불변 규칙
- look-ahead 방지: 각 리밸런싱 시점 t에서, t까지 실제 공시된 재무만 사용
  (metrics_fn이 t 시점 유효 지표를 돌려줄 책임을 진다)
- 생존편향 제거: universe_fn이 t 시점 실제 상장 종목(상폐 포함)을 돌려준다
- 거래비용 반드시 반영: 회전율 × (수수료·거래세·슬리피지)

데이터 접근은 콜백으로 추상화해 더미/실 DB 모두 동일 엔진으로 검증한다.
"""
from __future__ import annotations

from .performance import performance
from .selection import select_stocks

PERIODS_PER_YEAR = {"monthly": 12, "quarterly": 4, "semiannual": 2, "annual": 1}


def run_backtest(
    rebalance_dates: list[str],
    metrics_fn,            # (date) -> list[rows]  (look-ahead 적용된 시점 지표)
    price_fn,              # (date, code) -> close | None
    params: dict,
    benchmark_fn=None,     # (date) -> index level | None
    weights: dict | None = None,  # {종목:비중} 외부 비중벡터 입력 모드 (None이면 기존 동작 그대로)
) -> dict:
    """포트폴리오 리밸런싱 시뮬레이션.

    - weights=None(기본): criteria로 종목을 선정해 동일가중 리밸런싱(기존 동작, 하위호환).
    - weights={종목:비중}: 외부에서 계산된 비중벡터(예: optimize_weights)로 buy&hold 시뮬레이션.
      engine은 공매도를 표현할 수 없으므로 음수 비중이 오면 조용히 틀린 결과를 내지 않고
      명시적으로 거부(ValueError)한다.
    """
    combine = params.get("combine", "zscore")
    sectors = params.get("sectors")
    markets = params.get("markets")        # 시장 필터 ['KOSPI','KOSDAQ'] 또는 None(전체)
    rebalance = params.get("rebalance", "quarterly")

    fee = params.get("fee_rate", 0.00015)
    tax = params.get("tax_rate", 0.0018)
    slip = params.get("slippage_rate", 0.0010)
    # 1회 교체(매도+매수) 비용: 매수(수수료+슬리피지)+매도(수수료+거래세+슬리피지)
    cost_per_turn = (fee + slip) + (fee + tax + slip)

    if weights is not None:
        return _run_weighted_backtest(rebalance_dates, price_fn, weights, cost_per_turn,
                                      benchmark_fn, rebalance)

    n = params.get("n", 20)
    criteria = params["criteria"]
    nav = 1.0
    navs = [1.0]
    out_dates = [rebalance_dates[0]]
    bench = [benchmark_fn(rebalance_dates[0])] if benchmark_fn else None
    prev_codes: set[str] = set()
    turnovers: list[float] = []
    holdings_log = []

    for i in range(len(rebalance_dates) - 1):
        t, t_next = rebalance_dates[i], rebalance_dates[i + 1]
        rows = metrics_fn(t)
        selected = select_stocks(rows, criteria, combine, n, sectors, markets)
        codes = [r["stock_code"] for r in selected]
        holdings_log.append({"date": t, "codes": codes})
        if not codes:
            holdings_log[-1]["period_return"] = 0.0  # 편입 종목 없음 → 구간수익 0
            navs.append(nav)
            out_dates.append(t_next)
            if bench is not None:
                bench.append(benchmark_fn(t_next))
            continue

        new = set(codes)
        # 회전율: 직전 보유 대비 교체 비중 (0~1)
        if prev_codes:
            turnover = len(new - prev_codes) / len(new)
        else:
            turnover = 1.0  # 최초 편입
        turnovers.append(turnover)
        nav *= (1 - turnover * cost_per_turn)

        # 보유 수익 (t → t_next, 동일가중)
        rets = []
        for c in codes:
            p0, p1 = price_fn(t, c), price_fn(t_next, c)
            if p0 and p1 and p0 > 0:
                rets.append(p1 / p0 - 1)
        period_ret = sum(rets) / len(rets) if rets else 0.0
        holdings_log[-1]["period_return"] = period_ret  # 이 리밸런싱 구간의 보유 수익률(순수 추가)
        nav *= (1 + period_ret)

        navs.append(nav)
        out_dates.append(t_next)
        if bench is not None:
            bench.append(benchmark_fn(t_next))
        prev_codes = new

    ppy = PERIODS_PER_YEAR.get(rebalance, 4)
    bench_clean = bench if (bench and all(b is not None for b in bench)) else None
    perf = performance(navs, ppy, benchmark=bench_clean)
    perf["avg_turnover"] = round(sum(turnovers) / len(turnovers) * 100, 1) if turnovers else 0.0

    return {
        "dates": out_dates,
        "navs": navs,
        "benchmark": bench_clean,
        "performance": perf,
        "holdings": holdings_log,
    }


def _run_weighted_backtest(rebalance_dates, price_fn, weights, cost_per_turn, benchmark_fn, rebalance):
    """외부 비중벡터로 buy&hold 시뮬레이션. 음수 비중은 거부(공매도 불가)."""
    neg = {k: v for k, v in weights.items() if v is not None and v < 0}
    if neg:
        raise ValueError(f"음수 비중(공매도)은 run_backtest에서 지원하지 않습니다: {neg}")
    total = sum(v for v in weights.values() if v is not None)
    if total <= 0:
        raise ValueError("비중 합이 0 이하라 백테스트할 수 없습니다")

    codes = [c for c, v in weights.items() if v]
    # 각 종목 슬리브 가치(초기 = 정규화 목표비중, 총합 1) — 진입비용(turnover=1) 1회 반영
    sleeves = {c: (weights[c] / total) * (1 - cost_per_turn) for c in codes}
    navs = [1.0]
    out_dates = [rebalance_dates[0]]
    bench = [benchmark_fn(rebalance_dates[0])] if benchmark_fn else None

    for i in range(len(rebalance_dates) - 1):
        t, t_next = rebalance_dates[i], rebalance_dates[i + 1]
        for c in codes:
            p0, p1 = price_fn(t, c), price_fn(t_next, c)
            if p0 and p1 and p0 > 0:
                sleeves[c] *= (p1 / p0)  # buy&hold 복리 성장
        navs.append(sum(sleeves.values()))
        out_dates.append(t_next)
        if bench is not None:
            bench.append(benchmark_fn(t_next))

    ppy = PERIODS_PER_YEAR.get(rebalance, 4)
    bench_clean = bench if (bench and all(b is not None for b in bench)) else None
    perf = performance(navs, ppy, benchmark=bench_clean)
    perf["avg_turnover"] = 100.0  # 진입 시 1회 전액 매수(이후 buy&hold)
    return {
        "dates": out_dates,
        "navs": navs,
        "benchmark": bench_clean,
        "performance": perf,
        "holdings": [{"date": rebalance_dates[0], "codes": codes}],
        "weights": {c: weights[c] / total for c in codes},
    }


def save_backtest_run(conn, name: str, params: dict, perf: dict, start_year: int, end_year: int) -> int:
    import json

    from ..version import now_iso

    cur = conn.execute(
        """INSERT INTO backtest_runs(name, params_json, cost_json, start_year, end_year, result_json, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (
            name,
            json.dumps({k: v for k, v in params.items() if k != "criteria"} | {"criteria": params.get("criteria")}, ensure_ascii=False),
            json.dumps({k: params.get(k) for k in ("fee_rate", "tax_rate", "slippage_rate")}, ensure_ascii=False),
            start_year,
            end_year,
            json.dumps(perf, ensure_ascii=False),
            now_iso(),
        ),
    )
    conn.commit()
    return cur.lastrowid
