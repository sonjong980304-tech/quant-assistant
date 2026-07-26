"""run_backtest()가 리밸런싱 시점마다 실제 연도별 거래세율(stt_rate_at)을 적용하는지 검증.

기존 동작(tax_rate 미지정 시 2024년 세율 0.0018 고정)은 다년 백테스트에서 과거 구간
비용을 최대 40%까지 과소평가했다(references/securities_transaction_tax_rate_history.md).
- tax_rate를 아예 안 주면(None) 각 리밸런싱 시점의 실제 세율을 조회해야 한다.
- tax_rate를 명시하면(0.0 포함) 그 값이 전 구간에 고정 적용되는 기존 동작을 유지해야 한다.
"""
from __future__ import annotations

import pytest

from src.backtest.engine import run_backtest


def test_criteria_mode_applies_year_specific_tax_across_periods():
    # 두 구간이 2019-06-03 세율 인하 경계를 사이에 두고 걸쳐 있다: 0.30% -> 0.25%.
    dates = ["2019-06-02", "2019-08-02", "2019-10-02"]
    scores = {
        "2019-06-02": {"AAA": 10, "BBB": 1},   # 1구간: AAA 선정
        "2019-08-02": {"AAA": 1, "BBB": 10},   # 2구간: BBB로 전량 교체(회전율 100%)
    }

    def metrics_fn(date):
        s = scores[date]
        return [{"stock_code": "AAA", "score": s["AAA"]},
                {"stock_code": "BBB", "score": s["BBB"]}]

    def price_fn(date, code):
        return 100.0  # 가격 변화 없음: NAV 변화는 거래비용만 반영

    res = run_backtest(
        dates, metrics_fn, price_fn,
        params={"criteria": [{"key": "score", "direction": "high", "weight": 1}],
                "n": 1, "fee_rate": 0.0, "slippage_rate": 0.0},  # tax_rate 미지정
    )

    expected_nav = (1 - 0.0030) * (1 - 0.0025)
    assert res["navs"][-1] == pytest.approx(expected_nav, rel=1e-9)


def test_criteria_mode_explicit_tax_rate_stays_fixed_across_periods():
    dates = ["2019-06-02", "2019-08-02", "2019-10-02"]
    scores = {
        "2019-06-02": {"AAA": 10, "BBB": 1},
        "2019-08-02": {"AAA": 1, "BBB": 10},
    }

    def metrics_fn(date):
        s = scores[date]
        return [{"stock_code": "AAA", "score": s["AAA"]},
                {"stock_code": "BBB", "score": s["BBB"]}]

    def price_fn(date, code):
        return 100.0

    res = run_backtest(
        dates, metrics_fn, price_fn,
        params={"criteria": [{"key": "score", "direction": "high", "weight": 1}],
                "n": 1, "fee_rate": 0.0, "slippage_rate": 0.0, "tax_rate": 0.05},
    )

    expected_nav = (1 - 0.05) * (1 - 0.05)
    assert res["navs"][-1] == pytest.approx(expected_nav, rel=1e-9)


def test_weights_mode_uses_entry_date_for_tax_schedule_lookup():
    prices = {"AAA": 100.0, "BBB": 100.0}

    def price_fn(date, code):
        return prices[code]

    pre_cut = run_backtest(
        ["2019-06-02", "2019-07-02"], metrics_fn=lambda d: [], price_fn=price_fn,
        params={"fee_rate": 0.0, "slippage_rate": 0.0}, weights={"AAA": 0.6, "BBB": 0.4},
    )
    post_cut = run_backtest(
        ["2019-06-03", "2019-07-03"], metrics_fn=lambda d: [], price_fn=price_fn,
        params={"fee_rate": 0.0, "slippage_rate": 0.0}, weights={"AAA": 0.6, "BBB": 0.4},
    )

    assert pre_cut["navs"][-1] == pytest.approx(1 - 0.0030, rel=1e-9)
    assert post_cut["navs"][-1] == pytest.approx(1 - 0.0025, rel=1e-9)
