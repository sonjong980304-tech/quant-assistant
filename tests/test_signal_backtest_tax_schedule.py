"""run_signal_backtest()가 매매 체결일마다 실제 연도별 거래세율(stt_rate_at)을 적용하는지 검증.

engine.run_backtest와 동일한 버그: tax_rate 미지정 시 체결일과 무관하게 2024년 세율
(0.0018)을 고정 사용했다. 이 테스트는 진입/청산 체결이 2019-06-03 세율 인하 경계를
사이에 두고 걸쳐 있을 때 각각 다른 세율이 적용되는지 확인한다.
"""
from __future__ import annotations

import pytest

from src.backtest.signal_engine import run_signal_backtest


def _history_fn_factory(series_by_code: dict):
    def fn(conn, codes, asof, lookback_days):
        return {
            c: [{"date": d, "close": px} for d, px in series_by_code.get(c, [])]
            for c in codes
        }
    return fn


def _indicator_fn_factory(arrays_by_name_period: dict):
    def fn(name, closes, period=None):
        return list(arrays_by_name_period[(name, period)])
    return fn


# 진입(entry) 크로스는 인덱스2에서, 청산(exit) 크로스는 인덱스6에서 발생 → 2봉 지연으로
# 실제 체결(보유집합 변경)은 인덱스4(진입)/인덱스8(청산)에서 일어난다. 종가는 전 구간
# 고정 100이라 보유수익은 0 — NAV 변화는 거래비용만 반영한다.
_DATES10 = ["2019-05-01", "2019-05-02", "2019-05-03", "2019-05-04", "2019-06-02",
            "2019-06-04", "2019-06-05", "2019-06-06", "2019-08-08", "2019-08-09"]
_SHORT = [1, 1, 3, 3, 3, 3, 1, 1, 1, 1]
_LONG = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
_CLOSES = [100.0] * 10

_ENTRY_RULE = {"left": {"kind": "indicator", "name": "sma", "period": 2}, "op": "cross_above",
               "right": {"kind": "indicator", "name": "sma", "period": 3}}
_EXIT_RULE = {"left": {"kind": "indicator", "name": "sma", "period": 2}, "op": "cross_below",
              "right": {"kind": "indicator", "name": "sma", "period": 3}}


def _run(params):
    history_fn = _history_fn_factory({"AAA": list(zip(_DATES10, _CLOSES))})
    indicator_fn = _indicator_fn_factory({("sma", 2): _SHORT, ("sma", 3): _LONG})
    return run_signal_backtest(
        conn=None, stock_codes=["AAA"], start_date=_DATES10[0], end_date=_DATES10[-1],
        entry_rule=_ENTRY_RULE, exit_rule=_EXIT_RULE,
        params=params, price_history_fn=history_fn, indicator_series_fn=indicator_fn,
    )


def test_entry_and_exit_use_year_specific_tax_at_execution_date():
    # 진입 체결일 2019-06-02(인하 전, 0.30%) / 청산 체결일 2019-08-08(인하 후, 0.25%).
    out = _run({"fee_rate": 0.0, "slippage_rate": 0.0})  # tax_rate 미지정

    assert out["holdings"] == [{"date": "2019-06-02", "codes": ["AAA"],
                                 "period_return": pytest.approx(0.0)}]
    expected_nav = (1 - 0.0030) * (1 - 0.0025)
    assert out["navs"][-1] == pytest.approx(expected_nav, rel=1e-9)


def test_explicit_tax_rate_stays_fixed_across_entry_and_exit():
    out = _run({"fee_rate": 0.0, "slippage_rate": 0.0, "tax_rate": 0.05})

    expected_nav = (1 - 0.05) * (1 - 0.05)
    assert out["navs"][-1] == pytest.approx(expected_nav, rel=1e-9)
