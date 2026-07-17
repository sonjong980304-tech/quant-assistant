"""run_backtest() 외부 비중벡터 입력 모드 + 차트 PNG 출력 테스트 (US-12).

- optimize_weights가 낸 {종목:비중}을 넘겨 실제 수익률 시뮬레이션을 수행한다.
- 기존 호출부(criteria 기반)는 하위호환 유지(weights=None이면 기존 동작 그대로).
- 음수 비중(공매도)은 명시적으로 거부(ValueError) — engine은 공매도를 표현할 수 없다.
- 차트는 matplotlib으로 PNG 파일 생성 후 경로 반환(터미널에 직접 그리지 않음).
"""
from __future__ import annotations

import pytest

from src.backtest.chart import save_nav_chart
from src.backtest.engine import run_backtest


# 합성 가격: (date, code) -> close
_PRICES = {
    ("2026-01-31", "AAA"): 100.0, ("2026-02-28", "AAA"): 110.0, ("2026-03-31", "AAA"): 121.0,
    ("2026-01-31", "BBB"): 50.0,  ("2026-02-28", "BBB"): 55.0,  ("2026-03-31", "BBB"): 60.5,
}
_DATES = ["2026-01-31", "2026-02-28", "2026-03-31"]


def _price_fn(date, code):
    return _PRICES.get((date, code))


def test_weights_mode_runs_simulation_and_returns_performance():
    res = run_backtest(_DATES, metrics_fn=lambda d: [], price_fn=_price_fn,
                       params={"fee_rate": 0.0, "tax_rate": 0.0, "slippage_rate": 0.0},
                       weights={"AAA": 0.6, "BBB": 0.4})
    assert res["navs"][0] == 1.0
    # 두 종목 모두 +10%/월 → nav도 약 +10%/월(비용 0)
    assert res["navs"][-1] == pytest.approx(1.21, rel=1e-6)
    assert "cagr" in res["performance"]
    assert res["dates"] == _DATES


def test_weights_mode_rejects_negative_weights():
    with pytest.raises(ValueError):
        run_backtest(_DATES, metrics_fn=lambda d: [], price_fn=_price_fn,
                     params={}, weights={"AAA": 1.3, "BBB": -0.3})


def test_weights_mode_rejects_nonpositive_sum():
    with pytest.raises(ValueError):
        run_backtest(_DATES, metrics_fn=lambda d: [], price_fn=_price_fn,
                     params={}, weights={"AAA": 0.0, "BBB": 0.0})


def test_backward_compat_criteria_mode_unaffected():
    """weights=None이면 기존 criteria 기반 동작(select_stocks 경로)이 그대로여야 한다."""
    rows = [
        {"stock_code": "AAA", "sector": "화학", "market": "KOSPI", "per": 5.0},
        {"stock_code": "BBB", "sector": "화학", "market": "KOSPI", "per": 9.0},
    ]
    res = run_backtest(_DATES, metrics_fn=lambda d: rows, price_fn=_price_fn,
                       params={"criteria": [{"key": "per", "direction": "low", "weight": 1.0}],
                               "n": 2, "fee_rate": 0.0, "tax_rate": 0.0, "slippage_rate": 0.0})
    assert res["navs"][0] == 1.0
    assert "cagr" in res["performance"]
    assert len(res["dates"]) == 3


# --------------------------------------------------------------------------
# 구간수익률(period_return) 저장 — 리밸런싱 구간별 보유종목 수익률을 holdings_log에 남긴다.
# nav/performance 계산 공식은 불변, period_return은 순수 추가 필드다(기존 감사 로직 무손상).
# --------------------------------------------------------------------------
def test_holdings_log_records_period_return_each_rebalance():
    dates = ["2026-01-31", "2026-02-28", "2026-03-31"]
    rows = [
        {"stock_code": "AAA", "sector": "화학", "market": "KOSPI", "per": 5.0},
        {"stock_code": "BBB", "sector": "화학", "market": "KOSPI", "per": 9.0},
    ]
    prices = {
        ("2026-01-31", "AAA"): 100.0, ("2026-02-28", "AAA"): 110.0, ("2026-03-31", "AAA"): 121.0,
        ("2026-01-31", "BBB"): 100.0, ("2026-02-28", "BBB"): 90.0,  ("2026-03-31", "BBB"): 108.0,
    }
    res = run_backtest(
        dates, metrics_fn=lambda d: rows, price_fn=lambda d, c: prices.get((d, c)),
        params={"criteria": [{"key": "per", "direction": "low", "weight": 1.0}], "n": 2,
                "fee_rate": 0.0, "tax_rate": 0.0, "slippage_rate": 0.0},
    )
    holdings = res["holdings"]
    assert len(holdings) == 2
    # 구간0 [01-31→02-28]: AAA +10%, BBB -10% → 동일가중 0.0
    assert holdings[0]["period_return"] == pytest.approx(0.0)
    # 구간1 [02-28→03-31]: AAA +10%, BBB +20% → 동일가중 0.15
    assert holdings[1]["period_return"] == pytest.approx(0.15)


# --------------------------------------------------------------------------
# 차트 PNG
# --------------------------------------------------------------------------
def test_save_nav_chart_creates_png_file(tmp_path):
    path = tmp_path / "nav.png"
    out = save_nav_chart(_DATES, [1.0, 1.1, 1.21], str(path),
                         benchmark=[1.0, 1.05, 1.09], title="Test Strategy")
    assert out == str(path)
    assert path.exists()
    assert path.stat().st_size > 0
    # PNG 매직바이트
    assert path.read_bytes()[:4] == b"\x89PNG"
