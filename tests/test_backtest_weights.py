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


def test_engine_excludes_newly_admin_stock_from_first_rebalance():
    """관리종목이 최초 리밸런싱(직전 보유 없음)에서는 1위여도 신규매수로 제외돼야 한다."""
    rows = [
        {"stock_code": "A", "sector": "화학", "market": "KOSPI", "per": 1.0, "is_admin": True},
        {"stock_code": "B", "sector": "화학", "market": "KOSPI", "per": 5.0, "is_admin": False},
    ]
    res = run_backtest(
        _DATES, metrics_fn=lambda d: rows, price_fn=lambda d, c: 100.0,
        params={"criteria": [{"key": "per", "direction": "low", "weight": 1.0}], "n": 1,
                "fee_rate": 0.0, "tax_rate": 0.0, "slippage_rate": 0.0},
    )
    assert res["holdings"][0]["codes"] == ["B"]  # A(관리종목, per 1위)는 신규매수라 제외됨


def test_engine_keeps_admin_stock_held_from_previous_rebalance():
    """직전 리밸런싱에 보유했던 종목이 이후 관리종목이 돼도, 더 우수한 신규 관리종목
    후보(E)보다 우선해 계속 선정돼야 한다 — engine이 prev_codes를 실제로 select_stocks에
    넘기지 않으면(버그) E가 admin 여부와 무관하게 최우수라 그냥 선정돼 이 테스트가 실패한다.
    """
    rows_t0 = [
        {"stock_code": "A", "sector": "화학", "market": "KOSPI", "per": 1.0, "is_admin": False},
        {"stock_code": "C", "sector": "화학", "market": "KOSPI", "per": 5.0, "is_admin": False},
    ]
    rows_t1 = [
        {"stock_code": "A", "sector": "화학", "market": "KOSPI", "per": 1.0, "is_admin": True},
        # E는 A보다도 더 우수(per 낮음)한 신규 관리종목 후보 — 직전 보유가 아니므로 제외돼야 함.
        {"stock_code": "E", "sector": "화학", "market": "KOSPI", "per": 0.5, "is_admin": True},
    ]
    rows_by_date = {_DATES[0]: rows_t0, _DATES[1]: rows_t1, _DATES[2]: rows_t1}
    res = run_backtest(
        _DATES, metrics_fn=lambda d: rows_by_date[d], price_fn=lambda d, c: 100.0,
        params={"criteria": [{"key": "per", "direction": "low", "weight": 1.0}], "n": 1,
                "fee_rate": 0.0, "tax_rate": 0.0, "slippage_rate": 0.0},
    )
    assert res["holdings"][0]["codes"] == ["A"]  # t0: 정상종목으로 선정
    assert res["holdings"][1]["codes"] == ["A"]  # t1: E(신규 관리종목)보다 우선해 직전 보유 A 유지


def test_engine_forwards_winsorize_z_param_to_select_stocks():
    """params["winsorize_z"]가 select_stocks까지 배선돼 선정 종목이 바뀌는지 검증(엔진 배선).

    극단치 종목 X는 winsorize 없으면 1위로 선정되지만, params에 winsorize_z=3.0을 주면
    z-score가 잘려 Y가 1위로 뒤바뀐다 — 백테스트 홀딩스가 그 결과를 그대로 반영해야 한다.
    """
    rows = [{"stock_code": f"F{i:02d}", "sector": "화학", "market": "KOSPI",
             "a": float(i + 1), "b": 5.0} for i in range(24)]
    rows.append({"stock_code": "Y", "sector": "화학", "market": "KOSPI", "a": 40.0, "b": 5.0})
    rows.append({"stock_code": "X", "sector": "화학", "market": "KOSPI", "a": 8.0, "b": 4000.0})
    criteria = [{"key": "a", "direction": "high", "weight": 1.0},
                {"key": "b", "direction": "high", "weight": 1.0}]
    base = {"criteria": criteria, "n": 1, "fee_rate": 0.0, "tax_rate": 0.0, "slippage_rate": 0.0}

    without = run_backtest(_DATES, metrics_fn=lambda d: rows, price_fn=lambda d, c: 100.0,
                           params=base)
    assert without["holdings"][0]["codes"][0] == "X"

    withw = run_backtest(_DATES, metrics_fn=lambda d: rows, price_fn=lambda d, c: 100.0,
                         params={**base, "winsorize_z": 3.0})
    assert withw["holdings"][0]["codes"][0] == "Y"


def test_engine_forwards_winsorize_pct_param_to_select_stocks():
    """params["winsorize_pct"]가 select_stocks까지 배선되는지 검증(퍼센타일 방식 엔진 배선).

    단일 기준 n=1: 클리핑 없으면 원본 최댓값 Q가 선정되지만, winsorize_pct=0.02면 P·Q가
    같은 경계로 눌려 동점이 돼 순서상 앞선 P가 선정된다 — 홀딩스가 그 결과를 반영해야 한다.
    """
    rows = [{"stock_code": f"F{i:02d}", "sector": "화학", "market": "KOSPI", "a": 1.0}
            for i in range(98)]
    rows.append({"stock_code": "P", "sector": "화학", "market": "KOSPI", "a": 100.0})
    rows.append({"stock_code": "Q", "sector": "화학", "market": "KOSPI", "a": 1000.0})
    criteria = [{"key": "a", "direction": "high", "weight": 1.0}]
    base = {"criteria": criteria, "n": 1, "fee_rate": 0.0, "tax_rate": 0.0, "slippage_rate": 0.0}

    without = run_backtest(_DATES, metrics_fn=lambda d: rows, price_fn=lambda d, c: 100.0,
                           params=base)
    assert without["holdings"][0]["codes"][0] == "Q"

    withp = run_backtest(_DATES, metrics_fn=lambda d: rows, price_fn=lambda d, c: 100.0,
                         params={**base, "winsorize_pct": 0.02})
    assert withp["holdings"][0]["codes"][0] == "P"


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


def test_engine_records_empty_periods_when_criterion_all_none():
    """은행/증권처럼 선택한 지표가 전 종목 None이면 매 리밸런싱마다 0개 선정되는데,
    이걸 조용히 NAV 고정으로만 넘기지 않고 empty_periods에 남겨 원인을 알 수 있게 한다."""
    dates = ["2026-01-31", "2026-02-28", "2026-03-31"]
    rows = [
        {"stock_code": "AAA", "sector": "은행", "market": "KOSPI", "gp_a": None},
        {"stock_code": "BBB", "sector": "은행", "market": "KOSPI", "gp_a": None},
    ]
    res = run_backtest(
        dates, metrics_fn=lambda d: rows, price_fn=lambda d, c: 100.0,
        params={"criteria": [{"key": "gp_a", "direction": "high", "weight": 1.0}], "n": 2,
                "fee_rate": 0.0, "tax_rate": 0.0, "slippage_rate": 0.0},
    )
    assert res["navs"] == [1.0, 1.0, 1.0]
    assert res["empty_periods"] == ["2026-01-31", "2026-02-28"]


def test_engine_tracks_second_benchmark_alongside_first_when_no_stock_selected():
    """종목이 하나도 안 뽑히는 구간(편입 종목 없음 분기)에서도 benchmark_fn2가 매 시점 기록된다."""
    kospi_levels = {"2026-01-31": 1.0, "2026-02-28": 1.1, "2026-03-31": 1.2}
    kosdaq_levels = {"2026-01-31": 1.0, "2026-02-28": 0.9, "2026-03-31": 0.8}
    res = run_backtest(
        _DATES, metrics_fn=lambda d: [], price_fn=_price_fn,
        params={"criteria": [{"key": "per", "direction": "low", "weight": 1.0}], "n": 2,
                "fee_rate": 0.0, "tax_rate": 0.0, "slippage_rate": 0.0},
        benchmark_fn=lambda d: kospi_levels[d],
        benchmark_fn2=lambda d: kosdaq_levels[d],
    )
    assert res["benchmark"] == [1.0, 1.1, 1.2]
    assert res["benchmark2"] == [1.0, 0.9, 0.8]


def test_engine_tracks_second_benchmark_alongside_first_when_stocks_are_selected():
    """종목이 정상적으로 편입되는 구간(정상 리밸런싱 분기)에서도 benchmark_fn2가 매 시점 기록돼야
    한다 — 편입 종목 없음 분기만 고쳐두고 이 정상 분기를 빠뜨리는 회귀를 잡기 위한 테스트."""
    dates = ["2026-01-31", "2026-02-28", "2026-03-31"]
    rows = [
        {"stock_code": "AAA", "sector": "화학", "market": "KOSPI", "per": 5.0},
        {"stock_code": "BBB", "sector": "화학", "market": "KOSPI", "per": 9.0},
    ]
    kospi_levels = {"2026-01-31": 1.0, "2026-02-28": 1.1, "2026-03-31": 1.2}
    kosdaq_levels = {"2026-01-31": 1.0, "2026-02-28": 0.9, "2026-03-31": 0.8}
    res = run_backtest(
        dates, metrics_fn=lambda d: rows, price_fn=_price_fn,
        params={"criteria": [{"key": "per", "direction": "low", "weight": 1.0}], "n": 2,
                "fee_rate": 0.0, "tax_rate": 0.0, "slippage_rate": 0.0},
        benchmark_fn=lambda d: kospi_levels[d],
        benchmark_fn2=lambda d: kosdaq_levels[d],
    )
    assert res["benchmark"] == [1.0, 1.1, 1.2]
    assert res["benchmark2"] == [1.0, 0.9, 0.8]


def test_weights_mode_always_returns_empty_periods_key():
    """weights 모드는 select_stocks를 안 쓰므로 항상 빈 리스트지만, 프런트가 항상 안전하게
    d.empty_periods를 읽을 수 있도록 키 자체는 criteria 모드와 동일하게 존재해야 한다."""
    res = run_backtest(_DATES, metrics_fn=lambda d: [], price_fn=_price_fn,
                       params={"fee_rate": 0.0, "tax_rate": 0.0, "slippage_rate": 0.0},
                       weights={"AAA": 0.6, "BBB": 0.4})
    assert res["empty_periods"] == []


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
