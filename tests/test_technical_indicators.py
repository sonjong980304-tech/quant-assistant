"""TA-Lib 기술지표 통합 테스트 (brainstorming-talib-reverse-backtest.md AC1-AC8).

- price_history_batch: 여러 종목의 연속 가격 시계열을 SQL 1회(IN절)로 배치 조회.
  기존 _price_at류(단일시점 스냅샷)와 달리 날짜 범위로 조회하는 최초의 함수다.
"""
from __future__ import annotations

import sqlite3

import pytest

from src.backtest.data_access import price_history_batch
from src.db import init_db


class _CountingConn:
    """conn.execute 호출 횟수를 세는 프록시. sqlite3.Connection은 속성 재할당이
    안 되므로(read-only) 얇은 위임 프록시로 감싼다."""

    def __init__(self, real_conn: sqlite3.Connection) -> None:
        self._real = real_conn
        self.execute_calls = 0

    def execute(self, *args, **kwargs):
        self.execute_calls += 1
        return self._real.execute(*args, **kwargs)


@pytest.fixture
def price_db(tmp_path):
    db = tmp_path / "test_market.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    codes = ["000001", "000002", "000003"]
    for code in codes:
        conn.execute(
            "INSERT INTO company(stock_code, name, market, sector) VALUES (?,?,?,?)",
            (code, code, "KOSPI", "기타"),
        )
    dates = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    for code in codes:
        for i, d in enumerate(dates):
            conn.execute(
                "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,?)",
                (code, d, 100.0 + i, 1e12),
            )
    conn.commit()
    return conn, codes, dates


def test_price_history_batch_uses_single_query_for_multiple_codes(price_db):
    conn, codes, dates = price_db
    counting = _CountingConn(conn)
    result = price_history_batch(counting, codes, asof="2026-01-08", lookback_days=30)
    assert counting.execute_calls == 1


def test_price_history_batch_returns_ascending_series_per_code(price_db):
    conn, codes, dates = price_db
    result = price_history_batch(conn, codes, asof="2026-01-08", lookback_days=30)
    assert set(result.keys()) == set(codes)
    series = result["000001"]
    assert [row["date"] for row in series] == dates
    assert series[0]["close"] == 100.0
    assert series[-1]["close"] == 104.0


def test_price_history_batch_respects_lookback_window(price_db):
    conn, codes, dates = price_db
    # lookback_days=1이면 2026-01-08 기준 하루 전(01-07)까지만 포함되어야 함
    result = price_history_batch(conn, codes, asof="2026-01-08", lookback_days=1)
    series = result["000001"]
    returned_dates = [row["date"] for row in series]
    assert "2026-01-02" not in returned_dates
    assert "2026-01-08" in returned_dates


def test_price_history_batch_excludes_dates_after_asof(price_db):
    conn, codes, dates = price_db
    result = price_history_batch(conn, codes, asof="2026-01-06", lookback_days=30)
    series = result["000001"]
    returned_dates = [row["date"] for row in series]
    assert "2026-01-07" not in returned_dates
    assert "2026-01-08" not in returned_dates


def test_price_history_batch_missing_code_returns_empty_list(price_db):
    conn, codes, dates = price_db
    result = price_history_batch(conn, ["999999"], asof="2026-01-08", lookback_days=30)
    assert result == {"999999": []}


# --------------------------------------------------------------------------
# compute_technical_indicator (AC2-AC5) — 순수 파라미터 해석 로직
# --------------------------------------------------------------------------
from src.backtest.primitives import _resolve_indicator_spec, compute_technical_indicator


def test_resolve_indicator_spec_sma_uses_default_period_when_unspecified():
    resolved = _resolve_indicator_spec({"name": "sma"})
    assert resolved == {"name": "sma", "params": {"period": 20}}


def test_resolve_indicator_spec_sma_uses_given_period():
    resolved = _resolve_indicator_spec({"name": "sma", "period": 50})
    assert resolved == {"name": "sma", "params": {"period": 50}}


def test_resolve_indicator_spec_ema_adjustable_like_sma():
    resolved = _resolve_indicator_spec({"name": "ema", "period": 10})
    assert resolved == {"name": "ema", "params": {"period": 10}}


def test_resolve_indicator_spec_rsi_ignores_period_param_fixed_14():
    resolved = _resolve_indicator_spec({"name": "rsi", "period": 99})
    assert resolved == {"name": "rsi", "params": {"period": 14}}


def test_resolve_indicator_spec_macd_uses_defaults_when_unspecified():
    resolved = _resolve_indicator_spec({"name": "macd"})
    assert resolved == {"name": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}}


def test_resolve_indicator_spec_macd_uses_given_params():
    resolved = _resolve_indicator_spec({"name": "macd", "fast": 5, "slow": 10, "signal": 3})
    assert resolved == {"name": "macd", "params": {"fast": 5, "slow": 10, "signal": 3}}


def test_resolve_indicator_spec_bollinger_ignores_period_param_fixed_20_2():
    resolved = _resolve_indicator_spec({"name": "bollinger", "period": 99})
    assert resolved == {"name": "bollinger", "params": {"period": 20, "nbdev": 2}}


def test_resolve_indicator_spec_rejects_unsupported_name():
    with pytest.raises(ValueError):
        _resolve_indicator_spec({"name": "stochastic"})


# --------------------------------------------------------------------------
# compute_technical_indicator (AC2-AC5) — 배선(history_fn/compute_fn DI로 talib 실제
# 계산 정확성은 검증하지 않고, 파라미터 전달/필드명/기존 필드 보존만 확인)
# --------------------------------------------------------------------------
def _fake_history_fn_factory(series_by_code):
    calls = []

    def fake_history_fn(conn, codes, asof, lookback_days):
        calls.append({"conn": conn, "codes": list(codes), "asof": asof, "lookback_days": lookback_days})
        return {c: series_by_code.get(c, []) for c in codes}

    fake_history_fn.calls = calls
    return fake_history_fn


def _fake_compute_fn_factory():
    calls = []

    def fake_compute_fn(name, closes, params):
        calls.append({"name": name, "closes": list(closes), "params": dict(params)})
        if name == "rsi":
            return {"rsi_14": 55.5}
        if name == "sma":
            return {f"sma_{params['period']}": 100.0}
        return {}

    fake_compute_fn.calls = calls
    return fake_compute_fn


def test_compute_technical_indicator_calls_history_fn_once_with_all_codes():
    rows = [{"stock_code": "000001"}, {"stock_code": "000002"}]
    history_fn = _fake_history_fn_factory({"000001": [{"date": "2026-01-01", "close": 100.0}]})
    compute_fn = _fake_compute_fn_factory()
    compute_technical_indicator(
        conn=object(), rows=rows, asof="2026-06-30", indicators=[{"name": "rsi"}],
        history_fn=history_fn, compute_fn=compute_fn,
    )
    assert len(history_fn.calls) == 1
    assert set(history_fn.calls[0]["codes"]) == {"000001", "000002"}


def test_compute_technical_indicator_preserves_original_row_fields():
    rows = [{"stock_code": "000001", "per": 10.5, "name": "가나전자"}]
    history_fn = _fake_history_fn_factory({"000001": [{"date": "2026-01-01", "close": 100.0}]})
    compute_fn = _fake_compute_fn_factory()
    out = compute_technical_indicator(
        conn=object(), rows=rows, asof="2026-06-30", indicators=[{"name": "rsi"}],
        history_fn=history_fn, compute_fn=compute_fn,
    )
    assert out[0]["stock_code"] == "000001"
    assert out[0]["per"] == 10.5
    assert out[0]["name"] == "가나전자"
    assert out[0]["rsi_14"] == 55.5


def test_compute_technical_indicator_does_not_mutate_input_rows():
    rows = [{"stock_code": "000001", "per": 10.5}]
    history_fn = _fake_history_fn_factory({"000001": []})
    compute_fn = _fake_compute_fn_factory()
    compute_technical_indicator(
        conn=object(), rows=rows, asof="2026-06-30", indicators=[{"name": "rsi"}],
        history_fn=history_fn, compute_fn=compute_fn,
    )
    assert rows[0] == {"stock_code": "000001", "per": 10.5}


def test_compute_technical_indicator_calls_compute_fn_per_row_per_indicator():
    rows = [{"stock_code": "000001"}, {"stock_code": "000002"}]
    history_fn = _fake_history_fn_factory({
        "000001": [{"date": "2026-01-01", "close": 100.0}],
        "000002": [{"date": "2026-01-01", "close": 200.0}],
    })
    compute_fn = _fake_compute_fn_factory()
    compute_technical_indicator(
        conn=object(), rows=rows, asof="2026-06-30",
        indicators=[{"name": "rsi"}, {"name": "sma", "period": 5}],
        history_fn=history_fn, compute_fn=compute_fn,
    )
    # 2 rows x 2 indicators = 4회 호출
    assert len(compute_fn.calls) == 4
    names_called = {c["name"] for c in compute_fn.calls}
    assert names_called == {"rsi", "sma"}


def test_compute_technical_indicator_passes_resolved_params_to_compute_fn():
    rows = [{"stock_code": "000001"}]
    history_fn = _fake_history_fn_factory({"000001": [{"date": "2026-01-01", "close": 100.0}]})
    compute_fn = _fake_compute_fn_factory()
    compute_technical_indicator(
        conn=object(), rows=rows, asof="2026-06-30",
        indicators=[{"name": "rsi", "period": 99}],  # RSI는 파라미터 무시되고 14 고정이어야 함
        history_fn=history_fn, compute_fn=compute_fn,
    )
    assert compute_fn.calls[0]["params"] == {"period": 14}


def test_compute_technical_indicator_lookback_grows_with_macd_slow_signal():
    rows = [{"stock_code": "000001"}]
    history_fn = _fake_history_fn_factory({"000001": []})
    compute_fn = _fake_compute_fn_factory()
    compute_technical_indicator(
        conn=object(), rows=rows, asof="2026-06-30",
        indicators=[{"name": "macd", "slow": 100, "signal": 50}],
        history_fn=history_fn, compute_fn=compute_fn,
    )
    # slow+signal=150 기준으로 넉넉한 lookback(최소 150*2=300일 이상)이어야 함
    assert history_fn.calls[0]["lookback_days"] >= 300


def test_compute_technical_indicator_rejects_unsupported_indicator_before_history_fetch():
    rows = [{"stock_code": "000001"}]
    history_fn = _fake_history_fn_factory({})
    with pytest.raises(ValueError):
        compute_technical_indicator(
            conn=object(), rows=rows, asof="2026-06-30", indicators=[{"name": "stochastic"}],
            history_fn=history_fn,
        )
    assert len(history_fn.calls) == 0  # 무거운 연산(가격조회) 전에 거부돼야 함


# --------------------------------------------------------------------------
# compute_technical_indicator → combine 병합 통합테스트 (AC4, AC5)
# --------------------------------------------------------------------------
from src.backtest.primitives import combine


def test_compute_technical_indicator_output_feeds_directly_into_combine():
    """get_cross_section 스타일 rows에 compute_technical_indicator로 지표를 추가한 뒤,
    기존 combine()에 그대로 넣어 원래 필드(per)와 신규 지표(rsi_14)를 함께 criteria로
    쓸 수 있는지 확인한다(TA-Lib 수치 자체가 아니라 배선을 검증)."""
    xs = [
        {"stock_code": "000001", "name": "A", "per": 5.0},
        {"stock_code": "000002", "name": "B", "per": 10.0},
        {"stock_code": "000003", "name": "C", "per": 15.0},
        {"stock_code": "000004", "name": "D", "per": 20.0},
    ]
    rsi_by_code = {"000001": 20.0, "000002": 30.0, "000003": 80.0, "000004": 90.0}

    def fake_compute_fn(name, closes, params):
        return {}

    def fake_history_fn(conn, codes, asof, lookback_days):
        return {c: [] for c in codes}

    enriched = compute_technical_indicator(
        conn=object(), rows=xs, asof="2026-06-30", indicators=[{"name": "rsi"}],
        history_fn=fake_history_fn, compute_fn=fake_compute_fn,
    )
    # fake_compute_fn이 빈 dict를 반환하므로 수동으로 rsi_14를 채워 시나리오를 완성한다
    for row in enriched:
        row["rsi_14"] = rsi_by_code[row["stock_code"]]

    picked = combine(
        enriched,
        criteria=[
            {"key": "per", "direction": "low", "weight": 0.5},
            {"key": "rsi_14", "direction": "low", "weight": 0.5},
        ],
        n=2,
    )
    picked_codes = {r["stock_code"] for r in picked}
    # per도 낮고 rsi도 낮은(과매도 아님) 000001/000002가 뽑혀야 함
    assert picked_codes == {"000001", "000002"}
    assert all("per" in r and "rsi_14" in r for r in picked)


def test_compute_technical_indicator_chains_after_get_cross_section_in_pipeline():
    """실제 get_cross_section 출력 형태(stock_code/name/sector/market/quarter 포함)에도
    compute_technical_indicator가 그대로 적용 가능한지 확인(파이프라인 연쇄 배선)."""
    xs_style_rows = [
        {"stock_code": "000001", "name": "가나전자", "sector": "반도체", "market": "KOSPI",
         "quarter": "2025Q4", "per": 8.0},
    ]

    def fake_history_fn(conn, codes, asof, lookback_days):
        return {c: [{"date": "2026-01-01", "close": 100.0}] for c in codes}

    def fake_compute_fn(name, closes, params):
        return {"rsi_14": 45.0}

    out = compute_technical_indicator(
        conn=object(), rows=xs_style_rows, asof="2026-06-30", indicators=[{"name": "rsi"}],
        history_fn=fake_history_fn, compute_fn=fake_compute_fn,
    )
    assert out[0]["sector"] == "반도체"
    assert out[0]["market"] == "KOSPI"
    assert out[0]["quarter"] == "2025Q4"
    assert out[0]["rsi_14"] == 45.0


def test_compute_technical_indicator_default_compute_fn_uses_real_talib():
    """compute_fn을 주입하지 않으면 실제 TA-Lib으로 계산한다(기본 경로 스모크 테스트).

    TA-Lib 자체의 수치 정확성은 검증 대상이 아니므로(스펙 결정), 필드가 존재하고
    합리적 범위(RSI 0~100)인지만 확인한다.
    """
    talib = pytest.importorskip("talib")
    rows = [{"stock_code": "000001"}]
    closes = [100.0 + i * 0.5 for i in range(40)]
    history_fn = _fake_history_fn_factory({
        "000001": [{"date": f"2026-01-{i+1:02d}", "close": c} for i, c in enumerate(closes)]
    })
    out = compute_technical_indicator(
        conn=object(), rows=rows, asof="2026-06-30", indicators=[{"name": "rsi"}],
        history_fn=history_fn,
    )
    assert out[0]["rsi_14"] is not None
    assert 0.0 <= out[0]["rsi_14"] <= 100.0
