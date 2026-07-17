"""올웨더 배치 파이프라인 테스트 (AC10/AC12/AC13/AC16 통합).

배치는 4종목 가격 조회 → 몬테카를로+walk-forward 계산 → all_weather_snapshot 저장 → 텔레그램
알림을 순서대로 엮는다. 수집 fetcher/몬테카를로/전송 함수는 모두 주입 가능해 네트워크·10만회
시뮬레이션 없이 오케스트레이션(저장·이력·알림)만 검증한다(macro_pipeline과 동일 DI 관례).
"""
from __future__ import annotations

import sqlite3
from datetime import date

import numpy as np
import pandas as pd

from src.allweather.pipeline import run_all_weather_pipeline
from src.db import init_db

_DATES = pd.date_range("2014-01-01", periods=10 * 252, freq="B")


def _fake_yf(dates):
    rng = np.random.default_rng(3)

    def f(ticker):
        px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, size=len(dates))))
        return pd.Series(px, index=dates)

    return f


def _fake_irx(dates):
    def f(ticker=None):
        return pd.Series([5.0] * len(dates), index=dates)

    return f


def _fast_mc(prices, **kw):
    cols = list(prices.columns)
    return {
        "weights": {c: 1.0 / len(cols) for c in cols},
        "annual_return": 0.1,
        "annual_vol": 0.1,
        "sharpe": 1.0,
        "period_start": str(prices.index[0].date()),
        "period_end": str(prices.index[-1].date()),
    }


def test_pipeline_persists_all_metrics_and_sends_telegram(tmp_path):
    # AC10 + AC12: DB에 비중/MDD/CAGR/누적수익률/샤프 저장 + 텔레그램 전송.
    db = str(tmp_path / "pipe.db")
    init_db(db)
    dates = _DATES
    sent = []
    run_all_weather_pipeline(
        db_path=db, fetch_yf=_fake_yf(dates), fetch_irx=_fake_irx(dates),
        monte_carlo_fn=_fast_mc, send_fn=lambda m: sent.append(m),
        today=date(2024, 1, 1), n_simulations=10,
    )
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT weights, cagr, mdd, sharpe, cumulative_return, backtest_curve "
            "FROM all_weather_snapshot ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    weights, cagr, mdd, sharpe, cumret, curve = row
    assert weights and cagr is not None and mdd is not None
    assert sharpe is not None and cumret is not None and curve
    assert len(sent) == 1  # AC12: 텔레그램 1회 전송


def test_pipeline_appends_history_two_runs(tmp_path):
    # AC16: 두 번 실행하면 이력 2행.
    db = str(tmp_path / "pipe2.db")
    init_db(db)
    dates = _DATES
    for _ in range(2):
        run_all_weather_pipeline(
            db_path=db, fetch_yf=_fake_yf(dates), fetch_irx=_fake_irx(dates),
            monte_carlo_fn=_fast_mc, send_fn=lambda m: None,
            today=date(2024, 1, 1), n_simulations=10,
        )
    conn = sqlite3.connect(db)
    try:
        n = conn.execute("SELECT COUNT(*) FROM all_weather_snapshot").fetchone()[0]
    finally:
        conn.close()
    assert n == 2


def test_pipeline_second_run_message_has_delta(tmp_path):
    # AC13: 직전 달 저장값이 있으면 두 번째 실행 알림에 델타(%p)가 들어간다.
    db = str(tmp_path / "pipe3.db")
    init_db(db)
    dates = _DATES
    msgs = []
    run_all_weather_pipeline(
        db_path=db, fetch_yf=_fake_yf(dates), fetch_irx=_fake_irx(dates),
        monte_carlo_fn=_fast_mc, send_fn=lambda m: msgs.append(m),
        today=date(2024, 1, 1), n_simulations=10,
    )
    run_all_weather_pipeline(
        db_path=db, fetch_yf=_fake_yf(dates), fetch_irx=_fake_irx(dates),
        monte_carlo_fn=_fast_mc, send_fn=lambda m: msgs.append(m),
        today=date(2024, 2, 1), n_simulations=10,
    )
    assert len(msgs) == 2
    assert "%p" in msgs[1]  # 두 번째 알림엔 직전 대비 델타 표기


def test_pipeline_defaults_send_fn_to_send_telegram(tmp_path, monkeypatch):
    # send_fn 미주입 시 allweather.notify.send_telegram이 기본값으로 쓰인다.
    db = str(tmp_path / "pipe4.db")
    init_db(db)
    dates = _DATES
    import src.allweather.pipeline as pl

    sent = []
    monkeypatch.setattr(pl, "send_telegram", lambda msg: sent.append(msg) or True)
    run_all_weather_pipeline(
        db_path=db, fetch_yf=_fake_yf(dates), fetch_irx=_fake_irx(dates),
        monte_carlo_fn=_fast_mc, today=date(2024, 1, 1), n_simulations=10,
    )
    assert len(sent) == 1


def test_send_telegram_uses_telegram_env_vars(monkeypatch):
    # AC12: 전송 함수가 quant_trader와 동일한 env var(TELEGRAM_BOT_TOKEN/CHAT_ID)를 쓴다.
    import src.allweather.notify as noti

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TESTTOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "TESTCHAT")
    captured = {}

    class _Resp:
        status_code = 200
        text = "ok"

    def fake_post(url, data=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        return _Resp()

    monkeypatch.setattr(noti.requests, "post", fake_post)
    ok = noti.send_telegram("hello")
    assert ok is True
    assert "TESTTOKEN" in captured["url"]
    assert captured["data"]["chat_id"] == "TESTCHAT"


def test_send_telegram_noop_when_env_missing(monkeypatch):
    # 토큰/채팅ID 미설정 시 조용히 False(전송 시도 없음) — quant_trader notifier와 동일.
    import src.allweather.notify as noti

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    called = []
    monkeypatch.setattr(noti.requests, "post", lambda *a, **k: called.append(1))
    assert noti.send_telegram("x") is False
    assert called == []
