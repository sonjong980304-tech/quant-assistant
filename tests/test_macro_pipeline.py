"""매크로 파이프라인 오케스트레이션 테스트 (MAC-4).

.omc/specs/brainstorming-macro-indicator-agent.md AC13 관련.
run_macro_pipeline은 MAC-2 수집(ingest_macro_indicators) → MAC-3 신호판정(run_signal)
→ 신호 변경 시에만 Slack 알림(send_slack_alert)을 순서대로 엮는다.

수집 fetcher와 알림 함수(alert_fn)는 모두 주입 가능해 네트워크 없이 검증한다(DI 관례).
"""
from __future__ import annotations

from datetime import date

import src.ingest.macro_indicators as macro_indicators
from src.db import connect, init_db
from src.ingest.macro_pipeline import run_macro_pipeline


def _fakes(spread=0.6, vix=14.2, cnn=55, vkospi=84.89):
    """네 지표를 고정값으로 반환하는 fake fetcher 4종."""
    return (
        lambda today=None: ("2026-07-14", spread),
        lambda today=None: ("2026-07-14", vix),
        lambda today=None: ("2026-07-14", cnn),
        lambda today=None: ("2026-07-14", vkospi),
    )


def test_pipeline_ingests_then_persists_signal(tmp_path, monkeypatch):
    # 수집 → macro_indicators 3행, 판정 → macro_signal 1행(overall은 스프레드 레짐 기반).
    db = str(tmp_path / "pipe1.db")
    init_db(db)
    monkeypatch.setattr(macro_indicators, "send_slack_alert", lambda *a, **k: None)
    fs, fv, fc, fk = _fakes(spread=0.6, vix=14.2, cnn=55)
    alerts = []
    out = run_macro_pipeline(
        db_path=db, fetch_spread=fs, fetch_vix=fv, fetch_cnn=fc, fetch_vkospi=fk,
        today=date(2026, 7, 14), alert_fn=lambda msg: alerts.append(msg),
    )
    assert set(out["ingest"]["succeeded"]) == {"T10Y2Y", "VIXCLS", "CNN_FNG", "VKOSPI"}
    assert out["signal"]["overall"] == "GREEN"          # spread 0.6 → 정상 → GREEN
    assert out["signal"]["spread_regime"] == "정상"

    conn = connect(db)
    try:
        n_ind = conn.execute("SELECT COUNT(*) FROM macro_indicators").fetchone()[0]
        n_sig = conn.execute("SELECT COUNT(*) FROM macro_signal").fetchone()[0]
    finally:
        conn.close()
    assert n_ind == 4
    assert n_sig == 1


def test_pipeline_alerts_when_signal_changes(tmp_path, monkeypatch):
    # AC13: 직전 판정과 이번 판정이 다르면 알림 발송.
    db = str(tmp_path / "pipe2.db")
    init_db(db)
    monkeypatch.setattr(macro_indicators, "send_slack_alert", lambda *a, **k: None)
    alerts = []
    rec = lambda msg: alerts.append(msg)

    fs, fv, fc, fk = _fakes(spread=0.6)   # GREEN
    run_macro_pipeline(db_path=db, fetch_spread=fs, fetch_vix=fv, fetch_cnn=fc, fetch_vkospi=fk,
                       today=date(2026, 7, 14), alert_fn=rec)
    assert len(alerts) == 1           # 최초 판정(None→GREEN)도 변경으로 간주해 발송

    fs2, fv2, fc2, fk2 = _fakes(spread=-0.2)   # RED (GREEN→RED 변경)
    out2 = run_macro_pipeline(db_path=db, fetch_spread=fs2, fetch_vix=fv2, fetch_cnn=fc2, fetch_vkospi=fk2,
                              today=date(2026, 7, 15), alert_fn=rec)
    assert out2["signal"]["overall"] == "RED"
    assert out2["alerted"] is True
    assert len(alerts) == 2


def test_pipeline_no_alert_when_signal_unchanged(tmp_path, monkeypatch):
    # AC13: 직전과 같은 판정이면 알림 미발송.
    db = str(tmp_path / "pipe3.db")
    init_db(db)
    monkeypatch.setattr(macro_indicators, "send_slack_alert", lambda *a, **k: None)
    alerts = []
    rec = lambda msg: alerts.append(msg)

    fs, fv, fc, fk = _fakes(spread=0.6)   # GREEN
    run_macro_pipeline(db_path=db, fetch_spread=fs, fetch_vix=fv, fetch_cnn=fc, fetch_vkospi=fk,
                       today=date(2026, 7, 14), alert_fn=rec)
    assert len(alerts) == 1

    fs2, fv2, fc2, fk2 = _fakes(spread=0.7)   # 여전히 GREEN → 미발송
    out2 = run_macro_pipeline(db_path=db, fetch_spread=fs2, fetch_vix=fv2, fetch_cnn=fc2, fetch_vkospi=fk2,
                              today=date(2026, 7, 15), alert_fn=rec)
    assert out2["signal"]["overall"] == "GREEN"
    assert out2["alerted"] is False
    assert len(alerts) == 1           # 두 번째 실행에선 알림이 추가되지 않는다


def test_pipeline_spread_failure_keeps_prev_and_no_new_signal(tmp_path, monkeypatch):
    # AC11+AC13: 금리차 수집 실패 → 직전 신호 유지(데이터없음) + 판정 미변경이라 알림 없음.
    db = str(tmp_path / "pipe4.db")
    init_db(db)
    monkeypatch.setattr(macro_indicators, "send_slack_alert", lambda *a, **k: None)
    alerts = []
    rec = lambda msg: alerts.append(msg)

    fs, fv, fc, fk = _fakes(spread=0.6)   # 1일차 GREEN
    run_macro_pipeline(db_path=db, fetch_spread=fs, fetch_vix=fv, fetch_cnn=fc, fetch_vkospi=fk,
                       today=date(2026, 7, 14), alert_fn=rec)
    assert len(alerts) == 1

    def failing_spread(today=None):
        raise ConnectionError("FRED 실패(mock)")

    _, fv2, fc2, fk2 = _fakes(vix=15.0, cnn=60)
    out2 = run_macro_pipeline(db_path=db, fetch_spread=failing_spread, fetch_vix=fv2, fetch_cnn=fc2, fetch_vkospi=fk2,
                              today=date(2026, 7, 15), alert_fn=rec)
    assert out2["signal"]["spread_regime"] == "데이터없음"
    assert out2["signal"]["overall"] == "GREEN"      # 직전 신호 유지
    assert out2["alerted"] is False
    assert len(alerts) == 1
    assert "T10Y2Y" in out2["ingest"]["failed"]
    assert set(out2["ingest"]["succeeded"]) == {"VIXCLS", "CNN_FNG", "VKOSPI"}   # 나머지는 계속 수집

    conn = connect(db)
    try:
        n_sig = conn.execute("SELECT COUNT(*) FROM macro_signal").fetchone()[0]
    finally:
        conn.close()
    assert n_sig == 2   # 판정은 여전히 날짜별 append(직전 신호 유지 행)


def test_pipeline_defaults_alert_fn_to_send_slack_alert(tmp_path, monkeypatch):
    # alert_fn 미주입 시 send_slack_alert가 기본값으로 쓰인다(모듈 속성 monkeypatch로 확인).
    db = str(tmp_path / "pipe5.db")
    init_db(db)
    monkeypatch.setattr(macro_indicators, "send_slack_alert", lambda *a, **k: None)
    import src.ingest.macro_pipeline as macro_pipeline

    sent = []
    monkeypatch.setattr(macro_pipeline, "send_slack_alert", lambda msg, **k: sent.append(msg) or True)
    fs, fv, fc, fk = _fakes(spread=-0.3)   # RED
    run_macro_pipeline(db_path=db, fetch_spread=fs, fetch_vix=fv, fetch_cnn=fc, fetch_vkospi=fk,
                       today=date(2026, 7, 14))
    assert len(sent) == 1              # 기본 alert_fn(send_slack_alert)이 호출됨
