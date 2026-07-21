"""매크로 지표 파이프라인 오케스트레이션 (MAC-4).

.omc/specs/brainstorming-macro-indicator-agent.md 참고 (접근방식 A: 단순 스크립트).

세 단계를 한 번에 엮는 얇은 오케스트레이터:
  1) 수집(ingest)   : macro_indicators.ingest_macro_indicators — FRED/CNN 3지표를 upsert
  2) 판정(signal)   : macro_signal.run_signal — 최신 지표값으로 GREEN/YELLOW/RED 판정 append
  3) 알림(notify)   : send_slack_alert — 직전 판정과 다를 때만 발송(AC13)

설계 원칙(기존 ingest 관례 준수)
--------------------------------
- 실제 네트워크 의존(수집 fetcher, Slack 알림)은 모두 주입 가능한 인자로 분리 —
  네트워크 없이 오케스트레이션 순서·분기만 단위테스트한다(DI 관례).
- 금리차(T10Y2Y) 수집이 실패하면 그날은 새로 판정하지 않고 직전 신호를 유지한다(AC11).
  이 판단은 ingest 결과의 succeeded 목록으로 하며(그 지표가 이번 실행에서 실제로
  갱신됐는지), DB에 남아있을 수 있는 과거 값에 오염되지 않는다.
"""
from __future__ import annotations

from datetime import date
from typing import Callable

from ..db import connect
from .macro_indicators import ingest_macro_indicators
from .macro_signal import run_signal
from .notify import send_slack_alert

# 지표 키 → macro_signal 판정 입력 이름
_INDICATORS = {"T10Y2Y": "spread", "VIXCLS": "vix", "CNN_FNG": "cnn"}


def _latest_value(conn, indicator: str) -> float | None:
    """macro_indicators에서 해당 지표의 최신(date 기준) 값. 없으면 None."""
    row = conn.execute(
        "SELECT value FROM macro_indicators WHERE indicator=? ORDER BY date DESC LIMIT 1",
        (indicator,),
    ).fetchone()
    return row[0] if row else None


def _alert_message(signal: dict) -> str:
    """신호 변경 알림 문구(한국어). prev→overall 전이와 판정 기준일을 담는다."""
    prev = signal.get("prev_overall") or "없음"
    return (
        f"[매크로 신호] 종합신호 변경: {prev} → {signal['overall']} "
        f"(금리차 레짐={signal['spread_regime']}, 기준일={signal['as_of']})"
    )


def run_macro_pipeline(
    db_path: str | None = None,
    fetch_spread: Callable | None = None,
    fetch_vix: Callable | None = None,
    fetch_cnn: Callable | None = None,
    fetch_vkospi: Callable | None = None,
    today: date | None = None,
    alert_fn: Callable | None = None,
) -> dict:
    """수집 → 판정 → (변경 시)알림을 순서대로 실행하고 요약 dict를 반환한다.

    VKOSPI는 참고 표시 전용이라 이 판정(_INDICATORS)에는 들어가지 않는다 — 수집만 되고
    macro_indicators에 저장돼 /api/macro/vkospi가 그대로 노출한다(신호 계산 미반영).
    반환: {"ingest": {succeeded/failed}, "signal": 판정 dict, "alerted": bool}.
    alert_fn을 주입하지 않으면 send_slack_alert(웹훅 미설정 시 조용히 스킵)를 쓴다.
    """
    today = today or date.today()
    alert_fn = alert_fn or send_slack_alert

    # 1) 수집 (지표별 재시도·부분실패 격리는 ingest 내부가 처리)
    ingest_result = ingest_macro_indicators(
        db_path=db_path,
        fetch_spread=fetch_spread,
        fetch_vix=fetch_vix,
        fetch_cnn=fetch_cnn,
        fetch_vkospi=fetch_vkospi,
        today=today,
    )
    succeeded = set(ingest_result["succeeded"])

    # 2) 판정 — 이번 실행에서 갱신에 성공한 지표만 값으로 반영(실패 지표는 None)
    conn = connect(db_path)
    try:
        values = {
            name: (_latest_value(conn, ind) if ind in succeeded else None)
            for ind, name in _INDICATORS.items()
        }
        signal = run_signal(
            conn,
            spread=values["spread"],
            cnn=values["cnn"],
            vix=values["vix"],
            as_of=today.strftime("%Y-%m-%d"),
        )
    finally:
        conn.close()

    # 3) 알림 — 직전 판정과 다를 때만(AC13)
    alerted = signal["overall"] != signal["prev_overall"]
    if alerted:
        alert_fn(_alert_message(signal))

    return {"ingest": ingest_result, "signal": signal, "alerted": alerted}
