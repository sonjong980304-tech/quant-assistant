"""올웨더 배치 파이프라인 오케스트레이션.

.omc/specs/brainstorming-all-weather-portfolio.md AC10/AC12/AC13/AC16 참고.

네 단계를 한 번에 엮는 얇은 오케스트레이터(macro_pipeline과 동일 정신):
  1) 데이터   : build_price_panel — QQQ/삼성전자/TLT(yfinance) + 411060.KS(GLD×환율 합성+실데이터 스플라이스)
  2) 계산     : run_walk_forward — 매 리밸런싱 시점 몬테카를로 재계산(look-ahead 없음)
  3) 저장     : persist_snapshot — all_weather_snapshot에 이력 append(AC16)
  4) 알림     : send_telegram — 직전 달 대비 비중 델타 포함(AC12/AC13)

실제 네트워크 의존(yfinance 수집, ^IRX 조회, 텔레그램 전송)과 10만회 몬테카를로는 모두 주입
가능한 인자로 분리 — 네트워크·중연산 없이 오케스트레이션(저장·이력·알림)만 단위테스트한다(DI 관례).
이 파이프라인은 계산 결과의 저장·표시·알림까지만 담당하고 실거래 주문은 하지 않는다(AC9).
"""
from __future__ import annotations

from datetime import date
from typing import Callable

from ..db import connect, init_db
from .backtest import run_walk_forward
from .data import build_price_panel, fetch_irx_series
from .montecarlo import N_SIMULATIONS, run_monte_carlo
from .notify import build_delta_message, send_telegram
from .store import get_latest_snapshot, persist_snapshot


def run_all_weather_pipeline(
    db_path: str | None = None,
    fetch_yf: Callable | None = None,
    fetch_irx: Callable | None = None,
    monte_carlo_fn: Callable | None = None,
    send_fn: Callable | None = None,
    today: date | None = None,
    n_simulations: int = N_SIMULATIONS,
) -> dict:
    """수집 → walk-forward 계산 → 저장 → 알림을 순서대로 실행하고 요약 dict를 반환한다.

    반환: {"snapshot": 스냅샷 dict, "message": 전송 메시지, "alerted": bool}.
    send_fn 미주입 시 send_telegram(미설정 시 조용히 스킵)을 쓴다.
    """
    today = today or date.today()
    # 2026-07 사용자 결정: 순수 샤프비율 극대화(무제약)를 그대로 유지 — MDD 제약은 걸지 않는다.
    # MDD -20% 제약 버전(run_monte_carlo_mdd_constrained, montecarlo.py)은 별도로 만들어뒀지만
    # 이 배치의 기본값으로는 쓰지 않는다(필요하면 monte_carlo_fn 인자로 주입해서 쓸 수 있다).
    monte_carlo_fn = monte_carlo_fn or run_monte_carlo
    send_fn = send_fn or send_telegram

    init_db(db_path)
    conn = connect(db_path)
    try:
        panel = build_price_panel(fetch_fn=fetch_yf)
        irx = fetch_irx_series(fetch_fn=fetch_irx)
        snapshot = run_walk_forward(
            panel, irx, monte_carlo_fn=monte_carlo_fn, n_simulations=n_simulations, today=today,
        )
        # 저장 전에 직전 스냅샷을 읽어 델타(AC13)를 계산한다.
        previous = get_latest_snapshot(conn)
        persist_snapshot(conn, snapshot)
    finally:
        conn.close()

    message = build_delta_message(snapshot, previous)
    alerted = bool(send_fn(message))
    return {"snapshot": snapshot, "message": message, "alerted": alerted}
