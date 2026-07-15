"""US 재무제표 — yfinance 손익/재무상태/현금흐름 매퍼 + 오케스트레이터.

`Ticker(symbol).income_stmt`/`.balance_sheet`/`.cashflow`(및 `quarterly_*`)가
반환하는 pandas DataFrame(행=항목명, 열=기간말일 Timestamp)을 `us_financials`
EAV 스키마 행으로 변환한다.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd

from ..db import connect, init_db
from .notify import send_slack_alert
from .robust import log_ingest

# yfinance는 재무제표에 실제 공시일을 주지 않고 기간말일만 준다 → 보수적 고정지연으로 근사한다.
# SEC 제출기한 기준: 10-Q(분기) 약 45일, 10-K(연간) 약 90일. 실제 EDGAR 조회는 범위 밖.
_DISCLOSURE_LAG_DAYS = {"quarterly": 45, "annual": 90}


def _approx_disclosed_date(as_of_date: str, period_type: str) -> str | None:
    """기간말일 + 고정지연으로 공시일을 근사한다(look-ahead 방지용). 알 수 없는 period_type이면 None."""
    lag = _DISCLOSURE_LAG_DAYS.get(period_type)
    if lag is None:
        return None
    return (datetime.strptime(as_of_date, "%Y-%m-%d") + timedelta(days=lag)).strftime("%Y-%m-%d")


def normalize_financial_statement(
    stock_code: str,
    statement_type: str,
    period_type: str,
    df: pd.DataFrame,
) -> list[dict]:
    """yfinance 재무제표 wide DataFrame을 EAV 행 리스트로 변환. NaN 값은 제외."""
    rows: list[dict] = []
    for item_key, series in df.iterrows():
        for period, value in series.items():
            if pd.isna(value):
                continue
            as_of_date = period.strftime("%Y-%m-%d")
            rows.append({
                "stock_code": stock_code,
                "as_of_date": as_of_date,
                "period_type": period_type,
                "statement_type": statement_type,
                "item_key": item_key,
                "item_value": float(value),
                "disclosed_date": _approx_disclosed_date(as_of_date, period_type),
            })
    return rows


def _fetch_yf_statements(stock_code: str) -> dict:
    import yfinance as yf  # 지연 import — pykrx.py의 기존 패턴과 동일

    t = yf.Ticker(stock_code)
    return {
        ("income_stmt", "annual"): t.income_stmt,
        ("income_stmt", "quarterly"): t.quarterly_income_stmt,
        ("balance_sheet", "annual"): t.balance_sheet,
        ("balance_sheet", "quarterly"): t.quarterly_balance_sheet,
        ("cashflow", "annual"): t.cashflow,
        ("cashflow", "quarterly"): t.quarterly_cashflow,
    }


def ingest_us_financials(
    db_path: str | None = None,
    fetch_statements: Optional[Callable[[str], dict]] = None,
    commit_every: int = 1,
    codes: Optional[list[str]] = None,
) -> dict:
    """us_company 전종목(또는 codes로 지정한 부분집합)의 손익/재무상태/현금흐름
    (quarterly+annual)을 yfinance에서 받아 us_financials에 upsert. 실패 종목은
    건너뛰고 Slack 알림만 보낸다.

    commit_every개 종목마다 주기적으로 commit한다. 기본값 1(종목마다 즉시 commit) —
    이 백필은 종목당 네트워크 호출로 수 초씩 걸려, commit_every가 크면 그만큼 오래
    쓰기 트랜잭션이 열려있어 동시에 뜬 웹 서버의 다른 쓰기(질의 기록 저장 등)가
    `sqlite3.OperationalError: database is locked`로 실패한다(busy_timeout 15초를
    초과). 종목마다 commit하면 잠금 구간이 수 밀리초로 줄어 동시 접근과 공존 가능하다.
    (전체 루프 끝에 1번만 commit하면 수천 종목 백필 중 프로세스가 죽었을 때 그때까지의
    진행이 통째로 날아가는 문제도 있었음 — INSERT OR REPLACE라 재실행 자체는 안전하지만
    처음부터 다시 하는 비용을 줄이기 위함이기도 하다.)

    codes를 지정하면 그 목록만 처리한다(기본 None이면 us_company 전종목). 이미 수집된
    종목을 건너뛰고 이어서 재시작할 때 등에 쓴다.
    """
    fetch_statements = fetch_statements or _fetch_yf_statements
    init_db(db_path)
    conn = connect(db_path)
    collected_at = datetime.now(timezone.utc).isoformat()
    try:
        if codes is None:
            codes = [r["stock_code"] for r in conn.execute("SELECT stock_code FROM us_company").fetchall()]
        succeeded = 0
        failed: list[str] = []
        metric_rows = 0
        for code in codes:
            try:
                statements = fetch_statements(code)
                rows: list[dict] = []
                for (statement_type, period_type), df in statements.items():
                    rows.extend(normalize_financial_statement(code, statement_type, period_type, df))
                if not rows:
                    raise ValueError("빈 응답(필드 누락)")
            except Exception as exc:  # noqa: BLE001 — 종목별 실패를 격리해 다음 종목으로 계속
                failed.append(code)
                log_ingest({"source": "us_financials", "stock_code": code, "status": "fail", "error": str(exc)})
                send_slack_alert(f"[us_financials] {code} 수집 실패: {exc}")
                continue
            for row in rows:
                conn.execute(
                    "INSERT OR REPLACE INTO us_financials"
                    "(stock_code, as_of_date, period_type, statement_type, item_key, item_value, disclosed_date, source, collected_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        code,
                        row["as_of_date"],
                        row["period_type"],
                        row["statement_type"],
                        row["item_key"],
                        row["item_value"],
                        row.get("disclosed_date"),
                        "yfinance",
                        collected_at,
                    ),
                )
                metric_rows += 1
            succeeded += 1
            if succeeded % commit_every == 0:
                conn.commit()
        conn.commit()
        return {"tickers": len(codes), "succeeded": succeeded, "failed": failed, "metric_rows": metric_rows}
    finally:
        conn.close()
