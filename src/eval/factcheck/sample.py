"""factcheck 표본 선정 (US-2).

.omc/specs/brainstorming-factcheck-eval.md Round 11: 재무제표·주가 표본은
DB에서 동적으로 시가총액 상위 N종목을 조회한다 — 하드코딩 금지(순위가 바뀌어도
재현 가능해야 함). "최신 시가총액"의 정의는 src/eval/goldset.py의 관례
(p.date=(SELECT MAX(date) FROM prices))를 그대로 따른다 — prices는 하루 1회
일괄 갱신되어 전 종목이 같은 최신 거래일을 공유한다는 전제다.
"""
from __future__ import annotations

import sqlite3


def top_market_cap_stocks(conn: sqlite3.Connection, n: int) -> list[dict]:
    """시가총액(prices 최신일 기준) 상위 n개 종목을 내림차순으로 반환한다.

    반환: [{"stock_code":..., "name":..., "market_cap":...}, ...] (최대 n개)
    """
    rows = conn.execute(
        "SELECT c.stock_code AS stock_code, c.name AS name, p.market_cap AS market_cap "
        "FROM prices p "
        "JOIN company c ON c.stock_code = p.stock_code "
        "WHERE p.date = (SELECT MAX(date) FROM prices) AND p.market_cap IS NOT NULL "
        "ORDER BY p.market_cap DESC "
        "LIMIT ?",
        (n,),
    ).fetchall()
    return [dict(r) for r in rows]
