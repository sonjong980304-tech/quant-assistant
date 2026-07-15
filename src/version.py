"""data_version 유효시점 로직.

핵심 원칙
---------
1) 분기 재무는 "날짜로 단순 추정"하지 않는다. 분기보고서는 분기 종료 후
   약 45일 뒤(사업보고서는 ~90일) 공시되므로, 실제 DART에 공시된
   가장 최근 분기를 data_version으로 쓴다 (DB 적재 메타 우선).
2) 주가 포함 지표는 종가 스냅샷(일 1회) 기준. 같은 날 같은 질문은 히트,
   다음날이면 종가날짜가 바뀌어 자동 미스.

data_version 형식
-----------------
- 순수 재무 지표(부채비율/영업이익률/ROE): "2025Q1"           (분기 단위)
- 주가 포함 지표(PER/PBR/시총):           "2025Q1_20260622"  (분기+종가날짜)
- router 결과(route)로 형식이 결정된다.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone


# ---------------------------------------------------------------------------
# 시각 헬퍼 (외부에서 date.today()를 주입할 수 있게 today 인자 제공)
# ---------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def today_str(d: date | None = None) -> str:
    return (d or date.today()).strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# 분기 라벨 유틸 ("2025Q1" 형식)
# ---------------------------------------------------------------------------
def quarter_to_tuple(q: str) -> tuple[int, int]:
    y, n = q.upper().split("Q")
    return int(y), int(n)


def tuple_to_quarter(y: int, n: int) -> str:
    return f"{y}Q{n}"


def shift_quarter(q: str, delta: int) -> str:
    """분기 라벨을 delta 만큼 이동 (음수=과거)."""
    y, n = quarter_to_tuple(q)
    idx = (y * 4 + (n - 1)) + delta
    return tuple_to_quarter(idx // 4, idx % 4 + 1)


def recent_quarters(latest_q: str, count: int) -> list[str]:
    """latest_q 포함, 과거→최신 순으로 count개 분기 라벨."""
    return [shift_quarter(latest_q, -i) for i in range(count - 1, -1, -1)]


def quarter_end_date(q: str) -> date:
    y, n = quarter_to_tuple(q)
    return {1: date(y, 3, 31), 2: date(y, 6, 30), 3: date(y, 9, 30), 4: date(y, 12, 31)}[n]


# ---------------------------------------------------------------------------
# 공시 가능 분기 추정 (적재 시 어느 분기까지 만들지 결정하는 "보조" 함수)
# 실제 data_version은 DB 적재 메타를 우선 사용한다.
# ---------------------------------------------------------------------------
DISCLOSURE_LAG_DAYS = 45  # 분기보고서 공시 지연 (사업보고서 Q4는 익년 3/31로 별도 처리)


def estimate_available_quarter(d: date | None = None) -> str:
    """주어진 날짜 기준, 공시되었을 것으로 추정되는 최신 분기.

    Q1/Q2/Q3는 분기말 +45일, Q4(사업보고서)는 익년 3월 말로 가정.
    """
    d = d or date.today()
    cands: list[tuple[date, str]] = []
    for y in (d.year - 1, d.year):
        cands.append((date(y, 3, 31).replace(month=5, day=15), f"{y}Q1"))
        cands.append((date(y, 8, 14), f"{y}Q2"))
        cands.append((date(y, 11, 14), f"{y}Q3"))
        cands.append((date(y + 1, 3, 31), f"{y}Q4"))  # 사업보고서
    avail = sorted([(dd, q) for dd, q in cands if dd <= d])
    if not avail:
        # 아주 이른 날짜 방어
        return f"{d.year - 1}Q3"
    return avail[-1][1]


def estimate_disclosed_date(q: str) -> str:
    """분기 라벨의 추정 공시일 (YYYY-MM-DD). 더미 적재용."""
    end = quarter_end_date(q)
    y, n = quarter_to_tuple(q)
    if n == 4:
        return date(y + 1, 3, 31).isoformat()
    # +45일
    from datetime import timedelta

    return (end + timedelta(days=DISCLOSURE_LAG_DAYS)).isoformat()


# ---------------------------------------------------------------------------
# DB 기반 "유효 시점" 조회 (실제 공시/적재된 최신값 우선)
# ---------------------------------------------------------------------------
def effective_quarter(conn: sqlite3.Connection, d: date | None = None) -> str:
    """실제 공시(적재)된 가장 최근 분기.

    우선순위: ingest_meta('latest_disclosed_quarter') → financials MAX → 추정.
    """
    from .db import get_meta

    meta = get_meta(conn, "latest_disclosed_quarter")
    if meta:
        return meta
    row = conn.execute("SELECT MAX(quarter) AS q FROM financials").fetchone()
    if row and row["q"]:
        return row["q"]
    return estimate_available_quarter(d)


def effective_price_date(conn: sqlite3.Connection, d: date | None = None) -> str:
    """최신 종가 스냅샷 날짜 (YYYYMMDD).

    우선순위: ingest_meta('latest_price_date') → prices MAX → 오늘.
    """
    from .db import get_meta

    meta = get_meta(conn, "latest_price_date")
    if meta:
        return meta.replace("-", "")
    row = conn.execute("SELECT MAX(date) AS d FROM prices").fetchone()
    if row and row["d"]:
        return row["d"].replace("-", "")
    return today_str(d)


# route ∈ {'financial', 'price', 'both'}
PRICE_ROUTES = {"price", "both"}


def build_data_version(route: str, conn: sqlite3.Connection, d: date | None = None) -> str:
    """router 결과(route)에 따라 data_version 키 형식을 결정.

    - financial      → "2025Q1"
    - price | both   → "2025Q1_20260622"
    """
    q = effective_quarter(conn, d)
    if route in PRICE_ROUTES:
        return f"{q}_{effective_price_date(conn, d)}"
    return q


if __name__ == "__main__":
    today = date(2026, 6, 22)
    print("estimate_available_quarter(2026-06-22) =", estimate_available_quarter(today))
    print("recent_quarters(2026Q1, 12) =", recent_quarters("2026Q1", 12))
    print("estimate_disclosed_date(2026Q1) =", estimate_disclosed_date("2026Q1"))
    print("shift_quarter(2026Q1, -1) =", shift_quarter("2026Q1", -1))
