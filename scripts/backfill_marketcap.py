"""과거 시가총액 백필 (prices.market_cap = 종가 × 그 시점 유효 상장주식수).

⚠️ 반드시 "분기별 상장주식수 수집(backfill_full / update_financials)" 후에 실행할 것.
   현재 financials의 shares_outstanding 행이 거의 없으면(초기 11행) 대부분 NULL로 남는다.
   주식수가 충분히 쌓인 뒤 실행해야 의미 있는 결과가 나온다.

이 스크립트는 DART/pykrx를 호출하지 않는다(이미 적재된 financials.shares_outstanding만 사용)
→ DART 일일 한도와 무관하다. 다만 위 사유로 "주식수 수집 후"에 실행한다.

동작
----
- prices의 각 (stock_code, date) 행에 대해
  market_cap = close × (그 시점에 유효한 상장주식수)를 채운다.
- "그 시점 유효 주식수": 해당 주가일(date)에 **실제로 공시돼 있던**(disclosed_date <= date)
  shares_outstanding 중 가장 최근 값. 실적 분기 라벨이 아니라 **공시일 기준**이라
  백테스트에서 미래 데이터(look-ahead)를 쓰지 않는다.
  예: 2024-05-31 주가에는 2024Q1(5/15 공시) 또는 그 이전 주식수만 쓰고, 2024Q2(8월 공시)는 안 씀.
- 종목별로 shares_outstanding 시계열을 한 번만 읽어 메모리에서 매칭(효율적).
- UPDATE prices SET market_cap=... (DART 호출 없음). idempotent: 다시 돌려도 안전.

실행: python3 scripts/backfill_marketcap.py   ← 분기별 주식수 수집 후에만!
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import connect, init_db


def _shares_series(conn) -> dict:
    """종목별 상장주식수 시계열: {stock_code: [(disclosed_date'YYYYMMDD', shares), ...]} (공시일 오름차순).

    disclosed_date가 없는 행은 시점을 알 수 없어 제외(look-ahead 방지).
    """
    rows = conn.execute(
        """SELECT stock_code, disclosed_date, amount
             FROM financials
            WHERE account_key = 'shares_outstanding' AND amount IS NOT NULL AND amount > 0
              AND disclosed_date IS NOT NULL AND disclosed_date != ''
            ORDER BY stock_code, disclosed_date"""
    ).fetchall()
    series: dict = {}
    for r in rows:
        series.setdefault(r["stock_code"], []).append((str(r["disclosed_date"]), r["amount"]))
    for code in series:
        series[code].sort(key=lambda t: t[0])  # 'YYYYMMDD' 문자열 정렬 = 시간순
    return series


def _effective_shares(series: list, on_compact: str):
    """series(공시일 오름차순)에서 on_compact('YYYYMMDD') 시점까지 **공시된** 최신 주식수.

    공시일(disclosed_date) 기준이라 미래에 공시될 주식수는 쓰지 않는다(look-ahead 방지). 없으면 None.
    """
    best = None
    for disc, shares in series:
        if disc <= on_compact:
            best = shares  # 오름차순이라 마지막으로 통과한 값이 "그 시점까지 공시된 최신"
        else:
            break
    return best


def backfill_marketcap(db_path: str | None = None) -> dict:
    """prices.market_cap을 종가 × (그 시점 공시된 상장주식수)로 채운다(UPDATE)."""
    init_db(db_path)
    conn = connect(db_path)
    try:
        series = _shares_series(conn)
        price_rows = conn.execute(
            "SELECT stock_code, date, close FROM prices WHERE close IS NOT NULL"
        ).fetchall()

        updated = 0
        no_shares = 0
        for r in price_rows:
            ser = series.get(r["stock_code"])
            if not ser:
                no_shares += 1
                continue
            # prices.date·disclosed_date 모두 'YYYY-MM-DD' → 변환 없이 직접 비교(사전식=시간순)
            shares = _effective_shares(ser, r["date"])
            if not shares:
                no_shares += 1
                continue
            cap = round(r["close"] * shares)
            conn.execute(
                "UPDATE prices SET market_cap = ? WHERE stock_code = ? AND date = ?",
                (cap, r["stock_code"], r["date"]),
            )
            updated += 1
        conn.commit()

        report = {
            "price_rows": len(price_rows),
            "market_cap_updated": updated,
            "no_shares": no_shares,
            "codes_with_shares": len(series),
        }
        print(f"시가총액 백필 완료: {report}")
        return report
    finally:
        conn.close()


if __name__ == "__main__":
    # ⚠️ 분기별 상장주식수 수집 완료 후에만 실행할 것.
    print(backfill_marketcap())
