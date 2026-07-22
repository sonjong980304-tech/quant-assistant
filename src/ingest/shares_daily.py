"""일자별 상장주식수 수집 (pykrx get_market_cap_by_date → daily_shares 테이블).

액면분할·무상증자가 분기 결산일 '사이'에 나면 financials.shares_outstanding(분기 재무제표
딸림)이 다음 분기 보고서까지 옛 값에 멈춰, prices.market_cap 이 '분할 전 주식수 × 분할 후
가격'으로 축소되는 구조적 버그를 해소하기 위한 정확한 일자별 상장주식수 소스다.
pykrx get_market_cap_by_date 는 종목당 API 호출 1회로 기간 전체의 '그 날 실제 발효 중이던'
상장주식수를 준다(결산일 지연·look-ahead 없음 — 그 값 자체가 그 날짜의 실제 값).

⚠️ pykrx 호출 전에 반드시 src.config.CONFIG 가 import 돼 있어야 한다(.env 의 KRX_ID/KRX_PW
로그인 자격을 pykrx 가 환경변수에서 읽는다). 이 모듈은 상단에서 CONFIG 를 import 해 보장한다.
"""
from __future__ import annotations

import sqlite3

from ..config import CONFIG  # noqa: F401 — .env 로드(pykrx KRX 로그인 자격) 보장을 위한 import


def fetch_daily_shares(code: str, fromdate: str, todate: str) -> list[dict]:
    """pykrx 로 종목의 일자별 상장주식수를 조회해 [{"date","shares_outstanding"}, ...] 반환.

    fromdate/todate 는 pykrx 형식(YYYYMMDD). 반환 date 는 'YYYY-MM-DD' 로 정규화한다.
    데이터가 없으면(상장 전 구간/빈 응답) 빈 리스트를 반환한다. 네트워크/종목없음 등의
    예외는 호출부(백필 루프)가 종목 단위로 격리하도록 그대로 전파한다
    (backfill_prices._ingest_one 과 동일 규약 — 빈 df 는 빈 리스트, 오류는 전파).
    """
    from pykrx import stock  # 지연 import (네트워크 의존)

    df = stock.get_market_cap_by_date(fromdate, todate, code)
    if df is None or len(df) == 0 or "상장주식수" not in df.columns:
        return []
    rows: list[dict] = []
    for dt, row in df.iterrows():
        rows.append({
            "date": dt.strftime("%Y-%m-%d"),
            "shares_outstanding": row["상장주식수"],
        })
    return rows


def upsert_daily_shares(conn: sqlite3.Connection, code: str, rows: list[dict], source: str = "pykrx") -> int:
    """rows 를 daily_shares 에 INSERT OR REPLACE(UNIQUE(stock_code,date)). 적재 행수 반환.

    값이 없거나(None/NaN) 0 이하인 행은 스킵한다 — pykrx 는 KRX 정식 데이터라 이상치 가드가
    거의 불필요하지만, 상장 전/거래정지 등으로 0 이 섞일 수 있어 비양수만 걸러낸다.
    """
    n = 0
    for r in rows:
        val = r.get("shares_outstanding")
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if fval != fval or fval <= 0:  # NaN(자기 자신과 다름) 또는 비양수 스킵
            continue
        conn.execute(
            "INSERT OR REPLACE INTO daily_shares(stock_code, date, shares_outstanding, source) "
            "VALUES (?,?,?,?)",
            (code, r["date"], fval, source),
        )
        n += 1
    return n
