"""네이버 fchart 수정주가 일별 시세 크롤러.

소스: https://fchart.stock.naver.com/sise.nhn?symbol={code}&timeframe=day&count={n}&requestType=0
(네이버 차트 위젯이 쓰는 내부 API — 사람이 보는 finance.naver.com/item/sise_day.naver 페이지는
수정주가가 아니라서 채택하지 않음. 2018-05-04 삼성전자 50:1 액면분할 구간으로 직접 검증함.)

응답은 EUC-KR XML이며 <item data="YYYYMMDD|open|high|low|close|volume" /> 형태로
거래일별 1건씩 들어있다. 액면분할 전후 거래정지일은 open/high/low/volume이 전부 0으로
찍히는데(close만 유효), 이 경우를 실제 가격 0원이 아니라 결측으로 취급해 None으로 바꾼다.
"""
from __future__ import annotations

import re

from ..db import connect, init_db
from .http_fetch import ThrottledFetcher
from .notify import send_slack_alert
from .robust import log_ingest

_ITEM_RE = re.compile(r'<item\s+data="([^"]+)"\s*/>')
_FCHART_URL = "https://fchart.stock.naver.com/sise.nhn?symbol={symbol}&timeframe=day&count={count}&requestType=0"


def _to_date(raw: str) -> str:
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def parse_fchart_xml(xml_text: str) -> list[dict]:
    """fchart 응답 XML을 {date, open, high, low, close, volume} 딕셔너리 리스트로 변환."""
    rows = []
    for raw in _ITEM_RE.findall(xml_text):
        date_raw, open_, high, low, close, volume = raw.split("|")
        open_i, high_i, low_i, volume_i = int(open_), int(high), int(low), int(volume)
        halted = open_i == 0 and high_i == 0 and low_i == 0
        rows.append(
            {
                "date": _to_date(date_raw),
                "open": None if halted else open_i,
                "high": None if halted else high_i,
                "low": None if halted else low_i,
                "close": int(close),
                "volume": None if halted else volume_i,
            }
        )
    return rows


def fetch_daily_prices(stock_code: str, count: int = 3000, fetcher: ThrottledFetcher | None = None) -> list[dict]:
    """네이버 fchart에서 일별 수정주가를 받아 파싱까지 마친 행 리스트로 반환."""
    fetcher = fetcher or ThrottledFetcher()
    url = _FCHART_URL.format(symbol=stock_code, count=count)
    response = fetcher.get(url)
    response.encoding = "euc-kr"
    return parse_fchart_xml(response.text)


def ingest_naver_prices(
    db_path: str | None = None,
    fetcher: ThrottledFetcher | None = None,
    count: int = 3000,
    commit_every: int = 1,
    codes: list[str] | None = None,
) -> dict:
    """company 테이블 전종목의 OHLCV를 네이버 fchart에서 받아 prices에 upsert.

    codes를 주면 company 전체 대신 그 종목들만 순회한다(액면병합 등으로 오염된
    과거 종가를 골라 재수집하는 복구용). None(기본값)이면 기존대로 company 전체를
    순회한다(하위호환).

    실패 종목(HTTP 오류/파싱 실패/빈 응답)은 건너뛰고 Slack 알림만 보낸다
    (pykrx 자동 폴백 없음, AC6). 겹치는 (stock_code, date)는 네이버 값으로
    전체 덮어쓴다(INSERT OR REPLACE, AC3 — 네이버가 source of truth).

    commit_every개 종목마다 주기적으로 commit한다. 기본값 1(종목마다 즉시 commit) —
    전종목 순회는 종목당 2초 지연이 있어 수천 종목 기준 수시간 걸릴 수 있는데,
    commit_every가 크면 그만큼 오래 쓰기 트랜잭션이 열려있어 동시에 뜬 웹 서버의 다른
    쓰기가 `database is locked`로 실패한다(us_financials.py에서 실제로 발생·확인됨).
    종목마다 commit하면 (1) 프로세스가 죽어도(예: 전원 종료) 진행이 안 날아가고,
    (2) 잠금 구간이 수 밀리초로 줄어 동시 접근과 공존 가능하다.
    """
    fetcher = fetcher or ThrottledFetcher()
    init_db(db_path)
    conn = connect(db_path)
    try:
        if codes is None:
            codes = [r["stock_code"] for r in conn.execute("SELECT stock_code FROM company").fetchall()]
        succeeded = 0
        failed: list[str] = []
        price_rows = 0
        for code in codes:
            try:
                rows = fetch_daily_prices(code, count=count, fetcher=fetcher)
                if not rows:
                    raise ValueError("빈 응답(필드 누락)")
            except Exception as exc:  # noqa: BLE001 — 종목별 실패를 격리해 다음 종목으로 계속
                failed.append(code)
                log_ingest({"source": "naver_prices", "stock_code": code, "status": "fail", "error": str(exc)})
                send_slack_alert(f"[naver_prices] {code} 수집 실패: {exc}")
                continue
            for row in rows:
                # INSERT OR REPLACE는 행 전체를 갈아끼워 pykrx(krx.py)가 채운
                # market_cap을 NULL로 지운다 — ON CONFLICT DO UPDATE로 네이버가
                # 소유한 OHLCV 컬럼만 갱신하고 market_cap은 보존한다.
                conn.execute(
                    "INSERT INTO prices(stock_code, date, open, high, low, close, volume) "
                    "VALUES (?,?,?,?,?,?,?) "
                    "ON CONFLICT(stock_code, date) DO UPDATE SET "
                    "open=excluded.open, high=excluded.high, low=excluded.low, "
                    "close=excluded.close, volume=excluded.volume",
                    (code, row["date"], row["open"], row["high"], row["low"], row["close"], row["volume"]),
                )
                price_rows += 1
            succeeded += 1
            if succeeded % commit_every == 0:
                conn.commit()
        conn.commit()
        return {"tickers": len(codes), "succeeded": succeeded, "failed": failed, "price_rows": price_rows}
    finally:
        conn.close()
