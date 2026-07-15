"""한국 주가 데이터 에이전트 — prices 테이블 조회 + 기술지표 부착 (HA-3).

이 프로젝트의 한국 주가는 이미 `prices` 테이블 하나로 병합되어 있다(src/ingest/krx.py가
종가·시가총액, src/ingest/naver_prices.py가 시가/고가/저가/거래량을 각각 ON CONFLICT
DO UPDATE로 채움). **이 파일은 신규 병합 로직을 만들지 않는다** — prices 테이블을
execute_sql(HA-1 실행기, src/agents/exec_runtime.py)로 그대로 조회할 뿐이다.

기술지표(이동평균/RSI/MACD/볼린저밴드)는 src/backtest/primitives.py의
compute_technical_indicator(TA-Lib 기반, 이전 세션에 이미 완성)를 그대로 재사용한다 —
TA-Lib 계산 로직을 새로 만들지 않는다.

안전: SQL은 반드시 execute_sql()을 경유한다(conn.execute() 직접 호출 금지 — conn은
connect_readonly()로 만든 읽기전용 연결이어야 함). execute_sql은 파라미터 바인딩을
지원하지 않으므로(sql 문자열 하나만 받음) 종목코드를 SQL에 직접 문자열로 끼워 넣기 전에
반드시 6자리 숫자 형식인지 검증한다(주입 방지) — 형식이 아닌 입력은 조용히 걸러진다.

반환은 get_cross_section(src/backtest/primitives.py)과 동일한 계약으로 통일한다 —
stock_code 필드를 가진 list[dict]이며, HA-2(재무데이터)와 stock_code 키로 자연스럽게
merge 가능하고 compute_technical_indicator의 rows 인자로 그대로 넣을 수 있다.
"""
from __future__ import annotations

import re
from typing import Callable

from src.agents.exec_runtime import execute_sql
from src.backtest.primitives import compute_technical_indicator

_STOCK_CODE_RE = re.compile(r"^\d{6}$")

_PRICE_COLUMNS = "p.stock_code, p.date, p.close, p.market_cap, p.open, p.high, p.low, p.volume"


def _normalize_codes(stock_codes: str | list[str]) -> list[str]:
    """단일 종목코드(str) 또는 종목코드 리스트를 받아 6자리 숫자 형식만 남긴다.

    execute_sql은 파라미터 바인딩이 없어 종목코드를 SQL 문자열에 직접 끼워 넣어야 하므로,
    형식이 아닌 입력(SQL 인젝션 시도 포함)은 쿼리를 만들기 전에 걸러낸다.
    """
    codes = [stock_codes] if isinstance(stock_codes, str) else list(stock_codes)
    return [c for c in codes if isinstance(c, str) and _STOCK_CODE_RE.match(c)]


def get_latest_price_kr(
    conn,
    stock_codes: str | list[str],
    execute_sql_fn: Callable | None = None,
) -> list[dict]:
    """한국 종목코드(6자리, 단일 str 또는 list)를 받아 prices 테이블에서 종목별 최신
    스냅샷(종가/시가총액/시가/고가/저가/거래량)을 조회한다.

    prices 테이블은 이미 병합 완료된 단일 테이블이라(모듈 docstring 참고) 여기서는 새
    병합 로직 없이 종목별 최신 date 행만 골라 반환한다. SQL은 execute_sql()로만 실행한다
    (conn.execute() 직접 호출 금지 — conn은 connect_readonly() 읽기전용 연결이어야 함).

    execute_sql_fn은 테스트 주입용(기본=execute_sql, HA-1 실행기) — get_cross_section의
    metrics_fn 관례와 동일한 DI 패턴이다.

    반환: get_cross_section과 동일한 계약의 list[dict](stock_code 필드 포함). 데이터가
    없거나 형식이 맞지 않는 종목코드는 결과에서 빠진다(순서 보장 안 함).
    """
    execute_sql_fn = execute_sql_fn or execute_sql
    codes = _normalize_codes(stock_codes)
    if not codes:
        return []
    code_list_sql = ",".join(f"'{c}'" for c in codes)
    sql = (
        f"SELECT {_PRICE_COLUMNS} FROM prices p "
        "INNER JOIN (SELECT stock_code, MAX(date) AS max_date FROM prices "
        f"WHERE stock_code IN ({code_list_sql}) GROUP BY stock_code) latest "
        "ON p.stock_code = latest.stock_code AND p.date = latest.max_date"
    )
    result = execute_sql_fn(sql, conn)
    if not result.get("ok"):
        return []
    return result["rows"]


def get_price_history_kr(
    conn,
    stock_code: str,
    days: int = 365,
    execute_sql_fn: Callable | None = None,
) -> list[dict]:
    """단일 종목(6자리)의 최근 종가 시계열을 과거→최신 순으로 반환한다(차트용).

    get_latest_price_kr이 종목별 최신 스냅샷 1행만 주는 것과 달리, 여기서는 최근 days
    거래일의 (stock_code, date, close) 시계열을 반환한다. web/app.py의 api_macro_history와
    동일한 "DESC LIMIT 후 reversed" 패턴이지만, 도메인 에이전트 관례대로 conn.execute()가
    아니라 execute_sql(HA-1 실행기)을 경유한다.

    execute_sql은 파라미터 바인딩이 없어 종목코드를 SQL 문자열에 직접 끼워 넣으므로, 6자리
    숫자 형식이 아닌 입력(SQL 인젝션 시도 포함)은 쿼리를 만들기 전에 걸러진다(빈 리스트 반환).
    """
    execute_sql_fn = execute_sql_fn or execute_sql
    codes = _normalize_codes(stock_code)
    if not codes:
        return []
    code = codes[0]
    try:
        limit = int(days)
    except (TypeError, ValueError):
        limit = 365
    if limit <= 0:
        return []
    sql = (
        f"SELECT stock_code, date, close FROM prices WHERE stock_code = '{code}' "
        f"ORDER BY date DESC LIMIT {limit}"
    )
    result = execute_sql_fn(sql, conn)
    if not result.get("ok"):
        return []
    return list(reversed(result["rows"]))  # 최신순 조회 → 뒤집어 과거→최신(그리기 순서)


def get_price_snapshot_kr(
    conn,
    stock_codes: str | list[str],
    asof: str | None = None,
    indicators: list[dict] | None = None,
    execute_sql_fn: Callable | None = None,
    indicator_fn: Callable | None = None,
) -> list[dict]:
    """get_latest_price_kr() 스냅샷에 (옵션) 기술지표를 붙여 반환하는 통합 진입점.

    indicators가 주어지면 compute_technical_indicator(이전 세션에 이미 완성된 TA-Lib
    프리미티브, src/backtest/primitives.py)를 그대로 호출해 지표 필드를 추가한다 — 지표
    계산 로직은 새로 만들지 않는다. indicator_fn은 테스트 주입용(기본=compute_technical_
    indicator).

    asof를 지정하지 않으면 조회된 스냅샷 중 가장 최신 date를 기준시점으로 사용한다(실제
    캘린더 오늘 날짜가 아니라 DB에 실제로 존재하는 최신 데이터 시점 — 주말/공휴일이나
    DB 미갱신 상황에서도 결정론적으로 동작).

    조회 결과가 없으면(빈 rows) 무거운 연산(가격 시계열 재조회 + TA-Lib 계산)인
    compute_technical_indicator를 호출하지 않고 빈 리스트를 그대로 반환한다.
    """
    indicator_fn = indicator_fn or compute_technical_indicator
    rows = get_latest_price_kr(conn, stock_codes, execute_sql_fn=execute_sql_fn)
    if not indicators or not rows:
        return rows
    resolved_asof = asof or max(r["date"] for r in rows if r.get("date"))
    return indicator_fn(conn, rows, resolved_asof, indicators)
