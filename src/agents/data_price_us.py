"""미국 주가 데이터 에이전트 — us_prices/us_company 조회 + 기술지표 부착 (HA-4).

이 프로젝트의 미국 주가는 이미 단일 출처(yfinance, src/ingest/us_prices.py)로
채워진 `us_prices` 테이블 하나다 — 한국(prices, KRX+네이버 병합)과 달리 이종 소스를
합칠 필요가 없다. **이 파일은 신규 병합 로직을 만들지 않는다** — us_prices/us_company
(src/backtest/data_access_us.py와 동일 테이블)를 execute_sql(HA-1 실행기,
src/agents/exec_runtime.py)로 그대로 조회할 뿐이다. 시가총액은 us_prices에 없고
us_company.market_cap 근사치를 쓴다(data_access_us._price_at_us와 동일 근거).

기술지표(이동평균/RSI/MACD/볼린저밴드 등)는 src/backtest/primitives.py의
compute_technical_indicator(TA-Lib 기반, 이전 세션에 이미 완성)를 그대로 재사용한다 —
TA-Lib 계산 로직을 새로 만들지 않는다. 다만 그 기본 history_fn(price_history_batch)은
한국 prices 테이블을 조회하므로, US에서는 us_price_history_batch(신규,
src/backtest/data_access_us.py)를 history_fn으로 주입한다.

안전: SQL은 반드시 execute_sql()을 경유한다(conn.execute() 직접 호출 금지 — conn은
connect_readonly()로 만든 읽기전용 연결이어야 함). execute_sql은 파라미터 바인딩을
지원하지 않으므로(sql 문자열 하나만 받음) 티커를 SQL에 직접 문자열로 끼워 넣기 전에
반드시 US 티커 형식인지 검증한다(주입 방지) — 형식이 아닌 입력은 조용히 걸러진다.

반환은 src/agents/data_price_kr.py(HA-3)의 get_latest_price_kr/get_price_snapshot_kr와
동일한 계약으로 통일한다 — stock_code 필드를 가진 list[dict]이며, HA-2(재무데이터)와
stock_code 키로 자연스럽게 merge 가능하고 compute_technical_indicator의 rows 인자로
그대로 넣을 수 있다. HA-7(미국주식 도메인 에이전트)이 HA-6과 대칭으로 호출한다.
"""
from __future__ import annotations

import re
from typing import Callable

from src.agents.exec_runtime import execute_sql
from src.backtest.data_access_us import us_price_history_batch
from src.backtest.primitives import compute_technical_indicator

# 미국 티커 형식: 대문자 1~6자 + 옵션으로 '.'+대문자 1~3자(예: AAPL, BRK.B, BF.B).
_TICKER_RE = re.compile(r"^[A-Z]{1,6}(\.[A-Z]{1,3})?$")

_PRICE_COLUMNS = (
    "p.stock_code, p.date, p.close, p.open, p.high, p.low, p.volume, "
    "c.name, c.exchange, c.sector, c.market_cap"
)


def _normalize_codes(stock_codes: str | list[str]) -> list[str]:
    """단일 티커(str) 또는 티커 리스트를 받아 형식이 유효한 것만 대문자로 남긴다.

    execute_sql은 파라미터 바인딩이 없어 티커를 SQL 문자열에 직접 끼워 넣어야 하므로,
    형식이 아닌 입력(SQL 인젝션 시도 포함)은 쿼리를 만들기 전에 걸러낸다.
    """
    codes = [stock_codes] if isinstance(stock_codes, str) else list(stock_codes)
    upper = [c.strip().upper() for c in codes if isinstance(c, str)]
    return [c for c in upper if _TICKER_RE.match(c)]


def get_latest_price_us(
    conn,
    stock_codes: str | list[str],
    execute_sql_fn: Callable | None = None,
) -> list[dict]:
    """미국 티커(단일 str 또는 list)를 받아 종목별 최신 스냅샷(종가/시가/고가/저가/거래량
    + name/exchange/sector/market_cap)을 조회한다.

    us_prices(가격)와 us_company(회사정보)는 원래도 별개 테이블이라(모듈 docstring
    참고) 여기서는 새 병합 로직 없이 LEFT JOIN으로 함께 읽어올 뿐이다. SQL은
    execute_sql()로만 실행한다(conn.execute() 직접 호출 금지 — conn은
    connect_readonly() 읽기전용 연결이어야 함).

    execute_sql_fn은 테스트 주입용(기본=execute_sql, HA-1 실행기) — get_latest_price_kr의
    관례와 동일한 DI 패턴이다.

    반환: get_latest_price_kr과 동일한 계약의 list[dict](stock_code 필드 포함). 데이터가
    없거나 형식이 맞지 않는 티커는 결과에서 빠진다(순서 보장 안 함).
    """
    execute_sql_fn = execute_sql_fn or execute_sql
    codes = _normalize_codes(stock_codes)
    if not codes:
        return []
    code_list_sql = ",".join(f"'{c}'" for c in codes)
    sql = (
        f"SELECT {_PRICE_COLUMNS} FROM us_prices p "
        "INNER JOIN (SELECT stock_code, MAX(date) AS max_date FROM us_prices "
        f"WHERE stock_code IN ({code_list_sql}) GROUP BY stock_code) latest "
        "ON p.stock_code = latest.stock_code AND p.date = latest.max_date "
        "LEFT JOIN us_company c ON c.stock_code = p.stock_code"
    )
    result = execute_sql_fn(sql, conn)
    if not result.get("ok"):
        return []
    return result["rows"]


def get_price_history_us(
    conn,
    stock_code: str,
    days: int = 365,
    execute_sql_fn: Callable | None = None,
) -> list[dict]:
    """단일 티커의 최근 종가 시계열을 과거→최신 순으로 반환한다(차트용).

    HA-3(data_price_kr.get_price_history_kr)와 동일 계약이며 유일한 차이는 대상 테이블
    (us_prices)이다. execute_sql은 파라미터 바인딩이 없어 티커를 SQL 문자열에 직접 끼워
    넣으므로, US 티커 형식이 아닌 입력(SQL 인젝션 시도 포함)은 걸러진다(빈 리스트 반환).
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
        f"SELECT stock_code, date, close FROM us_prices WHERE stock_code = '{code}' "
        f"ORDER BY date DESC LIMIT {limit}"
    )
    result = execute_sql_fn(sql, conn)
    if not result.get("ok"):
        return []
    return list(reversed(result["rows"]))  # 최신순 조회 → 뒤집어 과거→최신(그리기 순서)


def get_price_snapshot_us(
    conn,
    stock_codes: str | list[str],
    asof: str | None = None,
    indicators: list[dict] | None = None,
    execute_sql_fn: Callable | None = None,
    indicator_fn: Callable | None = None,
    history_fn: Callable | None = None,
) -> list[dict]:
    """get_latest_price_us() 스냅샷에 (옵션) 기술지표를 붙여 반환하는 통합 진입점.

    indicators가 주어지면 compute_technical_indicator(이전 세션에 이미 완성된 TA-Lib
    프리미티브, src/backtest/primitives.py)를 그대로 호출해 지표 필드를 추가한다 — 지표
    계산 로직은 새로 만들지 않는다. history_fn 기본값은 us_price_history_batch
    (src/backtest/data_access_us.py, US 전용 가격 시계열 배치조회) — 한국판(data_price_kr)
    의 기본 history_fn(price_history_batch, KR prices 테이블)과 달리 US는 us_prices
    테이블을 봐야 하므로 명시적으로 주입한다. indicator_fn은 테스트 주입용
    (기본=compute_technical_indicator).

    asof를 지정하지 않으면 조회된 스냅샷 중 가장 최신 date를 기준시점으로 사용한다(실제
    캘린더 오늘 날짜가 아니라 DB에 실제로 존재하는 최신 데이터 시점).

    조회 결과가 없으면(빈 rows) 무거운 연산(가격 시계열 재조회 + TA-Lib 계산)인
    compute_technical_indicator를 호출하지 않고 빈 리스트를 그대로 반환한다.
    """
    indicator_fn = indicator_fn or compute_technical_indicator
    history_fn = history_fn or us_price_history_batch
    rows = get_latest_price_us(conn, stock_codes, execute_sql_fn=execute_sql_fn)
    if not indicators or not rows:
        return rows
    resolved_asof = asof or max(r["date"] for r in rows if r.get("date"))
    return indicator_fn(conn, rows, resolved_asof, indicators, history_fn=history_fn)
