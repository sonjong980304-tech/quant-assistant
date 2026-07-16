"""미국판 백테스트 DB 어댑터 (data_access.py의 US 대응물).

엔진(engine.run_backtest)에 넘길 콜백을 us_company/us_prices/us_financials(EAV)에서 구현한다.
한국판 data_access.py와 시그니처를 맞추되, 미국 데이터 특성상 아래가 다르다:
- look-ahead 방지: disclosed_date는 실제 공시일이 아니라 기말+45/90일 근사값(us_financials.py 참고).
- 생존편향: 미국은 상장폐지 추적 데이터가 없다 → _is_alive_us는 bool이 아니라 문자열
  "unverifiable"(검증불가)을 반환한다. KR의 _is_alive(True/False)와 반환 타입이 다르므로
  호출부에서 "통과"로 오인하지 않게 명시적으로 구분한다.
- 시가총액: us_prices에 상장주식수가 없어 정밀 시총 계산 불가 → us_company.market_cap으로 근사한다.
- 벤치마크: 동일가중 유니버스 벤치마크는 KR의 build_benchmark_fn(콜백 기반)을 그대로 재사용하고,
  S&P500 실제지수(^GSPC)는 build_sp500_benchmark_fn으로 별도 계산한다.
"""
from __future__ import annotations

from ..ingest.metrics import _div
from .data_access import _one_year_before  # return_12m 기준일 계산 재사용(KR과 동일 규약)

# 미국은 상장폐지 추적 데이터가 없어 생존편향 검증이 원천적으로 불가능하다(버그가 아니라 알려진 한계).
UNVERIFIABLE = "unverifiable"


def effective_quarter_at_us(conn, code: str, asof: str) -> str | None:
    """asof 시점까지 공시(근사)된 최신 quarterly의 as_of_date (look-ahead 방지).

    us_financials.disclosed_date는 기말+45/90일 근사값이다(실제 EDGAR 공시일 아님).
    반환하는 as_of_date 문자열을 KR의 quarter처럼 '유효분기 식별자'로 쓴다.
    """
    row = conn.execute(
        "SELECT as_of_date FROM us_financials WHERE stock_code=? AND period_type='quarterly' "
        "AND disclosed_date IS NOT NULL AND disclosed_date<=? "
        "ORDER BY as_of_date DESC LIMIT 1",
        (code, asof),
    ).fetchone()
    return row["as_of_date"] if row else None


def _price_at_us(conn, code: str, asof: str):
    """(종가, 시가총액). 종가는 us_prices의 asof 이하 최신 종가.

    시가총액은 us_company.market_cap 근사치다 — us_prices엔 상장주식수가 없어 주가 변동을
    반영한 정밀 시총을 계산할 수 없다(정밀 계산엔 상장주식수 필요, 미수집). 근사임에 유의.
    """
    prow = conn.execute(
        "SELECT close FROM us_prices WHERE stock_code=? AND date<=? ORDER BY date DESC LIMIT 1",
        (code, asof),
    ).fetchone()
    if not prow or prow["close"] is None:
        return (None, None)
    crow = conn.execute("SELECT market_cap FROM us_company WHERE stock_code=?", (code,)).fetchone()
    cap = crow["market_cap"] if crow else None
    return (prow["close"], cap)


def us_price_history_batch(conn, codes: list[str], asof: str, lookback_days: int) -> dict[str, list[dict]]:
    """여러 US 티커의 연속 가격 시계열을 SQL 1회(IN절)로 배치 조회한다.

    data_access.py의 price_history_batch(KR)와 동일한 계약(반환 dict[stock_code, list[dict]],
    각 dict는 {"date","close"})이지만 us_prices 테이블을 조회한다. compute_technical_indicator
    (src/backtest/primitives.py)의 history_fn으로 주입해 US 종목에도 그대로 재사용하기 위함
    — TA-Lib 계산 로직 자체는 새로 만들지 않는다.
    """
    import datetime

    out: dict[str, list[dict]] = {code: [] for code in codes}
    if not codes:
        return out
    start_date = (
        datetime.date.fromisoformat(asof) - datetime.timedelta(days=lookback_days)
    ).isoformat()
    placeholders = ",".join("?" for _ in codes)
    rows = conn.execute(
        f"SELECT stock_code, date, close FROM us_prices "
        f"WHERE stock_code IN ({placeholders}) AND date>=? AND date<=? "
        f"ORDER BY stock_code, date ASC",
        (*codes, start_date, asof),
    ).fetchall()
    for r in rows:
        out[r["stock_code"]].append({"date": r["date"], "close": r["close"]})
    return out


def _is_alive_us(conn, code: str, asof: str) -> str:
    """미국은 상장폐지 데이터가 없어 생존 여부를 검증할 수 없다 → 항상 "unverifiable".

    KR의 _is_alive는 bool(True/False)을 돌려주지만 이 함수는 문자열을 돌려준다 —
    "검증불가"를 "통과(True)"와 혼동하지 않게 하기 위한 의도적 타입 차이다.
    """
    return UNVERIFIABLE


def _us_item(conn, code: str, as_of_date: str, statement_type: str, item_key: str):
    """특정 as_of_date의 단일 quarterly 항목값(EAV 단건 조회)."""
    row = conn.execute(
        "SELECT item_value FROM us_financials WHERE stock_code=? AND as_of_date=? "
        "AND period_type='quarterly' AND statement_type=? AND item_key=?",
        (code, as_of_date, statement_type, item_key),
    ).fetchone()
    return row["item_value"] if row and row["item_value"] is not None else None


def _ttm_net_income(conn, code: str, as_of_date: str):
    """최근 4개 quarterly Net Income 합(TTM). 4개 미만이면 추정하지 않고 None(KR _sum_ttm과 동일 원칙)."""
    rows = conn.execute(
        "SELECT item_value FROM us_financials WHERE stock_code=? AND period_type='quarterly' "
        "AND statement_type='income_stmt' AND item_key='Net Income' AND as_of_date<=? "
        "ORDER BY as_of_date DESC LIMIT 4",
        (code, as_of_date),
    ).fetchall()
    vals = [r["item_value"] for r in rows if r["item_value"] is not None]
    if len(vals) < 4:
        return None
    return sum(vals)


# KR data_access.METRIC_FIELD_DESCRIPTIONS 와 동일한 목적의 US 전용 단일 정의처(canonical
# source) — US 유니버스는 필드셋이 KR보다 좁다(revenue_growth/op_growth/ni_growth 없음,
# psr/pcr/ev_ebitda 등 없음, yfinance 소스 한계). domain_us.py의 _US_SCREEN_FIELDS가 이
# dict의 key만 파생해서 쓴다 — 새 지표 추가 시 여기 한 곳만 고치면 된다.
METRIC_FIELD_DESCRIPTIONS_US: dict[str, str] = {
    "per": "주가수익비율(PER)",
    "pbr": "주가순자산비율(PBR)",
    "roe": "자기자본이익률(ROE, %)",
    "operating_margin": "영업이익률(%, 해당분기)",
    "net_margin": "순이익률(%, 해당분기)",
    "return_12m": "최근 12개월 주가 수익률(모멘텀, %)",
    "operating_profit": "영업이익(달러 절대값, 해당분기, TTM 아님)",
    "revenue": "매출액(달러 절대값, 해당분기, TTM 아님)",
    "net_income": "순이익(달러 절대값, 해당분기, TTM 아님)",
    "market_cap": "시가총액(달러 절대값, 종가×상장주식수, 해당 기준시점)",
}


def metrics_at_us(conn, asof: str) -> list[dict]:
    """시점 asof의 유효 US 지표 행 목록 (look-ahead 방지 + KR과 동일 이상치가드).

    PER=시총/순이익(TTM), PBR=시총/자기자본, ROE=순이익(TTM)/자기자본,
    영업이익률=영업이익/매출(단일분기), 순이익률=순이익/매출(단일분기).
    검증된 item_key만 사용한다(prompts.py 매핑): Total Revenue/Operating Income/Net Income/Stockholders Equity.
    """
    companies = conn.execute(
        "SELECT stock_code, name, sector, exchange, security_type, financial_currency FROM us_company"
    ).fetchall()
    out = []
    for c in companies:
        code = c["stock_code"]
        q = effective_quarter_at_us(conn, code, asof)
        if not q:
            continue
        close, cap = _price_at_us(conn, code, asof)
        # 그 시점 실제 거래되는(주가가 있는) 종목만
        if not close:
            continue

        rev_q = _us_item(conn, code, q, "income_stmt", "Total Revenue")
        op_q = _us_item(conn, code, q, "income_stmt", "Operating Income")
        ni_q = _us_item(conn, code, q, "income_stmt", "Net Income")
        equity = _us_item(conn, code, q, "balance_sheet", "Stockholders Equity")
        ni_ttm = _ttm_net_income(conn, code, q)

        # 이상치가드는 KR metrics_at과 동일 임계값 재사용(과최적화 금지).
        # 순이익이 자기자본 대비 비현실적 거대(원본 오류)면 순이익 기반 지표 무효화.
        if ni_ttm is not None and equity and equity > 0 and abs(ni_ttm) > equity * 20:
            ni_ttm = None
        # 시총이 자기자본의 100배 초과(PBR>100) → 데이터 오류 의심 → 시총 기반 지표 무효.
        if cap is not None and equity and equity > 0 and cap > equity * 100:
            cap = None

        # ROE = 순이익(TTM)/자기자본. 양수이고 자기자본 미만(<100%)일 때만 인정(일회성 폭발 제외).
        roe = _div(ni_ttm, equity, pct=True) if (ni_ttm and equity and equity > 0 and ni_ttm < equity) else None
        # 영업이익률: 단일분기 op<rev(마진<100%)일 때만.
        op_margin = _div(op_q, rev_q, pct=True) if (rev_q and rev_q > 0 and op_q is not None and op_q < rev_q) else None
        # 순이익률: 단일분기(적자면 음수=유효).
        net_margin = _div(ni_q, rev_q, pct=True) if (rev_q and rev_q > 0 and ni_q is not None) else None

        # 12개월 가격 수익률(모멘텀, KR metrics_at과 동일 정의/규약). _price_at_us는 date<=기준일만
        # 보므로 asof 이후 종가를 참조하지 않는다(look-ahead 방지). 1년 전 종가 없으면 None.
        prev_close, _ = _price_at_us(conn, code, _one_year_before(asof))
        return_12m = _div(close - prev_close, prev_close, pct=True) if (prev_close and prev_close > 0) else None

        row = {
            "stock_code": code, "name": c["name"], "sector": c["sector"], "market": c["exchange"],
            "security_type": c["security_type"],
            "quarter": q, "close": close, "market_cap": cap,
            "per": _div(cap, ni_ttm) if (cap and ni_ttm and ni_ttm > 0) else None,
            "pbr": _div(cap, equity) if (cap and equity and equity > 0) else None,
            "roe": roe,
            "operating_margin": op_margin,
            "net_margin": net_margin,
            # 절대값(달러, 단일분기 — TTM 아님). KR metrics_at과 동일하게 마진 계산에만
            # 쓰던 변수(op_q/rev_q/ni_q)를 출력에도 그대로 노출한다(새 계산 없음).
            "operating_profit": op_q,
            "revenue": rev_q,
            "net_income": ni_q,
            "return_12m": return_12m,
        }
        # 재무제표 보고통화가 확인됐고(financial_currency NOT NULL) USD가 아니면(예: SKM=KRW)
        # 재무 파생 필드 전부를 무효화한다 — 시총(항상 달러)을 원화 순이익으로 나누는 식의
        # 통화 불일치 계산오류(PER=0.035류)를 막는다. NULL(미수집, 배치 스크립트 실행 전)은
        # 기존 동작 그대로 둔다(scripts/backfill_us_financial_currency.py 참고). 가격/시총
        # 자체는 정상 달러 데이터라 건드리지 않는다 — 종목을 숨기는 게 아니라 "재무비율
        # 계산에만 안 쓴다"는 원칙(src/data_quality.py의 종목단위 제외보다 좁은 필드단위 적용).
        financial_currency = c["financial_currency"]
        if financial_currency is not None and financial_currency != "USD":
            for key in (
                "per", "pbr", "roe", "operating_margin", "net_margin",
                "operating_profit", "revenue", "net_income",
            ):
                row[key] = None
        out.append(row)
    return out


def build_callbacks_us(conn):
    """엔진용 (metrics_fn, price_fn) 생성. 시점 metrics는 한 번 계산 후 캐시(KR build_callbacks와 동일 시그니처)."""
    cache: dict[str, list[dict]] = {}

    def metrics_fn(asof: str) -> list[dict]:
        if asof not in cache:
            cache[asof] = metrics_at_us(conn, asof)
        return cache[asof]

    def price_fn(asof: str, code: str):
        c, _ = _price_at_us(conn, code, asof)
        return c

    return metrics_fn, price_fn


def _fetch_gspc_history(start: str, end: str) -> dict[str, float]:
    """yfinance ^GSPC(S&P500 실제지수) 종가 히스토리를 {날짜: 종가} dict로 반환한다.

    build_sp500_benchmark_fn의 기본 fetch_fn(테스트에선 주입으로 대체). us_prices의
    지연 import 관례를 따른다.
    """
    from datetime import datetime, timedelta

    import pandas as pd
    import yfinance as yf

    end_excl = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.Ticker("^GSPC").history(start=start, end=end_excl, auto_adjust=True)
    return {
        ts.strftime("%Y-%m-%d"): float(row["Close"])
        for ts, row in df.iterrows()
        if not pd.isna(row.get("Close"))
    }


def build_sp500_benchmark_fn(dates: list[str], fetch_fn=None):
    """S&P500 실제지수(^GSPC) 레벨 시계열(레벨) 벤치마크 함수를 생성한다.

    각 리밸런싱 날짜의 종가(휴장이면 그 이전 최근 거래일 종가 = on-or-before)로 레벨을 만들고,
    첫 시점=1.0 기준으로 정규화한다. fetch_fn(start,end)->{날짜:종가}는 테스트 주입용
    (기본=_fetch_gspc_history). 해당 날짜의 종가를 못 구하면 그 시점 레벨은 None.
    """
    fetch_fn = fetch_fn or _fetch_gspc_history
    history = fetch_fn(dates[0], dates[-1])
    sorted_days = sorted(history.keys())

    def _close_on_or_before(d: str):
        candidates = [k for k in sorted_days if k <= d]
        return history[candidates[-1]] if candidates else None

    levels: dict[str, float | None] = {}
    base = None
    for d in dates:
        px = _close_on_or_before(d)
        if px is None:
            levels[d] = None
            continue
        if base is None:
            base = px
        levels[d] = px / base
    return lambda d: levels.get(d)
