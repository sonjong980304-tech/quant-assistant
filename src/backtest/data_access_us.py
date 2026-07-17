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


def _ttm_us(conn, code: str, as_of_date: str, statement_type: str, item_key: str):
    """as_of_date 이하 최근 4개 quarterly 항목값 합(TTM). 4개 미만이면 추정 없이 None.

    KR _sum_ttm과 동일 원칙(4분기 중 하나라도 없으면 억지 추정 안 함). _ttm_net_income을
    임의 (statement_type,item_key)로 일반화한 것 — psr(매출)/pcr(영업현금흐름)/gp_a(매출총이익)/
    ev_ebitda(EBITDA)/interest_coverage(영업이익·이자비용) 등 TTM 기반 지표가 공유한다.
    """
    rows = conn.execute(
        "SELECT item_value FROM us_financials WHERE stock_code=? AND period_type='quarterly' "
        "AND statement_type=? AND item_key=? AND as_of_date<=? "
        "ORDER BY as_of_date DESC LIMIT 4",
        (code, statement_type, item_key, as_of_date),
    ).fetchall()
    vals = [r["item_value"] for r in rows if r["item_value"] is not None]
    if len(vals) < 4:
        return None
    return sum(vals)


def _ttm_net_income(conn, code: str, as_of_date: str):
    """최근 4개 quarterly Net Income 합(TTM). 4개 미만이면 추정하지 않고 None(KR _sum_ttm과 동일 원칙)."""
    return _ttm_us(conn, code, as_of_date, "income_stmt", "Net Income")


def _yoy_us(conn, code: str, as_of_date: str, statement_type: str, item_key: str):
    """전년동기 대비 성장률(%). 단일분기 기준(KR ingest.metrics._yoy와 동일 정의).

    현재 분기(as_of_date) 단일값 대비, 정확히 4개 quarterly 전(OFFSET 4) 단일값의 증가율.
    yfinance quarterly는 연속 분기라 'OFFSET 4 = 1년 전 동일분기'다(KR의 shift_quarter(-4)에
    대응). 전년동기가 없거나(신규상장·데이터한계) 0이면 억지 계산 없이 None을 돌려준다.
    us_financials 종목당 quarterly가 5개 안팎이라(data/market.db 확인) 최신 분기에서만
    전년동기 비교가 가능한 경우가 흔하며, 그 경우 과거 시점 성장률은 자연히 None이 된다.
    """
    cur_row = conn.execute(
        "SELECT item_value FROM us_financials WHERE stock_code=? AND period_type='quarterly' "
        "AND statement_type=? AND item_key=? AND as_of_date=?",
        (code, statement_type, item_key, as_of_date),
    ).fetchone()
    prev_row = conn.execute(
        "SELECT item_value FROM us_financials WHERE stock_code=? AND period_type='quarterly' "
        "AND statement_type=? AND item_key=? AND as_of_date<=? "
        "ORDER BY as_of_date DESC LIMIT 1 OFFSET 4",
        (code, statement_type, item_key, as_of_date),
    ).fetchone()
    if cur_row is None or prev_row is None:
        return None
    cur, prev = cur_row["item_value"], prev_row["item_value"]
    if cur is None or prev is None or prev == 0:
        return None
    return (cur - prev) / abs(prev) * 100.0


# KR data_access.METRIC_FIELD_DESCRIPTIONS 와 동일한 목적의 US 전용 단일 정의처(canonical
# source). domain_us.py의 _US_SCREEN_FIELDS가 이 dict의 key만 파생해서 쓴다 — 새 지표
# 추가 시 여기 한 곳(과 아래 metrics_at_us의 출력 dict)만 고치면 된다.
#
# 초기엔 "yfinance 소스 한계로 KR보다 좁다"고 12개만 노출했으나, us_financials(EAV) 원본을
# data/market.db에서 직접 조회한 결과 Total Assets/Total Debt/Total Liabilities/Current
# Assets·Liabilities/Operating Cash Flow/EBITDA/Interest Expense/Gross Profit 등이 이미
# 수집돼 있어(대부분 종목이 quarterly 5개 보유 → TTM/YoY 계산 가능) 공식만 추가하면 되는
# 상황이었다. 백테스트 UI 체크박스(db.py METRIC_DEFS)는 국가 구분 없이 20개를 노출하는데
# US가 그중 일부만 계산해 selection._validate_criteria_keys가 크래시시키던 문제를 없애기 위해
# 파생 지표군(밸류/수익성/안정성/성장)을 KR과 동일 정의로 확장했다.
# (momentum은 web/app.py에서 return_12m으로 별칭 치환되고, dividend_yield는 별도 처리 중.)
METRIC_FIELD_DESCRIPTIONS_US: dict[str, str] = {
    "per": "주가수익비율(PER)",
    "pbr": "주가순자산비율(PBR)",
    "psr": "주가매출비율(PSR, 시총 ÷ 매출TTM)",
    "pcr": "주가현금흐름비율(PCR, 시총 ÷ 영업활동현금흐름TTM)",
    "ev_ebitda": "EV/EBITDA(기업가치[시총+총부채-현금] ÷ EBITDA TTM, 낮을수록 우수)",
    "peg": "PEG(PER ÷ 순이익성장률, 낮을수록 우수)",
    "roe": "자기자본이익률(ROE, %)",
    "roa": "총자산이익률(ROA, %, 순이익TTM ÷ 총자산)",
    "operating_margin": "영업이익률(%, 해당분기)",
    "net_margin": "순이익률(%, 해당분기)",
    "gross_margin": "매출총이익률(%, 해당분기, 매출총이익 ÷ 매출액, 높을수록 우수)",
    "cogs_ratio": "매출원가율(%, 해당분기, 매출원가 ÷ 매출액, 낮을수록 우수)",
    "gp_a": "매출총이익/총자산(GPA, %, 매출총이익TTM ÷ 총자산, 높을수록 우수)",
    "debt_ratio": "부채비율(%, 부채총계 ÷ 자기자본)",
    "current_ratio": "유동비율(%, 유동자산 ÷ 유동부채, 높을수록 우수)",
    "interest_coverage": "이자보상배율(영업이익TTM ÷ 이자비용TTM, 높을수록 우수)",
    "revenue_growth": "매출 성장률(YoY, %, 전년동기 대비)",
    "op_growth": "영업이익 성장률(YoY, %, 전년동기 대비)",
    "ni_growth": "순이익 성장률(YoY, %, 전년동기 대비)",
    "return_12m": "최근 12개월 주가 수익률(모멘텀, %)",
    "operating_profit": "영업이익(달러 절대값, 해당분기, TTM 아님)",
    "revenue": "매출액(달러 절대값, 해당분기, TTM 아님)",
    "net_income": "순이익(달러 절대값, 해당분기, TTM 아님)",
    "market_cap": "시가총액(달러 절대값, 종가×상장주식수, 해당 기준시점)",
}


def metrics_at_us(conn, asof: str) -> list[dict]:
    """시점 asof의 유효 US 지표 행 목록 (look-ahead 방지 + KR과 동일 이상치가드).

    밸류: PER=시총/순이익TTM, PBR=시총/자기자본, PSR=시총/매출TTM, PCR=시총/영업현금흐름TTM,
      EV/EBITDA=(시총+총부채-현금)/EBITDA TTM, PEG=PER/순이익성장률.
    수익성: ROE=순이익TTM/자기자본, ROA=순이익TTM/총자산, 영업/순이익률(단일분기),
      매출총이익률/매출원가율(단일분기), GP/A=매출총이익TTM/총자산.
    안정성: 부채비율=부채/자본, 유동비율=유동자산/유동부채, 이자보상배율=영업이익TTM/이자비용TTM.
    성장(YoY 단일분기): 매출/영업이익/순이익 성장률. 그 외 return_12m(가격 모멘텀)·절대값.
    us_financials(yfinance EAV)에 이미 수집된 item_key만 쓴다(원본 존재는 data/market.db 직접
    조회로 확인). 원본이 없는 종목은 해당 지표가 None(억지 추정 없음).
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
        # 매출총이익률·매출원가율 입력(단일분기 원본 항목). yfinance item_key(prompts.py 매핑):
        # 매출총이익=Gross Profit, 매출원가=Cost Of Revenue.
        gp_q = _us_item(conn, code, q, "income_stmt", "Gross Profit")
        cogs_q = _us_item(conn, code, q, "income_stmt", "Cost Of Revenue")
        equity = _us_item(conn, code, q, "balance_sheet", "Stockholders Equity")
        ni_ttm = _ttm_net_income(conn, code, q)

        # ── 확장 파생 지표 입력(us_financials에 이미 수집된 원본; DB 직접조회로 존재 확인) ──
        # 재무상태(BS)는 최신분기 스냅샷, 손익/현금흐름은 TTM(4분기 합) — KR data_access.py 규약과 동일.
        assets = _us_item(conn, code, q, "balance_sheet", "Total Assets")                          # roa/gp_a 분모
        total_liab = _us_item(conn, code, q, "balance_sheet", "Total Liabilities Net Minority Interest")  # debt_ratio 분자
        cur_assets = _us_item(conn, code, q, "balance_sheet", "Current Assets")                    # current_ratio 분자
        cur_liab = _us_item(conn, code, q, "balance_sheet", "Current Liabilities")                 # current_ratio 분모
        total_debt = _us_item(conn, code, q, "balance_sheet", "Total Debt")                        # EV 구성(순부채)
        cash = _us_item(conn, code, q, "balance_sheet", "Cash And Cash Equivalents")               # EV 구성(현금차감)
        rev_ttm = _ttm_us(conn, code, q, "income_stmt", "Total Revenue")                           # psr 분모
        ocf_ttm = _ttm_us(conn, code, q, "cashflow", "Operating Cash Flow")                        # pcr 분모
        gp_ttm = _ttm_us(conn, code, q, "income_stmt", "Gross Profit")                             # gp_a 분자
        ebitda_ttm = _ttm_us(conn, code, q, "income_stmt", "EBITDA")                               # ev_ebitda 분모
        op_ttm = _ttm_us(conn, code, q, "income_stmt", "Operating Income")                         # interest_coverage 분자
        int_ttm = _ttm_us(conn, code, q, "income_stmt", "Interest Expense")                        # interest_coverage 분모

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

        # 매출총이익률/매출원가율(KR data_access.metrics_at과 동일 정의/근사 규약). 원본 계정을
        # 우선 쓰고 없으면 항등식(매출=매출원가+매출총이익)으로 유도, {metric}_estimated로 표시.
        gp_for_margin = gp_q if gp_q is not None else (
            (rev_q - cogs_q) if (rev_q is not None and cogs_q is not None) else None
        )
        gross_margin = _div(gp_for_margin, rev_q, pct=True) if (
            rev_q and rev_q > 0 and gp_for_margin is not None
        ) else None
        gross_margin_estimated = (gp_q is None) if gross_margin is not None else None
        cogs_for_ratio = cogs_q if cogs_q is not None else (
            (rev_q - gp_q) if (rev_q is not None and gp_q is not None) else None
        )
        cogs_ratio = _div(cogs_for_ratio, rev_q, pct=True) if (
            rev_q and rev_q > 0 and cogs_for_ratio is not None
        ) else None
        cogs_ratio_estimated = (cogs_q is None) if cogs_ratio is not None else None

        # 12개월 가격 수익률(모멘텀, KR metrics_at과 동일 정의/규약). _price_at_us는 date<=기준일만
        # 보므로 asof 이후 종가를 참조하지 않는다(look-ahead 방지). 1년 전 종가 없으면 None.
        prev_close, _ = _price_at_us(conn, code, _one_year_before(asof))
        return_12m = _div(close - prev_close, prev_close, pct=True) if (prev_close and prev_close > 0) else None

        # ── 확장 파생 지표 계산(KR data_access.metrics_at/ingest.metrics와 동일 정의) ──
        # PER은 peg에도 쓰이므로 변수로 뽑는다(분자=순이익TTM, 위 이상치가드 반영된 cap/ni_ttm 사용).
        per = _div(cap, ni_ttm) if (cap and ni_ttm and ni_ttm > 0) else None
        # 밸류: psr=시총/매출TTM, pcr=시총/영업현금흐름TTM(둘 다 분모>0 요구, 음수 현금흐름은 무의미→제외).
        psr = _div(cap, rev_ttm) if (cap and rev_ttm and rev_ttm > 0) else None
        pcr = _div(cap, ocf_ttm) if (cap and ocf_ttm and ocf_ttm > 0) else None
        # EV = 시총 + 총부채 - 현금(교과서 정의). yfinance에 Total Debt/Cash 원본이 있어 KR의
        # 총부채 근사(data_access.earnings_yield) 대신 표준 정의를 쓴다. 셋 중 하나라도 없으면 None.
        ev = (cap + total_debt - cash) if (
            cap is not None and total_debt is not None and cash is not None
        ) else None
        ev_ebitda = _div(ev, ebitda_ttm) if (ev is not None and ebitda_ttm and ebitda_ttm > 0) else None
        # 성장(YoY 단일분기) — peg 분모로도 쓰이므로 먼저 계산.
        revenue_growth = _yoy_us(conn, code, q, "income_stmt", "Total Revenue")
        op_growth = _yoy_us(conn, code, q, "income_stmt", "Operating Income")
        ni_growth = _yoy_us(conn, code, q, "income_stmt", "Net Income")
        # PEG = PER / 순이익성장률(양수일 때만, KR ingest.metrics와 동일 규약).
        peg = (per / ni_growth) if (per is not None and ni_growth and ni_growth > 0) else None
        # 수익성: roa=순이익TTM/총자산, gp_a=매출총이익TTM/총자산(부호 보존, 분모>0만 요구).
        roa = _div(ni_ttm, assets, pct=True) if (ni_ttm and assets and assets > 0) else None
        gp_a = _div(gp_ttm, assets, pct=True) if (gp_ttm is not None and assets and assets > 0) else None
        # 안정성: debt_ratio=부채/자본, current_ratio=유동자산/유동부채, interest_coverage=영업이익TTM/이자비용TTM.
        debt_ratio = _div(total_liab, equity, pct=True) if (total_liab is not None and equity and equity > 0) else None
        current_ratio = _div(cur_assets, cur_liab, pct=True) if (cur_assets is not None and cur_liab and cur_liab > 0) else None
        interest_coverage = _div(op_ttm, int_ttm) if (op_ttm is not None and int_ttm and int_ttm > 0) else None

        row = {
            "stock_code": code, "name": c["name"], "sector": c["sector"], "market": c["exchange"],
            "security_type": c["security_type"],
            "quarter": q, "close": close, "market_cap": cap,
            "per": per,
            "pbr": _div(cap, equity) if (cap and equity and equity > 0) else None,
            "psr": psr,
            "pcr": pcr,
            "ev_ebitda": ev_ebitda,
            "peg": peg,
            "roe": roe,
            "roa": roa,
            "operating_margin": op_margin,
            "net_margin": net_margin,
            "gross_margin": gross_margin,
            "gross_margin_estimated": gross_margin_estimated,
            "cogs_ratio": cogs_ratio,
            "cogs_ratio_estimated": cogs_ratio_estimated,
            "gp_a": gp_a,
            "debt_ratio": debt_ratio,
            "current_ratio": current_ratio,
            "interest_coverage": interest_coverage,
            "revenue_growth": revenue_growth,
            "op_growth": op_growth,
            "ni_growth": ni_growth,
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
                "gross_margin", "gross_margin_estimated", "cogs_ratio", "cogs_ratio_estimated",
                "operating_profit", "revenue", "net_income",
                # 확장 파생 지표도 동일하게 무효화한다. 성장률·부채비율 등 일부는 통화가 약분되는
                # 순수 비율이지만, 비USD 종목은 재무 데이터셋 전체를 신뢰하지 않는다는 기존 설계
                # (roe/operating_margin 같은 순수 비율도 이미 무효화)를 그대로 따른다(보수적·일관).
                "psr", "pcr", "ev_ebitda", "peg", "roa", "gp_a",
                "debt_ratio", "current_ratio", "interest_coverage",
                "revenue_growth", "op_growth", "ni_growth",
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
