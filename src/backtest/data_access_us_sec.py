"""SEC XBRL 소스 백테스트 지표 어댑터 (data_access_us.py 의 SEC 대응물).

기존 data_access_us.py(yfinance us_financials EAV)의 검증된 계산 로직·이상치 가드·지표
정의를 그대로 미러링하되, 데이터 소스만 새 테이블 us_financials_sec(SEC 원시 XBRL 팩트)로
바꾼다(스펙: "기존 검증된 로직 최대한 재사용, 데이터 소스만 교체"). 핵심 차이:

- 항목 식별: yfinance item_key(예: 'Total Revenue') 대신 XBRL 태그(예: 'Revenues')를 쓴다.
  태그는 회사·연도별로 변형이 있어(예: Revenues ↔ RevenueFromContractWithCustomerExcluding
  AssessedTax) 우선순위 목록으로 첫 번째로 데이터가 있는 태그를 고른다.
- 분기 식별: us_financials 는 quarterly/annual period_type 로 구분했지만, XBRL 은 duration
  팩트의 기간 길이(period_end - period_start ≈ 3개월)로 단일분기를 식별하고, instant 팩트
  (period_start 없음/빈문자열)로 재무상태 스냅샷을 식별한다.
- look-ahead 방지(AC7): us_financials 의 기말+45/90일 근사 대신 SEC 실제 filed(제출일)를
  써서 asof 이후 제출된 팩트를 제외한다(더 정확).
- 통화: 태그별 unit='USD' 팩트만 조회해 통화 불일치를 소스에서 차단한다(yfinance 경로가
  financial_currency 캐시로 사후 무효화하던 것을 원천 차단 — 더 단순·정확).
"""
from __future__ import annotations

from ..ingest.metrics import _div
from .data_access import _one_year_before  # return_12m 기준일(KR과 동일 규약)
# 지표 설명 단일 정의처는 yfinance 경로와 동일한 지표군이라 그대로 재사용한다(중복 정의 금지).
# _is_alive_us(생존편향 사전필터)도 yfinance 경로와 동일한 us_delisting 판정을 공유한다.
from .data_access_us import METRIC_FIELD_DESCRIPTIONS_US, _is_alive_us  # noqa: F401  (재노출용)

# ── XBRL 태그 우선순위(개념 → 후보 태그). 첫 번째로 데이터가 있는 태그를 쓴다. ──
# 손익/현금흐름(duration)
REVENUE_TAGS = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet")
OPERATING_INCOME_TAGS = ("OperatingIncomeLoss",)
NET_INCOME_TAGS = ("NetIncomeLoss", "ProfitLoss")
GROSS_PROFIT_TAGS = ("GrossProfit",)
COST_OF_REVENUE_TAGS = ("CostOfRevenue", "CostOfGoodsAndServicesSold")
INTEREST_EXPENSE_TAGS = ("InterestExpense", "InterestExpenseNonoperating")
DEPRECIATION_TAGS = ("DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet", "DepreciationAndAmortization")
OPERATING_CASHFLOW_TAGS = ("NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")
# 재무상태(instant)
EQUITY_TAGS = ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
ASSETS_TAGS = ("Assets",)
LIABILITIES_TAGS = ("Liabilities",)
CURRENT_ASSETS_TAGS = ("AssetsCurrent",)
CURRENT_LIABILITIES_TAGS = ("LiabilitiesCurrent",)
CASH_TAGS = ("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents")
LONG_TERM_DEBT_TAGS = ("LongTermDebtNoncurrent", "LongTermDebt")
SHORT_TERM_DEBT_TAGS = ("LongTermDebtCurrent", "DebtCurrent")

# 단일분기 duration 판정 범위(일). 표준 회계분기 ≈ 91일 → 80~100일을 단일분기로 본다.
# 이 범위 밖(YTD 반기/9개월)은 제외해 이중계산을 막는다.
_QUARTER_MIN_DAYS = 80
_QUARTER_MAX_DAYS = 100
# 연간(FY, 12개월) duration 판정 범위(일). 52/53주 회계연도 편차를 감안해 350~380일.
# Q4 역산(연간 − Q1~Q3 합)에 쓴다.
_ANNUAL_MIN_DAYS = 350
_ANNUAL_MAX_DAYS = 380
# 발행주식수(instant) 태그 우선순위 — 시점별 시가총액(종가×주식수) 계산용.
SHARES_OUTSTANDING_TAGS = (
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
)
# YoY 전년동기 짝짓기 허용 오차(일). 회계연도 52/53주 편차로 1년 전 기말이 며칠씩 밀리므로
# ±45일까지 '같은 분기'로 인정한다(인접 분기는 ~91일 떨어져 있어 오인 없음). critic MAJOR-1.
_YOY_TOLERANCE_DAYS = 45
# 발행주식수 stale 판정 임계(일). 분기 발행주식수가 asof 대비 이보다 오래됐으면(예: 데이터가
# 오래전 끊긴 종목) 그 낡은 값 대신 us_company.market_cap 으로 폴백한다. 연간만 보고하는
# 종목까지 감안해 ~400일(1년+여유). critic MAJOR-2 후속.
_SHARES_STALE_DAYS = 400


def _duration_rows(conn, code: str, tag: str, min_days: int, max_days: int,
                   period_end_max: str | None = None, filed_max: str | None = None):
    """단일 태그의 duration 팩트(min~max일)를 period_end 별 최신 filed 하나로 정리해 DESC 반환.

    unit='USD' 만, filed_max/period_end_max 로 look-ahead·범위 제한. 각 원소는 sqlite Row
    (period_start, period_end, value, filed).
    """
    sql = (
        "SELECT period_start, period_end, value, filed FROM us_financials_sec "
        "WHERE stock_code=? AND tag=? AND unit='USD' AND period_start<>'' "
        "AND julianday(period_end)-julianday(period_start) BETWEEN ? AND ? "
    )
    params: list = [code, tag, min_days, max_days]
    if filed_max is not None:
        sql += "AND filed IS NOT NULL AND filed<=? "
        params.append(filed_max)
    if period_end_max is not None:
        sql += "AND period_end<=? "
        params.append(period_end_max)
    sql += "ORDER BY period_end DESC, filed DESC"
    seen: set[str] = set()
    out = []
    for r in conn.execute(sql, params).fetchall():
        pe = r["period_end"]
        if pe in seen or r["value"] is None:
            continue
        seen.add(pe)
        out.append(r)
    return out


def _quarter_series(conn, code: str, tags, period_end_max: str | None = None, filed_max: str | None = None):
    """단일분기 값 시계열 [(period_end, value)] DESC. 실제 단독분기(Q1~Q3) + 역산 Q4.

    SEC 기업은 4분기를 단독 제출하지 않는다(10-K 연간에만 포함, Q1~Q3만 10-Q 단독). 그래서
    단독 3개월 팩트만 쓰면 연말분기가 통째로 빠지고 TTM 이 조용히 틀린다(작년 Q3를 끼워넣음).
    이를 막기 위해 연간(FY, 12개월) 팩트에서 Q4 = FY − (그 회계연도의 Q1+Q2+Q3) 로 역산해
    시계열에 합친다(표준 SEC-XBRL Q4 유도 방식). filed_max/period_end_max 로 look-ahead 제한.
    """
    for tag in tags:
        q_rows = _duration_rows(conn, code, tag, _QUARTER_MIN_DAYS, _QUARTER_MAX_DAYS,
                                period_end_max, filed_max)
        if not q_rows:
            continue
        q_map: dict[str, float] = {r["period_end"]: r["value"] for r in q_rows}
        # 연간 팩트로 Q4 역산(단독 Q4가 이미 있으면 건너뜀 — 드물게 단독 제출하는 회사 대비).
        for a in _duration_rows(conn, code, tag, _ANNUAL_MIN_DAYS, _ANNUAL_MAX_DAYS,
                                period_end_max, filed_max):
            ae, a_start, av = a["period_end"], a["period_start"], a["value"]
            if ae in q_map:
                continue
            inner = [
                r for r in q_rows
                if r["period_start"] >= a_start and r["period_end"] <= ae and r["period_end"] != ae
            ]
            if len(inner) == 3:  # Q1+Q2+Q3 가 그 연도 안에 정확히 3개 있을 때만 역산
                q_map[ae] = av - sum(r["value"] for r in inner)
        if q_map:
            return sorted(q_map.items(), key=lambda kv: kv[0], reverse=True)
    return []


def _bs_snapshot(conn, code: str, quarter: str, tags, filed_max: str | None = None):
    """재무상태(instant) 스냅샷: period_end==quarter 인 팩트값(태그 우선순위, unit='USD').

    us_financials 경로의 _us_item(as_of_date 정확일치)과 동일 원칙 — 재무상태는 그 분기말
    시점값을 그대로 쓴다(TTM 아님). 수정제출 중복은 최신 filed 하나만.
    """
    for tag in tags:
        sql = (
            "SELECT value FROM us_financials_sec "
            "WHERE stock_code=? AND tag=? AND unit='USD' AND period_start='' AND period_end=? "
        )
        params: list = [code, tag, quarter]
        if filed_max is not None:
            sql += "AND filed IS NOT NULL AND filed<=? "
            params.append(filed_max)
        sql += "ORDER BY filed DESC LIMIT 1"
        row = conn.execute(sql, params).fetchone()
        if row and row["value"] is not None:
            return row["value"]
    return None


def effective_quarter_at_us_sec(conn, code: str, asof: str) -> str | None:
    """asof 시점까지 제출(filed<=asof)된 최신 단일분기의 period_end (look-ahead 방지, AC7).

    순이익(NetIncomeLoss) 단일분기 팩트를 기준으로 유효분기를 정한다(핵심 지표). filed 는
    SEC 실제 제출일이라 us_financials 의 기말+45일 근사보다 정확하다.
    """
    series = _quarter_series(conn, code, NET_INCOME_TAGS, filed_max=asof)
    return series[0][0] if series else None


def _single_q(conn, code: str, quarter: str, tags, filed_max: str | None = None):
    """quarter(period_end) 단일분기 값(태그 우선순위, Q4는 역산). 없으면 None.

    filed_max 로 look-ahead 제한을 일관되게 전달한다(_bs_snapshot 과 동일 원칙, critic M1).
    """
    for tag in tags:
        series = _quarter_series(conn, code, (tag,), period_end_max=quarter, filed_max=filed_max)
        if series and series[0][0] == quarter:
            return series[0][1]
    return None


def _ttm_sec(conn, code: str, quarter: str, tags, filed_max: str | None = None):
    """quarter 이하 최근 4개 단일분기 합(TTM, Q4 역산 포함). 4개 미만이면 추정 없이 None.

    filed_max 로 look-ahead 제한을 전달해 나중 정정(10-K/A 등)이 과거 asof 조회에 새는 것을
    막는다(critic M1 — _bs_snapshot/effective_quarter 와 동일 기준).
    """
    series = _quarter_series(conn, code, tags, period_end_max=quarter, filed_max=filed_max)
    vals = [v for _, v in series[:4]]
    if len(vals) < 4:
        return None
    return sum(vals)


def _yoy_sec(conn, code: str, quarter: str, tags, filed_max: str | None = None):
    """전년동기 대비 성장률(%). '실제 1년 전 분기'와 짝지어 계산(위치 인덱스 맹신 금지).

    series[4](4번째 앞)를 '4분기 전'으로 가정하면, 중간 분기가 누락돼 시계열에 구멍이 생겼을
    때 엉뚱한 분기와 비교해 그럴듯하게 틀린 값을 낸다(critic MAJOR-1). 그래서 기준 분기의
    정확히 1년 전(±45일 허용) period_end 를 가진 항목을 날짜로 찾아 비교하고, 없으면 None.
    Q4 역산 포함 시계열을 쓰며 filed_max 로 look-ahead 를 제한한다(M1).
    """
    from datetime import date

    series = _quarter_series(conn, code, tags, period_end_max=quarter, filed_max=filed_max)
    if not series or series[0][0] != quarter:
        return None
    cur = series[0][1]
    target = date.fromisoformat(_one_year_before(quarter))
    prev = None
    for pe, val in series[1:]:
        if abs((date.fromisoformat(pe) - target).days) <= _YOY_TOLERANCE_DAYS:
            prev = val
            break
    if cur is None or prev is None or prev == 0:
        return None
    return (cur - prev) / abs(prev) * 100.0


def _shares_outstanding_at(conn, code: str, asof: str):
    """asof 시점 기준 유효 발행주식수(instant, unit='shares'). 없거나 stale 하면 None.

    시점별 시가총액(종가×주식수) 계산용(critic M2). dei/us-gaap 태그 우선순위, filed<=asof
    (look-ahead). 실측(critic MAJOR-2 후속): SEC companyfacts API 는 클래스별 차원을 제거하고
    연결 총계 한 팩트만 준다(Alphabet 총계 12.1B 정상, 같은 (end,accn) 복수 팩트 0건 확인) —
    즉 '여러 클래스 중 하나만 집어 과소계상'하는 문제는 발생하지 않는다. 다만 일부 종목
    (예: 버크셔)은 발행주식수 팩트가 오래돼(2011까지) 있고 단일클래스라, 그 낡은 값을 최신
    시총에 쓰면 오히려 틀린다. 그래서 팩트의 period_end 가 asof 대비 _SHARES_STALE_DAYS 를
    넘으면 stale 로 보고 None 을 돌려 metrics_at_us_sec 가 us_company.market_cap 으로 폴백하게
    한다(과소·오류보다 폴백이 낫다는 critic 가이드).
    """
    from datetime import date

    asof_d = date.fromisoformat(asof)
    for taxonomy, tag in SHARES_OUTSTANDING_TAGS:
        row = conn.execute(
            "SELECT value, period_end FROM us_financials_sec "
            "WHERE stock_code=? AND taxonomy=? AND tag=? AND unit='shares' AND period_start='' "
            "AND period_end<=? AND filed IS NOT NULL AND filed<=? "
            "ORDER BY period_end DESC, filed DESC LIMIT 1",
            (code, taxonomy, tag, asof, asof),
        ).fetchone()
        if row and row["value"]:
            if (asof_d - date.fromisoformat(row["period_end"])).days > _SHARES_STALE_DAYS:
                continue  # stale — 다음 태그 시도, 없으면 None(회사 스냅샷 폴백)
            return row["value"]
    return None


def _price_at_us_sec(conn, code: str, asof: str):
    """(종가, 시가총액). data_access_us._price_at_us 와 동일(us_prices/us_company 공유)."""
    prow = conn.execute(
        "SELECT close FROM us_prices WHERE stock_code=? AND date<=? ORDER BY date DESC LIMIT 1",
        (code, asof),
    ).fetchone()
    if not prow or prow["close"] is None:
        return (None, None)
    crow = conn.execute("SELECT market_cap FROM us_company WHERE stock_code=?", (code,)).fetchone()
    cap = crow["market_cap"] if crow else None
    return (prow["close"], cap)


def metrics_at_us_sec(conn, asof: str) -> list[dict]:
    """시점 asof 의 유효 US 지표 행 목록 — SEC XBRL 소스판(metrics_at_us 와 동일 지표군·가드).

    밸류/수익성/안정성/성장 지표를 data_access_us.metrics_at_us 와 동일한 공식·이상치가드로
    계산하되 XBRL 태그를 소스로 쓴다. filed 기준 look-ahead 방지(AC7). EBITDA 는 표준 XBRL
    태그가 없어 KR 정의(영업이익+감가상각, 둘 다 TTM)로 유도한다.
    """
    companies = conn.execute(
        "SELECT stock_code, name, sector, exchange, cik FROM us_company WHERE cik IS NOT NULL"
    ).fetchall()
    out = []
    for c in companies:
        code = c["stock_code"]
        # 생존편향 사전필터(yfinance 경로 metrics_at_us와 동일, us_delisting 구간 기반, AC6/AC12).
        if not _is_alive_us(conn, code, asof):
            continue
        q = effective_quarter_at_us_sec(conn, code, asof)
        if not q:
            continue
        close, fallback_cap = _price_at_us_sec(conn, code, asof)
        if not close:
            continue
        # 시점별 시가총액 = 그 시점 종가 × 그 시점 발행주식수(critic M2). 발행주식수 XBRL 이
        # 없을 때만 us_company.market_cap(현재 스냅샷)으로 폴백한다. 이렇게 해야 2010년 asof
        # 조회가 2026년 시총을 쓰는 오류(PER/PBR 등 밸류지표 전부 무의미)를 막는다.
        shares = _shares_outstanding_at(conn, code, asof)
        cap = (close * shares) if shares else fallback_cap

        # 단일분기 손익(filed_max=asof 로 look-ahead 일관 제한, critic M1)
        rev_q = _single_q(conn, code, q, REVENUE_TAGS, filed_max=asof)
        op_q = _single_q(conn, code, q, OPERATING_INCOME_TAGS, filed_max=asof)
        ni_q = _single_q(conn, code, q, NET_INCOME_TAGS, filed_max=asof)
        gp_q = _single_q(conn, code, q, GROSS_PROFIT_TAGS, filed_max=asof)
        cogs_q = _single_q(conn, code, q, COST_OF_REVENUE_TAGS, filed_max=asof)
        # 재무상태 스냅샷(instant)
        equity = _bs_snapshot(conn, code, q, EQUITY_TAGS, filed_max=asof)
        assets = _bs_snapshot(conn, code, q, ASSETS_TAGS, filed_max=asof)
        total_liab = _bs_snapshot(conn, code, q, LIABILITIES_TAGS, filed_max=asof)
        cur_assets = _bs_snapshot(conn, code, q, CURRENT_ASSETS_TAGS, filed_max=asof)
        cur_liab = _bs_snapshot(conn, code, q, CURRENT_LIABILITIES_TAGS, filed_max=asof)
        cash = _bs_snapshot(conn, code, q, CASH_TAGS, filed_max=asof)
        ltd = _bs_snapshot(conn, code, q, LONG_TERM_DEBT_TAGS, filed_max=asof)
        std = _bs_snapshot(conn, code, q, SHORT_TERM_DEBT_TAGS, filed_max=asof)
        total_debt = (ltd or 0.0) + (std or 0.0) if (ltd is not None or std is not None) else None
        # TTM (filed_max=asof 로 look-ahead 일관 제한, critic M1)
        ni_ttm = _ttm_sec(conn, code, q, NET_INCOME_TAGS, filed_max=asof)
        rev_ttm = _ttm_sec(conn, code, q, REVENUE_TAGS, filed_max=asof)
        ocf_ttm = _ttm_sec(conn, code, q, OPERATING_CASHFLOW_TAGS, filed_max=asof)
        gp_ttm = _ttm_sec(conn, code, q, GROSS_PROFIT_TAGS, filed_max=asof)
        op_ttm = _ttm_sec(conn, code, q, OPERATING_INCOME_TAGS, filed_max=asof)
        int_ttm = _ttm_sec(conn, code, q, INTEREST_EXPENSE_TAGS, filed_max=asof)
        dep_ttm = _ttm_sec(conn, code, q, DEPRECIATION_TAGS, filed_max=asof)
        # EBITDA = 영업이익 + 감가상각(둘 다 TTM, KR ingest.metrics 정의). 둘 중 하나라도 없으면 None.
        ebitda_ttm = (op_ttm + dep_ttm) if (op_ttm is not None and dep_ttm is not None) else None

        # ── 이상치가드(KR/US metrics_at 과 동일 임계값) ──
        if ni_ttm is not None and equity and equity > 0 and abs(ni_ttm) > equity * 20:
            ni_ttm = None
        if cap is not None and equity and equity > 0 and cap > equity * 100:
            cap = None

        roe = _div(ni_ttm, equity, pct=True) if (ni_ttm and equity and equity > 0 and ni_ttm < equity) else None
        op_margin = _div(op_q, rev_q, pct=True) if (rev_q and rev_q > 0 and op_q is not None and op_q < rev_q) else None
        net_margin = _div(ni_q, rev_q, pct=True) if (rev_q and rev_q > 0 and ni_q is not None) else None

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

        prev_close, _ = _price_at_us_sec(conn, code, _one_year_before(asof))
        return_12m = _div(close - prev_close, prev_close, pct=True) if (prev_close and prev_close > 0) else None

        per = _div(cap, ni_ttm) if (cap and ni_ttm and ni_ttm > 0) else None
        psr = _div(cap, rev_ttm) if (cap and rev_ttm and rev_ttm > 0) else None
        pcr = _div(cap, ocf_ttm) if (cap and ocf_ttm and ocf_ttm > 0) else None
        ev = (cap + total_debt - cash) if (
            cap is not None and total_debt is not None and cash is not None
        ) else None
        ev_ebitda = _div(ev, ebitda_ttm) if (ev is not None and ebitda_ttm and ebitda_ttm > 0) else None
        revenue_growth = _yoy_sec(conn, code, q, REVENUE_TAGS, filed_max=asof)
        op_growth = _yoy_sec(conn, code, q, OPERATING_INCOME_TAGS, filed_max=asof)
        ni_growth = _yoy_sec(conn, code, q, NET_INCOME_TAGS, filed_max=asof)
        peg = (per / ni_growth) if (per is not None and ni_growth and ni_growth > 0) else None
        roa = _div(ni_ttm, assets, pct=True) if (ni_ttm and assets and assets > 0) else None
        gp_a = _div(gp_ttm, assets, pct=True) if (gp_ttm is not None and assets and assets > 0) else None
        debt_ratio = _div(total_liab, equity, pct=True) if (total_liab is not None and equity and equity > 0) else None
        current_ratio = _div(cur_assets, cur_liab, pct=True) if (cur_assets is not None and cur_liab and cur_liab > 0) else None
        interest_coverage = _div(op_ttm, int_ttm) if (op_ttm is not None and int_ttm and int_ttm > 0) else None

        out.append({
            "stock_code": code, "name": c["name"], "sector": c["sector"], "market": c["exchange"],
            "quarter": q, "close": close, "market_cap": cap,
            "per": per,
            "pbr": _div(cap, equity) if (cap and equity and equity > 0) else None,
            "psr": psr, "pcr": pcr, "ev_ebitda": ev_ebitda, "peg": peg,
            "roe": roe, "roa": roa,
            "operating_margin": op_margin, "net_margin": net_margin,
            "gross_margin": gross_margin, "gross_margin_estimated": gross_margin_estimated,
            "cogs_ratio": cogs_ratio, "cogs_ratio_estimated": cogs_ratio_estimated,
            "gp_a": gp_a,
            "debt_ratio": debt_ratio, "current_ratio": current_ratio, "interest_coverage": interest_coverage,
            "revenue_growth": revenue_growth, "op_growth": op_growth, "ni_growth": ni_growth,
            "operating_profit": op_q, "revenue": rev_q, "net_income": ni_q,
            "return_12m": return_12m,
        })
    return out


def build_callbacks_us_sec(conn):
    """엔진용 (metrics_fn, price_fn) 생성 + 시점 캐시(data_access_us.build_callbacks_us 와 동일 시그니처)."""
    cache: dict[str, list[dict]] = {}

    def metrics_fn(asof: str) -> list[dict]:
        if asof not in cache:
            cache[asof] = metrics_at_us_sec(conn, asof)
        return cache[asof]

    def price_fn(asof: str, code: str):
        c, _ = _price_at_us_sec(conn, code, asof)
        return c

    return metrics_fn, price_fn


def ttm_coverage(conn) -> dict:
    """추적 종목 중 TTM(직전4분기합, Q4 역산 포함)이 실제로 계산 가능한 종목 비율.

    us_financials_sec.quarterly_coverage(단순 quarterly 팩트 존재율)는 Q1~Q3만 있어도
    '커버됨'으로 잡혀 C1(Q4 미존재로 TTM이 틀리는) 문제를 가려버린다. 이 리포트는 순이익
    TTM 이 4분기(Q4 역산 성공 포함) 확보되는 종목만 세어 '진짜 백테스트 가능' 커버리지를
    보고한다(critic 부수지적). 백필 후 이 값과 quarterly_coverage 를 함께 보고한다.
    """
    codes = [
        r["stock_code"]
        for r in conn.execute("SELECT stock_code FROM us_company WHERE cik IS NOT NULL").fetchall()
    ]
    ok = 0
    for code in codes:
        series = _quarter_series(conn, code, NET_INCOME_TAGS)
        if len(series) >= 4:
            ok += 1
    total = len(codes)
    return {
        "total_tracked": total,
        "ttm_computable": ok,
        "ttm_coverage_rate": (ok / total) if total else 0.0,
    }
