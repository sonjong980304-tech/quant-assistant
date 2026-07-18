"""백테스트용 DB 어댑터.

엔진(engine.run_backtest)에 넘길 콜백을 DB에서 구현한다.
- look-ahead 방지: 시점 t에서 disclosed_date <= t 인 최신 분기 재무만 사용
- 생존편향 제거: delisting.delisting_date > t 인 종목만 (상폐 전까지 포함)
- 시점 t 지표는 그 시점 유효 재무 + t 이하 최근 종가로 즉석 계산
"""
from __future__ import annotations

import calendar

from ..data_quality import get_price_quality_excluded_codes
from ..ingest.metrics import (
    _div, _fin, _sum_ttm, _yoy,
    controlling_equity, avg_controlling_equity,
    NI_TO_EQUITY_MAX_RATIO, CAP_TO_EQUITY_MAX_RATIO,
)


def _month_end(y: int, m: int) -> str:
    return f"{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"


def _one_year_before(asof: str) -> str:
    """asof(YYYY-MM-DD)의 정확히 1년 전 날짜. 2월 29일이면 2월 28일로 보정한다.

    return_12m(12개월 가격 수익률)의 기준시점을 잡는 데 쓴다 — 이 날짜 '이하'의 가장 가까운
    거래일 종가를 조회하므로(_price_at의 date<=asof 규약) 미래참조 없이 look-ahead가 방지된다.
    """
    import datetime

    d = datetime.date.fromisoformat(asof)
    try:
        return d.replace(year=d.year - 1).isoformat()
    except ValueError:  # 2/29 → 전년 2/28
        return d.replace(year=d.year - 1, day=28).isoformat()


def _months_before(asof: str, months: int) -> str:
    """asof(YYYY-MM-DD)로부터 정확히 months개월 전 날짜(YYYY-MM-DD).

    _one_year_before(고정 12개월)의 임의 개월수 일반화다. 연/월 캐리를 처리하고,
    말일 보정(calendar.monthrange)으로 존재하지 않는 날짜(예: 3/31의 1개월 전 2/31)를
    그 달 말일(평년 2/28·윤년 2/29)로 당긴다 — _one_year_before가 2/29→전년 2/28로
    보정하는 것과 같은 이유. 실제로 months=12를 넣으면 _one_year_before와 항상 동일한
    결과가 나온다(tests/test_backtest_data_access.py가 회귀로 고정).
    """
    import datetime

    d = datetime.date.fromisoformat(asof)
    total = d.year * 12 + (d.month - 1) - months
    y, m = divmod(total, 12)
    m += 1  # divmod의 0-based 월 인덱스를 1~12로 되돌린다
    day = min(d.day, calendar.monthrange(y, m)[1])  # 말일 보정
    return datetime.date(y, m, day).isoformat()


def rebalance_dates(start_year: int, end_year: int, freq: str = "quarterly") -> list[str]:
    step = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}[freq]
    out, y, m = [], start_year, 1
    while y < end_year or (y == end_year and m <= 12):
        out.append(_month_end(y, m))
        m += step
        while m > 12:
            m -= 12
            y += 1
    return out


def effective_quarter_at(conn, code: str, asof: str) -> str | None:
    """asof 시점까지 공시된 최신 분기 (look-ahead 방지).

    정정공시(재무제표 재작성)로 financials 원본이 덮어써지면 그 분기의 disclosed_date 가
    정정일(미래)로 바뀌어, 정정 전 asof 에서는 그 분기가 통째로 사라진다. 그래서 정정이력을
    보존하는 financials_revision(disclosed_date<=asof 인 버전이 하나라도 있으면 그 분기는
    "그때 알 수 있던 분기")을 우선 본다. 정정테이블 도입 전 적재분(revision 행 없음)은 기존
    financials.disclosed_date 로 폴백해 과거 데이터 백테스트가 깨지지 않게 한다. 두 소스를
    UNION 해 "그 시점까지 공시된 분기" 집합의 최대 quarter 를 고른다(quarter 는 'YYYYQn' 이라
    사전식 정렬=시간 정렬).
    """
    row = conn.execute(
        "SELECT MAX(quarter) AS quarter FROM ("
        "  SELECT quarter FROM financials_revision "
        "    WHERE stock_code=? AND disclosed_date IS NOT NULL AND disclosed_date<=? "
        "  UNION "
        "  SELECT quarter FROM financials "
        "    WHERE stock_code=? AND disclosed_date IS NOT NULL AND disclosed_date<=? "
        ")",
        (code, asof, code, asof),
    ).fetchone()
    return row["quarter"] if row and row["quarter"] else None


def mode_financial_quarter_at(conn, asof: str) -> str | None:
    """asof 시점 기준, 유효 재무데이터를 가진 종목들 중 가장 많이 쓰인 재무 기준분기(최빈값).

    get_cross_section(_qvm)류 파이프라인(correlation/quantile_bucket_means 등)은 result에
    종목별 quarter가 남지 않아(집계값만 반환) domain_backtest._build_data_asof가 재무
    기준분기를 못 붙였다. metrics_at()을 통째로 재실행하는 대신(전종목 순회 비용 큼),
    effective_quarter_at과 동일한 look-ahead 방지 조건(disclosed_date<=asof)만으로 종목별
    최신분기를 SQL 윈도우함수 한 번에 구해 최빈값을 반환한다. 국내 상장사는 공시 마감기한이
    비슷해 특정 시점엔 대부분 같은 분기로 수렴하므로 대표값으로 유의미하다.
    financials가 계정과목(account_key)별로 여러 행을 가지므로 종목당 1표만 세도록 먼저
    종목당 최신 분기 1개씩만 남긴 뒤 GROUP BY한다. effective_quarter_at과 동일하게
    financials_revision(정정이력 보존)과 financials(도입 전 폴백)를 UNION해 그 시점까지
    공시된 분기를 모은다. 데이터가 전혀 없으면 None.
    """
    row = conn.execute(
        """
        SELECT quarter FROM (
            SELECT stock_code, MAX(quarter) AS quarter FROM (
                SELECT stock_code, quarter FROM financials_revision
                    WHERE disclosed_date IS NOT NULL AND disclosed_date <= ?
                UNION
                SELECT stock_code, quarter FROM financials
                    WHERE disclosed_date IS NOT NULL AND disclosed_date <= ?
            )
            GROUP BY stock_code
        )
        GROUP BY quarter
        ORDER BY COUNT(*) DESC
        LIMIT 1
        """,
        (asof, asof),
    ).fetchone()
    return row["quarter"] if row else None


def _price_at(conn, code: str, asof: str):
    row = conn.execute(
        "SELECT close, market_cap FROM prices WHERE stock_code=? AND date<=? ORDER BY date DESC LIMIT 1",
        (code, asof),
    ).fetchone()
    return (row["close"], row["market_cap"]) if row else (None, None)


def price_return_over_months(conn, code: str, asof: str, months: int) -> dict:
    """asof 기준 최근 months개월 가격 수익률(%)을 SQL로 결정론적으로 계산한다.

    return_12m(metrics_at의 고정 12개월)과 동일한 "date<=시점 최근 거래일 종가" 원칙을
    임의 개월수로 일반화한다 — LLM 코드생성 폴백(날짜 구간을 즉석에서 스스로 판단해 8년 전
    데이터로 계산하던 실서버 버그)을 대체한다. _price_at(2-tuple 반환)의 시그니처를 바꾸지
    않으려고 date까지 함께 SELECT하는 새 쿼리를 쓴다.

    start/end 각각 요청한 날짜가 아니라 **실제 매칭된 거래일**(휴장일이면 그 이하 가장 가까운
    거래일)을 그대로 노출해 투명성을 유지한다. start 시점 이전 데이터가 아예 없으면(상장 전 등)
    start_date/start_close/return_pct가 None이 된다 — 예외를 던지거나 잘못된 값을 만들지 않고
    조용히 None으로 "계산 불가"를 알린다(호출부가 안내할 수 있게).

    반환: {"stock_code","months","start_target_date","start_date","start_close",
    "end_date","end_close","return_pct"}.
    """
    start_target = _months_before(asof, months)
    q = "SELECT date, close FROM prices WHERE stock_code=? AND date<=? ORDER BY date DESC LIMIT 1"
    end_row = conn.execute(q, (code, asof)).fetchone()
    start_row = conn.execute(q, (code, start_target)).fetchone()
    end_date = end_row["date"] if end_row else None
    end_close = end_row["close"] if end_row else None
    start_date = start_row["date"] if start_row else None
    start_close = start_row["close"] if start_row else None
    return_pct = (
        _div(end_close - start_close, start_close, pct=True)
        if (start_close and start_close > 0 and end_close is not None)
        else None
    )
    return {
        "stock_code": code,
        "months": months,
        "start_target_date": start_target,
        "start_date": start_date,
        "start_close": start_close,
        "end_date": end_date,
        "end_close": end_close,
        "return_pct": return_pct,
    }


def price_history_batch(conn, codes: list[str], asof: str, lookback_days: int) -> dict[str, list[dict]]:
    """여러 종목의 연속 가격 시계열을 SQL 1회(IN절)로 배치 조회한다.

    _price_at류(단일시점 스냅샷)와 달리 [asof-lookback_days, asof] 구간의 시계열을
    날짜 오름차순으로 반환한다 — TA-Lib 등 연속 기간이 필요한 지표 계산에 사용한다.
    종목별 개별 쿼리(N+1)를 피하려고 IN절로 한 번에 가져온다.
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
        f"SELECT stock_code, date, close FROM prices "
        f"WHERE stock_code IN ({placeholders}) AND date>=? AND date<=? "
        f"ORDER BY stock_code, date ASC",
        (*codes, start_date, asof),
    ).fetchall()
    for r in rows:
        out[r["stock_code"]].append({"date": r["date"], "close": r["close"]})
    return out


def _latest_close_batch(conn, codes: list[str], cutoff: str) -> dict[str, float]:
    """여러 종목의 'cutoff 이하 가장 가까운 거래일 종가'를 SQL 1회로 배치 조회한다.

    _price_at(단일종목 스냅샷)의 배치판 — 종목별 개별 쿼리(N+1)를 피하려고 서브쿼리로
    종목별 MAX(date)를 한 번에 구한 뒤 그 날짜의 종가를 JOIN한다(idx_price_code_d 활용).
    momentum_12_1_batch가 시작/끝 두 컷오프에 각각 1회씩만 호출하므로, 전종목 모멘텀을
    상수 회수(2회)의 SQL로 계산할 수 있다(종목 수에 비례한 반복 조회를 만들지 않는다).
    """
    out: dict[str, float] = {}
    if not codes:
        return out
    placeholders = ",".join("?" for _ in codes)
    rows = conn.execute(
        f"SELECT p.stock_code AS stock_code, p.close AS close FROM prices p "
        f"JOIN (SELECT stock_code, MAX(date) AS md FROM prices "
        f"WHERE stock_code IN ({placeholders}) AND date<=? GROUP BY stock_code) m "
        f"ON p.stock_code=m.stock_code AND p.date=m.md",
        (*codes, cutoff),
    ).fetchall()
    for r in rows:
        out[r["stock_code"]] = r["close"]
    return out


def momentum_12_1_batch(conn, codes: list[str], asof: str) -> dict[str, float | None]:
    """12-1 모멘텀(최근 1개월 제외 12개월 수익률, %)을 전종목 배치로 계산한다.

    12개월 전 종가 대비 '1개월 전' 종가의 수익률(%). 최근 1개월(asof~1개월전)을 제외하는
    표준 12-1 모멘텀이다(단순 return_12m과 다름 — 사용자가 명시적으로 12-1을 선택).
    각 컷오프(12개월전/1개월전)의 종가는 그 날짜 '이하' 가장 가까운 거래일 종가를 쓴다
    (미래참조 없음, price_return_over_months/return_12m과 동일 규약). 시작 종가가 없거나
    0 이하이면 계산 불가로 None(억지 추정하지 않음).

    크로스섹션(수천 종목)에 쓰이므로 종목별 반복 SQL이 아니라 _latest_close_batch를 시작/끝
    각 1회씩(총 2회)만 호출해 배치로 계산한다 — 기존 kr 스크리닝 지연 문제를 악화시키지 않는다.
    """
    out: dict[str, float | None] = {code: None for code in codes}
    if not codes:
        return out
    start_px = _latest_close_batch(conn, codes, _months_before(asof, 12))
    end_px = _latest_close_batch(conn, codes, _months_before(asof, 1))
    for code in codes:
        s, e = start_px.get(code), end_px.get(code)
        out[code] = _div(e - s, s, pct=True) if (s and s > 0 and e is not None) else None
    return out


def momentum_12_1(conn, code: str, asof: str) -> float | None:
    """단일종목 12-1 모멘텀(%). momentum_12_1_batch의 얇은 래퍼(로직 중복 방지)."""
    return momentum_12_1_batch(conn, [code], asof).get(code)


def _is_alive(conn, code: str, asof: str) -> bool:
    row = conn.execute("SELECT delisting_date FROM delisting WHERE stock_code=?", (code,)).fetchone()
    if row and row["delisting_date"]:
        return row["delisting_date"] > asof
    return True


# 스크리닝 LLM 프롬프트에 노출할 "재무/파생 지표" 필드의 단일 정의처(canonical source).
# 새 지표를 추가할 때 metrics_at()의 반환 dict에 필드를 넣는 것과 함께 여기(key: 한글설명)만
# 갱신하면, domain_kr.py의 _KR_SCREEN_FIELDS와 스크리닝 프롬프트(_screening_prompt)가
# 자동으로 따라온다 — 과거 return_12m/operating_profit 추가 때마다 별도 목록(_KR_SCREEN_FIELDS)에
# 손으로 다시 베껴 적는 걸 깜빡해 스크리닝에서 새 지표를 못 쓰던 반복 버그의 근본 원인을
# 없앤다. stock_code/name/sector/market/quarter/close/market_cap 같은 식별자·원시값은
# "지표"가 아니므로 포함하지 않는다. tests/test_screening_field_descriptions.py 가 이
# dict의 key 집합과 metrics_at() 실제 출력의 재무 필드 key 집합이 어긋나면 실패하도록
# 강제한다(두 곳이 따로 놀 위험을 테스트로 차단 — 완전 런타임 introspection은 과설계).
METRIC_FIELD_DESCRIPTIONS: dict[str, str] = {
    "per": "주가수익비율(PER)",
    "pbr": "주가순자산비율(PBR)",
    "psr": "주가매출비율(PSR)",
    "pcr": "주가현금흐름비율(PCR, 시가총액 ÷ 영업활동현금흐름 TTM). 밸류 팩터(낮을수록 저평가·우수)",
    "ev_ebitda": "EV/EBITDA(기업가치 ÷ EBITDA, EBITDA=EBIT+감가상각비 TTM). 밸류 팩터(낮을수록 저평가·우수). 감가상각비 미수집 종목은 None",
    "peg": "PEG(PER ÷ 순이익성장률). 성장 대비 밸류 팩터(낮을수록 우수). 순이익성장률>0일 때만 계산",
    "roe": "자기자본이익률(ROE, %)",
    "roa": "총자산이익률(ROA, %)",
    "operating_margin": "영업이익률(%, 해당분기)",
    "net_margin": "순이익률(%, 해당분기)",
    "gross_margin": "매출총이익률(%, 해당분기, 매출총이익 ÷ 매출액). GPA(gp_a=매출총이익÷총자산)와 분모가 다른 별개 지표(높을수록 우수)",
    "cogs_ratio": "매출원가율(%, 해당분기, 매출원가 ÷ 매출액). 원가효율(낮을수록 우수)",
    "debt_ratio": "부채비율(%)",
    "current_ratio": "유동비율(%, 유동자산 ÷ 유동부채). 안정성 팩터(높을수록 우수)",
    "interest_coverage": "이자보상배율(영업이익 TTM ÷ 이자비용 TTM, 배율). 안정성 팩터(높을수록 우수)",
    "revenue_growth": "매출 성장률(YoY, %)",
    "op_growth": "영업이익 성장률(YoY, %)",
    "ni_growth": "순이익 성장률(YoY, %)",
    "return_12m": "최근 12개월 주가 수익률(모멘텀, %)",
    "operating_profit": "영업이익(원화 절대값, 해당분기, TTM 아님)",
    "revenue": "매출액(원화 절대값, 해당분기, TTM 아님)",
    "net_income": "순이익(원화 절대값, 해당분기, TTM 아님)",
    "gross_profit": "매출총이익(원화 절대값, TTM 합산)",
    "total_assets": "총자산(원화 절대값, 해당분기 잔액)",
    "gp_a": "매출총이익/총자산(GPA, %, TTM 매출총이익 ÷ 해당분기 총자산). 수익성 팩터(높을수록 우수)",
    "cfo_ratio": "영업활동현금흐름/총자산(CFO비율, %, TTM 영업현금흐름 ÷ 해당분기 총자산). 질(Quality) 팩터(높을수록 우수)",
    "earnings_yield": "이익수익률(EY, %, 마법공식 EBIT ÷ 기업가치EV). 밸류 팩터(높을수록 저평가·우수)",
    "roc": "투하자본수익률(ROC, %, 마법공식 EBIT ÷ 투하자본IC). 수익성 팩터(높을수록 우수)",
    # "총자산"(total_assets)과 혼동되지 않게 시가총액임을 명시한다 — 실서버 재현 버그:
    # "시가총액"을 스크리닝 가능한 지표로 노출하지 않아 LLM이 total_assets로 잘못 골랐다.
    "market_cap": "시가총액(원화 절대값, 종가×상장주식수, 해당 기준시점)",
}


def metrics_at(conn, asof: str) -> list[dict]:
    """시점 asof의 유효 지표 행 목록 (look-ahead/생존편향/이상치 가드 적용)."""
    companies = conn.execute("SELECT stock_code, name, sector, market FROM company").fetchall()
    # 가격 이력이 신뢰 못 할(액면분할/병합 미반영 등으로 과거 종가가 불연속한) 종목은
    # 종목 단위로 통째로 제외한다(src/data_quality.py). 백테스트 유니버스·스크리닝
    # (get_cross_section이 이 함수를 기본 metrics_fn으로 씀) 양쪽에 동시 적용된다.
    excluded_codes = get_price_quality_excluded_codes(conn)
    out = []
    for c in companies:
        code = c["stock_code"]
        name = c["name"]
        if not _is_alive(conn, code, asof):
            continue
        # 스팩(기업인수목적회사)은 합병 전 현금성 자산 덩어리라 PER/ROE가 무의미 → 제외
        if "스팩" in name or "기업인수목적" in name:
            continue
        if code in excluded_codes:
            continue
        q = effective_quarter_at(conn, code, asof)
        if not q:
            continue
        close, cap = _price_at(conn, code, asof)
        # 그 시점에 실제 거래되는(주가가 있는) 종목만 → 상폐·비상장 종목(예: 대선조선) 제외
        if not close:
            continue

        # look-ahead 방지: asof 를 전달해 각 값을 financials_revision 의 "disclosed_date<=asof
        # 최신 버전"으로 조회한다(정정공시 전 시점엔 원본값, 정정 후엔 정정값). 정정이력이 없는
        # 도입 전 적재분은 _fin 내부에서 기존 financials 로 폴백한다.
        equity = _fin(conn, code, q, "total_equity", asof=asof)        # 부채비율 분모(자본총계)
        book = controlling_equity(conn, code, q, asof=asof)           # ① PBR 분모(지배주주지분)
        avg_eq = avg_controlling_equity(conn, code, q, asof=asof)     # ② ROE 분모(4분기 평균 지배주주지분)
        liab = _fin(conn, code, q, "total_liabilities", asof=asof)
        assets = _fin(conn, code, q, "total_assets", asof=asof)
        ni_ttm = _sum_ttm(conn, code, q, "net_income", asof=asof)               # 연결 전체 (ROA용)
        gp_ttm = _sum_ttm(conn, code, q, "gross_profit", asof=asof)             # GPA(수익성 팩터) 분자
        ocf_ttm = _sum_ttm(conn, code, q, "operating_cashflow", asof=asof)      # CFO비율(질 팩터) 분자(TTM 영업현금흐름)
        op_ttm = _sum_ttm(conn, code, q, "operating_profit", asof=asof)         # 이자보상배율 분자(TTM 영업이익)
        # PER·ROE 분자는 지배주주 귀속 순이익. 미수집 시 연결 순이익으로 폴백.
        ctrl_ni_ttm = _sum_ttm(conn, code, q, "controlling_net_income", asof=asof)
        if ctrl_ni_ttm is None:
            ctrl_ni_ttm = ni_ttm
        rev_ttm = _sum_ttm(conn, code, q, "revenue", asof=asof)
        # ③ 마진(영업이익률·순이익률)은 단일분기로 통일 (SoT 혼용금지, 질의 경로와 동일 정의)
        op_q = _fin(conn, code, q, "operating_profit", asof=asof)
        rev_q = _fin(conn, code, q, "revenue", asof=asof)
        ni_q = _fin(conn, code, q, "net_income", asof=asof)
        # 매출총이익률·매출원가율 입력 — 마진(operating_margin/net_margin)과 동일하게 단일분기.
        gp_q = _fin(conn, code, q, "gross_profit", asof=asof)     # 매출총이익(단일분기 원본 계정)
        cogs_q = _fin(conn, code, q, "cost_of_sales", asof=asof)  # 매출원가(단일분기 원본 계정)

        # 마법공식(EY/ROC) 입력 — 손익계산서 항목은 TTM(4분기 합), 재무상태표 항목은 시점 스냅샷.
        tax_ttm = _sum_ttm(conn, code, q, "tax_expense", asof=asof)          # 법인세비용(TTM)
        int_ttm = _sum_ttm(conn, code, q, "interest_expense", asof=asof)     # 이자비용(TTM)
        cur_assets = _fin(conn, code, q, "current_assets", asof=asof)        # 유동자산(스냅샷)
        cur_liab = _fin(conn, code, q, "current_liabilities", asof=asof)     # 유동부채(스냅샷)
        noncur_assets = _fin(conn, code, q, "non_current_assets", asof=asof) # 비유동자산(스냅샷)
        cash = _fin(conn, code, q, "cash", asof=asof)                        # 현금및현금성자산(스냅샷)
        dep = _fin(conn, code, q, "depreciation", asof=asof)                 # 감가상각비(스냅샷, ROC 투하자본용)
        dep_ttm = _sum_ttm(conn, code, q, "depreciation", asof=asof)         # 감가상각비(TTM, EBITDA용 — EBIT과 기간 일치)

        # 순이익 데이터 오류(자기자본 대비 비현실적 거대 = Q4 차분 폭발/원본 오류) 방어.
        # 순이익 기반 지표(PER·ROE·ROA)를 무효화해 다원넥스뷰 5.7경 같은 종목을 배제.
        if ni_ttm is not None and equity and equity > 0 and abs(ni_ttm) > equity * NI_TO_EQUITY_MAX_RATIO:
            ni_ttm = None
        if (
            ctrl_ni_ttm is not None
            and equity
            and equity > 0
            and abs(ctrl_ni_ttm) > equity * NI_TO_EQUITY_MAX_RATIO
        ):
            ctrl_ni_ttm = None
        # 시총이 자기자본의 100배 초과(PBR>100) → 상장주식수/주가 오류 의심 → 시총 기반 지표 무효
        if cap is not None and equity and equity > 0 and cap > equity * CAP_TO_EQUITY_MAX_RATIO:
            cap = None

        # ROE = 지배주주순이익(TTM) ÷ 지배주주지분 평균. 부호는 그대로 인정(적자면 음수=유효,
        # net_margin과 동일 관례). 자본평균 초과(≥100%, 일회성 폭발)만 이상치로 제외.
        roe = _div(ctrl_ni_ttm, avg_eq, pct=True) if (ctrl_ni_ttm and avg_eq and avg_eq > 0 and ctrl_ni_ttm < avg_eq) else None
        # 영업이익률은 단일분기 op<rev(마진<100%)일 때만 — 지주사·금융 지분법이익 폭발 제외
        op_margin = _div(op_q, rev_q, pct=True) if (rev_q and rev_q > 0 and op_q is not None and op_q < rev_q) else None
        # 순이익률도 단일분기 기준 (적자면 음수=유효)
        net_margin = _div(ni_q, rev_q, pct=True) if (rev_q and rev_q > 0 and ni_q is not None) else None

        # ── 매출총이익률(gross_margin)/매출원가율(cogs_ratio) — 단일분기, 분모=매출액 ──
        # GPA(gp_a=매출총이익÷총자산)와 분모가 완전히 다른 별개 지표다(혼동 금지). 원본 계정
        # (매출총이익/매출원가)을 우선 쓰고, 없으면 항등식(매출액=매출원가+매출총이익)으로 유도한다:
        # 매출총이익이 없으면 매출액-매출원가로, 매출원가가 없으면 매출액-매출총이익으로 유도하고
        # '{metric}_estimated'로 근사 여부를 노출한다(마법공식 roc_estimated 패턴 그대로).
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

        # 12개월 가격 수익률(모멘텀). 기준시점 종가 대비 정확히 1년 전(가장 가까운 이전 거래일)
        # 종가의 변화율(%). 미래참조 금지: _price_at은 date<=기준일만 보므로 asof 이후 종가를
        # 절대 참조하지 않는다. 1년 전 종가가 없으면 None(억지 추정하지 않음, SoT 데이터 원칙).
        prev_close, _ = _price_at(conn, code, _one_year_before(asof))
        return_12m = _div(close - prev_close, prev_close, pct=True) if (prev_close and prev_close > 0) else None

        # ── 마법공식 이익수익률(EY)/투하자본수익률(ROC) — GPA와 동일하게 크로스섹션 노출 ──
        # EBIT = 당기순이익(TTM)+법인세비용(TTM)+이자비용(TTM). 손익계산서 항목만 TTM.
        # 셋 중 하나라도 없으면 EBIT None(추정 안 함). ni_ttm은 위 이상치 가드로 None일 수
        # 있고, 그 경우 EBIT도 자연히 None → EY/ROC None(가드가 그대로 반영된다).
        ebit = (ni_ttm + tax_ttm + int_ttm) if (
            ni_ttm is not None and tax_ttm is not None and int_ttm is not None
        ) else None
        # 여유자금(excess cash): 운전자본 소요분 max(0, 유동부채-유동자산+현금)을 현금에서
        # 뺀 그린블라트 표준 방식. 현금/유동자산/유동부채가 하나라도 없으면 계산 불가(None).
        if cash is not None and cur_assets is not None and cur_liab is not None:
            excess_cash = cash - max(0.0, cur_liab - cur_assets + cash)
        else:
            excess_cash = None
        # 기업가치 EV = 시총 + 총부채 - 여유자금. EV<=0이면 EY 무의미 → None(0/음수 나누기 방지).
        ev = (cap + liab - excess_cash) if (
            cap is not None and liab is not None and excess_cash is not None
        ) else None
        earnings_yield = _div(ebit, ev, pct=True) if (ebit is not None and ev is not None and ev > 0) else None
        # EV/EBITDA — 마법공식 EV·EBIT 인프라를 그대로 재사용한다. EBITDA = EBIT + 감가상각비(TTM,
        # EBIT과 기간을 맞춤). roc(IC용 dep는 0으로 근사)와 달리 감가상각비가 없으면 EBITDA 자체를
        # 근사하지 않고 None으로 둔다 — DART 표준 API에 감가상각비 계정이 없는 종목(삼성전자 등
        # 대형주 다수)은 EV/EBITDA가 None이 된다. src/ingest/metrics.py의 ev_ebitda와 취지는
        # 같지만(감가상각비 없으면 None) EBIT/EV 구성 자체는 다르다(여기는 마법공식의
        # EBIT=ni+tax+int, EV=시총+총부채-여유자금 인프라를 재사용 — ingest는 영업이익 기반).
        ebitda = (ebit + dep_ttm) if (ebit is not None and dep_ttm is not None) else None
        # 투하자본 IC = (유동자산-유동부채)+(비유동자산-감가상각비). IC<=0이면 ROC 무의미 → None.
        # 감가상각비는 DART 표준 API에 계정 자체가 없는 종목이 많다(삼성전자 등 대형주 다수 —
        # 사업보고서 주석에서만 확인 가능하나 이 프로젝트가 수집하는 API 범위 밖). 감가상각비가
        # 없다고 ROC를 통째로 포기하는 대신 0으로 근사한다(비유동자산을 깎지 않으므로 IC가
        # 실제보다 커져 ROC는 실제보다 낮게=보수적으로 나오는 안전한 방향의 근사). roc_estimated로
        # 이 근사가 적용됐는지 항상 표시한다.
        dep_for_ic = dep if dep is not None else 0.0
        ic = ((cur_assets - cur_liab) + (noncur_assets - dep_for_ic)) if (
            cur_assets is not None and cur_liab is not None and noncur_assets is not None
        ) else None
        roc = _div(ebit, ic, pct=True) if (ebit is not None and ic is not None and ic > 0) else None
        roc_estimated = (dep is None) if roc is not None else None

        # PER·순이익성장률(YoY)을 출력 dict에서 계산하기 전에 변수로 뽑아 PEG(=PER/순이익성장률)에
        # 재사용한다(값 정의는 아래 dict와 100% 동일). PEG는 성장 대비 밸류라 성장률>0일 때만
        # 의미가 있다(성장률<=0이면 None — src/ingest/metrics.py의 peg와 동일 관례).
        per = _div(cap, ctrl_ni_ttm) if (cap and ctrl_ni_ttm and ctrl_ni_ttm > 0) else None
        ni_growth = _yoy(conn, code, q, "net_income", asof=asof)
        peg = (per / ni_growth) if (per is not None and ni_growth and ni_growth > 0) else None

        out.append({
            "stock_code": code, "name": c["name"], "sector": c["sector"], "market": c["market"],
            "quarter": q, "close": close, "market_cap": cap,
            "per": per,
            "pbr": _div(cap, book) if (cap and book and book > 0) else None,
            "psr": _div(cap, rev_ttm) if (cap and rev_ttm and rev_ttm > 0) else None,
            # PCR = 시가총액 ÷ 영업활동현금흐름(TTM). ocf_ttm은 이미 CFO비율에서 조회한 값 재사용.
            # PER의 음수순이익 처리와 동일하게 현금흐름 TTM>0일 때만(음수=현금소진은 밸류 무의미) 계산.
            "pcr": _div(cap, ocf_ttm) if (cap and ocf_ttm and ocf_ttm > 0) else None,
            # EV/EBITDA = 마법공식 EV ÷ EBITDA(=EBIT+감가상각비 TTM). EBITDA<=0 또는 EV/EBITDA 입력
            # (감가상각비) 결측이면 None. 밸류 팩터(낮을수록 저평가).
            "ev_ebitda": _div(ev, ebitda) if (ev is not None and ebitda is not None and ebitda > 0) else None,
            "peg": peg,
            "roe": roe,
            "roa": _div(ni_ttm, assets, pct=True) if (ni_ttm and assets and assets > 0) else None,
            "operating_margin": op_margin,
            "net_margin": net_margin,
            "gross_margin": gross_margin,
            "gross_margin_estimated": gross_margin_estimated,
            "cogs_ratio": cogs_ratio,
            "cogs_ratio_estimated": cogs_ratio_estimated,
            # 절대값(원화, 단일분기 — TTM 아님). "영업이익 가장 높은 기업"처럼 절대값 랭킹
            # 질문에 답하려면 마진 계산에만 쓰던 이 변수들을 출력에도 그대로 노출해야 한다
            # (이미 DB에서 조회해 메모리에 있는 값 — 새 계산/새 컬럼 없음).
            "operating_profit": op_q,
            "revenue": rev_q,
            "net_income": ni_q,
            "gross_profit": gp_ttm,
            "total_assets": assets,
            "gp_a": _div(gp_ttm, assets, pct=True) if (gp_ttm is not None and assets and assets > 0) else None,
            # CFO비율 = TTM 영업활동현금흐름 ÷ 해당분기 총자산(%). gp_a와 동일 패턴. 음수(현금소진)는
            # 유효한 저품질 신호이므로 그대로 인정(부호 보존) — 분모 총자산>0만 요구한다.
            "cfo_ratio": _div(ocf_ttm, assets, pct=True) if (ocf_ttm is not None and assets and assets > 0) else None,
            "earnings_yield": earnings_yield,
            "roc": roc,
            "roc_estimated": roc_estimated,
            "debt_ratio": _div(liab, equity, pct=True) if (equity and equity > 0) else None,
            # 유동비율 = 유동자산 ÷ 유동부채(%). cur_assets/cur_liab는 마법공식 IC에서 이미 조회한 값 재사용.
            "current_ratio": _div(cur_assets, cur_liab, pct=True) if (cur_liab and cur_liab > 0) else None,
            # 이자보상배율 = 영업이익(TTM) ÷ 이자비용(TTM). 배율이라 %가 아님. 이자비용>0일 때만(무이자
            # 기업은 이자보상배율 정의상 해당 없음 → None). int_ttm은 마법공식 EBIT에서 이미 조회한 값 재사용.
            "interest_coverage": _div(op_ttm, int_ttm) if (int_ttm and int_ttm > 0) else None,
            "revenue_growth": _yoy(conn, code, q, "revenue", asof=asof),
            "op_growth": _yoy(conn, code, q, "operating_profit", asof=asof),
            "ni_growth": ni_growth,
            "return_12m": return_12m,
        })
    return out


def build_callbacks(conn):
    """엔진용 (metrics_fn, price_fn) 생성. 시점 metrics는 한 번 계산 후 캐시."""
    cache: dict[str, list[dict]] = {}

    def metrics_fn(asof: str) -> list[dict]:
        if asof not in cache:
            cache[asof] = metrics_at(conn, asof)
        return cache[asof]

    def price_fn(asof: str, code: str):
        c, _ = _price_at(conn, code, asof)
        return c

    return metrics_fn, price_fn


def build_benchmark_fn(dates: list[str], metrics_fn, price_fn):
    """시장 동일가중 벤치마크 지수(레벨 시계열) 생성.

    각 시점의 유니버스(가드를 통과한 거래종목 전체)를 동일가중 보유했을 때의 NAV.
    전략 포트폴리오와 '같은 유니버스·같은 가격경로'를 쓰므로, 전략이 시장(동일가중)을
    이겼는지 공정하게 비교할 수 있다. 첫 시점=1.0, 이후 구간 평균수익률을 누적한다.
    """
    levels = {dates[0]: 1.0}
    cur = 1.0
    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        codes = [r["stock_code"] for r in metrics_fn(t)]
        rets = []
        for code in codes:
            p0, p1 = price_fn(t, code), price_fn(t1, code)
            if p0 and p1 and p0 > 0:
                rets.append(p1 / p0 - 1)
        cur *= (1 + (sum(rets) / len(rets) if rets else 0.0))
        levels[t1] = cur
    return lambda d: levels.get(d)
