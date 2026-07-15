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
    """asof 시점까지 공시된 최신 분기 (look-ahead 방지)."""
    row = conn.execute(
        "SELECT quarter FROM financials WHERE stock_code=? AND disclosed_date IS NOT NULL "
        "AND disclosed_date<=? ORDER BY quarter DESC LIMIT 1",
        (code, asof),
    ).fetchone()
    return row["quarter"] if row else None


def _price_at(conn, code: str, asof: str):
    row = conn.execute(
        "SELECT close, market_cap FROM prices WHERE stock_code=? AND date<=? ORDER BY date DESC LIMIT 1",
        (code, asof),
    ).fetchone()
    return (row["close"], row["market_cap"]) if row else (None, None)


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
    "roe": "자기자본이익률(ROE, %)",
    "roa": "총자산이익률(ROA, %)",
    "operating_margin": "영업이익률(%, 해당분기)",
    "net_margin": "순이익률(%, 해당분기)",
    "debt_ratio": "부채비율(%)",
    "revenue_growth": "매출 성장률(YoY, %)",
    "op_growth": "영업이익 성장률(YoY, %)",
    "ni_growth": "순이익 성장률(YoY, %)",
    "return_12m": "최근 12개월 주가 수익률(모멘텀, %)",
    "operating_profit": "영업이익(원화 절대값, 해당분기, TTM 아님)",
    "revenue": "매출액(원화 절대값, 해당분기, TTM 아님)",
    "net_income": "순이익(원화 절대값, 해당분기, TTM 아님)",
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

        equity = _fin(conn, code, q, "total_equity")        # 부채비율 분모(자본총계)
        book = controlling_equity(conn, code, q)            # ① PBR 분모(지배주주지분)
        avg_eq = avg_controlling_equity(conn, code, q)      # ② ROE 분모(4분기 평균 지배주주지분)
        liab = _fin(conn, code, q, "total_liabilities")
        assets = _fin(conn, code, q, "total_assets")
        ni_ttm = _sum_ttm(conn, code, q, "net_income")               # 연결 전체 (ROA용)
        # PER·ROE 분자는 지배주주 귀속 순이익. 미수집 시 연결 순이익으로 폴백.
        ctrl_ni_ttm = _sum_ttm(conn, code, q, "controlling_net_income")
        if ctrl_ni_ttm is None:
            ctrl_ni_ttm = ni_ttm
        rev_ttm = _sum_ttm(conn, code, q, "revenue")
        # ③ 마진(영업이익률·순이익률)은 단일분기로 통일 (SoT 혼용금지, 질의 경로와 동일 정의)
        op_q = _fin(conn, code, q, "operating_profit")
        rev_q = _fin(conn, code, q, "revenue")
        ni_q = _fin(conn, code, q, "net_income")

        # 순이익 데이터 오류(자기자본 대비 비현실적 거대 = Q4 차분 폭발/원본 오류) 방어.
        # 순이익 기반 지표(PER·ROE·ROA)를 무효화해 다원넥스뷰 5.7경 같은 종목을 배제.
        if ni_ttm is not None and equity and equity > 0 and abs(ni_ttm) > equity * 20:
            ni_ttm = None
        if ctrl_ni_ttm is not None and equity and equity > 0 and abs(ctrl_ni_ttm) > equity * 20:
            ctrl_ni_ttm = None
        # 시총이 자기자본의 100배 초과(PBR>100) → 상장주식수/주가 오류 의심 → 시총 기반 지표 무효
        if cap is not None and equity and equity > 0 and cap > equity * 100:
            cap = None

        # ROE = 지배주주순이익(TTM) ÷ 지배주주지분 평균. 양수이고 자본평균 미만(<100%)일 때만 인정.
        roe = _div(ctrl_ni_ttm, avg_eq, pct=True) if (ctrl_ni_ttm and avg_eq and avg_eq > 0 and ctrl_ni_ttm < avg_eq) else None
        # 영업이익률은 단일분기 op<rev(마진<100%)일 때만 — 지주사·금융 지분법이익 폭발 제외
        op_margin = _div(op_q, rev_q, pct=True) if (rev_q and rev_q > 0 and op_q is not None and op_q < rev_q) else None
        # 순이익률도 단일분기 기준 (적자면 음수=유효)
        net_margin = _div(ni_q, rev_q, pct=True) if (rev_q and rev_q > 0 and ni_q is not None) else None

        # 12개월 가격 수익률(모멘텀). 기준시점 종가 대비 정확히 1년 전(가장 가까운 이전 거래일)
        # 종가의 변화율(%). 미래참조 금지: _price_at은 date<=기준일만 보므로 asof 이후 종가를
        # 절대 참조하지 않는다. 1년 전 종가가 없으면 None(억지 추정하지 않음, SoT 데이터 원칙).
        prev_close, _ = _price_at(conn, code, _one_year_before(asof))
        return_12m = _div(close - prev_close, prev_close, pct=True) if (prev_close and prev_close > 0) else None

        out.append({
            "stock_code": code, "name": c["name"], "sector": c["sector"], "market": c["market"],
            "quarter": q, "close": close, "market_cap": cap,
            "per": _div(cap, ctrl_ni_ttm) if (cap and ctrl_ni_ttm and ctrl_ni_ttm > 0) else None,
            "pbr": _div(cap, book) if (cap and book and book > 0) else None,
            "psr": _div(cap, rev_ttm) if (cap and rev_ttm and rev_ttm > 0) else None,
            "roe": roe,
            "roa": _div(ni_ttm, assets, pct=True) if (ni_ttm and assets and assets > 0) else None,
            "operating_margin": op_margin,
            "net_margin": net_margin,
            # 절대값(원화, 단일분기 — TTM 아님). "영업이익 가장 높은 기업"처럼 절대값 랭킹
            # 질문에 답하려면 마진 계산에만 쓰던 이 변수들을 출력에도 그대로 노출해야 한다
            # (이미 DB에서 조회해 메모리에 있는 값 — 새 계산/새 컬럼 없음).
            "operating_profit": op_q,
            "revenue": rev_q,
            "net_income": ni_q,
            "debt_ratio": _div(liab, equity, pct=True) if (equity and equity > 0) else None,
            "revenue_growth": _yoy(conn, code, q, "revenue"),
            "op_growth": _yoy(conn, code, q, "operating_profit"),
            "ni_growth": _yoy(conn, code, q, "net_income"),
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
