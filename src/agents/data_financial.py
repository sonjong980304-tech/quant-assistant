"""재무데이터 에이전트 — 지표별로 값을 어디서 볼지(DART vs FnGuide) 판단한다.

이 프로젝트의 재무데이터는 완전히 분리된 두 테이블에서 온다:
- DART(`src/ingest/dart.py`가 채움) → `financials`(표준 재무제표: 매출/영업이익/순이익/
  자산/부채/자본 등)와 계산 지표 `metrics`(per/pbr/roe 등).
- FnGuide(`src/ingest/fnguide_metrics.py`가 채움) → `fnguide_metrics`(별도 테이블, 병합 안 됨).
  컨센서스 목표주가(consensus_target_price)·투자의견(consensus_opinion_score) 등
  FnGuide에만 있는 지표가 존재한다.

핵심 책임: 질문에 나온 지표명을 보고
1) 규칙기반 매핑표(METRIC_SOURCE_MAP)로 소스를 정하고,
2) 매핑표에 없으면 주입된 llm_fn 판단 경로로 위임하며,
3) DART·FnGuide 둘 다에 값이 있으면 대표값(value/source/period)은 DART를 우선 채택하되,
   두 소스 값이 다를 수 있으므로 dart_value/fnguide_value에 양쪽을 모두 담아 최종 답변에서
   "DART는 X, FnGuide는 Y" 식으로 병기할 수 있게 하고,
4) 반환 dict에 항상 어느 소스를 썼는지 `source`('DART'|'FnGuide')를 담는다.

섹터(company.sector)의 최종 출처는 KRX로 이미 확정되어 있어(scripts/backfill_sector_krx.py)
이 에이전트는 섹터 라우팅을 다루지 않는다.
"""
from __future__ import annotations

import sqlite3
from typing import Callable

DART = "DART"
FNGUIDE = "FnGuide"

# 지표명(소문자) → 소스. 규칙기반 1차 라우팅표.
# - financials(DART 표준 재무제표) 계정 + metrics(DART 계산 지표)는 DART.
# - fnguide_metrics 전용 지표(컨센서스 목표주가·투자의견·수정주가)는 FnGuide.
METRIC_SOURCE_MAP: dict[str, str] = {
    # --- DART: financials 표준 재무제표 계정(account_key) ---
    "revenue": DART,
    "cost_of_sales": DART,
    "gross_profit": DART,
    "sga": DART,
    "operating_profit": DART,
    "net_income": DART,
    "interest_expense": DART,
    "operating_cashflow": DART,
    "depreciation": DART,
    "dividend": DART,
    "total_assets": DART,
    "total_liabilities": DART,
    "total_equity": DART,
    "shares_outstanding": DART,
    # --- DART: metrics 계산 지표(컬럼형) ---
    "per": DART,
    "pbr": DART,
    "roe": DART,
    "operating_margin": DART,
    "debt_ratio": DART,
    "market_cap": DART,
    # --- DART: metrics 계산 지표(컬럼형, 시가총액 파생 밸류 팩터) — per/pbr과 완전히 동일한
    # 메커니즘(ingest 시 metrics 테이블에 사전계산돼 있고, _fetch_dart/_fetch_dart_at_quarter가
    # _METRICS_TABLE_COLS 키를 범용적으로 처리하므로 새 계산 로직 없이 등록만으로 direct-WHERE
    # 분기 매치가 된다). 실서버 재현 버그: 이 등록이 빠져 있어 _COMPUTED_ONLY_FIELDS(asof 기반
    # cross-section 경로)로 잘못 분류돼 있었다.
    "psr": DART,
    "pcr": DART,
    "ev_ebitda": DART,
    "peg": DART,
    # --- DART: _FLOW_RATIO_ACCOUNTS(EAV 두 계정 즉석 비율계산, operating_margin과 동일 경로) ---
    # 실서버 재현 버그: 이 등록이 빠져 있어 gross_margin 등이 _COMPUTED_ONLY_FIELDS(asof/
    # look-ahead 경로)로 잘못 분류됐다. _fetch_flow_ratio는 이미 이 셋을 지원했는데
    # (test_resolve_metric_net_gross_cogs_ratios_fall_back_to_eav) 라우팅이 안 태웠다.
    "gross_margin": DART,
    "net_margin": DART,
    "cogs_ratio": DART,
    # --- DART: _RATIO_TTM_ACCOUNTS(TTM/스냅샷 혼합 비율, 가격 불필요 → EAV 직접 quarter 매치) ---
    "roa": DART,
    "gp_a": DART,
    "interest_coverage": DART,
    "current_ratio": DART,
    # --- DART: _YOY_GROWTH_ACCOUNTS(전년 동기 대비 성장률, 가격 불필요 → EAV 직접 매치) ---
    "revenue_growth": DART,
    "op_growth": DART,
    "ni_growth": DART,
    # --- FnGuide 전용 지표(fnguide_metrics.metric_key) ---
    "target_price": FNGUIDE,
    "consensus_target_price": FNGUIDE,
    "analyst_opinion": FNGUIDE,
    "consensus_opinion_score": FNGUIDE,
    "adjusted_close": FNGUIDE,
}

# 일상 지표명 → fnguide_metrics.metric_key 실제 컬럼값 별칭
# (fnguide_metrics.py의 _TARGET_CHART_METRIC_KEYS 참고).
_FNGUIDE_KEY_ALIASES: dict[str, str] = {
    "target_price": "consensus_target_price",
    "analyst_opinion": "consensus_opinion_score",
}

# DART 계산 지표(metrics 테이블 컬럼) 화이트리스트 — SQL 식별자로 직접 쓰므로
# 반드시 이 집합 안의 값만 사용한다(주입 방지).
_METRICS_TABLE_COLS = {
    "per", "pbr", "roe", "operating_margin", "debt_ratio", "market_cap",
    "psr", "pcr", "ev_ebitda", "peg",
}

# 위 metrics 컬럼 중 "주가에 의존하는" 지표. 이 지표들은 metrics 행에 함께 적재된
# price_date(종가 기준일, 적재 시 version.effective_price_date로 확정)를 계산에 실제로 쓰므로,
# 조회 결과에 그 price_date를 함께 실어 사용자가 "어느 종가 기준 값이냐"를 검증할 수 있게 한다.
# roe/operating_margin/debt_ratio는 순수 재무비율이라 price_date가 무의미하므로 제외한다.
# psr/pcr/ev_ebitda는 시가총액을, peg는 per(시가총액 기반)를 분자에 쓰므로 전부 포함한다.
_PRICE_BASED_METRICS = {"per", "pbr", "market_cap", "psr", "pcr", "ev_ebitda", "peg"}

# 연간(annual) 요청 시 4개 분기를 합산해야 하는 손익계산서/현금흐름 흐름값 계정.
# 재무상태표 잔액(total_assets/total_liabilities/total_equity/shares_outstanding)이나
# 비율지표(per/pbr/roe 등)는 합산 대상이 아니라 연말(Q4) 스냅샷을 쓴다(잔액을 더하면 틀림).
_SUMMABLE_FLOW_ACCOUNTS: frozenset[str] = frozenset({
    "revenue", "cost_of_sales", "gross_profit", "sga", "operating_profit",
    "net_income", "interest_expense", "operating_cashflow", "depreciation", "dividend",
})

# 원본 EAV 흐름값 계정 두 개(분자, 분모)만으로 즉석 계산 가능한 단순 비율 지표(%).
# metrics 사전계산 테이블은 "가장 최근 인제스트 시점의 스냅샷 한 분기"만 유지하므로,
# 과거 분기를 지목한 질문(예: SK하이닉스 24/25년 영업이익률)은 그 분기 metrics 행이 없어
# null로 빠진다 — 원본 financials(EAV)엔 두 계정이 정상 존재하는데도. 그 경우 여기 등록된
# 지표만 EAV 두 계정을 직접 읽어 비율을 즉석 계산한다(metrics.py/_div와 동일하게 ×100 퍼센트).
# 분자·분모가 모두 _SUMMABLE_FLOW_ACCOUNTS 소속이라 분기값은 그 분기 두 계정을, 연간값은
# 각 계정의 4분기 합을 나눈다(분기별 비율의 평균이 아니라 연간 합계 기준 비율 — 재무 관례).
# per/pbr/시총(주가 필요)·roe(평균자본)·debt_ratio(잔액) 등은 EAV 두 흐름계정만으로
# 재현 불가라 제외한다.
_FLOW_RATIO_ACCOUNTS: dict[str, tuple[str, str]] = {
    "operating_margin": ("operating_profit", "revenue"),
    "net_margin": ("net_income", "revenue"),
    "gross_margin": ("gross_profit", "revenue"),
    "cogs_ratio": ("cost_of_sales", "revenue"),
}


def _normalize_source(value: str | None) -> str:
    """소스 문자열(대소문자/여백 섞임)을 표준 'DART'|'FnGuide'로 정규화."""
    key = (value or "").strip().lower()
    if key == "fnguide":
        return FNGUIDE
    return DART  # 그 외/불명확은 표준 소스 DART로 안전 폴백


def classify_source(metric: str, llm_fn: Callable[[str], str] | None = None) -> str:
    """지표명 → 'DART' | 'FnGuide'.

    매핑표에 있으면 규칙기반으로 즉시 결정한다. 없으면 llm_fn(metric)에 위임하고
    그 반환값을 정규화한다. llm_fn이 없고 매핑표에도 없으면 표준 소스 DART로 폴백한다.
    """
    key = metric.strip().lower()
    if key in METRIC_SOURCE_MAP:
        return METRIC_SOURCE_MAP[key]
    if llm_fn is not None:
        return _normalize_source(llm_fn(metric))
    return DART


def _fetch_dart_at_quarter(
    conn: sqlite3.Connection, key: str, stock_code: str, quarter: str
) -> tuple[float, str, str | None] | None:
    """특정 분기(예: '2025Q3')의 DART 값 — financials(EAV) 우선, 없으면 metrics 컬럼.

    반환: (value, quarter, price_date). price_date는 주가 기반 지표(_PRICE_BASED_METRICS,
    per/pbr/market_cap)를 metrics 테이블에서 읽었을 때 그 값 계산에 쓰인 종가 기준일
    (metrics.price_date)이며, 그 외(순수 재무비율/EAV 계정)에는 None이다. 한 분기에 종가
    스냅샷이 여러 개면 가장 최신(price_date DESC)을 쓴다.
    """
    row = conn.execute(
        "SELECT amount, quarter FROM financials WHERE stock_code=? AND account_key=? AND quarter=?",
        (stock_code, key, quarter),
    ).fetchone()
    if row is not None and row["amount"] is not None:
        return row["amount"], row["quarter"], None
    if key in _METRICS_TABLE_COLS:
        row = conn.execute(
            f"SELECT {key} AS v, quarter, price_date FROM metrics "
            "WHERE stock_code=? AND quarter=? ORDER BY price_date DESC LIMIT 1",
            (stock_code, quarter),
        ).fetchone()
        if row is not None and row["v"] is not None:
            price_date = row["price_date"] if key in _PRICE_BASED_METRICS else None
            return row["v"], row["quarter"], price_date
    return None


def _fetch_eav_amount(
    conn: sqlite3.Connection, key: str, stock_code: str, quarter: str
) -> float | None:
    """financials(EAV)에서 (stock_code, account_key, quarter)의 금액. 없으면 None."""
    row = conn.execute(
        "SELECT amount FROM financials WHERE stock_code=? AND account_key=? AND quarter=?",
        (stock_code, key, quarter),
    ).fetchone()
    return row["amount"] if row is not None else None


def _sum_four_quarters(
    conn: sqlite3.Connection, key: str, stock_code: str, year: int
) -> float | None:
    """year의 4개 분기(Q1~Q4) EAV 금액 합. 4개 모두 있을 때만 합, 하나라도 없으면 None.

    SoT(누락 분기 추정 금지): 4개 분기가 모두 존재할 때만 연간 합계를 낸다. 흐름값 계정의
    연간 합산(_fetch_dart annual)과 흐름비율의 연간 분자·분모 합산이 이 규칙을 공유한다.
    """
    quarters = [f"{year}Q{i}" for i in (1, 2, 3, 4)]
    placeholders = ",".join("?" for _ in quarters)
    rows = conn.execute(
        f"SELECT quarter, amount FROM financials WHERE stock_code=? AND account_key=? "
        f"AND quarter IN ({placeholders})",
        (stock_code, key, *quarters),
    ).fetchall()
    amounts = {r["quarter"]: r["amount"] for r in rows if r["amount"] is not None}
    return sum(amounts.values()) if len(amounts) == 4 else None


def _fetch_flow_ratio(
    conn: sqlite3.Connection, key: str, stock_code: str, period: dict
) -> tuple[float, str, str | None] | None:
    """단순 흐름비율 지표(_FLOW_RATIO_ACCOUNTS)를 원본 EAV 두 계정으로 즉석 계산(%).

    metrics 사전계산 스냅샷에 그 분기 값이 없을 때만 부르는 폴백(호출부가 우선순위 보장).
    분모가 0/None이거나 분자가 None이면 계산하지 않는다(SoT: 억지 추정 금지). 반환 형식은
    _fetch_dart_at_quarter와 동일한 (value, 기간라벨, price_date=None) — 순수 재무비율이라
    price_date는 항상 None이고, source는 호출부에서 여전히 'DART'로 채워진다(즉석 계산이라는
    사실을 별도로 노출하지 않는다).
    - quarter: 그 분기의 분자/분모 EAV를 나눈다.
    - annual: 분자 4분기 합 ÷ 분모 4분기 합(각 4개 모두 있을 때만, 없으면 None).
    """
    num_key, den_key = _FLOW_RATIO_ACCOUNTS[key]
    if period.get("kind") == "annual":
        year = period["year"]
        num = _sum_four_quarters(conn, num_key, stock_code, year)
        den = _sum_four_quarters(conn, den_key, stock_code, year)
        label = f"{year} 연간"
    else:  # quarter
        quarter = period["quarter"]
        num = _fetch_eav_amount(conn, num_key, stock_code, quarter)
        den = _fetch_eav_amount(conn, den_key, stock_code, quarter)
        label = quarter
    if num is None or not den:  # 분자 None 또는 분모 0/None → 계산 불가
        return None
    return num / den * 100.0, label, None


# 분자·분모 각각이 "TTM(추세 12개월, 4분기 합)"인지 "그 분기 스냅샷(잔액)"인지가 다른 비율
# 지표. src/backtest/data_access.py의 metrics_at() 실제 계산식과 동일 정의를 재사용한다
# (roa=ni_ttm/assets, gp_a=gp_ttm/assets, interest_coverage=op_ttm/int_ttm, current_ratio=
# cur_assets/cur_liab). scale=100.0(%)|1.0(배율, interest_coverage만 배수 표기).
_RATIO_TTM_ACCOUNTS: dict[str, tuple[tuple[bool, str], tuple[bool, str], float]] = {
    "roa": ((True, "net_income"), (False, "total_assets"), 100.0),
    "gp_a": ((True, "gross_profit"), (False, "total_assets"), 100.0),
    "interest_coverage": ((True, "operating_profit"), (True, "interest_expense"), 1.0),
    "current_ratio": ((False, "current_assets"), (False, "current_liabilities"), 100.0),
}


def _trailing_quarters(end_quarter: str, n: int = 4) -> list[str]:
    """end_quarter로 끝나는 최근 n개 분기 문자열(과거→최신)을 만든다("2026Q1"→[...,2026Q1])."""
    year, q = int(end_quarter[:4]), int(end_quarter[5])
    out = []
    for _ in range(n):
        out.append(f"{year}Q{q}")
        q -= 1
        if q == 0:
            q, year = 4, year - 1
    return list(reversed(out))


def _sum_trailing_four_quarters(conn: sqlite3.Connection, key: str, stock_code: str, end_quarter: str) -> float | None:
    """end_quarter로 끝나는 TTM(연간 캘린더가 아니라 그 분기 기준 최근 4분기) 합.

    _sum_four_quarters(연간=1~4월, 특정 연도 고정)와 달리 어느 분기에서든 "그 분기까지의
    최근 4분기"를 구한다(roa 등 TTM 지표가 분기 질문에서도 정확한 추세치를 내려면 필요).
    4개 분기 중 하나라도 없으면 None(SoT: 누락 분기 추정 금지, _sum_four_quarters와 동일 원칙).
    """
    quarters = _trailing_quarters(end_quarter, 4)
    placeholders = ",".join("?" for _ in quarters)
    rows = conn.execute(
        f"SELECT quarter, amount FROM financials WHERE stock_code=? AND account_key=? "
        f"AND quarter IN ({placeholders})",
        (stock_code, key, *quarters),
    ).fetchall()
    amounts = {r["quarter"]: r["amount"] for r in rows if r["amount"] is not None}
    return sum(amounts.values()) if len(amounts) == 4 else None


def _resolve_ratio_component(conn: sqlite3.Connection, stock_code: str, quarter: str, spec: tuple[bool, str]) -> float | None:
    is_ttm, account = spec
    if is_ttm:
        return _sum_trailing_four_quarters(conn, account, stock_code, quarter)
    return _fetch_eav_amount(conn, account, stock_code, quarter)


def _fetch_ratio_ttm(
    conn: sqlite3.Connection, key: str, stock_code: str, period: dict
) -> tuple[float, str, str | None] | None:
    """TTM/스냅샷 혼합 비율 지표(_RATIO_TTM_ACCOUNTS)를 그 분기(annual이면 연말 Q4) 기준으로
    즉석 계산한다. metrics_at()과 동일 정의를 EAV에서 직접 재현하되, 가격/시가총액은 전혀
    쓰지 않는 순수 재무비율이라 look-ahead용 asof 없이 지목된 분기를 그대로 쓸 수 있다."""
    quarter = period["quarter"] if period.get("kind") == "quarter" else f"{period['year']}Q4"
    num_spec, den_spec, scale = _RATIO_TTM_ACCOUNTS[key]
    num = _resolve_ratio_component(conn, stock_code, quarter, num_spec)
    den = _resolve_ratio_component(conn, stock_code, quarter, den_spec)
    if num is None or not den:
        return None
    return num / den * scale, quarter, None


# 전년 동기(YoY) 대비 성장률. 두 시점 모두 그 계정의 원본 값(quarter는 단일분기,
# annual은 4분기 합)이 있어야 하고, 전년 값이 0 이하면 계산하지 않는다(SoT: 적자→흑자
# 전환처럼 기준값이 0/음수인 성장률은 부호가 무의미해 억지 추정하지 않는다).
_YOY_GROWTH_ACCOUNTS: dict[str, str] = {
    "revenue_growth": "revenue",
    "op_growth": "operating_profit",
    "ni_growth": "net_income",
}


def _fetch_yoy_growth(
    conn: sqlite3.Connection, key: str, stock_code: str, period: dict
) -> tuple[float, str, str | None] | None:
    account = _YOY_GROWTH_ACCOUNTS[key]
    if period.get("kind") == "annual":
        year = period["year"]
        cur = _sum_four_quarters(conn, account, stock_code, year)
        prev = _sum_four_quarters(conn, account, stock_code, year - 1)
        label = f"{year} 연간"
    else:
        quarter = period["quarter"]
        prev_quarter = f"{int(quarter[:4]) - 1}{quarter[4:]}"
        cur = _fetch_eav_amount(conn, account, stock_code, quarter)
        prev = _fetch_eav_amount(conn, account, stock_code, prev_quarter)
        label = quarter
    if cur is None or not prev or prev <= 0:
        return None
    return (cur - prev) / prev * 100.0, label, None


def _latest_quarter_with_account(conn: sqlite3.Connection, stock_code: str, account_key: str) -> str | None:
    """그 계정이 실존하는 가장 최근 분기(period 미지정 질문의 '최신값' 폴백용)."""
    row = conn.execute(
        "SELECT MAX(quarter) AS q FROM financials WHERE stock_code=? AND account_key=? AND amount IS NOT NULL",
        (stock_code, account_key),
    ).fetchone()
    return row["q"] if row else None


def _fetch_dart(
    conn: sqlite3.Connection,
    stock_code: str,
    metric: str,
    period: dict | None = None,
) -> tuple[float, str, str | None] | None:
    """DART 값 조회: 먼저 financials(EAV), 없으면 metrics(계산 지표).

    반환: (value, quarter, price_date) — quarter는 조회된 값이 실제로 어느 기간 것인지
    항상 함께 돌려주고, price_date는 주가 기반 지표(per/pbr/market_cap)일 때 그 값 계산에
    쓰인 종가 기준일(그 외 None)이다. 이 라벨이 없으면 총괄 에이전트의 검증 단계가 "26년 1분기 영업이익"처럼
    특정 분기를 지목한 질문에 대해 값이 맞는 분기인지 확인할 방법이 없어 항상 검증
    실패(uncertain)로 빠진다(실사용 재현 버그).

    period(총괄→도메인 에이전트가 질문에서 파싱해 넘김)로 조회 기간을 고른다:
    - None: 기존과 동일하게 가장 최신 분기 1건(회귀 없음).
    - {"kind":"quarter","quarter":"2025Q3"}: 그 분기만 조회.
    - {"kind":"annual","year":2025}: 흐름값 계정(_SUMMABLE_FLOW_ACCOUNTS)이면 그 해 4개
      분기를 합산(4개 모두 있을 때만 — SoT: 누락 분기를 추정하지 않는다), 그 외(재무상태표
      잔액/비율)는 연말(Q4) 스냅샷을 쓴다.
    """
    key = metric.strip().lower()

    if period is None:
        row = conn.execute(
            "SELECT amount, quarter FROM financials WHERE stock_code=? AND account_key=? "
            "ORDER BY quarter DESC LIMIT 1",
            (stock_code, key),
        ).fetchone()
        if row is not None and row["amount"] is not None:
            return row["amount"], row["quarter"], None
        if key in _METRICS_TABLE_COLS:
            row = conn.execute(
                f"SELECT {key} AS v, quarter, price_date FROM metrics WHERE stock_code=? "
                "ORDER BY quarter DESC, price_date DESC LIMIT 1",
                (stock_code,),
            ).fetchone()
            if row is not None and row["v"] is not None:
                price_date = row["price_date"] if key in _PRICE_BASED_METRICS else None
                return row["v"], row["quarter"], price_date
        # 즉석계산 비율/성장률 지표(_FLOW_RATIO_ACCOUNTS/_RATIO_TTM_ACCOUNTS/_YOY_GROWTH_ACCOUNTS)는
        # financials/metrics에 그 이름의 컬럼이 없어 위에서 항상 못 찾는다 — "가장 최근 데이터가
        # 있는 분기"를 찾아 그 분기 기준으로 위임한다(period 명시 질문과 동일한 계산, 대상 분기만
        # 자동 산정). 앵커 계정은 되도록 스냅샷 쪽(분기마다 하나씩만 있어 최신판별이 명확)을 쓴다.
        if key in _FLOW_RATIO_ACCOUNTS:
            anchor = _latest_quarter_with_account(conn, stock_code, _FLOW_RATIO_ACCOUNTS[key][0])
            return _fetch_flow_ratio(conn, key, stock_code, {"kind": "quarter", "quarter": anchor}) if anchor else None
        if key in _RATIO_TTM_ACCOUNTS:
            num_spec, den_spec, _scale = _RATIO_TTM_ACCOUNTS[key]
            anchor_account = den_spec[1] if not den_spec[0] else num_spec[1]
            anchor = _latest_quarter_with_account(conn, stock_code, anchor_account)
            return _fetch_ratio_ttm(conn, key, stock_code, {"kind": "quarter", "quarter": anchor}) if anchor else None
        if key in _YOY_GROWTH_ACCOUNTS:
            anchor = _latest_quarter_with_account(conn, stock_code, _YOY_GROWTH_ACCOUNTS[key])
            return _fetch_yoy_growth(conn, key, stock_code, {"kind": "quarter", "quarter": anchor}) if anchor else None
        return None

    if period.get("kind") == "quarter":
        result = _fetch_dart_at_quarter(conn, key, stock_code, period["quarter"])
        # metrics 스냅샷에 없는 과거 분기라도 단순 흐름비율/TTM비율/성장률은 원본 EAV 계정
        # 으로 즉석 계산한다(폴백). metrics 값이 있으면 위에서 이미 반환되므로 사전계산값 최우선.
        if result is None and key in _FLOW_RATIO_ACCOUNTS:
            return _fetch_flow_ratio(conn, key, stock_code, period)
        if result is None and key in _RATIO_TTM_ACCOUNTS:
            return _fetch_ratio_ttm(conn, key, stock_code, period)
        if result is None and key in _YOY_GROWTH_ACCOUNTS:
            return _fetch_yoy_growth(conn, key, stock_code, period)
        return result

    # kind == "annual"
    year = period["year"]
    if key in _SUMMABLE_FLOW_ACCOUNTS:
        total = _sum_four_quarters(conn, key, stock_code, year)
        return (total, f"{year} 연간", None) if total is not None else None
    # 단순 흐름비율은 연말(Q4) 스냅샷이 아니라 분자 4분기 합 ÷ 분모 4분기 합으로 계산한다
    # (Q4 단일분기 마진을 연간 마진으로 오인하지 않게 — 재무 관례상 연간 합계 기준 비율).
    if key in _FLOW_RATIO_ACCOUNTS:
        return _fetch_flow_ratio(conn, key, stock_code, period)
    if key in _RATIO_TTM_ACCOUNTS:
        return _fetch_ratio_ttm(conn, key, stock_code, period)
    if key in _YOY_GROWTH_ACCOUNTS:
        return _fetch_yoy_growth(conn, key, stock_code, period)
    # 그 외 흐름값이 아닌 계정(잔액/주가비율)은 연말(Q4) 스냅샷.
    return _fetch_dart_at_quarter(conn, key, stock_code, f"{year}Q4")


def _fetch_fnguide(conn: sqlite3.Connection, stock_code: str, metric: str) -> tuple[float, str] | None:
    """FnGuide 값 조회: fnguide_metrics에서 지표명(+별칭)으로 최신 as_of_date 값.

    반환: (value, as_of_date) — _fetch_dart와 동일한 이유로 시점을 항상 함께 돌려준다.
    """
    key = metric.strip().lower()
    candidates = [key]
    alias = _FNGUIDE_KEY_ALIASES.get(key)
    if alias and alias not in candidates:
        candidates.append(alias)
    for cand in candidates:
        row = conn.execute(
            "SELECT metric_value, as_of_date FROM fnguide_metrics WHERE stock_code=? AND metric_key=? "
            "ORDER BY as_of_date DESC LIMIT 1",
            (stock_code, cand),
        ).fetchone()
        if row is not None and row["metric_value"] is not None:
            return row["metric_value"], row["as_of_date"]
    return None


def resolve_metric(
    conn: sqlite3.Connection,
    stock_code: str,
    metric: str,
    llm_fn: Callable[[str], str] | None = None,
    period: dict | None = None,
) -> dict:
    """지표 하나를 소스 판단 후 DB에서 조회. 항상 'source'/'period'를 담은 dict 반환.

    반환: {"stock_code", "metric", "value", "source", "period",
           "dart_value", "dart_period", "fnguide_value", "fnguide_period", "price_date"}.
    price_date는 주가 기반 지표(per/pbr/시총)일 때 그 값 계산에 쓰인 종가 기준일이고 그 외 None.
    - DART·FnGuide 둘 다 값이 있으면 대표값(value/source/period)은 DART를 우선 채택한다
      (rule #3, 회귀 없음). 단 이 경우 두 소스 값이 다를 수 있으므로(예: 같은 '매출액'이라도
      집계 기준이 달라 DART/FnGuide 수치가 어긋남) dart_value/fnguide_value에 양쪽을 모두
      담아 최종 답변에서 "DART는 X, FnGuide는 Y" 식으로 병기할 수 있게 한다.
    - 한쪽에만 값이 있으면 그 소스를 대표값으로 쓰고, 값이 없는 쪽의 *_value/*_period는 None.
    - 양쪽 다 값이 없으면 value=period=None, source엔 라우팅 판단 소스를 담는다.
    - period 인자(총괄→도메인 에이전트가 질문에서 파싱)로 DART 조회 기간을 고른다:
      None이면 기존처럼 최신 분기 1건(회귀 없음), {"kind":"quarter",...}면 그 분기,
      {"kind":"annual",...}면 그 해 연간(흐름값은 4분기 합, 그 외는 Q4). 반환 dict의
      "period"는 실제 조회된 기간 라벨(분기 문자열 또는 "YYYY 연간", FnGuide면 as_of_date)
      이며, 호출부(verify_answer)가 이걸 보고 질문이 지목한 기간과 맞는지 판정한다.
    """
    chosen = classify_source(metric, llm_fn=llm_fn)
    dart_result = _fetch_dart(conn, stock_code, metric, period=period)
    fnguide_result = _fetch_fnguide(conn, stock_code, metric)

    if dart_result is not None:  # DART 우선(둘 다 있어도 DART)
        value, resolved_period, source = dart_result[0], dart_result[1], DART
    elif fnguide_result is not None:
        value, resolved_period, source = fnguide_result[0], fnguide_result[1], FNGUIDE
    else:
        value, resolved_period, source = None, None, chosen

    return {
        "stock_code": stock_code, "metric": metric,
        "value": value, "source": source, "period": resolved_period,
        "dart_value": dart_result[0] if dart_result else None,
        "dart_period": dart_result[1] if dart_result else None,
        "fnguide_value": fnguide_result[0] if fnguide_result else None,
        "fnguide_period": fnguide_result[1] if fnguide_result else None,
        # 주가 기반 지표(per/pbr/시총)의 종가 기준일(metrics.price_date). 그 외 지표는 None.
        # 기간 미지정 질문에서 "이 값이 어느 종가 기준이냐"를 답변에 라벨링하는 데 재사용된다.
        "price_date": dart_result[2] if dart_result else None,
    }
