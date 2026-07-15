"""휴리스틱 라우팅 + 폴백 SQL 생성.

- route 판단(router_node)은 결정적 키워드 매칭으로 처리 (LLM 불필요, 빠름).
- 폴백 SQL은 LLM(키/데몬)이 없을 때만 사용한다. 실제 평가는 LLM 기준이며,
  폴백은 키 없이도 전체 파이프라인이 동작함을 보이기 위한 것이다.
"""
from __future__ import annotations

import re

from src.ingest.companies import COMPANIES

# 지표 → (테이블, 컬럼, route 힌트)
METRIC_COLUMNS = {
    "per": ("metrics", "per", "both"),
    "pbr": ("metrics", "pbr", "both"),
    "roe": ("metrics", "roe", "financial"),
    "영업이익률": ("metrics", "operating_margin", "financial"),
    "부채비율": ("metrics", "debt_ratio", "financial"),
    "시가총액": ("prices", "market_cap", "price"),
    "시총": ("prices", "market_cap", "price"),
    "종가": ("prices", "close", "price"),
    "매출": ("financials", "revenue", "financial"),
    "매출액": ("financials", "revenue", "financial"),
    "영업이익": ("financials", "operating_profit", "financial"),
    "순이익": ("financials", "net_income", "financial"),
    "당기순이익": ("financials", "net_income", "financial"),
}

# 긴 키워드 우선 매칭 (영업이익률 > 영업이익, 매출액 > 매출)
_METRIC_ORDER = sorted(METRIC_COLUMNS, key=len, reverse=True)

_DESC_WORDS = ["높은", "큰", "많은", "상위", "높", "best", "top", "고"]
_ASC_WORDS = ["낮은", "작은", "적은", "하위", "낮", "저"]

_SECTORS = sorted({c[3] for c in COMPANIES}, key=len, reverse=True)
_NAMES = sorted({c[1] for c in COMPANIES}, key=len, reverse=True)


# SQL로 표현 불가능한 통계/퀀트 분석(파이프라인 경로) 감지용 키워드.
# 기존 goldset 50문항(PER/PBR/ROE/부채/시총/매출/영업이익률/순이익/업종/집계)에는 없는
# 단어만 골라 오탐(false positive)이 나지 않게 한다. detect_route와 같은 결정적 키워드 방식.
_PIPELINE_KEYWORDS = [
    "k-ratio", "k_ratio", "케이레이쇼", "케이 레이쇼",
    "포트폴리오", "최적화", "최적 비중", "비중 계산", "비중을 구",
    "최대샤프", "최대 샤프", "샤프 최대", "max sharpe", "max_sharpe",
    "최소분산", "최소 분산", "min variance", "min_variance",
    "위험균형", "위험 균형", "리스크패리티", "리스크 패리티", "risk parity", "risk_parity",
    "회귀분석", "회귀 분석", "표준오차", "누적수익률", "수익률 회귀",
    "1년 수익률", "1년수익률", "섹터중립", "섹터 중립",
    "백테스트", "리밸런싱", "리밸런스",
    "rsi", "macd", "이동평균", "볼린저", "볼린저밴드", "골든크로스", "데드크로스", "기술적지표", "기술지표",
    "전략 찾아", "전략 탐색", "전략을 찾", "전략을 탐색", "최대손실폭",
]


def detect_pipeline(question: str) -> bool:
    """SQL로 표현 불가능한 통계/퀀트 분석 질문인지 결정적 키워드로 판정한다.

    기존 detect_route와 동일한 '휴리스틱 키워드' 관례를 따른다(LLM 불필요, 결정론).
    True면 router_node가 route="pipeline"으로 분기한다.
    """
    text = (question or "").lower()
    return any(kw in text for kw in _PIPELINE_KEYWORDS)


def detect_route(question: str, sql: str | None = None) -> str:
    """주가 필요/재무만/둘다 판단 → 'financial' | 'price' | 'both'."""
    text = f"{question} {sql or ''}".lower()
    if "per" in text or "pbr" in text or "주가수익" in text or "주가순자산" in text:
        return "both"
    has_price = any(k in text for k in ["시가총액", "시총", "종가", "주가", "market_cap"])
    has_fin = any(
        k in text
        for k in ["roe", "영업이익", "부채", "매출", "순이익", "자산", "자본", "자기자본"]
    )
    if has_price and has_fin:
        return "both"
    if has_price:
        return "price"
    return "financial"


def _detect_metric(q: str) -> tuple[str, str, str] | None:
    ql = q.lower()
    for kw in _METRIC_ORDER:
        if kw in ql:
            return METRIC_COLUMNS[kw]
    return None


def _detect_dir(q: str) -> str:
    ql = q.lower()
    if any(w in ql for w in _ASC_WORDS):
        return "ASC"
    if any(w in ql for w in _DESC_WORDS):
        return "DESC"
    return "DESC"


def _detect_n(q: str, default: int = 10) -> int:
    m = re.search(r"(\d+)\s*(개|곳|위|종목|companies|개사)?", q)
    if m:
        return max(1, min(50, int(m.group(1))))
    return default


def _detect_sector(q: str) -> str | None:
    for s in _SECTORS:
        if s in q:
            return s
    return None


def _detect_name(q: str) -> str | None:
    for n in _NAMES:
        if n in q:
            return n
    return None


def heuristic_sql(question: str, route: str) -> str:
    """키워드 기반 폴백 SQL. 매칭 실패 시 안전한 기본 쿼리."""
    metric = _detect_metric(question)
    direction = _detect_dir(question)
    n = _detect_n(question)
    sector = _detect_sector(question)
    name = _detect_name(question)

    if metric is None:
        # 지표 미상 → 회사 목록
        where = f"WHERE name = '{name}'" if name else ""
        return f"SELECT stock_code, name, market, sector FROM company {where} LIMIT {n}".strip()

    table, col, _ = metric

    if table == "metrics":
        conds = [f"m.{col} IS NOT NULL"]
        if sector:
            conds.append(f"c.sector = '{sector}'")
        if name:
            conds.append(f"c.name = '{name}'")
        where = "WHERE " + " AND ".join(conds)
        return (
            f"SELECT c.name, m.{col} FROM metrics m "
            f"JOIN company c ON c.stock_code = m.stock_code "
            f"{where} ORDER BY m.{col} {direction} LIMIT {n}"
        )

    if table == "prices":
        conds = ["p.date = (SELECT MAX(date) FROM prices)", f"p.{col} IS NOT NULL"]
        if sector:
            conds.append(f"c.sector = '{sector}'")
        if name:
            conds.append(f"c.name = '{name}'")
        where = "WHERE " + " AND ".join(conds)
        return (
            f"SELECT c.name, p.{col} FROM prices p "
            f"JOIN company c ON c.stock_code = p.stock_code "
            f"{where} ORDER BY p.{col} {direction} LIMIT {n}"
        )

    # financials
    conds = [
        f"f.account_key = '{col}'",
        "f.quarter = (SELECT MAX(quarter) FROM financials)",
    ]
    if sector:
        conds.append(f"c.sector = '{sector}'")
    if name:
        conds.append(f"c.name = '{name}'")
    where = "WHERE " + " AND ".join(conds)
    return (
        f"SELECT c.name, f.amount FROM financials f "
        f"JOIN company c ON c.stock_code = f.stock_code "
        f"{where} ORDER BY f.amount {direction} LIMIT {n}"
    )
