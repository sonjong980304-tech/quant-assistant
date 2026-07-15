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
3) DART·FnGuide 둘 다에 값이 있으면 DART 값을 우선 채택하고,
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
_METRICS_TABLE_COLS = {"per", "pbr", "roe", "operating_margin", "debt_ratio", "market_cap"}

# 연간(annual) 요청 시 4개 분기를 합산해야 하는 손익계산서/현금흐름 흐름값 계정.
# 재무상태표 잔액(total_assets/total_liabilities/total_equity/shares_outstanding)이나
# 비율지표(per/pbr/roe 등)는 합산 대상이 아니라 연말(Q4) 스냅샷을 쓴다(잔액을 더하면 틀림).
_SUMMABLE_FLOW_ACCOUNTS: frozenset[str] = frozenset({
    "revenue", "cost_of_sales", "gross_profit", "sga", "operating_profit",
    "net_income", "interest_expense", "operating_cashflow", "depreciation", "dividend",
})


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
) -> tuple[float, str] | None:
    """특정 분기(예: '2025Q3')의 DART 값 — financials(EAV) 우선, 없으면 metrics 컬럼."""
    row = conn.execute(
        "SELECT amount, quarter FROM financials WHERE stock_code=? AND account_key=? AND quarter=?",
        (stock_code, key, quarter),
    ).fetchone()
    if row is not None and row["amount"] is not None:
        return row["amount"], row["quarter"]
    if key in _METRICS_TABLE_COLS:
        row = conn.execute(
            f"SELECT {key} AS v, quarter FROM metrics WHERE stock_code=? AND quarter=?",
            (stock_code, quarter),
        ).fetchone()
        if row is not None and row["v"] is not None:
            return row["v"], row["quarter"]
    return None


def _fetch_dart(
    conn: sqlite3.Connection,
    stock_code: str,
    metric: str,
    period: dict | None = None,
) -> tuple[float, str] | None:
    """DART 값 조회: 먼저 financials(EAV), 없으면 metrics(계산 지표).

    반환: (value, quarter) — quarter는 조회된 값이 실제로 어느 기간 것인지 항상 함께
    돌려준다. 이 라벨이 없으면 총괄 에이전트의 검증 단계가 "26년 1분기 영업이익"처럼
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
            return row["amount"], row["quarter"]
        if key in _METRICS_TABLE_COLS:
            row = conn.execute(
                f"SELECT {key} AS v, quarter FROM metrics WHERE stock_code=? "
                "ORDER BY quarter DESC LIMIT 1",
                (stock_code,),
            ).fetchone()
            if row is not None and row["v"] is not None:
                return row["v"], row["quarter"]
        return None

    if period.get("kind") == "quarter":
        return _fetch_dart_at_quarter(conn, key, stock_code, period["quarter"])

    # kind == "annual"
    year = period["year"]
    if key in _SUMMABLE_FLOW_ACCOUNTS:
        quarters = [f"{year}Q{i}" for i in (1, 2, 3, 4)]
        placeholders = ",".join("?" for _ in quarters)
        rows = conn.execute(
            f"SELECT quarter, amount FROM financials WHERE stock_code=? AND account_key=? "
            f"AND quarter IN ({placeholders})",
            (stock_code, key, *quarters),
        ).fetchall()
        amounts = {r["quarter"]: r["amount"] for r in rows if r["amount"] is not None}
        if len(amounts) == 4:  # 4개 분기 모두 있을 때만 합산(추정 금지)
            return sum(amounts.values()), f"{year} 연간"
        return None
    # 흐름값이 아닌 계정(잔액/비율)은 연말(Q4) 스냅샷.
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

    반환: {"stock_code", "metric", "value", "source", "period"}.
    - DART·FnGuide 둘 다 값이 있으면 DART 값을 우선 채택한다(rule #3).
    - 한쪽에만 값이 있으면 그 소스를 쓴다.
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
        value, period, source = dart_result[0], dart_result[1], DART
    elif fnguide_result is not None:
        value, period, source = fnguide_result[0], fnguide_result[1], FNGUIDE
    else:
        value, period, source = None, None, chosen

    return {
        "stock_code": stock_code, "metric": metric,
        "value": value, "source": source, "period": period,
    }
