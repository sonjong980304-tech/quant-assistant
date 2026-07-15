"""종목 선정 로직 (백테스트 핵심).

지표 선정: 복수 지표 + 방향(낮을수록/높을수록 우수) + 가중치.
조합 방식 3가지:
  - 'and'      : 각 지표 상위군 교집합 (AND 필터)
  - 'rank_sum' : 각 지표 순위 합 (마법공식류, 낮을수록 우수)
  - 'zscore'   : 가중 z-score 합 (낮을수록 우수로 통일)

N/A(결측·적자) 종목은 선정 지표가 비면 제외한다.
산업 필터(sectors)와 종목 수(n)를 지원한다.
"""
from __future__ import annotations

import numpy as np

# criteria 예: [{"key": "per", "direction": "low", "weight": 0.5},
#               {"key": "roe", "direction": "high", "weight": 0.5}]


def _validate_criteria_keys(rows: list[dict], criteria: list[dict]) -> None:
    """criteria의 key가 실제 rows에 존재하는 필드인지 검증한다.

    LLM이 존재하지 않는 필드명(예: forward_12m_return)을 지어내면, 검증 없이는
    .get()이 매 행마다 None을 반환해 모든 종목이 조용히 제외되고 만다(에러 없이
    빈 결과). 화이트리스트를 별도로 두지 않고 rows[0]의 실제 키를 정답으로 쓴다
    (get_cross_section/metrics_at의 출력 스키마와 항상 동기화되도록).
    """
    if not rows:
        return
    valid_fields = set(rows[0].keys())
    unknown = [c["key"] for c in criteria if c["key"] not in valid_fields]
    if unknown:
        raise ValueError(
            f"존재하지 않는 필드: {unknown}. 사용 가능한 필드: {sorted(valid_fields)}"
        )


def _filter_valid(rows: list[dict], criteria: list[dict], sectors=None, markets=None) -> list[dict]:
    _validate_criteria_keys(rows, criteria)
    keys = [c["key"] for c in criteria]
    out = []
    for r in rows:
        if sectors and r.get("sector") not in sectors:
            continue
        if markets and r.get("market") not in markets:  # 시장(KOSPI/KOSDAQ) 필터
            continue
        if any(r.get(k) is None for k in keys):
            continue
        out.append(r)
    return out


def select_stocks(
    rows: list[dict],
    criteria: list[dict],
    combine: str = "zscore",
    n: int = 20,
    sectors=None,
    markets=None,
) -> list[dict]:
    """선정된 종목 리스트(점수 우수순)를 반환. markets=['KOSPI','KOSDAQ'] 또는 None(전체)."""
    valid = _filter_valid(rows, criteria, sectors, markets)
    if not valid or not criteria:
        return []

    if combine == "and":
        survivors = {r["stock_code"] for r in valid}
        for c in criteria:
            ranked = sorted(valid, key=lambda r: r[c["key"]], reverse=(c["direction"] == "high"))
            cutoff = max(n, len(ranked) // 2)
            top = {r["stock_code"] for r in ranked[:cutoff]}
            survivors &= top
        result = [r for r in valid if r["stock_code"] in survivors]
        result.sort(key=lambda r: r[criteria[0]["key"]], reverse=(criteria[0]["direction"] == "high"))
        return result[:n]

    # rank_sum / zscore : "점수 낮을수록 우수"로 통일
    scores = {r["stock_code"]: 0.0 for r in valid}
    total_w = sum(c.get("weight", 1.0) for c in criteria) or 1.0
    for c in criteria:
        key, high = c["key"], (c["direction"] == "high")
        w = c.get("weight", 1.0) / total_w
        vals = np.array([r[key] for r in valid], dtype=float)
        if combine == "rank_sum":
            # 0=best. high면 큰 값이 best.
            comp = np.argsort(np.argsort(-vals if high else vals)).astype(float)
        else:  # zscore
            mu, sd = vals.mean(), (vals.std() or 1.0)
            z = (vals - mu) / sd
            comp = (-z if high else z)  # 낮을수록 우수
        for r, cv in zip(valid, comp):
            scores[r["stock_code"]] += w * cv

    ranked = sorted(valid, key=lambda r: scores[r["stock_code"]])
    for r in ranked:
        r["_score"] = round(scores[r["stock_code"]], 4)
    return ranked[:n]
