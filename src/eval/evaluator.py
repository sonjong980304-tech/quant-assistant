"""3층 평가 로직.

Layer1 Execution Accuracy : 생성 SQL 결과 == 정답 SQL 결과 (denotation 비교)
Layer2 LLM-as-Judge       : SQL 적절성 1~5점 (LLM 없으면 미적용)
Layer3 실행 가능성         : 문법(안전 SELECT) / 실행 성공 / 빈결과 / 스키마 유효성
"""
from __future__ import annotations

from ..llm import extract_json
from ..sql_exec import is_safe_select, run_select
from ..legacy.graph import prompts


# ---------------------------------------------------------------------------
# 결과셋 비교 (denotation accuracy)
# ---------------------------------------------------------------------------
def _norm_value(v):
    # SQL 생성 프롬프트/goldset이 ROUND(x,2) 관례를 쓰므로 비교도 2자리로 맞춘다
    # (자릿수가 다르면 부동소수 오차로 오탐 불일치가 난다).
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, int):
        return round(float(v), 2)
    return v


def _sort_key(v):
    nv = _norm_value(v)
    # 타입 혼합 안전 정렬: 숫자 먼저, 그다음 문자열
    return (0, nv) if isinstance(nv, (int, float)) else (1, str(nv))


def _row_signature(row: dict) -> tuple:
    # 표준 denotation 비교: 컬럼명/별칭/순서는 무시하고 값만 비교한다.
    return tuple(sorted((_norm_value(v) for v in row.values()), key=_sort_key))


def compare_resultsets(pred: list[dict], gold: list[dict], order_sensitive: bool = True) -> bool:
    """예측/정답 결과 비교.

    order_sensitive=True : 행 순서까지 비교 (상위 N 정렬 질의에 적합)
    False                : multiset 비교 (순서 무관)
    """
    if len(pred) != len(gold):
        return False
    if order_sensitive:
        return [_row_signature(r) for r in pred] == [_row_signature(r) for r in gold]
    from collections import Counter

    return Counter(_row_signature(r) for r in pred) == Counter(_row_signature(r) for r in gold)


# ---------------------------------------------------------------------------
# Layer 3 — 실행 가능성
# ---------------------------------------------------------------------------
def eval_layer3(state: dict, deps) -> dict:
    sql = state.get("sql", "")
    safe, reason = is_safe_select(sql)
    executed = state.get("error") is None and safe
    nonempty = state.get("row_count", 0) > 0
    passed = bool(safe and executed)
    return {
        "safe_select": safe,
        "executed": executed,
        "nonempty": nonempty,
        "passed": passed,
        "reason": reason or state.get("error") or "",
    }


# ---------------------------------------------------------------------------
# Layer 2 — LLM-as-Judge
# ---------------------------------------------------------------------------
def eval_layer2(state: dict, deps) -> dict:
    if not deps.llm.available:
        return {"applicable": False, "score": None, "reason": "LLM 미사용"}
    rows = state.get("rows", [])
    sample = rows[:3]
    res = deps.llm.complete(
        prompts.JUDGE_USER.format(
            schema=deps.schema,
            question=state.get("question", ""),
            sql=state.get("sql", ""),
            row_count=state.get("row_count", 0),
            columns=state.get("columns", []),
            sample=sample,
        ),
        system=prompts.JUDGE_SYSTEM,
        role="judge",
        max_tokens=300,
    )
    if not res.ok:
        return {"applicable": False, "score": None, "reason": f"judge 실패: {res.error}"}
    data = extract_json(res.text)
    score = data.get("score")
    try:
        score = int(score)
        score = max(1, min(5, score))
    except (TypeError, ValueError):
        score = None
    return {"applicable": True, "score": score, "reason": data.get("reason", "")}


# ---------------------------------------------------------------------------
# Layer 1 — Execution Accuracy
# ---------------------------------------------------------------------------
def eval_layer1(state: dict, deps) -> dict:
    gold = state.get("gold_sql")
    if not gold:
        return {"applicable": False, "match": None}
    gold_res = run_select(deps.conn, gold)
    if not gold_res["ok"]:
        # 골드 SQL 실행실패는 모델 오답이 아니라 평가 하네스(골드셋/DB 정합성) 결함이므로
        # EX 분모(applicable)에서 제외한다. gold_error로 표시해 리포트에서 드러나게 한다.
        return {
            "applicable": False,
            "match": None,
            "gold_error": True,
            "reason": f"정답 SQL 실행실패: {gold_res['error']}",
        }
    pred_rows = state.get("rows", [])
    # 1차 판정은 순서 무관(멀티셋) 비교: goldset 질문은 대부분 "Top-N" 랭킹 질의라
    # 채점 목적은 올바른 N개 행을 뽑았는지이지, 동점 타이브레이크 순서 일치가 아니다.
    match = compare_resultsets(pred_rows, gold_res["rows"], order_sensitive=False)
    # 순서까지 완전히 일치하는지는 보조 진단 필드로만 남긴다.
    order_exact_match = compare_resultsets(pred_rows, gold_res["rows"], order_sensitive=True)
    return {
        "applicable": True,
        "match": match,
        "order_exact_match": order_exact_match,
        "gold_row_count": gold_res["row_count"],
        "pred_row_count": len(pred_rows),
    }


# ---------------------------------------------------------------------------
# 종합 (eval_node에서 호출)
# ---------------------------------------------------------------------------
def evaluate_state(state: dict, deps) -> dict:
    l3 = eval_layer3(state, deps)
    l2 = eval_layer2(state, deps)
    l1 = eval_layer1(state, deps)
    parts = [f"L3 {'통과' if l3['passed'] else '실패'}"]
    if l1.get("applicable"):
        parts.append(f"L1 {'정답' if l1['match'] else '오답'}")
    if l2.get("applicable"):
        parts.append(f"L2 {l2['score']}점")
    return {"layer1": l1, "layer2": l2, "layer3": l3, "summary": " | ".join(parts)}
