"""신규 계층형 아키텍처(run_hierarchical) 용 goldset 재실행 러너 (HA-14, AC18/AC20).

기존 legacy 평가(src/eval/runner.py.run_evaluation)는 손대지 않는다 — 그것이 'legacy
baseline' 역할을 그대로 유지한다(레거시 6단계 파이프라인 기준 EX 정확도). 이 파일은 같은
goldset 을 **신규 계층형 그래프(run_hierarchical)** 로 재실행하고, 그 결과를 정답 SQL 실행
결과와 대조해 정확도를 매기는 별도 신규 경로다.

판정 방식(AC18):
  legacy 는 결과셋 denotation 비교(evaluator.compare_resultsets)가 가능하지만, 신규 구조의
  산출물은 결과셋이 아니라 종합결론(conclusion) + 도메인별 원본결과(domain_results)라 직접
  행 비교가 불가능하다. 그래서 기존 Layer2(LLM judge, evaluator.eval_layer2) 프롬프트 스타일을
  재사용해 "정답 SQL 실행 결과"와 "신규 구조 답변"의 일치 여부를 LLM 에게 판정시킨다.

원본 DB 보호(AC18): 평가는 항상 원본 DB 의 격리 사본(runner._isolated_copy)에서 실행한다 —
새로 발명하지 않고 기존 온라인 백업 패턴을 그대로 import 해서 쓴다.

성능 실측(AC20): count_llm_calls 로 llm_fn 호출 횟수를 세고, 문항별 응답시간을 측정한다.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from ..llm import extract_json
from ..sql_exec import run_select
from .runner import _cleanup_copy, _isolated_copy


class CountingLLM:
    """llm_fn(Callable[[str], str])을 감싸 호출 횟수를 세는 얇은 카운터(성능 실측용, AC20).

    과설계 없이 호출 카운트만 센다. reset()으로 문항 경계마다 카운트를 0으로 되돌린다.
    llm_fn 이 None 이면 wrap 대상이 없으므로 count 는 항상 0으로 남는다.
    """

    def __init__(self, llm_fn: Callable[[str], str] | None):
        self._fn = llm_fn
        self.count = 0

    def __call__(self, prompt: str) -> str:
        self.count += 1
        if self._fn is None:
            return ""
        return self._fn(prompt)

    def reset(self) -> None:
        self.count = 0


def _hier_answer_text(hier: dict) -> str:
    """신규 구조 결과(dict)를 judge 프롬프트에 넣을 요약 텍스트로 만든다.

    conclusion(종합결론) + routes + 도메인별 원본결과를 함께 담는다 — 종합결론만 보면
    빈약할 수 있어 원본 도메인 데이터도 근거로 제공한다(단, 지나치게 길지 않게 자른다).
    """
    conclusion = hier.get("conclusion") or ""
    if hier.get("uncertain"):
        conclusion = f"[불확실] {hier.get('reason') or ''}"
    domain = str(hier.get("domain_results") or {})
    if len(domain) > 2000:  # judge 프롬프트 폭주 방지(원본 도메인 데이터가 클 수 있음)
        domain = domain[:2000] + "…(생략)"
    return f"결론: {conclusion}\n라우팅: {hier.get('routes')}\n도메인결과: {domain}"


def judge_hierarchical_answer(
    question: str,
    gold_rows: list[dict],
    hier: dict,
    judge_llm_fn: Callable[[str], str] | None,
    sample_n: int = 8,
) -> dict:
    """정답 SQL 결과(gold_rows)와 신규 구조 답변(hier)의 일치 여부를 LLM judge 로 판정한다.

    기존 evaluator.eval_layer2 의 judge 프롬프트 스타일(스키마/질문/샘플 제공 + JSON 반환)을
    재사용한다. judge_llm_fn 이 None 이면 판정 불가(applicable=False).

    반환: {"applicable": bool, "match": bool|None, "reason": str}.
    """
    if judge_llm_fn is None:
        return {"applicable": False, "match": None, "reason": "judge LLM 미가용"}
    gold_sample = gold_rows[:sample_n]
    prompt = (
        "당신은 Text-to-SQL 계층형 에이전트의 답변을 채점하는 평가자입니다.\n"
        "아래 '정답'은 정답 SQL 을 실행한 결과(행 목록)이고, '시스템 답변'은 신규 계층형\n"
        "구조가 낸 종합결론과 도메인별 원본결과입니다. 시스템 답변이 정답과 사실상 같은\n"
        "정보를 담고 있으면(핵심 종목/수치/순위가 일치) match=true, 아니면 match=false 입니다.\n"
        '반드시 JSON 으로만 답하세요: {"match": true/false, "reason": "간단한 이유"}\n\n'
        f"질문: {question}\n"
        f"정답(정답 SQL 실행 결과, 최대 {sample_n}행): {gold_sample}\n"
        f"정답 총 행수: {len(gold_rows)}\n"
        f"시스템 답변: {_hier_answer_text(hier)}\n답:"
    )
    try:
        raw = judge_llm_fn(prompt) or ""
    except Exception as exc:  # noqa: BLE001 — judge 실패는 판정불가로 흡수(문항 하나 때문에 배치 중단 방지)
        return {"applicable": False, "match": None, "reason": f"judge 호출 실패: {type(exc).__name__}"}
    data = extract_json(raw)
    if not isinstance(data, dict) or "match" not in data:
        return {"applicable": False, "match": None, "reason": f"judge 응답 파싱 실패: {raw[:150]}"}
    return {"applicable": True, "match": bool(data["match"]), "reason": str(data.get("reason") or "")}


def run_hierarchical_eval(
    items: list[dict],
    run_hier_fn: Callable[..., dict],
    llm_fn: Callable[[str], str] | None,
    judge_llm_fn: Callable[[str], str] | None,
    db_path: str | None = None,
    connect_ro_fn: Callable[[str], Any] | None = None,
) -> dict:
    """goldset items 를 신규 계층형 구조로 재실행하고 정확도/성능을 집계한다.

    각 문항:
      (a) 정답 SQL 을 격리 사본에서 실행해 기대 결과(gold_rows)를 얻고,
      (b) 같은 사본 읽기전용 연결에 run_hier_fn(question, conn, llm_fn=...) 을 실행하고,
      (c) judge_hierarchical_answer 로 (a)와 (b)의 일치를 판정한다.
    동시에 문항별 응답시간(초)과 llm_fn 호출횟수(CountingLLM)를 실측한다(AC20).

    run_hier_fn/connect_ro_fn 은 주입 가능(단위테스트에서 실제 그래프/DB 없이 검증).
    반환: {"n", "judged", "match", "accuracy_pct", "avg_latency_s", "avg_llm_calls", "rows": [...]}.
    """
    if connect_ro_fn is None:
        from ..db import connect_readonly

        connect_ro_fn = connect_readonly

    copy_path = _isolated_copy(db_path)
    counter = CountingLLM(llm_fn)
    rows: list[dict] = []
    try:
        gold_conn = connect_ro_fn(copy_path)
        try:
            for item in items:
                gold_res = run_select(gold_conn, item["sql"])
                gold_rows = gold_res["rows"] if gold_res["ok"] else []

                counter.reset()
                conn = connect_ro_fn(copy_path)
                t0 = time.perf_counter()
                try:
                    hier = run_hier_fn(item["question"], conn, llm_fn=counter)
                except Exception as exc:  # noqa: BLE001 — 한 문항 실패가 배치를 무너뜨리지 않게
                    hier = {"uncertain": True, "reason": f"실행예외: {type(exc).__name__}: {exc}"}
                finally:
                    latency = time.perf_counter() - t0
                    try:
                        conn.close()
                    except Exception:  # noqa: BLE001
                        pass

                verdict = judge_hierarchical_answer(
                    item["question"], gold_rows, hier, judge_llm_fn
                )
                rows.append({
                    "id": item.get("id"),
                    "question": item["question"],
                    "tags": item.get("tags"),
                    "gold_ok": gold_res["ok"],
                    "gold_row_count": gold_res["row_count"] if gold_res["ok"] else None,
                    "uncertain": bool(hier.get("uncertain")),
                    "routes": hier.get("routes"),
                    "judge_applicable": verdict["applicable"],
                    "match": verdict["match"],
                    "reason": verdict["reason"],
                    "latency_s": round(latency, 2),
                    "llm_calls": counter.count,
                })
        finally:
            gold_conn.close()
    finally:
        _cleanup_copy(copy_path)

    judged = [r for r in rows if r["judge_applicable"]]
    matches = [r for r in judged if r["match"]]
    latencies = [r["latency_s"] for r in rows]
    calls = [r["llm_calls"] for r in rows]
    return {
        "n": len(items),
        "judged": len(judged),
        "match": len(matches),
        "accuracy_pct": round(100 * len(matches) / len(judged), 1) if judged else None,
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "avg_llm_calls": round(sum(calls) / len(calls), 2) if calls else None,
        "rows": rows,
    }
