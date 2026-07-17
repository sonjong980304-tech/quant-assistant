"""HA-5: 백테스트 검증 배선 — src/backtest/auditor.py의 하드/소프트 판정 재사용.

.omc/state/sessions/f6b79d90-b7b2-4b4f-b4f7-405a8355724e/prd.json의 HA-5 참고.

이 모듈은 auditor.py의 판정 로직(하드차단 3종/소프트경고 4종)을 **한 줄도 수정하지 않고**,
새 계층형 아키텍처의 백테스트 도메인 에이전트(HA-9)가 쓸 수 있는 오케스트레이션 함수로
감싸는 순수 배선(wiring)이다. 기존 레거시 경로(src/graph/nodes.py의 _pipeline_execute,
234~304번째 줄)가 pre_audit → run_pipeline → post_audit 순서로 호출하고, 하드차단 시
정상 결과를 폐기, 소프트경고는 결과에 첨부하는 것과 동일한 정신을 새 구조에 맞게
함수 하나로 재구성했다(코드를 그대로 복사하지 않음).

실제 백테스트 실행 자체는 이 스토리의 몫이 아니다(HA-9가 담당) — run_pipeline_fn으로
실행을 주입받는다. 시그니처는 auditor.pre_audit이 이미 요구하는 것과 동일하다:
(steps, conn=...) -> dict — 사전검사의 $ref 비중 해석과 본 실행 양쪽에 그대로 재사용된다.
"""
from __future__ import annotations

from typing import Callable

from ..backtest import auditor

# 검사(7대죄악) 코드 → 실시간 트리용 한글 라벨. auditor.py의 "sin" 값과 정확히 일치해야 한다.
_SIN_LABELS_KO: dict[str, str] = {
    "survivorship": "생존편향",
    "lookahead": "미래참조",
    "short_positions": "공매도",
    "storytelling": "스토리텔링",
    "snooping": "데이터스누핑",
    "signal_decay": "신호감소",
    "outlier": "이상치",
}


def _report_hard_verdict(verdict: dict, on_progress: Callable[[str, str], None]) -> None:
    label = _SIN_LABELS_KO.get(verdict.get("sin"), verdict.get("sin", "?"))
    if verdict.get("blocked"):
        on_progress("audit", f"{label} 검사: 차단({verdict.get('reason', '')})")
    else:
        on_progress("audit", f"{label} 검사: 통과")


def _report_soft_verdict(verdict: dict, on_progress: Callable[[str, str], None]) -> None:
    label = _SIN_LABELS_KO.get(verdict.get("sin"), verdict.get("sin", "?"))
    if verdict.get("triggered"):
        on_progress("audit", f"{label} 검사: ⚠ 경고 - {verdict.get('message', '')}")
    else:
        on_progress("audit", f"{label} 검사: 통과")


def run_backtest_with_audit(
    steps: list[dict],
    conn,
    question: str,
    run_pipeline_fn: Callable,
    llm_fn: Callable | None = None,
    market: str = "KR",
    on_progress: Callable[[str, str], None] | None = None,
) -> dict:
    """백테스트를 감사 레이어로 감싸 실행한다: 사전(공매도) → 실행 → 사후(생존/미래참조+소프트경고 4종).

    흐름(레거시 src/graph/nodes.py._pipeline_execute와 동일한 정신):
    1) auditor.pre_audit으로 공매도(음수 비중) 사전 하드차단 — 걸리면 run_pipeline_fn을
       아예 호출하지 않는다(AC11).
    2) run_pipeline_fn(steps, conn=conn)으로 실제 백테스트를 실행한다.
    3) auditor.post_audit으로 생존편향/미래참조 사후 하드차단(AC11)을 수행하고, 통과 시
       소프트경고 4종(run_soft_inspectors)을 호출한다(AC12).

    반환:
        하드차단(사전 또는 사후) 시: {"blocked": True, "error": str, "result": None,
                                    "hard": [차단된 verdict...], "warnings": []}
        통과 시: {"blocked": False, "error": None, "result": dict(백테스트 결과),
                 "hard": [...], "warnings": [triggered된 소프트 verdict...]}

    안전 원칙(fail-closed, SoT §3.4/§3.5·AC9/AC11): 하드차단 검사(사전 공매도 / 사후
    생존편향·미래참조)가 그 자체로 예외를 던져 죽으면, "차단 안 됨=통과"로 새지 않고
    반드시 **안전측(결과 폐기·차단)**으로 귀결한다. 검사기가 고장 났을 때 검증 안 된 결과를
    통과시키는 것은 이 3대 하드 규칙의 존재 목적(방어적 안전망)과 정면 충돌하기 때문이다.
    (소프트경고 4종은 보조 자문이라 실패해도 정상 결과를 폐기하지 않는 fail-open이 맞다 —
    run_soft_inspectors가 자체 타임아웃/오류를 내부에서 흡수한다. 반면 하드차단은 fail-closed.)
    """
    try:
        pre = auditor.pre_audit(steps, conn, run_pipeline_fn)
    except Exception as exc:  # noqa: BLE001 — fail-closed: 사전 하드검사가 죽으면 안전측(차단)
        pre = {"sin": "short_positions", "blocked": True,
               "reason": f"사전 하드검사(공매도) 내부 오류 — fail-closed 차단: {exc}",
               "evidence": []}
    if pre and on_progress:
        _report_hard_verdict(pre, on_progress)
    if pre and pre.get("blocked"):
        return {"blocked": True, "error": auditor.format_hard_block([pre]),
                "result": None, "hard": [pre], "warnings": []}

    result = run_pipeline_fn(steps, conn=conn)

    try:
        audit = auditor.post_audit(result, conn, question, llm_fn, market=market)
    except Exception as exc:  # noqa: BLE001 — fail-closed: 사후 하드검사가 죽으면 안전측(차단)
        audit = {"blocked": True,
                 "hard": [{"sin": "hard_audit_error", "blocked": True,
                           "reason": f"사후 하드검사(생존편향/미래참조) 내부 오류 — "
                                     f"fail-closed 차단: {exc}",
                           "evidence": []}],
                 "soft": []}

    if on_progress:
        for v in audit.get("hard", []):
            _report_hard_verdict(v, on_progress)

    if audit.get("blocked"):
        return {"blocked": True, "error": auditor.format_hard_block(audit.get("hard", [])),
                "result": None, "hard": audit.get("hard", []), "warnings": []}

    if on_progress:
        for v in audit.get("soft", []):
            _report_soft_verdict(v, on_progress)

    warnings = [v for v in audit.get("soft", []) if v.get("triggered")]
    return {"blocked": False, "error": None, "result": result,
            "hard": audit.get("hard", []), "warnings": warnings}
