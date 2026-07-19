"""백테스트 "퀀트 7대 죄악" 자동 감사 레이어.

.omc/specs/brainstorming-backtest-auditor-7-sins.md 참고.

이 모듈은 기존 실행/안전 로직(pipeline_exec.py의 고정 dict 디스패치·상한검사,
run_backtest_primitive의 사전검사) 위에 얹는 "감사 레이어"다. 기존 안전장치는 전혀
수정하지 않고, 백테스트 실행 전후에 자동으로 개입해 편향을 검사한다.

판정 기준은 **탐지 신뢰도**다(스펙 §3.2):
- 결정론적 코드로 100% 확실히 잡히는 3종 → 하드차단(실행 차단 + 사유·증거 반환).
    ① 생존편향   check_survivorship  — delisting_date <= asof인 죽은 종목이 보유에 있으면 차단
    ② 미래참조   check_lookahead     — 사용된 재무의 disclosed_date > asof면 차단
    ⑦ 공매도비용 check_short_positions — 비중 벡터에 음수가 있으면 차단
- LLM의 주관적 판단이 필요해 오판 위험이 있는 4종 → 소프트경고(결과에 첨부만).
    ③ 스토리텔링 inspect_storytelling
    ④ 데이터스누핑 inspect_snooping   — 사용자 원본 질문을 "사전 등록 가설"로 취급
    ⑤ 신호감소/회전율 inspect_signal_decay
    ⑥ 이상치제어 inspect_outlier

하드차단 검사 결과 구조: {"sin", "blocked": bool, "reason": str, "evidence": list}
소프트경고 검사 결과 구조: {"sin", "triggered": bool, "message": str}

LLM 호출은 이 프로젝트 기존 DI 관례대로 주입 가능한 llm_fn(prompt)->str로 분리해,
실제 네트워크 없이 mock으로 단위테스트가 가능하다(fama_french.py/primitives.py와 동일).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Callable

from ..llm import extract_json
from .data_access import _is_alive, effective_quarter_at
from .pipeline_exec import MAX_TIMEOUT  # 새 상수 만들지 않고 기존 타임아웃 상한 재사용


# ==========================================================================
# 하드차단 검사 3종 (LLM 없음 · 결정론적)
# ==========================================================================
def check_survivorship(conn, holdings: list[dict], is_alive_fn: Callable | None = None,
                       market: str = "KR") -> dict:
    """① 생존편향 하드차단: 보유종목 중 그 시점 기준 이미 상장폐지된 종목이 있으면 차단한다.

    각 리밸런싱 시점(holdings의 date=asof)에 보유한 종목이 그 시점에 실제로 살아있었는지를
    상장폐지 테이블로 재검증한다. 정상 경로에서는 metrics_at의 _is_alive가
    이미 죽은 종목을 걸러내므로 사실상 항상 통과한다 — 이 함수는 그 가드가 회귀로 깨졌을 때를
    잡는 **방어적 이중화 안전망**이다(스펙 §3.2 ①). 위반이 하나라도 있으면 blocked=True로
    사유·증거(종목코드)를 반환.

    기본 판정 함수는 data_access._is_alive(delisting 테이블)이다(bool 반환).
    is_alive_fn을 명시 주입하면 그것을 우선한다(테스트 주입용).
    """
    if is_alive_fn is None:
        is_alive_fn = _is_alive
    evidence = []
    for h in holdings or []:
        asof = h.get("date")
        for code in h.get("codes", []) or []:
            if not is_alive_fn(conn, code, asof):
                evidence.append({"stock_code": code, "asof": asof})
    blocked = bool(evidence)
    reason = (
        f"상장폐지된 종목이 백테스트 보유에 포함됨(생존편향): {len(evidence)}건"
        if blocked else ""
    )
    return {"sin": "survivorship", "blocked": blocked, "reason": reason, "evidence": evidence}


def check_lookahead(conn, holdings: list[dict], quarter_fn: Callable | None = None) -> dict:
    """② 미래참조편향 하드차단: 사용된 재무의 공시일(disclosed_date)이 조회시점(asof)보다
    미래면 차단한다.

    각 보유종목에 대해 그 시점 유효분기(quarter_fn=effective_quarter_at)를 구한 뒤, 그 분기
    재무의 실제 disclosed_date가 asof보다 미래인지 독립적으로 재검증한다. effective_quarter_at은
    disclosed_date<=asof만 고르므로 정상 경로에서는 항상 통과한다 — 그 가드가 깨졌을 때
    (예: disclosed_date를 무시하는 코드로 회귀) 잡아내는 **방어적 이중화**다(스펙 §3.2 ②).

    quarter_fn은 테스트 주입용(기본=data_access.effective_quarter_at).
    """
    quarter_fn = quarter_fn or effective_quarter_at
    evidence = []
    for h in holdings or []:
        asof = h.get("date")
        for code in h.get("codes", []) or []:
            q = quarter_fn(conn, code, asof)
            if not q:
                continue
            # 그 분기가 asof 까지 실제로 공시됐는지 재검증한다. 정정공시가 있으면 financials 의
            # disclosed_date 는 정정일(미래)로 바뀌므로, financials 만 보면 정정 전 asof 에서
            # 정상 경로(effective_quarter_at=revision 기반)를 오탐(false-positive)한다. 정정이력을
            # 보존하는 financials_revision 과 financials 를 UNION 해 "가장 이른 공시일"이 asof 보다
            # 뒤일 때만(=그 분기를 asof 시점엔 어떤 버전으로도 알 수 없었을 때만) 차단한다.
            row = conn.execute(
                "SELECT MIN(d) FROM ("
                "  SELECT MIN(disclosed_date) AS d FROM financials_revision "
                "    WHERE stock_code=? AND quarter=? AND disclosed_date IS NOT NULL "
                "  UNION ALL "
                "  SELECT MIN(disclosed_date) AS d FROM financials "
                "    WHERE stock_code=? AND quarter=? AND disclosed_date IS NOT NULL "
                ")",
                (code, q, code, q),
            ).fetchone()
            disclosed = row[0] if row else None
            if disclosed and asof and disclosed > asof:
                evidence.append({"stock_code": code, "quarter": q,
                                 "disclosed_date": disclosed, "asof": asof})
    blocked = bool(evidence)
    reason = (
        f"공시 전 재무가 백테스트에 사용됨(미래참조편향): {len(evidence)}건"
        if blocked else ""
    )
    return {"sin": "lookahead", "blocked": blocked, "reason": reason, "evidence": evidence}


def check_short_positions(weights) -> dict:
    """⑦ 공매도비용 하드차단: 비중 벡터에 음수 값이 하나라도 있으면 차단한다.

    optimize_weights 등으로 계산된 비중은 데이터에 따라 매 실행마다 달라지므로, 이 검사는
    방어적 이중화가 아니라 **매 실행마다 실제로 필요한** 검사다(스펙 §3.2 ⑦). engine의
    _run_weighted_backtest도 음수 비중을 거부하지만, 여기서 사전에 사유·증거(종목·비중값)를
    명시적으로 만들어 실행 자체를 막는다. weights는 {종목:비중} dict 또는 (종목,비중) 시퀀스.
    """
    if isinstance(weights, dict):
        items = list(weights.items())
    else:
        items = list(weights or [])
    evidence = [
        {"asset": a, "weight": w}
        for a, w in items
        if isinstance(w, (int, float)) and w < 0
    ]
    blocked = bool(evidence)
    reason = (
        f"음수 비중(공매도)이 포함됨 — engine은 공매도를 표현할 수 없음: {len(evidence)}건"
        if blocked else ""
    )
    return {"sin": "short_positions", "blocked": blocked, "reason": reason, "evidence": evidence}


# ==========================================================================
# 소프트경고 검사관 4종 (LLM 판정 · DI로 mock 가능)
# ==========================================================================
_SOFT_SYSTEM = (
    "당신은 퀀트 백테스트의 방법론적 결함을 심사하는 감사관이다. 아래 지침의 '한 가지 죄악'만"
    " 판단하고, 반드시 JSON {\"triggered\": true|false, \"message\": \"근거 한두 문장\"}만 출력하라."
)


_PORTFOLIO_OPS = {"run_backtest", "run_qvm_backtest"}


def _ops_used(steps: list[dict] | None) -> list[str]:
    return [s.get("op") for s in (steps or []) if isinstance(s, dict) and s.get("op")]


def _steps_context(steps: list[dict] | None) -> str:
    """실제 파이프라인 연산 목록을 검사관 프롬프트에 넣을 문구로 만든다.

    실서버 재현 버그: "코스피 전종목 pbr/gpa 상관관계 5분위" 같은 get_cross_section→
    correlation/quantile_bucket_means 파이프라인(run_backtest 없음)은 result에 performance/
    holdings가 전혀 없어, 4개 소프트경고 검사관이 아무 정보 없이 판단해야 했다. 특히
    inspect_outlier의 "모르면 경고로 남겨라" 지시 때문에 사용자가 뭘 해도 이상치 경고가
    100% 뜨는 구조적 결함이었다(remove_outliers/winsorize 프리미티브는 이미 정상 등록돼
    실행 가능한 상태였음 — 검사 단계가 그걸 썼는지 확인할 방법이 없었을 뿐).
    실제 steps를 보여줘 검사관이 추측 대신 사실에 근거해 판단하게 한다."""
    ops = _ops_used(steps)
    if not ops:
        return "이 파이프라인이 실제로 사용한 연산 목록: (알 수 없음 — steps 정보가 전달되지 않음)"
    return f"이 파이프라인이 실제로 사용한 연산 목록: {ops}"


def _portfolio_scope_note(steps: list[dict] | None) -> str:
    """포트폴리오 개념(기간 다양성/회전율)이 적용되지 않는 파이프라인이면 자동 통과시키는 지시.

    run_backtest/run_qvm_backtest처럼 여러 리밸런싱 시점을 거치는 연산이 없다면(예: 특정
    asof 한 시점의 횡단면(cross-section) 통계 분석), '백테스트 기간이 다양한 국면을
    포함하는지'/'회전율이 과도한지' 같은 질문 자체가 성립하지 않는다 — 정보가 없다고
    무조건 경고로 남기면(기존 동작) 순수 통계질문에 매번 오탐이 뜬다."""
    ops = _ops_used(steps)
    if ops and not (set(ops) & _PORTFOLIO_OPS):
        return (
            "위 연산 목록에 run_backtest/run_qvm_backtest가 없다 — 이는 여러 리밸런싱 시점을"
            " 거치는 포트폴리오 백테스트가 아니라 특정 시점의 횡단면(cross-section) 통계"
            " 분석이다. 이 경우 이 검사가 다루는 개념(기간의 다양성·국면 포함 여부/회전율)"
            " 자체가 해당사항 없음이니 반드시 triggered=false로 판단하라."
        )
    return ""


def _run_inspector(sin: str, prompt: str, llm_fn: Callable) -> dict:
    """공통 실행부: 프롬프트를 llm_fn에 넘겨 JSON 판정을 파싱한다.

    llm_fn(prompt)->str 은 주입 가능(기본 없음). 응답이 JSON이 아니거나 비면 triggered=False로
    안전하게 무시한다(소프트경고는 오판보다 미탐이 안전 — 정상 결과를 가리지 않으므로).
    """
    text = llm_fn(prompt) or ""
    data = extract_json(text) or {}
    return {
        "sin": sin,
        "triggered": bool(data.get("triggered")),
        "message": str(data.get("message") or ""),
    }


def inspect_storytelling(result: dict, llm_fn: Callable, steps: list[dict] | None = None) -> dict:
    """③ 스토리텔링·데이터의 역사: 백테스트 기간이 짧거나 단일 국면에만 성과가 몰렸는지."""
    scope_note = _portfolio_scope_note(steps)
    prompt = (
        f"{_SOFT_SYSTEM}\n\n[검사: 스토리텔링·데이터의 역사]\n"
        "백테스트 기간이 충분히 길고 다양한 경제 국면(상승/하락/횡보)을 포함하는지, 특정 시기의"
        " 우연한 성과를 사후적으로 이야기로 포장한 흔적이 없는지 판단하라.\n"
        f"{_steps_context(steps)}\n"
        + (f"{scope_note}\n" if scope_note else "")
        + f"성과: {result.get('performance')}\n"
        f"리밸런싱 시점 수: {len(result.get('holdings') or [])}\n"
        f"기간(dates): {result.get('dates')}"
    )
    return _run_inspector("storytelling", prompt, llm_fn)


def inspect_snooping(result: dict, question: str, llm_fn: Callable, steps: list[dict] | None = None) -> dict:
    """④ 데이터마이닝·스누핑: 사용자 원본 질문을 '사전 등록 가설'로 취급해 사후정당화 여부 판단(AC11)."""
    prompt = (
        f"{_SOFT_SYSTEM}\n\n[검사: 데이터마이닝·데이터스누핑]\n"
        "아래 '사용자 원본 질문'을 사전에 등록된 투자 가설로 간주하라. 백테스트 결과가 이 가설을"
        " 정직하게 검증한 것인지, 아니면 좋은 성과가 나오도록 팩터/기간/종목수를 사후에 짜맞춘"
        " 데이터 스누핑의 흔적이 있는지 판단하라.\n"
        f"[사용자 원본 질문(사전등록 가설)]\n{question}\n\n"
        f"{_steps_context(steps)}\n"
        f"성과: {result.get('performance')}\n"
        f"보유종목: {result.get('holdings')}"
    )
    return _run_inspector("snooping", prompt, llm_fn)


def inspect_signal_decay(result: dict, llm_fn: Callable, steps: list[dict] | None = None) -> dict:
    """⑤ 신호감소·회전율: 회전율이 과도해 거래비용 차감 후 성과가 신기루가 아닌지."""
    perf = result.get("performance") or {}
    scope_note = _portfolio_scope_note(steps)
    prompt = (
        f"{_SOFT_SYSTEM}\n\n[검사: 신호의 감소와 회전율]\n"
        "팩터 신호가 빠르게 감소하거나 회전율이 과도해, 수수료·세금·슬리피지를 제한 뒤에도"
        " 성과가 유지되는지 판단하라. (engine은 회전율×거래비용을 이미 차감한다.)\n"
        f"{_steps_context(steps)}\n"
        + (f"{scope_note}\n" if scope_note else "")
        + f"회전율(avg_turnover): {perf.get('avg_turnover')}\n"
        f"성과: {perf}"
    )
    return _run_inspector("signal_decay", prompt, llm_fn)


def inspect_outlier(result: dict, llm_fn: Callable, steps: list[dict] | None = None) -> dict:
    """⑥ 이상치제어: 팩터 정규화·이상치 완화 방식이 이 결과의 성과를 왜곡하지 않았는지."""
    prompt = (
        f"{_SOFT_SYSTEM}\n\n[검사: 이상치 제어]\n"
        "멀티팩터 결합 전 각 팩터를 표준화했는지, 소수 극단치(이상치)가 성과를 좌우하지 않는지"
        " 판단하라. 현재 파이프라인은 combine(method=)으로 zscore(값 자체를 표준화, 극단치에"
        " 가장 취약)/rank_sum(순위만 사용, 극단치 크기에 영향받지 않음)을 지원하고, 사전에"
        " winsorize(IQR 기준 상하단을 벗어난 값만 경계로 눌러 붙임)로 극단치를 먼저 완화할 수도"
        " 있다.\n"
        f"{_steps_context(steps)}\n"
        "위 연산 목록에 winsorize/winsorize_pct/remove_outliers가 있으면 이상치를 실제로"
        " 완화한 것이고, combine이 있으면 그 method를 확인해 어떤 조합을 썼는지 판단하라 —"
        " 목록에 없는 연산은 쓰이지 않은 것으로 간주하고 추측하지 마라. steps 정보 자체가"
        " 전혀 없을 때만(위 목록이 '알 수 없음') 그 불확실성을 경고로 남겨라.\n"
        f"성과: {result.get('performance')}\n"
        f"보유종목: {result.get('holdings')}"
    )
    return _run_inspector("outlier", prompt, llm_fn)


# ==========================================================================
# AUD-6: 소프트경고 4종 병렬 오케스트레이터
# ==========================================================================
def run_soft_inspectors(
    result: dict,
    question: str,
    llm_fn: Callable,
    pool_factory: Callable | None = None,
    timeout_s: float = MAX_TIMEOUT,
    steps: list[dict] | None = None,
) -> list[dict]:
    """소프트경고 검사관 4종을 ThreadPoolExecutor(max_workers=4)로 동시 병렬 실행한다.

    pipeline_exec.py의 기존 ThreadPoolExecutor 관례를 확장하고, 기존 MAX_TIMEOUT 상한을 재사용한다
    (새 상수 금지 — 스펙 §3.3). 4개가 모두 끝날 때까지 기다린 뒤 결과 리스트를 반환한다(동기 병렬).
    pool_factory는 테스트 주입용(submit 호출 횟수 spy 등). steps(실제 파이프라인)를 넘기면
    4개 검사관 전부에 그대로 전달돼, result에 performance/holdings가 없는 순수 통계 파이프라인
    (correlation/quantile_bucket_means 등)도 추측이 아니라 실제 연산을 보고 판단한다.
    """
    tasks = [
        lambda: inspect_storytelling(result, llm_fn, steps=steps),
        lambda: inspect_snooping(result, question, llm_fn, steps=steps),
        lambda: inspect_signal_decay(result, llm_fn, steps=steps),
        lambda: inspect_outlier(result, llm_fn, steps=steps),
    ]
    fallback_sins = ["storytelling", "snooping", "signal_decay", "outlier"]
    pool_factory = pool_factory or (lambda: ThreadPoolExecutor(max_workers=4))
    out: list[dict] = []
    with pool_factory() as pool:
        futures = [pool.submit(t) for t in tasks]
        for sin, fut in zip(fallback_sins, futures):
            try:
                out.append(fut.result(timeout=timeout_s))
            except FuturesTimeout:
                out.append({"sin": sin, "triggered": False, "message": "검사 타임아웃"})
    return out


# ==========================================================================
# 오케스트레이션 훅 (자동 트리거 배선용 — AUD-7)
# ==========================================================================
def _resolve_backtest_weights(steps: list[dict], conn, run_pipeline_fn: Callable):
    """run_backtest 단계가 받을 비중 벡터를 사전에 해석한다(사전검사용).

    run_backtest의 weights 파라미터가
    - 리터럴 dict면 그대로,
    - {"$ref": name}이면 그 name을 out으로 만드는 앞 단계까지의 접두 파이프라인을
      run_pipeline_fn으로 실행해(= 그 producer 단계의 결과) 비중을 얻는다.
    해석 불가면 None(사전검사 생략 — 오탐으로 정상 실행을 막지 않는다).
    """
    idx = next((i for i, s in enumerate(steps or []) if s.get("op") == "run_backtest"), None)
    if idx is None:
        return None
    wparam = (steps[idx].get("params") or {}).get("weights")
    if isinstance(wparam, dict) and set(wparam.keys()) == {"$ref"}:
        ref = wparam["$ref"]
        producer = next((i for i, s in enumerate(steps[:idx]) if s.get("out") == ref), None)
        if producer is None:
            return None
        return run_pipeline_fn(steps[: producer + 1], conn=conn)
    if isinstance(wparam, dict):
        return wparam
    return None


def pre_audit(steps: list[dict], conn, run_pipeline_fn: Callable) -> dict | None:
    """사전검사(run_backtest 실행 직전): 공매도(음수 비중) 하드차단.

    weights를 쓰는 백테스트일 때만 의미가 있다. 비중을 해석하지 못하면 None을 반환한다
    (개입하지 않음). 반환이 blocked=True면 호출자는 run_backtest를 실행하지 말아야 한다.
    """
    weights = _resolve_backtest_weights(steps, conn, run_pipeline_fn)
    if not isinstance(weights, dict) or not weights:
        return None
    return check_short_positions(weights)


def post_audit(result: dict, conn, question: str, llm_fn: Callable | None = None,
               market: str = "KR", steps: list[dict] | None = None) -> dict:
    """사후검사(run_backtest 실행 직후): 하드차단 2종 + 소프트경고 4종.

    하드차단(생존편향/미래참조)이 하나라도 발동하면 blocked=True로 소프트검사는 생략한다
    (하드차단 시 정상 결과를 폐기하므로 소프트경고는 무의미). llm_fn이 없으면(LLM 미가용)
    소프트검사도 생략한다.

    생존편향은 delisting 테이블 기반 _is_alive로 실제 하드차단한다 — 상장폐지 이력이
    없으면 조용히 통과한다. steps(실제 파이프라인)를 넘기면 run_soft_inspectors에
    그대로 전달해, result에 performance/holdings가 없는 순수 통계 파이프라인도 정확히 판단하게 한다.
    반환: {"blocked": bool, "hard": [verdict...], "soft": [verdict...]}
    """
    holdings = result.get("holdings") if isinstance(result, dict) else None
    hard = [
        check_survivorship(conn, holdings or [], market=market),
        check_lookahead(conn, holdings or []),
    ]
    blocked = any(v["blocked"] for v in hard)
    soft: list[dict] = []
    if not blocked and llm_fn is not None:
        soft = run_soft_inspectors(result, question, llm_fn, steps=steps)
    return {"blocked": blocked, "hard": hard, "soft": soft}


def audit_search_strategy_result(
    results: list[dict], question: str, llm_fn: Callable | None,
) -> list[dict]:
    """search_strategy(역백테스트) 최종 결과에 스누핑 소프트경고만 1회 첨부한다.

    search_strategy는 내부에서 engine.run_backtest를 직접 호출해 op=="run_backtest" 자동
    감사 발동 조건(nodes.py의 has_backtest)에 걸리지 않는다. 후보(최대 20개) 각각을 4종
    소프트검사로 감사하면 LLM 호출이 최대 80회까지 치솟아 비용이 크므로, 가장 위험한 편향
    (데이터마이닝·스누핑 — 사후적으로 잘 맞는 조합을 20개 중 고르는 행위 자체가 이 편향에
    가장 취약함)만 최종 1위(rank_by 기준 최상위) 결과를 대상으로 1회 판단한다
    (architect 검토 MAJOR 권고 Option B).

    하드차단(생존편향/미래참조)은 run_backtest 정상 경로의 1차 가드(metrics_at의 _is_alive,
    effective_quarter_at의 disclosed_date<=asof)에 위임하고 여기서 다시 검사하지 않는다 —
    run_backtest 파이프라인의 하드차단처럼 방어적 이중화까지는 하지 않는, search_strategy
    전용 축소 스코프다.
    """
    if not results or llm_fn is None:
        return []
    top = results[0]
    summary = {"performance": top.get("performance"), "holdings": top.get("holdings")}
    verdict = inspect_snooping(summary, question, llm_fn)
    if verdict.get("triggered"):
        verdict["message"] = f"(총 {len(results)}개 후보 중 최상위 결과 기준) {verdict['message']}"
    return [verdict]


def format_hard_block(verdicts: list[dict]) -> str:
    """하드차단 사유·증거를 사용자용 에러 메시지 한 줄로 조립한다(AUD-8)."""
    parts = []
    for v in verdicts:
        if v and v.get("blocked"):
            parts.append(f"[{v['sin']}] {v['reason']} / 증거: {v['evidence']}")
    return "백테스트 감사 차단 — " + " | ".join(parts)
