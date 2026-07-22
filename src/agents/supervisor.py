"""총괄 에이전트 (HA-10) — 계층형 멀티에이전트의 최상위 "총괄 로직"을 순수 함수로 구현한다.

.omc/state/sessions/f6b79d90-b7b2-4b4f-b4f7-405a8355724e/prd.json 의 HA-10 참고.

계층: **이 총괄 에이전트** → 도메인 에이전트(HA-6 domain_kr /
HA-8 domain_macro / HA-9 domain_backtest) → 데이터 에이전트. 이 파일은 이미 완성된
도메인 에이전트의 answer_*_question() 을 **그대로 재사용**하며, 판정/재무/주가 계산 로직을
새로 만들지 않는다 — 오직 (1)라우팅 (2)도메인 실행·종합 (3)정합성 검증 (4)재시도 오케스트레이션
네 가지 "총괄" 책임만 담당한다.

--- 설계 의도(HA-11 인계용) ------------------------------------------------------
이번 스토리(HA-10)는 LangGraph StateGraph 노드로 직접 배선하지 않는다. 대신 아래 네 함수를
**명확한 입출력 시그니처를 가진 순수 함수**로 설계해, 다음 스토리(HA-11)가 이 함수들을 그대로
LangGraph 노드 함수(`def supervisor_node(state: GraphState) -> dict` 형태)로 감쌀 수 있게 한다.
즉 HA-11은 state(dict)에서 question/conn/llm_fn/steps 를 꺼내 answer_with_verification(...)
을 호출하고 반환 dict를 state 갱신분으로 돌려주기만 하면 된다 — 총괄 로직 자체는 여기서 끝난다.

- route_question(question, llm_fn) -> list[str]
      질문 → 도메인 리스트(["kr"] | ["macro"] | ["backtest"] ...).
- dispatch_domains(routes, question, conn, llm_fn, steps=None) -> dict
      각 도메인 answer_*_question 을 호출해 **원본 결과를 가공 없이** 도메인별 키로 보존.
- verify_answer(question, domain_results, llm_fn) -> {"valid": bool, "reason": str}
      도메인 결과가 원 질문과 부합하는지 판정(결정론적 규칙 우선 + LLM 판정).
- answer_with_verification(...) -> dict
      route→dispatch→verify 순서로 실행, 검증 실패 시 **정확히 max_retries(기본 2)회까지만**
      재시도(무한루프 없음). 실패 시 "불확실성 명시" 응답(uncertain=True), 통과 시 종합결론
      (synthesize_conclusion)과 원본 domain_results 를 함께 반환.

llm_fn 규약: 이 프로젝트의 도메인 에이전트들과 동일하게 `Callable[[str], str]`(prompt→text).
호출부(HA-11/nodes.py)는 `lambda p: (deps.llm.complete(p, role="judge").text or "")` 형태로
주입한다(src/graph/nodes.py 참고). llm_fn 이 None이면 결정론적 휴리스틱으로 폴백한다.
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Callable

from src.agents.chart_agent import build_chart_freeform
from src.agents.domain_backtest import answer_backtest_question
from src.agents.domain_kr import answer_kr_question
from src.agents.domain_macro import answer_macro_question
from src.agents.exec_fallback import run_free_exec_fallback
from src.llm import extract_json

# 정규 도메인 순서 — LLM 응답의 순서/중복과 무관하게 항상 이 순서로 정렬해 결정론을 보장한다.
_DOMAINS: tuple[str, ...] = ("kr", "macro", "backtest")

# on_progress 이벤트 라벨용 — web/static/index.html의 TREE_LABEL/treeDepth와 동일한 도메인
# 이름(step)을 그대로 써야 프론트가 별도 분기 없이 트리 깊이를 매긴다(graph.py의
# _DOMAIN_LABELS와 같은 매핑이지만, graph.py가 이 모듈을 import하므로 순환을 피해 로컬로 둔다).
_DOMAIN_LABELS_KO: dict[str, str] = {"kr": "한국", "macro": "매크로", "backtest": "백테스트"}

# llm_fn 미주입 시 사용하는 라우팅 휴리스틱. 도메인별 대표 키워드(부분일치).
_ROUTE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "backtest": (
        "백테스트", "backtest", "리밸런", "전략 수익", "전략을 검증", "전략 검증",
        # 팩터/지표 간 상관관계·분위수 분석도 backtest 도메인의 correlation/
        # quantile_bucket_means 프리미티브(top_n 상한 없이 전체 크로스섹션 처리)로만
        # 계산 가능하다 — "백테스트"라는 단어가 없어도 이 도메인으로 가야 한다.
        "상관관계", "상관계수", "산점도", "분위수", "5분위", "팩터분석", "요인분석",
        "correlation", "quantile",
        # 히스토그램(분포도)도 histogram_buckets 프리미티브(get_cross_section 기반)로만
        # 계산 가능하다 — "백테스트"라는 단어가 없어도 이 도메인으로 가야 한다.
        "히스토그램", "histogram", "분포도",
        # 지표를 N구간으로 나눠 각 구간의 평균/집계를 구하는 질문도 quantile_bucket_means
        # 프리미티브 전용이다("분위수"/"5분위"의 동의어인데 예전엔 이 표현만 빠져 있어서
        # kr로 잘못 라우팅되고 kr은 구간 집계를 표현할 방법이 없어 재시도만 헛돌았다 —
        # 실사용 재현: "PER을 10구간으로 나눠서 각 구간 평균 구해줘").
        "구간별", "구간으로", "등분",
        # QVM(퀄리티·밸류·모멘텀) 등 여러 팩터를 합성한 복합점수 스크리닝/백테스트도
        # compute_qvm_scores/run_qvm_backtest 프리미티브(backtest 전용)로만 계산 가능하다.
        # "모멘텀"/"밸류" 단어 하나만으로는 kr 도메인의 단일지표 스크리닝(예: "모멘텀 상위
        # 10개")과 구분이 안 되므로 표준 단일지표 질문과 겹치지 않는 복합/멀티팩터 표현만
        # 키워드로 쓴다(과잉매칭 방지).
        "멀티팩터", "다중팩터", "다중 팩터", "복합팩터", "복합 팩터",
        "복합점수", "복합 점수", "합성점수", "합성 점수", "qvm",
        "multi-factor", "multifactor", "quality value momentum",
        "퀄리티 밸류 모멘텀", "퀄리티·밸류·모멘텀",
    ),
    "macro": ("매크로", "금리차", "스프레드", "장단기", "공포탐욕", "vix", "레짐", "매크로 신호",
              "파마프렌치", "파마-프렌치", "fama french", "fama-french", "smb", "hml", "rmw", "cma"),
    # "삼성" 외 국내 시가총액 상위 종목의 흔한 약칭도 넣는다 — LLM 라우팅(_route_prompt)이
    # 일시적으로 실패해 이 휴리스틱으로 폴백했을 때 "하이닉스 12개월 수익률"처럼 시장 자체를
    # 언급하지 않는 개별 종목 질문이 라우팅되지 않던 실사용 버그 재현.
    "kr": ("삼성", "코스피", "코스닥", "국내", "한국주식", "하이닉스", "네이버", "카카오", "포스코", "현대차", "기아"),
}

# 차트 요청 감지용 키워드(_ROUTE_KEYWORDS와 동일 스타일의 부분일치). 명시적으로 그래프/
# 시각화를 요청했을 때만 이미지 차트를 붙인다(모든 질문에 자동으로 붙이지 않는다). "그려"는
# "그려줘/그려봐/그려주세요"를 모두 포괄하되 "알려줘"와는 겹치지 않는다(오탐 방지).
_CHART_KEYWORDS: tuple[str, ...] = (
    "그래프", "차트", "그려", "시각화", "plot", "chart", "추이 보여", "추이보여",
)

# verify_answer 결정론 폴백/파싱용 마커. 순서 주의: '불일치'는 '일치'를 부분문자열로 포함하므로
# 반드시 부정 마커를 먼저 검사한다.
_INVALID_MARKERS: tuple[str, ...] = (
    "불일치", "부적합", "불충분", "부합하지", "맞지 않", "관련 없", "무관", "mismatch", "invalid", "false",
)
_VALID_MARKERS: tuple[str, ...] = ("일치", "적합", "부합", "충분", "valid", "true", "관련 있")


def route_question(
    question: str,
    llm_fn: Callable[[str], str] | None,
    on_progress: Callable[[str, str], None] | None = None,
) -> list[str]:
    """질문을 보고 처리에 필요한 도메인 리스트(복수 가능)를 반환한다.

    llm_fn(주입 시)에 라우팅을 위임하고, 그 응답 텍스트에서 도메인 토큰(kr/macro/
    backtest)을 추출해 **정규 순서(_DOMAINS)로 정렬·중복제거**한다(예: "삼성전자 최근 종가와
    골든크로스 전략" → LLM이 'kr, backtest' 응답 → ["kr","backtest"]). LLM 응답이 JSON
    리스트든 콤마 나열이든 도메인 토큰만 뽑으므로 형식에 관대하다.

    llm_fn 이 None이거나 응답에서 도메인을 하나도 못 찾으면 키워드 휴리스틱으로 폴백하고,
    그래도 못 찾으면 빈 리스트([])를 반환한다(SoT [ROUTING]: 무관한 질문에 억지로 기본
    도메인을 배정하지 않는다 — answer_with_verification이 빈 리스트를 즉시 불확실
    응답으로 처리한다).

    on_progress(step, summary) 를 주면 라우팅이 확정되는 즉시 1건 통지한다(실시간 트리용,
    HA-12 확장). 생략하면(기본값 None) 기존과 완전히 동일하게 동작한다.
    """
    if llm_fn is not None:
        try:
            raw = llm_fn(_route_prompt(question)) or ""
        except Exception:  # noqa: BLE001 — LLM 실패는 휴리스틱 폴백으로 흡수
            raw = ""
        routes = _extract_domains(raw)
        if not routes:
            routes = _route_heuristic(question)
    else:
        routes = _route_heuristic(question)

    if on_progress:
        labels = "+".join(_DOMAIN_LABELS_KO.get(r, r) for r in routes)
        on_progress("supervisor", f"질문 분석 완료 → {labels} 도메인으로 라우팅")
    return routes


def _route_prompt(question: str) -> str:
    return (
        "다음 사용자 질문을 처리하려면 어떤 도메인 에이전트가 필요한지 고르세요.\n"
        "가능한 도메인: kr(국내주식 재무/주가), "
        "macro(매크로 신호/금리차, 그리고 파마프렌치/Fama-French 팩터 데이터 조회 — "
        "SMB/HML/RMW/CMA/모멘텀 팩터 값을 묻는 질문도 macro입니다), "
        "backtest(전략 백테스트, 그리고 전종목 횡단면 지표 분석 — "
        "팩터/지표 간 상관관계·산점도, 분위수, 히스토그램/분포도. 특정 지표(PBR 등)를 구간으로 "
        "나눠 히스토그램/분포를 그리거나, 각 구간의 평균·집계를 구하는 질문(예: '10구간으로 "
        "나눠서 각 구간 평균 구해줘')은 개별 종목 조회가 아니라 반드시 backtest입니다 — "
        "구간별 평균 계산은 '분위수'와 동일한 quantile_bucket_means 전용 계산입니다. "
        "퀄리티·밸류·모멘텀(QVM)처럼 여러 팩터를 윈저라이즈·섹터중립 z-score·가중합성해 "
        "하나의 복합/종합 점수로 스크리닝하거나 그 전략으로 리밸런싱 백테스트하는 질문도 "
        "backtest입니다 — 이건 kr의 단일 지표 기준 스크리닝(예: 'PER 낮은 순 10개')과는 "
        "다른 계산(compute_qvm_scores/run_qvm_backtest)이 필요하므로 절대 kr로 보내지 마세요).\n"
        "거꾸로, 특정 지표 하나를 기준으로 순서대로 나열/정렬해서 그 결과를 그래프로 보여달라는 "
        "질문(예: 'PBR 오름차순으로 나열해서 그래프 그려줘', 'PER 낮은 순 10개')은 분포/"
        "히스토그램/상관관계 분석이 아니라 단일 지표 기준 단순 스크리닝이므로 kr입니다 — "
        "'PBR'·'그래프' 같은 단어가 섞였다고 backtest로 보내지 마세요(분포·히스토그램·상관관계·"
        "분위수·QVM 같은 '지표를 구간/집계/합성해 분석'하는 신호가 있을 때만 backtest입니다). "
        "특히 '오름차순으로 나열해서'처럼 정렬 요청으로 문장이 시작해도, 그 뒤에 '구간으로 "
        "나눠 평균/집계'가 이어지면 정렬은 전처리일 뿐이므로 backtest입니다 — 문장 앞부분의 "
        "정렬 표현만 보고 kr로 판단하지 마세요.\n"
        "여러 도메인이 필요하면 모두 나열하세요(예: 국내 종목 스크리닝 + 백테스트 → kr, backtest).\n"
        "도메인 키워드만 콤마로 구분해 답하세요.\n\n"
        f"질문: {question}\n답:"
    )


def _extract_domains(text: str) -> list[str]:
    """임의의 텍스트(콤마 나열/JSON 리스트 등)에서 도메인 토큰만 뽑아 정규 순서로 정리."""
    tokens = set(re.findall(r"[a-z]+", (text or "").lower()))
    return [d for d in _DOMAINS if d in tokens]


def _route_heuristic(question: str) -> list[str]:
    """llm_fn 미가용 시 키워드 기반 라우팅. 아무것도 안 걸리면 빈 리스트(unknown)를 반환한다.

    예전엔 아무것도 안 걸리면 무조건 ["kr"]로 폴백했는데, 그러면 완전히 무관한 질문("오늘
    날씨 어때")까지 한국주식 조회로 이어졌다. 빈 리스트를 반환하면 answer_with_verification이
    이를 "질문을 이해하지 못함"으로 즉시 처리한다. "삼성" 같은 실제 키워드가 있는 질문은
    이 폴백과 무관하게 정상적으로 매치되어 영향받지 않는다.
    """
    q = (question or "").lower()
    found = {
        domain
        for domain, keywords in _ROUTE_KEYWORDS.items()
        if any(kw.lower() in q for kw in keywords)
    }
    return [d for d in _DOMAINS if d in found]


def wants_chart(question: str) -> bool:
    """질문이 명시적으로 그래프/차트를 요청하는지 결정론적으로 판단한다(_ROUTE_KEYWORDS 스타일).

    LLM을 쓰지 않고 키워드 부분일치로만 판단한다 — "차트를 그릴지 말지"는 파이썬 코드가
    결정한다(프롬프트에 코드를 짜라고 시키지 않는다). 재시도 시 실패 피드백이 덧붙은
    dispatch_question이 아니라 **원본 question**으로 판단해야 하므로(verify_fn/synthesize_fn이
    원본 question을 쓰는 것과 동일한 이유), 호출부는 항상 원본 question을 넘긴다.
    """
    q = (question or "").lower()
    return any(kw.lower() in q for kw in _CHART_KEYWORDS)


def _chartable_payload(domain_results: dict):
    """차트 폴백(build_chart_freeform)에 넘길 '실제로 그릴 데이터'를 domain_results에서 꺼낸다.

    domain_results는 {도메인: {..., "result": <실제데이터>}} 래퍼다. 이걸 그대로 넘기면
    build_chart_freeform이 받는 data가 도메인키 dict({"kr": {...}})가 돼, _summarize_data_shape가
    'dict, 최상위 키: [kr]'로만 요약한다 — LLM은 정작 그릴 리스트가 domain_results["kr"]["result"]에
    묻혀 있는 걸 못 보고 data를 리스트로 착각한 코드를 짜 실행에 실패한다(실측: 리스트로 가정한
    코드가 dict를 순회 → TypeError → chart_base64 미충족 → None). 멀티턴 경로
    (conversation._run_followup_step)가 직전 턴의 flat한 result를 그대로 넘겨 정상 동작하는 것과
    대칭이 되도록, 여기서도 각 도메인의 "result" payload만 꺼내 넘긴다.

    - 도메인이 하나면 그 payload를 그대로 준다(가장 흔한 스크리닝 경로: flat 리스트).
    - 여러 도메인이면 {도메인: payload} dict로 준다(요약이 도메인별 구조를 보여주게).
    - 꺼낼 result가 하나도 없으면(예상 밖 모양) 원본을 그대로 돌려준다(회귀 안전 — 기존보다
      나빠지지 않는다). "result" 키가 없거나 None인 도메인(에러 응답 등)은 건너뛴다.
    """
    if not isinstance(domain_results, dict):
        return domain_results
    payloads = {
        domain: value["result"]
        for domain, value in domain_results.items()
        if isinstance(value, dict) and value.get("result") is not None
    }
    if not payloads:
        return domain_results
    if len(payloads) == 1:
        return next(iter(payloads.values()))
    return payloads


def dispatch_domains(
    routes: list[str],
    question: str,
    conn,
    llm_fn: Callable[[str], str] | None,
    steps: list[dict] | None = None,
    on_progress: Callable[[str, str], None] | None = None,
) -> dict:
    """routes 의 각 도메인 answer_*_question 을 호출하고 **원본 결과를 가공 없이** 보존한다.

    반환: {"kr": {...}, "macro": {...}, "backtest": {...}} — routes 에 포함된
    도메인만 키로 담는다. 복합 도메인 질문에서 각 도메인의 원본 데이터를 그대로 노출해야
    하므로(나중에 종합결론과 별도로 병기됨) 여기서는 어떤 가공/요약도 하지 않는다.

    도메인 함수가 예외를 던지면 전파하지 않고 해당 도메인 값에 {"error": ...}만 담는다
    (한 도메인 실패가 전체 dispatch를 무너뜨리지 않게 — 도메인 에이전트들은 자체적으로
    예외를 흡수하지만 방어적으로 한 겹 더 감싼다).

    on_progress(step, summary) 를 주면 도메인마다 조회 시작/완료(또는 오류)를 실시간
    통지한다(step=도메인 코드, 예: "kr"). 생략하면 기존과 완전히 동일하게 동작한다.
    """
    results: dict = {}
    for domain in routes:
        label = _DOMAIN_LABELS_KO.get(domain, domain)
        if on_progress:
            on_progress(domain, f"{label} 도메인 조회 중…")
        domain_kwargs = {"on_progress": on_progress} if on_progress else {}
        try:
            if domain == "kr":
                results["kr"] = answer_kr_question(question, conn, llm_fn=llm_fn, **domain_kwargs)
            elif domain == "macro":
                results["macro"] = answer_macro_question(question, conn)
            elif domain == "backtest":
                results["backtest"] = answer_backtest_question(
                    question, steps or [], conn, llm_fn=llm_fn, **domain_kwargs
                )
            if on_progress:
                on_progress(domain, f"{label} 도메인 완료")
        except Exception as exc:  # noqa: BLE001 — 방어적: 도메인 예외를 총괄까지 올리지 않음
            results[domain] = {"error": f"{type(exc).__name__}: {exc}"}
            if on_progress:
                on_progress(domain, f"{label} 도메인 오류: {exc}")
    return results


def _domain_has_data(result: dict) -> bool:
    """도메인 결과가 '쓸만한 데이터'를 담고 있는지 대략 판정(결정론 폴백용).

    도메인마다 스키마가 다르므로 완전히 정밀할 필요는 없다 — errors만 있고 실질 데이터가
    전혀 없는 경우를 걸러내는 정도다.
    """
    if not isinstance(result, dict) or result.get("error"):
        return False
    # kr: financial/price 중 하나라도 있으면 데이터 있음.
    if result.get("financial") or result.get("price"):
        return True
    # kr 다중종목(named multi-entity, domain_kr._answer_kr_multi_entity_question): 최상위
    # financial/price는 항상 None이고 실제 데이터는 entities 리스트 안에 종목별로 담긴다.
    # entities 중 하나라도 financial/price가 있으면 데이터 있음으로 본다.
    entities = result.get("entities")
    if isinstance(entities, list) and any(
        isinstance(e, dict) and (e.get("financial") or e.get("price")) for e in entities
    ):
        return True
    # kr 다중분기(multi-period, domain_kr): 한 종목의 여러 분기를 조회하면 최상위 financial은
    # None이고 실제 데이터는 periods 리스트에 기간별로 담긴다(entities와 동일 관례). 하나라도
    # financial이 있으면 데이터 있음으로 본다.
    periods = result.get("periods")
    if isinstance(periods, list) and any(
        isinstance(p, dict) and p.get("financial") for p in periods
    ):
        return True
    # kr 다중지표(multi-metric, domain_kr._resolve_metrics): 한 종목의 여러 지표(예: "PER PBR
    # PSR")를 조회하면 최상위 financial은 None이고 실제 데이터는 metrics 리스트에 지표별로
    # 담긴다(periods/entities와 동일 관례). 하나라도 financial이 있으면 데이터 있음으로 본다.
    metrics = result.get("metrics")
    if isinstance(metrics, list) and any(
        isinstance(m, dict) and m.get("financial") for m in metrics
    ):
        return True
    # macro: available=True.
    if result.get("available"):
        return True
    # backtest: 차단되지 않고 결과가 있으면.
    if result.get("result") is not None and not result.get("blocked"):
        return True
    return False


def _any_domain_has_data(domain_results: dict) -> bool:
    return any(_domain_has_data(v) for v in domain_results.values())


def _is_best_effort_search(domain_results: dict) -> bool:
    """backtest 도메인이 search_signal_strategy(탐색형) 결과를 담고 있는지 판정한다.

    search_signal_strategy는 "성과 목표(MDD·수익률 등)를 만족하는 전략을 찾아줘"에 대해 여러
    후보를 시도해 실제 성과를 돌려주며, 제약을 만족하는 후보가 없으면 constraints_met=False로
    '가장 근접한 시도'를 정직하게 반환한다(에러도 빈손도 아님). 이 결과는 제약 충족 여부와
    무관하게 그 자체로 질문에 대한 유효한 답이므로, 검증 LLM이 '목표 미달=답변 실패'로 오판해
    불필요한 재시도→불확실 응답으로 빠지지 않도록 verify_answer가 결정론적으로 통과시킨다.

    탐색형 고유의 반환 형식(dict에 constraints_met 키 + 비어있지 않은 results)만 매칭하므로,
    단일 run_signal_backtest/run_backtest(그 키가 없음)나 크로스섹셔널 search_strategy(list 반환)는
    영향받지 않는다(기존 검증 경로 그대로).
    """
    bt = domain_results.get("backtest")
    if not isinstance(bt, dict) or bt.get("blocked"):
        return False
    res = bt.get("result")
    return isinstance(res, dict) and "constraints_met" in res and bool(res.get("results"))


def verify_answer(
    question: str,
    domain_results: dict,
    llm_fn: Callable[[str], str] | None,
) -> dict:
    """도메인 결과들이 원 질문과 부합하는지 판정한다.

    결정론적 규칙을 우선 적용하고(라우팅된 도메인이 없거나 모든 도메인에서 유효 데이터를
    얻지 못하면 즉시 valid=False), 그 뒤 llm_fn(주입 시)에 최종 판정을 위임한다. llm_fn 이
    없으면 "데이터가 존재하면 통과"라는 결정론 폴백을 쓴다.

    반환: {"valid": bool, "reason": str}.
    """
    if not domain_results:
        return {"valid": False, "reason": "라우팅된 도메인이 없어 답변을 구성할 수 없습니다."}
    if not _any_domain_has_data(domain_results):
        return {"valid": False, "reason": "모든 도메인에서 질문에 답할 유효한 데이터를 얻지 못했습니다."}
    # 탐색형 백테스트(search_signal_strategy) 결과는 제약 충족 여부와 무관하게 유효한 답이다 —
    # LLM 판정에 넘기면 '목표 미달'을 '답변 실패'로 오판하므로 결정론적으로 통과시킨다(재시도 방지).
    if _is_best_effort_search(domain_results):
        met = bool(domain_results["backtest"]["result"].get("constraints_met"))
        note = ("제약을 만족하는 후보를 찾음" if met
                else "제약을 만족하는 후보가 없어 가장 근접한 시도를 제시함")
        reason = f"전략 탐색(역백테스트) 결과가 유효합니다 — {note}."
        return {"valid": True, "reason": reason, "per_domain": {"backtest": {"valid": True, "reason": reason}}}
    if llm_fn is None:
        reason = "결정론적 검증 통과(도메인 데이터 존재, LLM 미가용)."
        return {
            "valid": True, "reason": reason,
            "per_domain": {d: {"valid": True, "reason": reason} for d in domain_results},
        }

    # 복합 도메인(2개 이상)이면 먼저 전체 도메인 결과를 한 번에 합산 검증한다. 질문이
    # 여러 도메인에 나눠 걸쳐 있으면(예: "SK하이닉스 10년 골든크로스 전략" → kr(종가)+
    # backtest(수익률)) 각 도메인은 질문의 "자기 몫"만 정상 수행해도 되는데, 도메인
    # 하나만 떼어 전체 질문 기준으로 판정하면 "이 도메인 혼자서는 질문을 다 못 채운다"는
    # 이유로 결정론적으로(재시도해도 절대 안 고쳐짐) 매번 실패한다(실서버 재현 버그).
    # 합산 검증이 통과하면 그걸로 끝 — 도메인별 개별 호출로 낭비하지 않는다.
    if len(domain_results) > 1:
        try:
            joint_raw = llm_fn(_verify_prompt(question, domain_results)) or ""
            if not joint_raw.strip():
                # web의 _build_llm_fn은 LLM 호출 실패(quota 소진 등)를 예외가 아니라
                # 빈 문자열로 전파한다(LLMClient.complete가 예외를 삼키고 text="" 반환).
                # 빈 응답은 "잘못된 판정"이 아니라 "판정 불가" 신호 — 아래 except와 동일하게
                # 검증 불가로 처리한다. 안 그러면 LLM 장애 시 멀쩡한 데이터를 두고 재시도를
                # 전부 소모한 뒤 불확실 응답으로 끝난다(실사용 재현).
                joint_verdict = {
                    "valid": True,
                    "reason": "검증 불가(LLM 응답 없음) — 데이터 존재 확인만으로 통과시킴.",
                    "verification_unavailable": True,
                }
            else:
                joint_verdict = _parse_verdict(joint_raw)
        except Exception as exc:  # noqa: BLE001 — 검증 불가로 구분(위 단일도메인 분기와 동일 원칙)
            joint_verdict = {
                "valid": True,
                "reason": f"검증 불가(LLM 장애: {type(exc).__name__}) — 데이터 존재 확인만으로 통과시킴.",
                "verification_unavailable": True,
            }
        if joint_verdict["valid"]:
            verdict = {
                "valid": True, "reason": joint_verdict["reason"],
                "per_domain": {d: dict(joint_verdict) for d in domain_results},
            }
            if joint_verdict.get("verification_unavailable"):
                verdict["verification_unavailable"] = True
            return verdict
        # 합산 검증이 실제로 실패하면(진짜 문제) 아래에서 도메인별로 세분화해 원인을
        # 특정한다 — 실패한 도메인만 부분 재-dispatch할 수 있도록.

    # 도메인별로 개별 판정한다 — 복합 도메인 질문(kr+backtest 등)에서 일부 도메인만 검증에
    # 실패해도 그 도메인만 부분 재-dispatch할 수 있도록(answer_with_verification이 이
    # per_domain을 읽어 실패한 도메인만 재시도한다. 이미 통과한 도메인을 매번 다시
    # 실행하는 낭비를 없앤다). 이 프로젝트 스펙(AC3)이 요구하는 "검증 실패 시 최대
    # max_retries회 재시도, 무한루프 없음"(기본값은 2 — 실사용 관찰로 3→2로 줄임,
    # test_answer_with_verification_default_max_retries_is_two 참고) 자체는 그대로
    # 지킨다 — 부분/전체 재-dispatch는 그 시도 횟수 계약과 무관한 구현 디테일이다.
    per_domain: dict[str, dict] = {}
    for domain, result in domain_results.items():
        try:
            raw = llm_fn(_verify_prompt(question, {domain: result})) or ""
        except Exception as exc:  # noqa: BLE001 — LLM 호출 자체의 장애는 "검증 실패"가 아니라
            # "검증 불가"로 구분한다. 이 함수 상단에서 이미 데이터 존재를 확인했으므로
            # (_any_domain_has_data), OpenAI 등 일시 장애 때문에 멀쩡한 답을 버리지 않는다.
            per_domain[domain] = {
                "valid": True,
                "reason": f"검증 불가(LLM 장애: {type(exc).__name__}) — 데이터 존재 확인만으로 통과시킴.",
                "verification_unavailable": True,
            }
            continue
        if not raw.strip():
            # 위 합산 검증 분기와 동일 — 빈 응답(LLM 장애가 빈 문자열로 전파된 것)은
            # '검증 실패(재시도)'가 아니라 '검증 불가(통과)'다.
            per_domain[domain] = {
                "valid": True,
                "reason": "검증 불가(LLM 응답 없음) — 데이터 존재 확인만으로 통과시킴.",
                "verification_unavailable": True,
            }
            continue
        per_domain[domain] = _parse_verdict(raw)

    overall_valid = all(v["valid"] for v in per_domain.values())
    if overall_valid:
        unavailable_notes = [
            f"{d}: {v['reason']}" for d, v in per_domain.items() if v.get("verification_unavailable")
        ]
        reason = " / ".join(unavailable_notes) if unavailable_notes else "모든 도메인 검증 통과."
    else:
        reason = " / ".join(f"{d}: {v['reason']}" for d, v in per_domain.items() if not v["valid"])
    verdict = {"valid": overall_valid, "reason": reason, "per_domain": per_domain}
    if any(v.get("verification_unavailable") for v in per_domain.values()):
        verdict["verification_unavailable"] = True
    return verdict


# 프롬프트에 넣을 리스트 하나당 앞부분으로 남길 개수. 스크리닝 rows는 최대 1000행,
# 백테스트 시계열(dates/navs 등)도 길 수 있어 통째로 넣으면 프롬프트가 폭발한다.
# top_n 필드가 없는 리스트(free_exec 폴백의 구간별 집계 결과 등)는 전부 이 기본값을
# 그대로 쓴다. 실측 회귀: "코스피 전종목 PER 10구간 평균" 요청이 kr 도메인 실패 후
# free_exec 폴백으로 정확히 10구간 평균을 계산했는데도, top_n이 없어 기존 기본값(5)에
# 잘려 최종 답변 LLM이 "10개 중 5개만 있어 불완전하다"고 잘못 보고했다. 10~20분위 정도의
# 구간분석은 안전하게 다 보존하면서도, 원래 이 축약 로직이 막던 900종목급 원본 리스트
# 폭발과는 규모가 비교가 안 되므로 30으로 올려도 안전하다.
_PROMPT_LIST_HEAD = 30

# top_n 특혜(아래 docstring)의 상한. top_n이 이 값보다 크면(예: "코스피 전체"를 kr
# 에이전트가 top_n=4000으로 해석한 경우) 그 값을 그대로 쓰지 않고 상한까지만 연다 —
# 그렇지 않으면 KOSPI 전종목(약 900개 x 약 30개 필드) 같은 결과가 통째로 프롬프트에
# 들어가 verify/synthesize 프롬프트가 각각 약 77만자(31.7만 토큰)까지 불어나고, 실제
# API 호출로 약 $4.8가 소진되는 것까지 실측 확인됐다.
_PROMPT_TOP_N_CAP = 100


def _truncate_for_prompt(value, head: int = _PROMPT_LIST_HEAD):
    """LLM 프롬프트에 넣기 전 긴 리스트를 앞부분 몇 개 + 총 개수 요약으로 축약한다.

    dict/list 구조는 그대로 보존하되 head개를 넘는 리스트만 줄인다(재귀). 원본
    domain_results는 건드리지 않고 새 구조를 반환한다 — answer_with_verification이
    사용자에게 그대로 병기하는 원본 데이터("가공 없음" 원칙)와는 별개로, 이건 LLM에게
    검증/요약을 요청하는 프롬프트 텍스트 안의 데이터만 줄이는 용도다.

    스크리닝 결과 dict는 자신이 실제로 요청받은 개수(top_n)를 이미 알고 있다 — 그 값이
    고정 축약 상수(_PROMPT_LIST_HEAD=5)보다 크면, 그 형제 리스트(result 등)는 top_n개까지
    축약하지 않는다. 그렇지 않으면 "상위 10개"를 요청해도 검증/종합결론 LLM은 5개만 보고
    "10개 중 5개만 표시"라며 요청을 못 채운 것처럼 부정확하게 답한다(실사용 확인된 버그) —
    다만 top_n 자체에 상한이 없으면 "전체"처럼 사실상 무제한인 값도 그대로 다 열어버리므로,
    _PROMPT_TOP_N_CAP으로 상한을 둔다(상한 이내의 top_n은 지금처럼 전부 보존된다). top_n
    자체가 없는 리스트(백테스트 시계열 등)는 기존과 동일하게 head로 축약된다.
    """
    if isinstance(value, dict):
        local_head = head
        top_n = value.get("top_n")
        if isinstance(top_n, int) and top_n > local_head:
            local_head = min(top_n, _PROMPT_TOP_N_CAP)
        return {k: _truncate_for_prompt(v, local_head) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) <= head:
            return [_truncate_for_prompt(v, head) for v in value]
        head_items = [_truncate_for_prompt(v, head) for v in value[:head]]
        return head_items + [f"...(총 {len(value)}개 중 {head}개만 표시, 나머지 생략)"]
    return value


def _finalize_success(
    question: str, domain_results: dict, routes: list[str], attempts: int,
    synthesize_fn: Callable, llm_fn, chart_fallback_fn: Callable, chart_llm_fn,
) -> dict:
    """검증 통과 시 종합결론+원본 domain_results+(요청 시)차트를 묶어 반환한다.

    정형 재시도 루프의 성공 분기와 backtest 추가시도(escalation)의 성공 분기가 완전히
    동일한 마무리 로직(synthesize+차트)을 공유하므로 헬퍼로 뽑았다 — 새 로직 추가가 아니라
    기존 인라인 블록을 그대로 옮긴 것뿐이다(동작 변화 없음).
    """
    conclusion = synthesize_fn(question, domain_results, llm_fn)
    result = {
        "uncertain": False,
        "conclusion": conclusion,
        "domain_results": domain_results,
        "attempts": attempts,
        "routes": routes,
    }
    if wants_chart(question):
        chart = chart_fallback_fn(
            question, _chartable_payload(domain_results), chart_llm_fn or llm_fn
        )
        if chart:
            b64, title = chart["chart_base64"], chart.get("chart_title")
            result["chart_base64"], result["chart_title"] = b64, title
            result["charts"] = [{"chart_base64": b64, "chart_title": title}]
    return result


def _domain_results_json_for_prompt(domain_results: dict) -> str:
    return json.dumps(_truncate_for_prompt(domain_results), ensure_ascii=False, default=str)


def _verify_prompt(question: str, domain_results: dict) -> str:
    today = date.today().isoformat()
    return (
        f"오늘 날짜는 {today}입니다. 도메인 결과에 담긴 날짜/기간이 질문의 '오늘/현재/"
        "실시간' 같은 상대적 시점 표현과 맞는지는 이 오늘 날짜를 기준으로 판단하세요 — "
        "모델 자신의 학습 시점 기준으로 '미래처럼 보인다'는 이유만으로 valid=false를 "
        "주지 마세요(실사용 회귀: 실제로는 정확한 오늘자 데이터인데 검증 LLM이 날짜를 "
        "미래로 오판해 매번 불확실 처리되던 문제).\n"
        "아래 도메인 에이전트 결과가 사용자 질문에 실제로 답하는지 판정하세요.\n"
        "결과가 질문의 대상(종목/지표/기간 등)과 부합하면 valid=true, 어긋나면 valid=false.\n"
        "업종(sector) 필터가 쓰인 경우, 결과의 sectors 값이 질문의 업종 표현과 문자열로 정확히 "
        "같은지는 판정 대상이 아닙니다 — 하위 데이터 에이전트가 이미 실제 DB의 유효한 업종명으로 "
        "해석·검증했으므로 그 구체적 업종명 선택 자체는 재심사하지 마세요(예: 질문이 '반도체'이고 "
        "결과가 KRX 분류상 '전기·전자'여도 그 자체만으로 valid=false를 주지 마세요). 업종명 외의 "
        "나머지 조건(지표/방향/개수/기간 등)이 질문과 맞는지만 판정하세요.\n"
        "도메인 결과의 리스트가 '...(총 N개 중 M개만 표시, 나머지 생략)'로 축약돼 있으면, "
        "그건 프롬프트 길이 때문이지 실제 결과가 부족해서가 아닙니다 — 그 자체로 valid=false를 "
        "주지 마세요.\n"
        "도메인 결과가 2개 이상이면, 질문이 여러 도메인에 걸쳐 나뉘어 있을 수 있습니다 — 각 "
        "도메인이 전부 개별적으로 질문 전체에 답할 필요는 없고, 도메인 결과들을 종합했을 때 "
        "질문이 요구하는 내용이 이미 어딘가에 다 있으면 valid=true입니다(예: 한 도메인이 이미 "
        "z-score·분위·그래프까지 완전히 계산했다면, 다른 도메인이 더 단순한 형태의 목록만 "
        "추가로 제공한다고 해서 valid=false를 주지 마세요 — 이미 답이 된 요구사항을 다른 "
        "도메인이 혼자서는 중복 못 했다고 실패시키면 안 됩니다). 다만 질문이 명시적으로 서로 "
        "다른 여러 대상(예: 두 종목 비교)을 요구하는데 그중 일부가 어느 도메인 결과에도 전혀 "
        "없다면 valid=false를 주세요.\n"
        'JSON으로만 답하세요: {"valid": true/false, "reason": "간단한 이유"}\n\n'
        f"질문: {question}\n"
        f"도메인 결과: {_domain_results_json_for_prompt(domain_results)}\n답:"
    )


def _parse_verdict(text: str) -> dict:
    """LLM 응답에서 {"valid": bool, "reason": str} 판정을 추출한다.

    1) JSON({"valid":...})으로 오면 그대로 사용. 2) 아니면 키워드 마커로 판정하되,
    '불일치'가 '일치'를 포함하므로 **부정 마커를 먼저** 검사한다.
    """
    parsed = extract_json(text)
    if isinstance(parsed, dict) and "valid" in parsed:
        reason = str(parsed.get("reason") or "").strip() or "LLM 판정."
        return {"valid": bool(parsed["valid"]), "reason": reason}
    low = (text or "").lower()
    for marker in _INVALID_MARKERS:
        if marker.lower() in low:
            return {"valid": False, "reason": f"검증 실패({marker}): {text.strip()[:200]}"}
    for marker in _VALID_MARKERS:
        if marker.lower() in low:
            return {"valid": True, "reason": f"검증 통과({marker})."}
    # 판정 신호를 못 읽으면 안전하게 실패로 보고 재시도를 유발한다.
    return {"valid": False, "reason": f"검증 응답을 해석하지 못함: {text.strip()[:200]}"}


def synthesize_conclusion(
    question: str,
    domain_results: dict,
    llm_fn: Callable[[str], str] | None,
) -> str:
    """검증을 통과한 도메인 결과들로 최종 종합결론(문자열)을 만든다.

    llm_fn 주입 시 LLM으로 자연어 종합결론을 생성하고, 없거나 빈 응답이면 도메인별 핵심을
    나열한 결정론적 요약으로 폴백한다. 이 함수는 domain_results 를 **읽기만** 하며 원본을
    바꾸지 않는다(원본은 answer_with_verification 이 그대로 병기한다).
    """
    text = ""
    if llm_fn is not None:
        try:
            text = (llm_fn(_synthesize_prompt(question, domain_results)) or "").strip()
        except Exception:  # noqa: BLE001 — LLM 실패는 결정론 요약으로 폴백
            text = ""
    if not text:
        text = _deterministic_summary(question, domain_results)
    # 백테스트 리밸런싱 구간별 보유종목·구간수익률은 LLM 재량과 무관하게 최종 결론에 항상
    # 덧붙인다(domain_backtest가 만든 결정론적 텍스트를 그대로 병기 — 요구: "항상 포함").
    rebalance_block = _backtest_rebalance_block(domain_results)
    if rebalance_block:
        text = f"{text}\n\n{rebalance_block}"
    return text


def _backtest_rebalance_block(domain_results: dict) -> str | None:
    """domain_backtest가 붙인 결정론적 rebalance_summary 텍스트를 꺼낸다(없으면 None).

    다중 리밸런싱 백테스트일 때만 존재하는 순수 추가 필드다(단일 리밸런싱/비백테스트엔 없음).
    문자열이라 _truncate_for_prompt에 잘리지 않고, 여기서 결론 텍스트에 직접 병기하므로
    LLM이 보유종목/구간수익률을 언급하지 않아도 최종 답변에 반드시 포함된다."""
    bt = domain_results.get("backtest") if isinstance(domain_results, dict) else None
    if not isinstance(bt, dict):
        return None
    summary = bt.get("rebalance_summary")
    return summary if isinstance(summary, str) and summary.strip() else None


def _synthesize_prompt(question: str, domain_results: dict) -> str:
    return (
        "아래 여러 도메인 에이전트의 결과를 사용자 질문에 대한 하나의 종합결론으로 요약하세요.\n"
        "각 도메인 데이터를 왜곡하지 말고, 서로 상충하면 그 사실도 함께 밝히세요.\n"
        "도메인 결과에 'data_asof'(예: {\"price_date\":..., \"financial_quarter\":...})가 있으면, "
        "질문에 기간이 명시되지 않아 시스템이 자동으로 고른 '실제 데이터 기준시점'입니다 — "
        "사용자가 어느 시점 데이터인지 검증할 수 있도록 그 가격 기준일/재무 기준분기를 "
        "종합결론에 반드시 함께 밝히세요.\n"
        "스크리닝 결과 행에 '_same_company': true가 있으면, 그 행은 바로 앞의 '_same_company':"
        " false인 행과 동일 회사의 다른 상장 주식 종류(예: Class A/C)입니다 — 서로 다른"
        " 회사가 아니라 같은 회사임을 종합결론에서 반드시 밝히세요.\n"
        "도메인 결과에 'free_exec'.'verification_warning'이 있으면, 이 결과는 정형 검증 경로가"
        " 모두 실패해 LLM이 직접 작성한 코드로 얻은 값이며 재검증에서도 그 사유로 걸렸다는"
        " 뜻입니다 — 답변 서두에 자동검증을 통과하지 못했다는 점과 그 사유를 명시하고,"
        " 참고용으로만 제시된 값임을 사용자에게 알리세요(값 자체를 숨기거나 임의로 고치지"
        " 마세요).\n\n"
        f"질문: {question}\n"
        f"도메인 결과: {_domain_results_json_for_prompt(domain_results)}\n종합결론:"
    )


def _deterministic_summary(question: str, domain_results: dict) -> str:
    """LLM 없이도 항상 비어있지 않은 종합결론을 만드는 폴백 요약."""
    parts = [f"질문: {question}"]
    for domain in _DOMAINS:
        if domain not in domain_results:
            continue
        parts.append(f"[{domain}] {_summarize_one(domain, domain_results[domain])}")
    return " | ".join(parts)


def _summarize_one(domain: str, result: dict) -> str:
    if not isinstance(result, dict):
        return str(result)
    if result.get("error"):
        return f"오류: {result['error']}"
    if domain == "macro":
        return f"종합신호={result.get('overall')} (스프레드 레짐={result.get('spread', {}).get('regime')})"
    if domain == "kr":
        fin = result.get("financial")
        price = result.get("price")
        bits = []
        if fin:
            bits.append(f"재무={fin.get('value') if isinstance(fin, dict) else fin}")
        if price:
            bits.append("주가 데이터 포함")
        entities = result.get("entities")
        if isinstance(entities, list) and entities:
            codes = [
                e.get("stock_code") for e in entities
                if isinstance(e, dict) and (e.get("financial") or e.get("price"))
            ]
            if codes:
                bits.append(f"다중종목 데이터 포함({', '.join(codes)})")
        # 기간 미지정 시 자동으로 정해진 실제 데이터 기준시점(가격 기준일/재무 기준분기)을
        # 결정론 요약에도 명시한다 — LLM 없이도 사용자가 "언제 기준 데이터냐"를 확인할 수 있게.
        asof = result.get("data_asof")
        if isinstance(asof, dict) and asof:
            asof_bits = []
            if asof.get("price_date"):
                asof_bits.append(f"가격 기준일 {asof['price_date']}")
            if asof.get("financial_quarter"):
                asof_bits.append(f"재무 기준분기 {asof['financial_quarter']}")
            if asof_bits:
                bits.append("기준시점: " + ", ".join(asof_bits))
        return ", ".join(bits) or "데이터 없음"
    if domain == "backtest":
        if result.get("blocked"):
            return "하드차단됨(결과 폐기)"
        bits = [f"백테스트 결과={result.get('result')}"]
        asof = result.get("data_asof")
        if isinstance(asof, dict) and asof:
            asof_bits = []
            if asof.get("price_date"):
                asof_bits.append(f"가격 기준일 {asof['price_date']}")
            if asof.get("financial_quarter"):
                asof_bits.append(f"재무 기준분기 {asof['financial_quarter']}")
            if asof_bits:
                bits.append("기준시점: " + ", ".join(asof_bits))
        return ", ".join(bits)
    return "데이터 있음"


def answer_with_verification(
    question: str,
    conn,
    llm_fn: Callable[[str], str] | None,
    max_retries: int = 2,
    route_fn: Callable | None = None,
    dispatch_fn: Callable | None = None,
    verify_fn: Callable | None = None,
    synthesize_fn: Callable | None = None,
    steps: list[dict] | None = None,
    on_progress: Callable[[str, str], None] | None = None,
    fallback_fn: Callable | None = None,
    chart_fallback_fn: Callable | None = None,
    chart_llm_fn: Callable[[str], str] | None = None,
) -> dict:
    """총괄 오케스트레이션: route→dispatch→verify, 실패 시 정확히 max_retries회까지 재시도.

    흐름:
      1) route_fn(question, llm_fn) 으로 도메인 라우팅(한 번만).
      2) 최대 max_retries(기본 2)회 반복: dispatch_fn → verify_fn.
         - 검증 통과 시: synthesize_fn 으로 종합결론을 만들고, **원본 domain_results 를
           가공 없이 그대로 병기**해 반환한다.
         - 검증 실패 시: 재시도. **정확히 max_retries회에서 멈춘다(무한루프 없음)** — 즉
           4번째 시도가 없다.
      3) max_retries회 모두 실패하면, 정형 검증 루프와는 별개인 마지막 안전망으로
         fallback_fn(question, conn, llm_fn, last_reason)을 **정확히 1회만** 시도한다
         (exec_fallback.run_free_exec_fallback — LLM이 SQL+Python을 직접 작성해
         exec_runtime.py의 안전장치 위에서 실행). 정형 어휘(스크리닝 criteria/top_n,
         파이프라인 연산)의 표현력 한계로 반복 실패하는 질문을 위한 최후 수단이라, 실패해도
         fallback_fn을 다시 시도하지는 않는다(무한루프 방지 원칙 유지). 성공하면 domain_results에
         "free_exec" 키로 결과를 추가하고, verify_fn을 **정확히 1회** 더 통과시켜 재검증한다
         (재시도는 없음 — 검증 실패해도 결과를 버리지 않고 "free_exec"."verification_warning"에
         사유만 남긴다. 최후 수단이라 대안이 없으므로 답 자체는 유지하되, synthesize_fn이 최종
         답변에 신뢰도 유보 문구를 붙일 근거를 제공한다 — 실사용 재현: "PER z-score 히스토그램"
         요청이 z-score 없이 원본 PER로만 나온 결과가 무검증으로 그대로 나갔던 문제).
         uncertain은 재검증 통과 여부와 무관하게 항상 False로 반환한다. fallback_fn 자체가
         실패하면(ok=False) 기존과 동일하게 "불확실성 명시" 응답을 반환한다(uncertain=True).

    복합 도메인(kr+backtest 등) 질문에서는 verify_fn이 반환하는 verdict["per_domain"]을 읽어,
    이미 검증을 통과한 도메인은 그대로 두고 **실패한 도메인만** 다음 시도에서 재-dispatch한다
    (매번 전체 도메인을 다시 실행하는 낭비를 없앤다). verify_fn이 per_domain을 안 주면(예:
    단위테스트가 주입하는 단순 fake) 기존처럼 전체 routes를 재-dispatch한다 — 하위호환.
    시도 횟수 상한(정확히 max_retries회, 무한루프 없음)은 이 부분/전체 재-dispatch 여부와
    무관하게 그대로 지켜진다.

    route_fn/dispatch_fn/verify_fn/synthesize_fn 은 모두 주입 가능하다(기본값은 이 모듈의
    구현) — 단위테스트가 재시도 횟수/원본 보존을 결정론적으로 검증할 수 있게 하고, HA-11이
    LangGraph 노드로 감쌀 때 부분 교체도 가능하게 한다.

    성공 반환:
        {"uncertain": False, "conclusion": str, "domain_results": {...},
         "attempts": int, "routes": [...]}
    실패(불확실) 반환:
        {"uncertain": True, "reason": str, "attempts": max_retries,
         "domain_results": {...}, "routes": [...]}
    """
    route_fn = route_fn or route_question
    dispatch_fn = dispatch_fn or dispatch_domains
    verify_fn = verify_fn or verify_answer
    synthesize_fn = synthesize_fn or synthesize_conclusion
    fallback_fn = fallback_fn or run_free_exec_fallback
    chart_fallback_fn = chart_fallback_fn or build_chart_freeform

    # on_progress가 없으면(대부분의 기존 호출부) 하위 함수에 그 키워드 자체를 안 넘긴다 —
    # 주입되는 fake route_fn/dispatch_fn(테스트)이 on_progress 파라미터를 몰라도 깨지지 않는다.
    progress_kwargs = {"on_progress": on_progress} if on_progress else {}

    routes = route_fn(question, llm_fn, **progress_kwargs)

    if not routes:
        # 라우팅이 도메인을 하나도 못 찾음(unknown) — dispatch/verify를 시도하는 낭비 없이
        # 즉시 불확실 응답으로 끝낸다(완전히 무관한 질문에 억지로 도메인을 갖다붙이지 않는다).
        if on_progress:
            on_progress("supervisor", "질문을 이해하지 못했습니다 — 처리 가능한 도메인을 찾지 못함")
        return {
            "uncertain": True,
            "reason": "질문을 이해하지 못했습니다. 국내 주식, 매크로 지표, 백테스트 중 "
                      "어떤 것에 대한 질문인지 좀 더 구체적으로 말씀해 주세요.",
            "attempts": 0,
            "domain_results": {},
            "routes": [],
        }

    domain_results: dict = {}
    last_reason: str | None = None
    attempts = 0
    routes_to_dispatch = list(routes)  # 1차는 전체, 이후 실패한 도메인만(부분 재시도)
    for attempt in range(1, max_retries + 1):
        attempts = attempt
        # 재시도(2회차부터)에는 직전 실패 사유를 도메인 실행 질문에 피드백으로 덧붙인다.
        # 안 붙이면 도메인 에이전트(LLM)가 매 시도마다 완전히 동일한 접근을 반복해 검증에
        # 또 실패하고 토큰만 낭비한다(실사용 재현: "직전 12개월 수익률" 질문에 매번 똑같이
        # revenue_growth로 잘못 스크리닝). 검증(verify_fn)/종합결론(synthesize_fn)에는
        # 피드백이 섞이지 않은 원본 question을 그대로 넘겨 판정 자체가 왜곡되지 않게 한다.
        dispatch_question = question
        if last_reason:
            dispatch_question = (
                f"{question}\n\n"
                f"[이전 시도 실패 피드백] 직전 시도가 다음 이유로 검증에 실패했습니다: {last_reason}\n"
                f"같은 방식을 그대로 반복하지 말고, 이 피드백을 반영해 다른 접근으로 다시 답하세요."
            )
        new_results = dispatch_fn(
            routes_to_dispatch, dispatch_question, conn, llm_fn, steps=steps, **progress_kwargs
        )
        # 부분 재시도 시 이전에 검증 통과한 도메인 결과는 그대로 유지하고, 이번에 (재)실행한
        # 도메인만 덮어쓴다(update). 1차 시도는 routes_to_dispatch가 전체이므로 기존과 동일.
        domain_results.update(new_results)
        verdict = verify_fn(question, domain_results, llm_fn)
        if verdict.get("valid"):
            if on_progress:
                on_progress("verify", f"{attempt}차 검증 통과")
            # 명시적 차트 요청("그래프/차트/그려줘" 등)이 원본 question에 있을 때만 이미지 차트를
            # 붙인다(모든 질문에 자동으로 붙이지 않음). 차트 종류 판정은 결정론적 패턴매칭 없이
            # 전적으로 LLM(chart_fallback_fn=build_chart_freeform)에 위임한다 — matplotlib 전체에서
            # 질문·데이터에 맞는 종류를 자유롭게 고른다. (옛 결정론 경로가 숫자 필드 1개짜리
            # 스크리닝 결과에서 같은 필드를 x·y에 둘 다 배정해 "pbr vs pbr 산점도"를 그리던 오판이
            # 이 구조 변경으로 근본 소거된다.) 차트 판단용 LLM은 chart_llm_fn(기본=llm_fn; web가
            # 저가 role="chart"로 별도 주입 가능)이 맡는다. llm_fn 미가용/코드생성·실행 실패 시
            # None → 차트 없이 텍스트 응답만(부가 기능이라 본문 응답을 절대 무너뜨리지 않는다).
            return _finalize_success(
                question, domain_results, routes, attempt,
                synthesize_fn, llm_fn, chart_fallback_fn, chart_llm_fn,
            )
        last_reason = verdict.get("reason")
        per_domain = verdict.get("per_domain") or {}
        if per_domain:
            routes_to_dispatch = [d for d in routes if not per_domain.get(d, {}).get("valid", False)]
        else:
            routes_to_dispatch = list(routes)  # per_domain 정보 없는 verify_fn 하위호환
        if on_progress:
            suffix = " → 재시도" if attempt < max_retries else ""
            on_progress("verify", f"{attempt}차 검증 실패: {last_reason}{suffix}")

    # 정형 검증 루프가 모두 실패했고, 원래 라우팅에 backtest가 없었다면 — kr처럼 "구간별
    # 집계" 같은 표현력이 없는 도메인은 피드백을 아무리 줘도 같은 종류의 답만 반복하므로,
    # free_exec(자유 코드 생성)으로 넘어가기 전에 backtest를 한 번 더 시도한다(정확히 1회,
    # 무한루프 없음 — 라우팅이 이미 backtest를 포함했다면 중복 시도하지 않는다. 실사용
    # 재현: "PER 10구간 평균"이 라우팅 실수로 kr에만 갔던 문제의 안전망).
    if "backtest" not in routes:
        if on_progress:
            on_progress(
                "supervisor", f"{max_retries}회 정형 검증 실패 → backtest 도메인 추가시도",
            )
        escalation_results = dispatch_fn(
            ["backtest"], question, conn, llm_fn, steps=steps, **progress_kwargs
        )
        domain_results.update(escalation_results)
        verdict = verify_fn(question, domain_results, llm_fn)
        if verdict.get("valid"):
            if on_progress:
                on_progress("verify", "backtest 추가시도 검증 통과")
            result = _finalize_success(
                question, domain_results, routes + ["backtest"], attempts,
                synthesize_fn, llm_fn, chart_fallback_fn, chart_llm_fn,
            )
            result["used_backtest_escalation"] = True
            return result
        last_reason = verdict.get("reason")
        if on_progress:
            on_progress("verify", f"backtest 추가시도도 검증 실패: {last_reason}")

    # 정형 검증 루프가 모두 실패 — 정형 어휘의 표현력 한계일 수 있으므로 마지막으로 딱 1회,
    # LLM이 SQL+Python을 직접 작성하는 자유 실행 폴백을 시도한다(재시도 루프와 무관, 검증도
    # 다시 거치지 않음 — 그러면 같은 이유로 다시 실패해 무한루프에 가까워진다).
    fallback = fallback_fn(question, conn, llm_fn, last_reason)
    if fallback.get("ok"):
        if on_progress:
            on_progress(
                "supervisor", f"{max_retries}회 정형 검증 실패 → 자유 코드 생성 폴백 성공",
            )
        domain_results["free_exec"] = {
            "fallback_used": True,
            "sql": fallback.get("sql"),
            "code": fallback.get("code"),
            "result": fallback.get("result"),
        }
        # 재검증 안전망: 폴백 자체는 재시도하지 않는다(무한루프 방지 원칙 유지 — fallback_fn은
        # 위에서 이미 정확히 1회만 호출됐다). 다만 검증을 아예 안 거치면, 정형 어휘 한계로
        # 실패한 게 아니라 폴백이 스스로 질문을 잘못 이해한 경우(예: "PER z-score 히스토그램"을
        # z-score 없이 원본 PER로만 답함)까지 조용히 "정상 답변"으로 나갈 수 있다. 실패해도
        # 이 결과를 버리지는 않고(최후 수단이라 버리면 대안이 없다) 사유만 남겨, 최종 답변에
        # 신뢰도 유보 문구를 붙일 근거를 synthesize_fn에 제공한다.
        fallback_verdict = verify_fn(question, domain_results, llm_fn)
        if not fallback_verdict.get("valid"):
            domain_results["free_exec"]["verification_warning"] = fallback_verdict.get("reason")
            if on_progress:
                on_progress(
                    "verify",
                    f"자유 코드 폴백 결과 재검증 실패(참고용으로 표시): {fallback_verdict.get('reason')}",
                )
        elif on_progress:
            on_progress("verify", "자유 코드 폴백 결과 재검증 통과")
        conclusion = synthesize_fn(question, domain_results, llm_fn)
        return {
            "uncertain": False,
            "conclusion": conclusion,
            "domain_results": domain_results,
            "attempts": attempts,
            "routes": routes,
            "used_fallback": True,
        }
    if on_progress:
        on_progress(
            "supervisor",
            f"{max_retries}회 정형 검증 실패 → 자유 코드 생성 폴백도 실패: {fallback.get('error')}",
        )

    return {
        "uncertain": True,
        "reason": (
            f"{max_retries}회 검증에 모두 실패했습니다. 확실한 답을 제시할 수 없습니다."
            + (f" (마지막 사유: {last_reason})" if last_reason else "")
            + (f" (자유 코드 생성 폴백도 실패: {fallback.get('error')})" if fallback.get("error") else "")
        ),
        "attempts": attempts,
        "domain_results": domain_results,
        "routes": routes,
    }
