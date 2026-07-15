"""총괄 에이전트 (HA-10) — 계층형 멀티에이전트의 최상위 "총괄 로직"을 순수 함수로 구현한다.

.omc/state/sessions/f6b79d90-b7b2-4b4f-b4f7-405a8355724e/prd.json 의 HA-10 참고.

계층: **이 총괄 에이전트** → 도메인 에이전트(HA-6 domain_kr / HA-7 domain_us /
HA-8 domain_macro / HA-9 domain_backtest) → 데이터 에이전트. 이 파일은 이미 완성된 4개
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
      질문 → 도메인 리스트(["kr"] | ["us"] | ["kr","us"] | ["macro"] | ["backtest"] ...).
- dispatch_domains(routes, question, conn, llm_fn, steps=None) -> dict
      각 도메인 answer_*_question 을 호출해 **원본 결과를 가공 없이** 도메인별 키로 보존.
- verify_answer(question, domain_results, llm_fn) -> {"valid": bool, "reason": str}
      도메인 결과가 원 질문과 부합하는지 판정(결정론적 규칙 우선 + LLM 판정).
- answer_with_verification(...) -> dict
      route→dispatch→verify 순서로 실행, 검증 실패 시 **정확히 max_retries(기본 3)회까지만**
      재시도(무한루프 없음). 실패 시 "불확실성 명시" 응답(uncertain=True), 통과 시 종합결론
      (synthesize_conclusion)과 원본 domain_results 를 함께 반환.

llm_fn 규약: 이 프로젝트의 도메인 에이전트들과 동일하게 `Callable[[str], str]`(prompt→text).
호출부(HA-11/nodes.py)는 `lambda p: (deps.llm.complete(p, role="judge").text or "")` 형태로
주입한다(src/graph/nodes.py 참고). llm_fn 이 None이면 결정론적 휴리스틱으로 폴백한다.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Callable

from src.agents.charting import render_line_chart_base64
from src.agents.data_price_kr import get_price_history_kr
from src.agents.data_price_us import get_price_history_us
from src.agents.domain_backtest import answer_backtest_question
from src.agents.domain_kr import answer_kr_question
from src.agents.domain_macro import answer_macro_question, get_macro_history
from src.agents.domain_us import answer_us_question
from src.llm import extract_json

# 정규 도메인 순서 — LLM 응답의 순서/중복과 무관하게 항상 이 순서로 정렬해 결정론을 보장한다.
_DOMAINS: tuple[str, ...] = ("kr", "us", "macro", "backtest")

# on_progress 이벤트 라벨용 — web/static/index.html의 TREE_LABEL/treeDepth와 동일한 도메인
# 이름(step)을 그대로 써야 프론트가 별도 분기 없이 트리 깊이를 매긴다(graph.py의
# _DOMAIN_LABELS와 같은 매핑이지만, graph.py가 이 모듈을 import하므로 순환을 피해 로컬로 둔다).
_DOMAIN_LABELS_KO: dict[str, str] = {"kr": "한국", "us": "미국", "macro": "매크로", "backtest": "백테스트"}

# llm_fn 미주입 시 사용하는 라우팅 휴리스틱. 도메인별 대표 키워드(부분일치).
_ROUTE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "backtest": ("백테스트", "backtest", "리밸런", "전략 수익", "전략을 검증", "전략 검증"),
    "macro": ("매크로", "금리차", "스프레드", "장단기", "공포탐욕", "vix", "레짐", "매크로 신호"),
    "us": ("애플", "apple", "미국", "나스닥", "s&p", "테슬라", "엔비디아", "aapl", "tsla", "nvda"),
    "kr": ("삼성", "코스피", "코스닥", "국내", "한국주식"),
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

    llm_fn(주입 시)에 라우팅을 위임하고, 그 응답 텍스트에서 도메인 토큰(kr/us/macro/
    backtest)을 추출해 **정규 순서(_DOMAINS)로 정렬·중복제거**한다(예: "삼성전자 vs 애플"
    → LLM이 'kr, us' 응답 → ["kr","us"]). LLM 응답이 JSON 리스트든 콤마 나열이든
    도메인 토큰만 뽑으므로 형식에 관대하다.

    llm_fn 이 None이거나 응답에서 도메인을 하나도 못 찾으면 키워드 휴리스틱으로 폴백하고,
    그래도 못 찾으면 안전 기본값 ["kr"](국내주식)로 폴백한다.

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
        "가능한 도메인: kr(국내주식 재무/주가), us(미국주식 재무/주가), "
        "macro(매크로 신호/금리차), backtest(전략 백테스트).\n"
        "여러 도메인이 필요하면 모두 나열하세요(예: 국내 vs 미국 비교 → kr, us).\n"
        "도메인 키워드만 콤마로 구분해 답하세요.\n\n"
        f"질문: {question}\n답:"
    )


def _extract_domains(text: str) -> list[str]:
    """임의의 텍스트(콤마 나열/JSON 리스트 등)에서 도메인 토큰만 뽑아 정규 순서로 정리."""
    tokens = set(re.findall(r"[a-z]+", (text or "").lower()))
    return [d for d in _DOMAINS if d in tokens]


def _route_heuristic(question: str) -> list[str]:
    """llm_fn 미가용 시 키워드 기반 라우팅. 아무것도 안 걸리면 ["kr"]로 폴백."""
    q = (question or "").lower()
    found = {
        domain
        for domain, keywords in _ROUTE_KEYWORDS.items()
        if any(kw.lower() in q for kw in keywords)
    }
    routes = [d for d in _DOMAINS if d in found]
    return routes or ["kr"]


def wants_chart(question: str) -> bool:
    """질문이 명시적으로 그래프/차트를 요청하는지 결정론적으로 판단한다(_ROUTE_KEYWORDS 스타일).

    LLM을 쓰지 않고 키워드 부분일치로만 판단한다 — "차트를 그릴지 말지"는 파이썬 코드가
    결정한다(프롬프트에 코드를 짜라고 시키지 않는다). 재시도 시 실패 피드백이 덧붙은
    dispatch_question이 아니라 **원본 question**으로 판단해야 하므로(verify_fn/synthesize_fn이
    원본 question을 쓰는 것과 동일한 이유), 호출부는 항상 원본 question을 넘긴다.
    """
    q = (question or "").lower()
    return any(kw.lower() in q for kw in _CHART_KEYWORDS)


def _series_from_history(rows: list[dict], date_key: str, value_key: str) -> tuple[list, list]:
    """히스토리 rows에서 (dates, values)를 뽑되 값이 None인 지점은 함께 건너뛴다(정렬 유지)."""
    dates: list = []
    values: list = []
    for r in rows:
        v = r.get(value_key)
        if v is None:
            continue
        dates.append(r.get(date_key))
        values.append(v)
    return dates, values


def _extract_chart_data(domain_results: dict, conn) -> tuple[list, dict, str] | None:
    """domain_results에서 그릴 시계열을 우선순위(backtest > kr/us 가격 > macro)로 하나만 고른다.

    반환: (dates, {"라벨": [값들], ...}, title). 그릴 데이터가 전혀 없으면(스크리닝 결과처럼
    단일 시계열이 아닌 경우 등) 조용히 None(에러 아님 — 차트 없이 텍스트 응답만).

    이번 스코프는 "질문당 차트 1개"이므로 여러 조건이 동시에 해당해도 위 우선순위로 하나만
    그린다. kr/us 가격 시계열은 성능상 domain_results에 없으므로 get_price_history_*로
    최근 1년 종가를 조회하고, macro는 spread 시계열 하나만 그린다(다른 지표 동시 표시 안 함).
    """
    # 1) 백테스트 — 이미 시계열 전체(dates/navs/benchmark)를 담고 있어 재조회 불필요.
    bt = domain_results.get("backtest")
    if isinstance(bt, dict):
        res = bt.get("result")
        if isinstance(res, dict) and res.get("dates") and res.get("navs"):
            series: dict = {"전략": list(res["navs"])}
            bench = res.get("benchmark")
            if bench:
                series["벤치마크"] = list(bench)
            return list(res["dates"]), series, "백테스트 결과"

    # 2) 단일 종목 가격 — kr/us 도메인 결과에 stock_code가 있으면(단일종목 조회 케이스)
    #    최근 1년 종가 시계열을 조회해 그린다(스크리닝 결과엔 stock_code가 없어 자연히 제외).
    for domain, history_fn in (("kr", get_price_history_kr), ("us", get_price_history_us)):
        d = domain_results.get(domain)
        if isinstance(d, dict) and d.get("stock_code"):
            code = d["stock_code"]
            dates, closes = _series_from_history(history_fn(conn, code), "date", "close")
            if closes:
                return dates, {f"{code} 종가": closes}, f"{code} 최근 종가 추이"

    # 3) 매크로 — spread(장단기금리차) 시계열 하나만.
    macro = domain_results.get("macro")
    if isinstance(macro, dict) and macro.get("available"):
        dates, spreads = _series_from_history(get_macro_history(conn), "as_of", "spread")
        if spreads:
            return dates, {"장단기금리차": spreads}, "장단기 금리차 추이"

    return None


def _build_chart(domain_results: dict, conn) -> tuple[str, str] | None:
    """_extract_chart_data로 데이터를 고르고 render_line_chart_base64로 PNG(base64)를 만든다.

    반환: (chart_base64, chart_title) 또는 None(그릴 데이터가 없거나 렌더링 실패 시). 차트는
    부가 기능이므로, 렌더링이 어떤 이유로든 실패해도 예외를 전파하지 않고 None으로 흡수해
    본문 텍스트 응답이 깨지지 않게 한다.
    """
    data = _extract_chart_data(domain_results, conn)
    if data is None:
        return None
    dates, series, title = data
    try:
        chart_base64 = render_line_chart_base64(dates, series, title)
    except Exception:  # noqa: BLE001 — 차트 실패가 본문 응답을 무너뜨리지 않게 한다
        return None
    return chart_base64, title


def dispatch_domains(
    routes: list[str],
    question: str,
    conn,
    llm_fn: Callable[[str], str] | None,
    steps: list[dict] | None = None,
    on_progress: Callable[[str, str], None] | None = None,
) -> dict:
    """routes 의 각 도메인 answer_*_question 을 호출하고 **원본 결과를 가공 없이** 보존한다.

    반환: {"kr": {...}, "us": {...}, "macro": {...}, "backtest": {...}} — routes 에 포함된
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
        try:
            if domain == "kr":
                results["kr"] = answer_kr_question(question, conn, llm_fn=llm_fn)
            elif domain == "us":
                results["us"] = answer_us_question(question, conn, llm_fn=llm_fn)
            elif domain == "macro":
                results["macro"] = answer_macro_question(question, conn)
            elif domain == "backtest":
                bt_kwargs = {"on_progress": on_progress} if on_progress else {}
                results["backtest"] = answer_backtest_question(
                    question, steps or [], conn, llm_fn=llm_fn, **bt_kwargs
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
    # macro: available=True.
    if result.get("available"):
        return True
    # backtest: 차단되지 않고 결과가 있으면.
    if result.get("result") is not None and not result.get("blocked"):
        return True
    # us: ok=True(스키마상 error 없음).
    if result.get("ok") and (result.get("financial") or result.get("price")):
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
        return {"valid": True, "reason": f"전략 탐색(역백테스트) 결과가 유효합니다 — {note}."}
    if llm_fn is None:
        return {"valid": True, "reason": "결정론적 검증 통과(도메인 데이터 존재, LLM 미가용)."}
    try:
        raw = llm_fn(_verify_prompt(question, domain_results)) or ""
    except Exception as exc:  # noqa: BLE001 — LLM 실패는 판정 불가로 처리(재시도 유발)
        return {"valid": False, "reason": f"검증 LLM 호출 실패: {type(exc).__name__}"}
    return _parse_verdict(raw)


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
        'JSON으로만 답하세요: {"valid": true/false, "reason": "간단한 이유"}\n\n'
        f"질문: {question}\n"
        f"도메인 결과: {domain_results}\n답:"
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
    if llm_fn is not None:
        try:
            text = (llm_fn(_synthesize_prompt(question, domain_results)) or "").strip()
        except Exception:  # noqa: BLE001 — LLM 실패는 결정론 요약으로 폴백
            text = ""
        if text:
            return text
    return _deterministic_summary(question, domain_results)


def _synthesize_prompt(question: str, domain_results: dict) -> str:
    return (
        "아래 여러 도메인 에이전트의 결과를 사용자 질문에 대한 하나의 종합결론으로 요약하세요.\n"
        "각 도메인 데이터를 왜곡하지 말고, 서로 상충하면 그 사실도 함께 밝히세요.\n"
        "스크리닝 결과 행에 '_same_company': true가 있으면, 그 행은 바로 앞의 '_same_company':"
        " false인 행과 동일 회사의 다른 상장 주식 종류(예: Class A/C)입니다 — 서로 다른"
        " 회사가 아니라 같은 회사임을 종합결론에서 반드시 밝히세요.\n\n"
        f"질문: {question}\n"
        f"도메인 결과: {domain_results}\n종합결론:"
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
    if domain in ("kr", "us"):
        fin = result.get("financial")
        price = result.get("price")
        bits = []
        if fin:
            bits.append(f"재무={fin.get('value') if isinstance(fin, dict) else fin}")
        if price:
            bits.append("주가 데이터 포함")
        return ", ".join(bits) or "데이터 없음"
    if domain == "backtest":
        if result.get("blocked"):
            return "하드차단됨(결과 폐기)"
        return f"백테스트 결과={result.get('result')}"
    return "데이터 있음"


def answer_with_verification(
    question: str,
    conn,
    llm_fn: Callable[[str], str] | None,
    max_retries: int = 3,
    route_fn: Callable | None = None,
    dispatch_fn: Callable | None = None,
    verify_fn: Callable | None = None,
    synthesize_fn: Callable | None = None,
    steps: list[dict] | None = None,
    on_progress: Callable[[str, str], None] | None = None,
) -> dict:
    """총괄 오케스트레이션: route→dispatch→verify, 실패 시 정확히 max_retries회까지 재시도.

    흐름:
      1) route_fn(question, llm_fn) 으로 도메인 라우팅(한 번만).
      2) 최대 max_retries(기본 3)회 반복: dispatch_fn → verify_fn.
         - 검증 통과 시: synthesize_fn 으로 종합결론을 만들고, **원본 domain_results 를
           가공 없이 그대로 병기**해 반환한다.
         - 검증 실패 시: 재시도. **정확히 max_retries회에서 멈춘다(무한루프 없음)** — 즉
           4번째 시도가 없다.
      3) max_retries회 모두 실패하면 "불확실성 명시" 응답을 반환한다(uncertain=True).

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

    # on_progress가 없으면(대부분의 기존 호출부) 하위 함수에 그 키워드 자체를 안 넘긴다 —
    # 주입되는 fake route_fn/dispatch_fn(테스트)이 on_progress 파라미터를 몰라도 깨지지 않는다.
    progress_kwargs = {"on_progress": on_progress} if on_progress else {}

    routes = route_fn(question, llm_fn, **progress_kwargs)

    domain_results: dict = {}
    last_reason: str | None = None
    attempts = 0
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
        domain_results = dispatch_fn(
            routes, dispatch_question, conn, llm_fn, steps=steps, **progress_kwargs
        )
        verdict = verify_fn(question, domain_results, llm_fn)
        if verdict.get("valid"):
            if on_progress:
                on_progress("verify", f"{attempt}차 검증 통과")
            conclusion = synthesize_fn(question, domain_results, llm_fn)
            result = {
                "uncertain": False,
                "conclusion": conclusion,
                "domain_results": domain_results,
                "attempts": attempt,
                "routes": routes,
            }
            # 명시적 차트 요청("그래프/차트/그려줘" 등)이 원본 question에 있을 때만 이미지 차트를
            # 붙인다(모든 질문에 자동으로 붙이지 않음). 그릴 데이터가 없으면 None(차트 없이 텍스트만).
            if wants_chart(question):
                chart = _build_chart(domain_results, conn)
                if chart is not None:
                    result["chart_base64"], result["chart_title"] = chart
            return result
        last_reason = verdict.get("reason")
        if on_progress:
            suffix = " → 재시도" if attempt < max_retries else ""
            on_progress("verify", f"{attempt}차 검증 실패: {last_reason}{suffix}")

    return {
        "uncertain": True,
        "reason": (
            f"{max_retries}회 검증에 모두 실패했습니다. 확실한 답을 제시할 수 없습니다."
            + (f" (마지막 사유: {last_reason})" if last_reason else "")
        ),
        "attempts": attempts,
        "domain_results": domain_results,
        "routes": routes,
    }
