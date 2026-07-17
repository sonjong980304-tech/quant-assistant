"""계층형 총괄 그래프 (HA-11) — HA-10의 총괄 로직을 LangGraph StateGraph 노드로 감싼다.

.omc/state/sessions/f6b79d90-b7b2-4b4f-b4f7-405a8355724e/prd.json 의 HA-11 참고.

HA-10(src/agents/supervisor.py)은 라우팅·도메인실행·정합성검증·재시도를 이미
`answer_with_verification(...)` 하나로 통합해 두었다. 이 파일은 **새 로직을 만들지 않고**
그 총괄 함수를 LangGraph 노드(`supervisor_node`)로 감싸 StateGraph 로 조립하고, 실행을
`.invoke()`가 아니라 `.stream()`으로 돌려 노드 완료 시점마다 진행 이벤트를 방출한다.

--- AC5(코드리뷰) 근거 --------------------------------------------------------
"총괄 에이전트의 라우팅·검증 로직이 LangGraph 노드로 구현되어 있다"는 것이 핵심이다.
그 라우팅(route_question)·검증(verify_answer)·재시도(answer_with_verification) 로직은
전부 supervisor_node 안에서 answer_with_verification 호출로 실행된다. 총괄 노드가 이미
route→dispatch→verify→retry 를 내부에서 수행하므로, 이 스토리에서는 그래프를 단일 노드
(START→supervisor→END)로 둔다 — 도메인 실행 자체는 dispatch_domains 내부 순수 함수
호출로 처리되므로 그래프 레벨에서 도메인마다 노드를 쪼갤 필요가 없다. 중요한 것은
`.stream()`으로 노드 완료마다 이벤트가 나오는 것이다.

--- 스트리밍 이벤트 스키마(HA-12 SSE 인계용) ----------------------------------
각 노드 완료 이벤트는 **단계 이름 + 핵심 결과 한 줄**만 담는다(SQL 전문/원본 rows/
결론 본문 같은 상세는 넣지 않는다).
    {"step": "supervisor", "summary": "한국+미국 도메인 라우팅, 검증 통과(2회 시도)"}
최종 결론/원본 도메인 결과는 이벤트가 아니라 그래프 최종 상태로 얻는다 — 동기 호출부는
run_hierarchical, 스트리밍 호출부는 run_streaming(..., out_final={})로 같은 실행 한 번에서
받는다(진행상황용/최종답변용으로 그래프를 두 번 돌리지 않기 위함, HA-12 후속 수정).

--- 상태 주의(Python 3.9 호환) ------------------------------------------------
src/graph/state.py 와 동일하게 LangGraph가 get_type_hints 로 런타임 평가하므로
`X | None` 대신 Optional[...] 을 쓴다.
"""
from __future__ import annotations

import operator
import queue
import threading
from typing import Annotated, Any, Callable, Dict, Iterator, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.supervisor import answer_with_verification

# 도메인 코드 → 이벤트 요약용 짧은 한글 라벨.
_DOMAIN_LABELS: Dict[str, str] = {
    "kr": "한국",
    "us": "미국",
    "macro": "매크로",
    "backtest": "백테스트",
}


class HierarchicalState(TypedDict, total=False):
    # 입력
    question: str                      # 사용자 질문
    steps: List[Dict[str, Any]]        # 백테스트 파이프라인 스텝(있으면 backtest 도메인이 소비)
    conn: Any                          # DB 연결(노드에 클로저로 주입되지 않을 때 state 폴백)
    llm_fn: Any                        # Callable[[str], str] (동일 폴백용)
    on_progress: Any                   # Callable[[str, str], None] — 실시간 진행 콜백(HA-12 확장)

    # 총괄 노드 산출
    routes: List[str]                  # 라우팅된 도메인(["kr"] | ["kr","us"] ...)
    domain_results: Dict[str, Any]     # 도메인별 원본 결과(가공 없음)
    conclusion: Optional[str]          # 검증 통과 시 종합결론
    uncertain: Optional[bool]          # 검증 실패(3회) 시 True
    attempts: Optional[int]            # 실제 시도 횟수
    reason: Optional[str]              # 불확실 사유(실패 시)
    chart_base64: Optional[str]        # 차트 요청 시 이미지(PNG base64, 접두사 없음), 아니면 None
    chart_title: Optional[str]         # 차트 제목(차트가 있을 때만)
    charts: Optional[List[Dict[str, Any]]]  # 차트가 여러 개(산점도+막대그래프 등)면 전부. 없으면 None
    used_fallback: Optional[bool]      # 정형 검증 3회 실패 후 자유 코드 생성 폴백으로 답했으면 True

    # 스트리밍 이벤트 누적(노드 완료 시점마다 append). 여러 노드로 확장돼도 누적되도록 reducer 지정.
    events: Annotated[List[Dict[str, Any]], operator.add]


def _summarize(result: dict) -> str:
    """총괄 결과 → 이벤트 요약 한 줄. 라우팅 도메인 + 검증 통과/실패 + 시도 횟수만 담는다.

    (SQL 전문/원본 rows/결론 본문 등 상세는 의도적으로 제외 — HA-12가 SSE로 그대로 흘려도
    민감/장문 데이터가 새지 않게 한다.)
    """
    routes = result.get("routes") or []
    labels = "+".join(_DOMAIN_LABELS.get(r, r) for r in routes)
    attempts = result.get("attempts")
    if result.get("uncertain"):
        status = f"검증 실패({attempts}회 시도, 불확실)"
    else:
        status = f"검증 통과({attempts}회 시도)"
    prefix = f"{labels} 도메인 라우팅" if labels else "도메인 라우팅"
    return f"{prefix}, {status}"


def supervisor_node(
    state: HierarchicalState,
    conn: Any = None,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> dict:
    """총괄 에이전트를 감싼 LangGraph 노드.

    HA-10의 answer_with_verification 을 호출한다 — 라우팅(route_question)·도메인 실행
    (dispatch_domains)·정합성 검증(verify_answer)·최대 3회 재시도가 이 한 번의 호출 안에서
    모두 수행된다. 즉 총괄 에이전트의 라우팅·검증 로직이 이 노드로 구현된 것이다(AC5).

    conn/llm_fn 은 build_hierarchical_graph 가 클로저로 주입한다. 인자로 안 넘어오면
    state 에서 폴백으로 읽어(직접 노드 등록도 지원) answer_with_verification 에 전달한다.

    반환은 state 갱신분(dict) — routes/domain_results/conclusion/uncertain/attempts/reason
    과 이번 노드의 진행 이벤트 한 건을 events 로 돌려준다(events reducer가 누적).
    """
    question = state["question"]
    conn = conn if conn is not None else state.get("conn")
    llm_fn = llm_fn if llm_fn is not None else state.get("llm_fn")
    steps = state.get("steps")
    on_progress = state.get("on_progress")

    result = answer_with_verification(question, conn, llm_fn, steps=steps, on_progress=on_progress)

    event = {"step": "supervisor", "summary": _summarize(result)}
    return {
        "routes": result.get("routes", []),
        "domain_results": result.get("domain_results", {}),
        "conclusion": result.get("conclusion"),
        "uncertain": result.get("uncertain"),
        "attempts": result.get("attempts"),
        "reason": result.get("reason"),
        # 차트 요청 시에만 채워지는 필드(그대로 pass-through — web/app.py가 {**result}로 노출).
        "chart_base64": result.get("chart_base64"),
        "chart_title": result.get("chart_title"),
        "charts": result.get("charts"),
        "used_fallback": result.get("used_fallback"),
        "events": [event],
    }


def build_hierarchical_graph(
    conn: Any,
    llm_fn: Optional[Callable[[str], str]] = None,
):
    """총괄 노드 하나로 StateGraph 를 조립·컴파일한다: START → supervisor → END.

    supervisor_node 에 conn/llm_fn 을 클로저로 바인딩해 등록한다(기존 src/graph/build.py 의
    make_nodes(deps) 바인딩 관례와 동일 철학). 반환은 컴파일된 그래프(.stream()/.invoke() 지원).
    """
    def _supervisor(state: HierarchicalState) -> dict:
        return supervisor_node(state, conn=conn, llm_fn=llm_fn)

    g = StateGraph(HierarchicalState)
    g.add_node("supervisor", _supervisor)
    g.add_edge(START, "supervisor")
    g.add_edge("supervisor", END)
    return g.compile()


def run_streaming(
    question: str,
    conn: Any,
    llm_fn: Optional[Callable[[str], str]] = None,
    steps: Optional[List[Dict[str, Any]]] = None,
    out_final: Optional[Dict[str, Any]] = None,
) -> Iterator[dict]:
    """총괄 노드를 별도 스레드에서 실행하며, 그 내부 진행 상황(on_progress)을 실시간으로
    큐에서 꺼내 하나씩 yield 한다(HA-12 확장 — 실시간 트리 상세화).

    기존에는 `.stream()`이 "노드가 끝난 시점"에만 이벤트를 방출해, route→dispatch(도메인
    N개)→verify→(최대 3회 재시도)가 전부 끝난 뒤 요약 한 줄만 나오는 문제가 있었다(사실상
    "완료 후 요약"이지 "실시간"이 아니었다). 그래프 구조(START→supervisor→END, AC5)는 그대로
    유지하되, supervisor_node가 answer_with_verification에 전달하는 on_progress 콜백이
    호출될 때마다(라우팅 확정/도메인별 조회 시작·완료/검증 시도별 결과) 즉시 큐에 넣고,
    이 제너레이터가 그 큐를 실시간으로 소비한다 — 그래프 실행(worker 스레드)과 이벤트 소비
    (메인 스레드)가 동시에 진행되므로 진짜 실시간 스트리밍이 된다.

    단일 워커 스레드가 on_progress를 순서대로 호출하고 메인이 큐에서 그 순서 그대로 꺼내
    yield하므로, 동일 입력에 대해 이벤트 순서는 결정론적이다(collect_stream과 동일 결과).

    out_final: 전달하면(가변 dict, 예: {}) 이 실행이 정상 종료된 뒤 그래프 최종 상태
    (routes/domain_results/conclusion/uncertain/attempts/...)로 채워진다. 이 실행 하나가
    진행 이벤트와 최종 답변을 모두 내어주므로, 호출부가 "진행상황용"과 "최종답변용"을
    별도로 두 번 실행할 필요가 없다(web/app.py의 GET /api/query/stream이 과거 이 최종
    상태를 버리고 POST /api/query를 한 번 더 호출해 동일 질문을 두 번 계산하던 문제의
    해결책 — 노드 완료 시점의 events 는 여전히 진행 이벤트로만 쓰고, 결과는 out_final로
    받는다).
    """
    event_queue: "queue.Queue[Optional[dict]]" = queue.Queue()

    def on_progress(step: str, summary: str, detail: Optional[Dict[str, Any]] = None) -> None:
        event: Dict[str, Any] = {"step": step, "summary": summary}
        if detail is not None:
            event["detail"] = detail
        event_queue.put(event)

    error_box: Dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            graph = build_hierarchical_graph(conn, llm_fn)
            init: dict = {"question": question, "on_progress": on_progress}
            if steps is not None:
                init["steps"] = steps
            final_snapshot: dict = init
            for snapshot in graph.stream(init, stream_mode="values"):
                final_snapshot = snapshot
            if out_final is not None:
                out_final.update(final_snapshot)
        except BaseException as exc:  # noqa: BLE001 — 스레드 예외를 메인 스레드로 전달
            error_box["exc"] = exc
        finally:
            event_queue.put(None)  # 종료 신호

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    while True:
        item = event_queue.get()
        if item is None:
            break
        yield item
    thread.join()
    if "exc" in error_box:
        raise error_box["exc"]


def collect_stream(
    question: str,
    conn: Any,
    llm_fn: Optional[Callable[[str], str]] = None,
    steps: Optional[List[Dict[str, Any]]] = None,
) -> List[dict]:
    """run_streaming 을 리스트로 모아 반환하는 테스트/동기 소비용 버전."""
    return list(run_streaming(question, conn, llm_fn=llm_fn, steps=steps))


def run_hierarchical(
    question: str,
    conn: Any,
    llm_fn: Optional[Callable[[str], str]] = None,
    steps: Optional[List[Dict[str, Any]]] = None,
) -> dict:
    """그래프를 `.stream()`(values 모드)으로 실행해 **최종 누적 상태**(dict)를 반환한다.

    스트리밍 이벤트(진행 표시)와 별개로 최종 결론/원본 도메인 결과가 필요한 호출부(HA-12의
    최종 SSE 메시지 등)를 위해, invoke 대신 stream 마지막 스냅샷을 최종 상태로 돌려준다.
    """
    graph = build_hierarchical_graph(conn, llm_fn)
    init: dict = {"question": question}
    if steps is not None:
        init["steps"] = steps
    final: dict = dict(init)
    for snapshot in graph.stream(init, stream_mode="values"):
        final = snapshot
    return final
