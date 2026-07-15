"""StateGraph 구성.

흐름(기록 전용, 캐시 분기 없음):
  START → refine → router → sql_gen → execute → record → (분기)
    - do_eval : → eval → END
    - else    : → END

과거의 wiki_check(유사도 캐시 도출)/wiki_save(캐시 저장) 분기를 폐기했다.
이제 모든 질의는 항상 새로 SQL을 생성·실행하고, 성공한 질의를 record 노드가
기록 로그(wiki 테이블)에 남긴다.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import Deps, make_nodes
from .state import GraphState


def _branch_after_record(state: GraphState) -> str:
    return "eval" if state.get("do_eval") else "end"


def _route_after_diagnose(state: GraphState) -> str:
    """진단 결과로 분기.

    none           → record (정상, 통과)
    sql/refine(고칠수있음) & 3회 미만 → 해당 노드로 되돌려 재시도
    data 또는 3회 초과 → human_review (사람 검토, HITL)
    """
    d = state.get("diagnosis", {})
    cause = d.get("cause", "none")
    attempt = state.get("attempt_count", 0)
    if cause == "none":
        return "record"
    if attempt >= 3:                 # 3회 생성했으면 더 시도 않고 사람에게
        return "human"
    if cause == "sql" and d.get("fixable"):
        return "sql_gen"
    if cause == "refine" and d.get("fixable"):
        return "refine"
    return "human"                   # data, 또는 자동으로 못 고치는 경우


def build_graph(deps: Deps):
    nodes = make_nodes(deps)
    g = StateGraph(GraphState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.add_edge(START, "refine_node")
    g.add_edge("refine_node", "router_node")
    g.add_edge("router_node", "sql_gen_node")
    # sql_gen → execute → diagnose 순. diagnose가 원인을 보고 통과/재시도/HITL로 분기한다.
    g.add_edge("sql_gen_node", "execute_node")
    g.add_edge("execute_node", "diagnose_node")
    g.add_conditional_edges(
        "diagnose_node",
        _route_after_diagnose,
        {
            "record": "record_node",
            "sql_gen": "sql_gen_node",   # SQL 문제 → 진단 피드백 주고 재생성
            "refine": "refine_node",     # 질문 모호 → 재정제
            "human": "human_review_node",  # 데이터 문제/3회 초과 → 사람 검토
        },
    )
    g.add_conditional_edges(
        "record_node",
        _branch_after_record,
        {"eval": "eval_node", "end": END},
    )
    g.add_edge("human_review_node", END)
    g.add_edge("eval_node", END)
    return g.compile()
