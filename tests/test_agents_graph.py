"""계층형 총괄 그래프(src/agents/graph.py) 테스트 (TDD, HA-11).

HA-10의 순수 함수(answer_with_verification 등)를 LangGraph StateGraph 노드로 감싼다.
여기서는 (1) 그래프 컴파일, (2) .stream() 노드 이벤트 방출(순서), (3) 이벤트 스키마
(단계명+한줄요약만, SQL 전문 등 상세 없음)을 검증한다.

- supervisor_node(state, conn, llm_fn) -> dict:
  answer_with_verification를 호출해 상태를 갱신하는 단일 총괄 노드(라우팅+검증+재시도 포함).
- build_hierarchical_graph(conn, llm_fn=None) -> CompiledGraph:
  START→supervisor→END 로 컴파일한다.
- run_streaming / collect_stream:
  .invoke()가 아니라 .stream()으로 실행해 노드 완료 이벤트를 방출/수집한다.
"""
from __future__ import annotations

import src.agents.graph as graph_mod
from src.agents.graph import (
    HierarchicalState,
    build_hierarchical_graph,
    collect_stream,
    run_streaming,
    supervisor_node,
)


# 라우팅/검증/종합 3종 프롬프트를 프롬프트 내용으로 구분해 응답하는 mock LLM.
def _multi_domain_fake_llm(prompt: str) -> str:
    if "도메인 키워드만" in prompt:          # route_question 프롬프트
        return "kr, us"
    if "valid" in prompt:                    # verify_answer 프롬프트(JSON 예시에 valid 포함)
        return '{"valid": true, "reason": "부합"}'
    return "삼성전자와 애플 종합 결론"        # synthesize_conclusion 프롬프트


def _seed_valid_domains(monkeypatch) -> None:
    import src.agents.supervisor as sup

    monkeypatch.setattr(
        sup, "answer_kr_question",
        lambda question, conn, llm_fn=None, on_progress=None: {"stock_code": "005930", "financial": {"value": 12.5}},
    )


# ── HierarchicalState — 최소 상태 필드 존재(TypedDict, total=False) ────────────

def test_hierarchical_state_declares_min_fields():
    keys = set(HierarchicalState.__annotations__)
    for field in ("question", "routes", "domain_results", "conclusion", "uncertain", "attempts", "events"):
        assert field in keys


# ── build_hierarchical_graph — StateGraph 컴파일(AC5 뒷받침) ──────────────────

def test_build_hierarchical_graph_compiles():
    graph = build_hierarchical_graph(conn=None, llm_fn=None)
    assert hasattr(graph, "stream")   # .stream() 지원(컴파일된 그래프)
    assert hasattr(graph, "invoke")


def test_build_hierarchical_graph_registers_supervisor_node():
    graph = build_hierarchical_graph(conn=None, llm_fn=None)
    node_names = set(graph.get_graph().nodes)
    assert "supervisor" in node_names


# ── supervisor_node — answer_with_verification 결과로 상태 갱신 ───────────────

def test_supervisor_node_wraps_answer_with_verification(monkeypatch):
    def fake_awv(question, conn, llm_fn, steps=None, on_progress=None):
        return {
            "uncertain": False,
            "conclusion": "종합결론",
            "domain_results": {"kr": {"financial": {"value": 12.5}}},
            "attempts": 1,
            "routes": ["kr"],
        }

    monkeypatch.setattr(graph_mod, "answer_with_verification", fake_awv)
    out = supervisor_node({"question": "삼성전자 PER"}, conn=None, llm_fn=None)

    assert out["routes"] == ["kr"]
    assert out["conclusion"] == "종합결론"
    assert out["uncertain"] is False
    assert out["attempts"] == 1
    assert out["domain_results"] == {"kr": {"financial": {"value": 12.5}}}


def test_supervisor_node_passes_through_chart_fields(monkeypatch):
    """answer_with_verification가 chart_base64/chart_title를 주면 노드가 그대로 통과시킨다."""
    def fake_awv(question, conn, llm_fn, steps=None, on_progress=None):
        return {
            "uncertain": False,
            "conclusion": "종합결론",
            "domain_results": {"backtest": {"result": {"navs": [1.0, 1.1]}}},
            "attempts": 1,
            "routes": ["backtest"],
            "chart_base64": "ZmFrZS1wbmc=",
            "chart_title": "백테스트 결과",
        }

    monkeypatch.setattr(graph_mod, "answer_with_verification", fake_awv)
    out = supervisor_node({"question": "전략 그래프 그려줘"}, conn=None, llm_fn=None)

    assert out["chart_base64"] == "ZmFrZS1wbmc="
    assert out["chart_title"] == "백테스트 결과"


def test_hierarchical_state_declares_chart_fields():
    keys = set(HierarchicalState.__annotations__)
    assert "chart_base64" in keys and "chart_title" in keys


# 실서버 재현 버그: pipeline_exec 다중산출물 수정 후 answer_with_verification이
# 산점도+막대그래프 둘 다 담은 charts(리스트) 필드를 돌려주는데, 이 그래프 노드가
# chart_base64/chart_title만 골라 통과시키고 charts는 그냥 버려서, 실제 응답(SSE/REST)에
# 산점도 1개만 보이고 막대그래프는 사라졌다.
def test_supervisor_node_passes_through_charts_list(monkeypatch):
    def fake_awv(question, conn, llm_fn, steps=None, on_progress=None):
        return {
            "uncertain": False,
            "conclusion": "종합결론",
            "domain_results": {"backtest": {"result": {}}},
            "attempts": 1,
            "routes": ["backtest"],
            "chart_base64": "c2NhdHRlcg==",
            "chart_title": "산점도",
            "charts": [
                {"chart_base64": "c2NhdHRlcg==", "chart_title": "산점도"},
                {"chart_base64": "YmFy", "chart_title": "막대그래프"},
            ],
        }

    monkeypatch.setattr(graph_mod, "answer_with_verification", fake_awv)
    out = supervisor_node({"question": "산점도랑 막대그래프 둘 다 그려줘"}, conn=None, llm_fn=None)

    assert out.get("charts") == [
        {"chart_base64": "c2NhdHRlcg==", "chart_title": "산점도"},
        {"chart_base64": "YmFy", "chart_title": "막대그래프"},
    ]


def test_hierarchical_state_declares_charts_field():
    keys = set(HierarchicalState.__annotations__)
    assert "charts" in keys


def test_supervisor_node_passes_through_used_fallback_flag(monkeypatch):
    """자유 실행 폴백(exec_fallback)이 쓰였는지를 total=False TypedDict가 조용히
    떨어뜨리지 않고 그래프 상태까지 그대로 전달하는지(과거 charts 필드 누락 회귀와 동일 유형)."""
    def fake_awv(question, conn, llm_fn, steps=None, on_progress=None):
        return {
            "uncertain": False,
            "conclusion": "종합결론(폴백)",
            "domain_results": {"free_exec": {"fallback_used": True, "result": {}}},
            "attempts": 3,
            "routes": ["kr"],
            "used_fallback": True,
        }

    monkeypatch.setattr(graph_mod, "answer_with_verification", fake_awv)
    out = supervisor_node({"question": "코스피 코스닥 각각 상위 10개"}, conn=None, llm_fn=None)

    assert out.get("used_fallback") is True


def test_hierarchical_state_declares_used_fallback_field():
    keys = set(HierarchicalState.__annotations__)
    assert "used_fallback" in keys


def test_supervisor_node_reads_conn_and_llm_from_state(monkeypatch):
    """conn/llm_fn을 명시 인자로 안 넘기면 state에서 읽는다(직접 노드 등록도 지원)."""
    captured: dict = {}

    def fake_awv(question, conn, llm_fn, steps=None, on_progress=None):
        captured["conn"] = conn
        captured["llm_fn"] = llm_fn
        return {"uncertain": True, "reason": "x", "attempts": 3, "routes": ["kr"], "domain_results": {}}

    monkeypatch.setattr(graph_mod, "answer_with_verification", fake_awv)
    sentinel_conn = object()
    sentinel_llm = lambda p: "kr"
    supervisor_node({"question": "q", "conn": sentinel_conn, "llm_fn": sentinel_llm})

    assert captured["conn"] is sentinel_conn
    assert captured["llm_fn"] is sentinel_llm


# ── 이벤트 스키마 — 단계명+한줄요약만, SQL 전문 등 상세 없음 ──────────────────

def test_event_has_only_step_and_summary_no_detail(monkeypatch):
    def fake_awv(question, conn, llm_fn, steps=None, on_progress=None):
        return {
            "uncertain": False,
            "conclusion": "이건 매우 긴 종합결론 본문입니다.",
            "domain_results": {"kr": {"sql": "SELECT * FROM metrics WHERE per < 10 ORDER BY per",
                                       "rows": [{"per": 8.0}], "financial": {"value": 12.5}}},
            "attempts": 2,
            "routes": ["kr"],
        }

    monkeypatch.setattr(graph_mod, "answer_with_verification", fake_awv)
    out = supervisor_node({"question": "삼성전자 PER"}, conn=None, llm_fn=None)

    events = out["events"]
    assert isinstance(events, list) and len(events) == 1
    ev = events[0]
    # 이벤트는 정확히 step/summary 두 키만 가진다.
    assert set(ev.keys()) == {"step", "summary"}
    assert ev["step"] == "supervisor"
    assert isinstance(ev["summary"], str)
    # SQL 전문/원본 rows/결론 본문 같은 상세가 요약에 새면 안 된다.
    assert "SELECT" not in ev["summary"]
    assert "sql" not in ev["summary"].lower()
    assert "종합결론 본문" not in ev["summary"]
    # 요약은 라우팅/검증 상태만 담는다(한 줄).
    assert "\n" not in ev["summary"]
    assert "검증 통과" in ev["summary"]
    assert "2회" in ev["summary"]


def test_event_summary_reports_uncertain_on_failure(monkeypatch):
    def fake_awv(question, conn, llm_fn, steps=None, on_progress=None):
        return {"uncertain": True, "reason": "3회 실패", "attempts": 3,
                "routes": ["kr", "backtest"], "domain_results": {}}

    monkeypatch.setattr(graph_mod, "answer_with_verification", fake_awv)
    out = supervisor_node({"question": "q"}, conn=None, llm_fn=None)
    summary = out["events"][0]["summary"]
    assert "한국" in summary and "백테스트" in summary   # 도메인 라벨
    assert "불확실" in summary or "실패" in summary
    assert "3회" in summary


# ── .stream() 통합 — 노드 완료 이벤트가 순서대로 방출됨(mock LLM 주입) ─────────

def test_run_streaming_emits_supervisor_event(monkeypatch):
    """HA-12 확장: 라우팅→도메인별→검증 단계별로 여러 이벤트가 실시간 순서대로 방출된다.
    라우팅에 없는 도메인 토큰은 무시되므로 실제 실행 단계에는 kr만 나타난다."""
    _seed_valid_domains(monkeypatch)
    events = list(run_streaming("삼성전자 종가 알려줘", conn=None, llm_fn=_multi_domain_fake_llm))

    steps_order = [e["step"] for e in events]
    assert len(events) >= 4                     # 라우팅1 + kr 시작/완료 + 검증 4건 이상
    assert steps_order[0] == "supervisor"        # 라우팅이 가장 먼저
    assert "한국" in events[0]["summary"]
    assert "kr" in steps_order and "us" not in steps_order  # 미국은 비활성화 → 실행되지 않음
    assert steps_order[-1] == "verify"           # 검증 결과가 마지막
    assert "통과" in events[-1]["summary"]


def test_collect_stream_matches_run_streaming(monkeypatch):
    _seed_valid_domains(monkeypatch)
    collected = collect_stream("삼성전자 vs 애플 비교", conn=None, llm_fn=_multi_domain_fake_llm)
    assert isinstance(collected, list)
    assert collected and collected[0]["step"] == "supervisor"
    # 리스트 버전과 이터레이터 버전이 동일 이벤트를 낸다.
    streamed = list(run_streaming("삼성전자 vs 애플 비교", conn=None, llm_fn=_multi_domain_fake_llm))
    assert collected == streamed


# ── on_progress(step, summary, detail=None) — 실시간 코드(SQL/파이프라인 JSON) 노출(HA-12 확장) ──
def test_run_streaming_forwards_detail_field_when_provided(monkeypatch):
    """도메인 에이전트가 on_progress를 detail 인자와 함께 호출하면, SSE로 나가는 이벤트
    dict에도 그 detail이 그대로 실려야 한다(생성된 조건 JSON/파이프라인을 프론트가 즉시 표시)."""
    def fake_awv(question, conn, llm_fn, steps=None, on_progress=None):
        if on_progress:
            on_progress("code", "조건 생성 완료", detail={"kind": "screening_spec", "spec": {"criteria": []}})
        return {"uncertain": False, "conclusion": "ok", "domain_results": {}, "attempts": 1, "routes": ["kr"]}

    monkeypatch.setattr(graph_mod, "answer_with_verification", fake_awv)
    events = list(run_streaming("q", conn=None, llm_fn=None))

    detail_events = [e for e in events if e.get("step") == "code"]
    assert len(detail_events) == 1
    assert detail_events[0]["detail"] == {"kind": "screening_spec", "spec": {"criteria": []}}


def test_run_streaming_omits_detail_key_when_not_provided(monkeypatch):
    """detail 없이 on_progress(step, summary)만 호출하면 기존과 동일하게 detail 키 자체가 없어야
    한다(기존 소비자가 이벤트에 없는 키를 신경 쓸 필요 없게, payload도 불필요하게 커지지 않게)."""
    def fake_awv(question, conn, llm_fn, steps=None, on_progress=None):
        if on_progress:
            on_progress("kr", "한국 도메인 조회 중…")
        return {"uncertain": False, "conclusion": "ok", "domain_results": {}, "attempts": 1, "routes": ["kr"]}

    monkeypatch.setattr(graph_mod, "answer_with_verification", fake_awv)
    events = list(run_streaming("q", conn=None, llm_fn=None))
    assert all("detail" not in e for e in events)


def test_run_streaming_uses_heuristic_route_without_llm(monkeypatch):
    """llm_fn 없이도(결정론) 노드 이벤트가 방출된다 — kr 휴리스틱 라우팅."""
    _seed_valid_domains(monkeypatch)
    events = collect_stream("삼성전자 PER 알려줘", conn=None, llm_fn=None)
    assert events and events[0]["step"] == "supervisor"
    assert "한국" in events[0]["summary"]


# ── out_final — 진행 이벤트와 최종 답변을 같은 실행 한 번에서 모두 얻는다 ─────────────
# (회귀 방지: web/app.py GET /api/query/stream 이 과거 이 최종 상태를 버리고 POST /api/query
# 를 한 번 더 호출해 동일 질문을 두 번 계산했다 — 비용 2배 + 화면 진행상황과 실제 최종
# 답변이 달라질 수 있는 문제였다. out_final 은 그 문제를 "실행을 하나로 합쳐" 해결한다.)

def test_run_streaming_populates_out_final_with_final_state(monkeypatch):
    """out_final(가변 dict)을 넘기면, 이 스트리밍 실행이 끝난 뒤 그래프 최종 상태로 채워진다."""
    _seed_valid_domains(monkeypatch)
    out_final: dict = {}
    events = list(run_streaming(
        "삼성전자 vs 애플 비교", conn=None, llm_fn=_multi_domain_fake_llm, out_final=out_final,
    ))

    assert events  # 진행 이벤트는 그대로 방출된다(기존 동작 불변)
    assert out_final.get("conclusion") == "삼성전자와 애플 종합 결론"
    assert out_final.get("routes") == ["kr"]   # 미국은 비활성화되어 제외됨(코드는 보존)
    assert out_final.get("uncertain") is False


def test_run_streaming_without_out_final_keeps_default_behavior(monkeypatch):
    """out_final 을 안 넘기면(기존 호출부) 동작이 전혀 바뀌지 않는다 — 하위호환."""
    _seed_valid_domains(monkeypatch)
    events = list(run_streaming("삼성전자 vs 애플 비교", conn=None, llm_fn=_multi_domain_fake_llm))
    assert events and events[0]["step"] == "supervisor"
