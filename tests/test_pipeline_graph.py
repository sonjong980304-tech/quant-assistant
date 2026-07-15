"""SQL/Python 파이프라인 그래프 통합 테스트 (US-11).

- router_node가 통계/퀀트 질문을 route="pipeline"으로 감지(기존 detect_route와 같은
  '휴리스틱 키워드' 관례, 결정론 → 키 불필요).
- goldset 50문항이 route="pipeline"으로 오분류되지 않는다(오탐 없음, 결정론).
- sql_gen_node가 route=="pipeline"일 때 PIPELINE_USER로 파이프라인 JSON을 생성한다.
- execute_node가 파이프라인을 실행해 결과 rows를 만든다.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.config import CONFIG
from src.eval.goldset import GOLDSET
from src.legacy.graph.heuristic import detect_pipeline
from src.legacy.graph.nodes import Deps, make_nodes
from tests.conftest import FakeLLM


# --------------------------------------------------------------------------
# detect_pipeline (휴리스틱, 결정론)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("q", [
    "삼성전자 최근 1년 수익률의 K-ratio를 알려줘",
    "저PER 고ROE 우량주로 최대샤프 포트폴리오 비중을 구해줘",
    "이 종목들로 위험균형 포트폴리오를 만들어줘",
    "누적수익률을 회귀분석해서 표준오차를 구해줘",
    "최소분산 포트폴리오 비중 계산",
])
def test_detect_pipeline_true_for_stat_quant_questions(q):
    assert detect_pipeline(q) is True


def test_detect_pipeline_false_for_plain_sql_questions():
    assert detect_pipeline("PER이 가장 낮은 10개 회사를 알려줘") is False
    assert detect_pipeline("삼성전자의 부채비율은?") is False


@pytest.mark.parametrize("q", [
    "2024년부터 2026년까지 매 분기 PER 낮은 10개 종목으로 리밸런싱했을 때 백테스트 결과 보여줘",
    "모멘텀 상위 20개로 월별 리밸런싱하면 수익률이 어떻게 돼",
    "이 전략 백테스트해줘",
])
def test_detect_pipeline_true_for_backtest_questions(q):
    """run_backtest 프리미티브(7번째)가 PIPELINE_USER에 추가됐지만 라우팅 키워드 갱신이
    빠져, 자연어 백테스트 질문이 route="pipeline"으로 안 가고 일반 SQL로 새던 버그
    (웹 프롬프트 통합 후 실측 curl로 발견)."""
    assert detect_pipeline(q) is True


@pytest.mark.parametrize("q", [
    "RSI가 30 이하인 과매도 종목 찾아줘",
    "MACD 골든크로스 난 종목 알려줘",
    "20일 이동평균선 위에 있는 종목만 골라줘",
    "볼린저밴드 하단 이탈한 종목 보여줘",
])
def test_detect_pipeline_true_for_technical_indicator_questions(q):
    """compute_technical_indicator 프리미티브(9번째)가 PIPELINE_USER에 추가됐지만 라우팅
    키워드 갱신이 빠지면, 자연어 기술지표 질문이 route="pipeline"으로 안 가고 일반 SQL로
    새는 버그가 재발한다(백테스트 키워드 누락 버그와 동일 패턴, 재발 방지)."""
    assert detect_pipeline(q) is True


@pytest.mark.parametrize("q", [
    "MDD -10% 이내이면서 샤프가 가장 높은 전략 찾아줘",
    "최대손실폭 제약 안에서 좋은 전략 탐색해줘",
    "역백테스트로 조건 만족하는 전략 골라줘",
])
def test_detect_pipeline_true_for_strategy_search_questions(q):
    """search_strategy 프리미티브(10번째)가 PIPELINE_USER에 추가됐지만 라우팅 키워드
    갱신이 빠지면, 자연어 전략탐색 질문이 route="pipeline"으로 안 가는 버그가 재발한다."""
    assert detect_pipeline(q) is True


def test_goldset_50_no_false_positive_pipeline_route():
    """기존 goldset 50문항이 새 pipeline 경로로 새지 않아야 한다(결정론 회귀)."""
    leaked = [item["id"] for item in GOLDSET if detect_pipeline(item["question"])]
    assert leaked == []


# --------------------------------------------------------------------------
# router_node / sql_gen_node / execute_node 분기
# --------------------------------------------------------------------------
def _deps(llm, db_path=None):
    from datetime import date

    if db_path:
        conn = sqlite3.connect(db_path)
    else:
        conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return Deps(conn=conn, store=None, schema="(schema)", today=date(2026, 6, 22), llm=llm)


def test_router_sets_pipeline_route_for_quant_question():
    nodes = make_nodes(_deps(FakeLLM("{}")))
    out = nodes["router_node"]({"question": "이 종목들로 최대샤프 포트폴리오 비중을 구해줘"})
    assert out["route"] == "pipeline"


def test_router_keeps_normal_route_for_sql_question(seeded_db):
    nodes = make_nodes(_deps(FakeLLM("{}"), db_path=seeded_db))
    out = nodes["router_node"]({"question": "PER이 낮은 10개 회사", "raw_question": "PER이 낮은 10개 회사"})
    assert out["route"] != "pipeline"


def test_sql_gen_uses_pipeline_prompt_when_route_pipeline():
    llm = FakeLLM('{"pipeline": [{"op": "regress", "params": {"y": [1,2,3,4]}, "out": "r"}]}')
    nodes = make_nodes(_deps(llm))
    out = nodes["sql_gen_node"]({"question": "누적수익률 회귀 K-ratio", "route": "pipeline"})
    # PIPELINE_USER 프롬프트가 쓰였는지(프리미티브 안내 문구 포함), SQL_USER는 아님
    assert any("프리미티브" in c["prompt"] for c in llm.calls)
    assert out.get("sql_source") == "pipeline"
    assert out["pipeline"] == [{"op": "regress", "params": {"y": [1, 2, 3, 4]}, "out": "r"}]


def test_execute_runs_pipeline_and_returns_rows():
    llm = FakeLLM("{}")
    nodes = make_nodes(_deps(llm))
    state = {
        "route": "pipeline",
        "pipeline": [{"op": "regress", "params": {"y": [1.0, 2.0, 3.1, 3.9, 5.0]}, "out": "reg"}],
    }
    out = nodes["execute_node"](state)
    assert out["error"] is None
    assert out["row_count"] >= 1
    # regress 결과(slope 등)가 rows에 담긴다
    flat = {k: v for r in out["rows"] for k, v in r.items()}
    assert "slope" in flat or any("slope" in str(r) for r in out["rows"])


def test_execute_generates_chart_png_when_result_has_navs(tmp_path, monkeypatch):
    """run_backtest 프리미티브 결과(navs 포함)면 PNG를 만들고 경로를 결과에 담는다(US-12 AC4)."""
    from src.backtest import pipeline_exec as pipeline_exec_module

    monkeypatch.setattr(
        pipeline_exec_module, "run_pipeline",
        lambda steps, conn=None: {"dates": ["2026-01-31", "2026-02-28"], "navs": [1.0, 1.1],
                                   "benchmark": None, "performance": {"cagr": 0.1}},
    )
    llm = FakeLLM("{}")
    deps = _deps(llm)
    deps.chart_dir = str(tmp_path)
    nodes = make_nodes(deps)
    state = {"route": "pipeline", "pipeline": [{"op": "run_backtest", "params": {}, "out": "bt"}]}
    out = nodes["execute_node"](state)
    assert out["error"] is None
    assert out.get("chart_path")
    assert Path(out["chart_path"]).exists()
    assert Path(out["chart_path"]).parent == tmp_path


def test_pipeline_failure_retries_via_diagnose_then_regenerates_pipeline():
    """파이프라인 실행 실패 → diagnose_node가 cause=sql로 분류 →
    _route_after_diagnose가 sql_gen으로 되돌려 파이프라인을 재생성한다(US-11 AC3, 최대 3회)."""
    from src.legacy.graph.build import _route_after_diagnose

    llm = FakeLLM('{"pipeline": [{"op": "regress", "params": {"y": [1,2,3]}, "out": "r"}]}')
    nodes = make_nodes(_deps(llm))
    # diagnose 자체는 LLM 미가용 규칙기반 경로(결정론)로 검증 — sql_error→cause=sql 매핑 확인
    diag_nodes = make_nodes(_deps(FakeLLM("", available=False)))

    exec_out = nodes["execute_node"]({
        "route": "pipeline",
        "pipeline": [{"op": "regress", "params": {}, "out": "r"}],  # y 누락 → 실행 실패
    })
    assert exec_out["error"] is not None

    state = {**exec_out, "route": "pipeline", "attempt_count": 1, "question": "누적수익률 K-ratio"}
    diag_out = diag_nodes["diagnose_node"](state)
    assert diag_out["diagnosis"]["cause"] == "sql"
    assert diag_out["diagnosis"]["fixable"] is True

    next_state = {**state, **diag_out}
    assert _route_after_diagnose(next_state) == "sql_gen"  # 3회 미만 → 재시도로 라우팅

    # sql_gen_node가 diagnose 피드백을 받아 새 파이프라인 JSON을 다시 생성한다
    regen = nodes["sql_gen_node"](next_state)
    assert regen["sql_source"] == "pipeline"
    assert regen["pipeline"] == [{"op": "regress", "params": {"y": [1, 2, 3]}, "out": "r"}]

    # 3회 초과면 사람 검토로 라우팅(무한 재시도 방지)
    exhausted_state = {**next_state, "attempt_count": 3}
    assert _route_after_diagnose(exhausted_state) == "human"


def test_execute_does_not_generate_chart_for_non_backtest_result():
    llm = FakeLLM("{}")
    nodes = make_nodes(_deps(llm))
    state = {
        "route": "pipeline",
        "pipeline": [{"op": "regress", "params": {"y": [1.0, 2.0, 3.1, 3.9, 5.0]}, "out": "reg"}],
    }
    out = nodes["execute_node"](state)
    assert not out.get("chart_path")


# --------------------------------------------------------------------------
# 실측(키 있으면): 파마프렌치 테스트와 동일한 skipif 패턴
# --------------------------------------------------------------------------
@pytest.mark.skipif(not CONFIG.has_openai_key, reason="OpenAI 키 필요(README 관례: 키 없으면 스킵)")
def test_goldset_no_pipeline_leak_is_deterministic_without_llm():
    # detect_pipeline은 LLM을 쓰지 않으므로 키 유무와 무관하게 결정론이다(문서화용 확인).
    assert all(not detect_pipeline(i["question"]) for i in GOLDSET)
