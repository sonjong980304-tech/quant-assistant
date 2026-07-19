"""백테스트 도메인 에이전트(src/agents/domain_backtest.py) 통합/단위 테스트 (TDD, HA-9).

.omc/state/sessions/f6b79d90-b7b2-4b4f-b4f7-405a8355724e/prd.json의 HA-9 참고.

검증 대상(AC6 백테스트 부분 + 부수 AC):
- 질문을 받아 (필요 시) 한국/미국 데이터 에이전트(get_price_snapshot_kr/us)로 시계열
  스냅샷을 준비하고, HA-5의 검증 배선(run_backtest_with_audit: 하드3+소프트4)을 통과한
  결과만 반환한다. 하드차단 fixture는 결과를 폐기(blocked=True), 통과 fixture는 정상 반환.
- 백테스트 실행 자체는 기존 pipeline_exec.run_pipeline에 위임(run_pipeline_fn 주입)하고,
  이 도메인 에이전트가 새로 만드는 실행 경로(데이터 준비용 보조 조회)는 HA-1 실행기
  (execute_sql)만 경유한다 — 자체 별도 실행경로(conn.execute()/eval/exec)를 만들지 않는다.
"""
from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

from src.agents.domain_backtest import (
    answer_backtest_question,
    generate_backtest_steps,
    validate_pipeline_steps,
)
from src.backtest.pipeline_exec import run_pipeline as real_run_pipeline
from src.db import connect, connect_readonly, init_db

_BT_RESULT = {
    "dates": ["2025-09-30", "2025-12-31"],
    "navs": [1.0, 1.1],
    "benchmark": None,
    "performance": {"cagr": 5.0, "avg_turnover": 0.2},
    "holdings": [{"date": "2025-12-31", "codes": ["000001"]}],
}


def _json_llm(triggered=True, message="경고"):
    return lambda prompt: json.dumps({"triggered": triggered, "message": message})


def _writable_conn(tmp_path) -> sqlite3.Connection:
    db = tmp_path / "bt.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def _seed_prices_ro(tmp_path, rows: list[tuple]) -> sqlite3.Connection:
    """rows: (stock_code, date, close, market_cap, open, high, low, volume). 읽기전용 연결 반환."""
    db = tmp_path / "prices.db"
    init_db(str(db))
    c = connect(str(db))
    try:
        c.executemany(
            "INSERT INTO prices(stock_code, date, close, market_cap, open, high, low, volume) "
            "VALUES(?,?,?,?,?,?,?,?)",
            rows,
        )
        c.commit()
    finally:
        c.close()
    return connect_readonly(str(db))


# ── 정적 가드: 자체 실행경로를 새로 만들지 않는다(HA-1 실행기만 경유) ──────────────
def _module_tree() -> ast.AST:
    src = Path(__file__).resolve().parent.parent / "src" / "agents" / "domain_backtest.py"
    return ast.parse(src.read_text(encoding="utf-8"))


def test_module_does_not_call_conn_execute_directly():
    """SQL 실행은 반드시 HA-1 실행기(execute_sql) 경유 — conn.execute()를 직접 호출하지 않는다."""
    for node in ast.walk(_module_tree()):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "execute"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "conn"
        ):
            raise AssertionError("conn.execute() 직접 호출 발견 — execute_sql()을 경유해야 한다")


def test_module_has_no_eval_or_exec():
    """자체 실행경로(임의 코드 실행)를 새로 만들지 않는다 — eval/exec 호출 금지."""
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("eval", "exec"), "eval/exec 직접 호출 발견"


# ── run_backtest_with_audit(HA-5)이 실제로 호출되고 run_pipeline이 주입된다 ──────
def test_run_backtest_with_audit_invoked_with_pipeline_exec_run_pipeline(tmp_path):
    """도메인 에이전트가 HA-5 배선을 호출하고, run_pipeline_fn 기본값이 실제
    pipeline_exec.run_pipeline임을 확인(백테스트 실행을 기존 엔진에 위임)."""
    conn = _writable_conn(tmp_path)
    captured: dict = {}

    def spy_audit(steps, c, question, run_pipeline_fn, llm_fn=None, market="KR"):
        captured.update(
            steps=steps, conn=c, question=question,
            run_pipeline_fn=run_pipeline_fn, llm_fn=llm_fn, market=market,
        )
        return {"blocked": False, "error": None, "result": dict(_BT_RESULT),
                "hard": [], "warnings": []}

    steps = [{"op": "run_backtest", "params": {}, "out": "bt"}]
    answer_backtest_question("저PER 백테스트", steps, conn, run_audit_fn=spy_audit)

    assert captured["question"] == "저PER 백테스트"
    assert captured["steps"] == steps
    assert captured["conn"] is conn
    # 백테스트 실행 자체는 기존 pipeline_exec.run_pipeline에 위임(재발명 금지)
    assert captured["run_pipeline_fn"] is real_run_pipeline


# ── 하드차단(생존편향) fixture: 결과 폐기 ────────────────────────────────────────
def test_hard_block_survivorship_discards_result(tmp_path):
    conn = _writable_conn(tmp_path)
    conn.execute("INSERT INTO delisting(stock_code, name, delisting_date) VALUES (?,?,?)",
                 ("000001", "죽은회사", "2020-01-01"))
    conn.commit()

    steps = [{"op": "run_backtest", "params": {}, "out": "bt"}]
    out = answer_backtest_question(
        "저PER 20개 종목 분기 리밸런싱 백테스트", steps, conn,
        llm_fn=_json_llm(True, "x"), market="KR",
        run_pipeline_fn=lambda s, conn=None: dict(_BT_RESULT),
    )

    assert out["blocked"] is True
    assert "survivorship" in out["error"]
    assert out["result"] is None
    assert out["warnings"] == []


# ── 통과 fixture: 정상 결과 + 소프트경고 첨부 ───────────────────────────────────
def test_pass_returns_result_and_soft_warnings(tmp_path):
    conn = _writable_conn(tmp_path)
    steps = [{"op": "run_backtest", "params": {}, "out": "bt"}]
    out = answer_backtest_question(
        "저PER 20개 종목 분기 리밸런싱 백테스트", steps, conn,
        llm_fn=_json_llm(True, "경고"), market="KR",
        run_pipeline_fn=lambda s, conn=None: dict(_BT_RESULT),
    )

    assert out["blocked"] is False
    assert out["result"]["performance"]["cagr"] == 5.0
    assert len(out["warnings"]) == 4
    assert all(w["triggered"] for w in out["warnings"])


# ── 데이터 준비: 스냅샷 조회가 HA-1 실행기(execute_sql)를 경유한다 ────────────────
def test_data_prep_fetches_snapshot_via_execute_sql(tmp_path):
    conn = _seed_prices_ro(tmp_path, [
        ("005930", "2026-07-11", 71000.0, 4.1e14, 70000.0, 71500.0, 69800.0, 1.2e7),
    ])
    sql_calls: list[str] = []
    from src.agents.exec_runtime import execute_sql as real_execute_sql

    def spy_execute_sql(sql, c, *a, **k):
        sql_calls.append(sql)
        return real_execute_sql(sql, c, *a, **k)

    try:
        out = answer_backtest_question(
            "삼성전자 백테스트", [{"op": "run_backtest", "params": {}, "out": "bt"}], conn,
            stock_codes="005930", market="KR",
            run_pipeline_fn=lambda s, conn=None: dict(_BT_RESULT),
            execute_sql_fn=spy_execute_sql,
        )
    finally:
        conn.close()

    # 데이터 준비가 HA-1 실행기(execute_sql)를 통해 실행됐다
    assert len(sql_calls) >= 1
    assert any("prices" in s for s in sql_calls)
    # 준비된 스냅샷이 응답에 담겨 있다
    assert isinstance(out["data"], list)
    assert out["data"][0]["stock_code"] == "005930"
    assert out["blocked"] is False


def test_no_stock_codes_skips_data_prep(tmp_path):
    conn = _writable_conn(tmp_path)
    snap_calls: list = []

    def spy_snapshot(*a, **k):
        snap_calls.append((a, k))
        return []

    out = answer_backtest_question(
        "저PER 백테스트", [{"op": "run_backtest", "params": {}, "out": "bt"}], conn,
        market="KR", snapshot_fn=spy_snapshot,
        run_pipeline_fn=lambda s, conn=None: dict(_BT_RESULT),
    )

    assert snap_calls == []      # stock_codes 없으면 스냅샷 조회를 하지 않는다
    assert out["data"] == []


# ── steps(파이프라인) 자동 생성 — 질문 → LLM → JSON 파이프라인 (기존에 완전히 누락돼
#    있던 배선: web/app.py가 steps를 넘기지 않아 항상 빈 파이프라인이 실행되던 버그 수정) ──
def _pipeline_llm(pipeline):
    return lambda prompt: json.dumps({"pipeline": pipeline})


def test_generate_backtest_steps_parses_llm_pipeline_json():
    pipeline = [{
        "op": "run_backtest",
        "params": {"start_year": 2024, "end_year": 2026,
                   "criteria": [{"key": "per", "direction": "low", "weight": 1.0}],
                   "n": 10, "rebalance": "quarterly"},
        "out": "bt",
    }]
    steps = generate_backtest_steps("저PER 10개 분기 리밸런싱 백테스트", _pipeline_llm(pipeline))
    assert steps == pipeline


def test_generate_backtest_steps_returns_empty_on_invalid_json():
    steps = generate_backtest_steps("이상한 질문", lambda prompt: "이건 JSON이 아닙니다")
    assert steps == []


def test_generate_backtest_steps_returns_empty_when_pipeline_key_missing():
    steps = generate_backtest_steps("질문", lambda prompt: json.dumps({"foo": "bar"}))
    assert steps == []


def test_generate_backtest_steps_prompt_includes_question_and_primitives():
    captured: dict = {}

    def fake_llm(prompt):
        captured["prompt"] = prompt
        return json.dumps({"pipeline": []})

    generate_backtest_steps("삼성전자 골든크로스 데드크로스 전략 최근 2년 MDD", fake_llm)
    assert "삼성전자 골든크로스 데드크로스 전략 최근 2년 MDD" in captured["prompt"]
    # 프리미티브 스펙(legacy PIPELINE_USER와 동일한 10종)이 프롬프트에 포함돼야 LLM이
    # 존재하지 않는 연산을 지어내지 않는다.
    assert "run_backtest" in captured["prompt"]
    assert "search_strategy" in captured["prompt"]
    assert "get_cross_section" in captured["prompt"]
    assert "winsorize" in captured["prompt"]  # 신규 이상치 완화 프리미티브도 LLM에 노출돼야 함
    assert "correlation" in captured["prompt"]
    assert "quantile_bucket_means" in captured["prompt"]
    assert "gp_a" in captured["prompt"]  # GPA 필드도 get_cross_section 필드 목록에 노출돼야 함


def test_pipeline_prompt_documents_scatter_and_outlier_primitives():
    """산점도/이상치제거 프리미티브와 마법공식 필드가 프롬프트에 노출돼야 LLM이 '이익수익률과
    투하자본수익률 산점도, 이상치 제거' 파이프라인을 스스로 조립할 수 있다."""
    captured: dict = {}

    def fake_llm(prompt):
        captured["prompt"] = prompt
        return json.dumps({"pipeline": []})

    generate_backtest_steps("이익수익률과 투하자본수익률 산점도 그려줘 이상치는 빼고", fake_llm)
    p = captured["prompt"]
    assert "remove_outliers" in p        # 이상치 행 제거 프리미티브
    assert "scatter_data" in p           # 산점도용 데이터 정리 프리미티브
    assert "earnings_yield" in p         # 마법공식 이익수익률 필드
    assert "roc" in p                    # 마법공식 투하자본수익률 필드


def test_pipeline_prompt_documents_get_cross_section_markets_filter():
    """get_cross_section에 markets 필터가 있다는 걸 프롬프트가 알려줘야, LLM이 '코스피
    전종목' 류 질문에서 존재하지 않는 파라미터(예: run_backtest의 markets를 잘못 전용)를
    지어내는 대신 실제로 지원되는 markets 파라미터를 쓴다."""
    captured: dict = {}

    def fake_llm(prompt):
        captured["prompt"] = prompt
        return json.dumps({"pipeline": []})

    generate_backtest_steps("코스피 전종목 PBR과 GPA 상관계수 구해줘", fake_llm)
    p = captured["prompt"]
    assert "get_cross_section(asof, markets)" in p
    assert "get_cross_section에는" in p  # markets 외 다른 파라미터를 지어내지 말라는 경고 문구


def test_pipeline_prompt_documents_neutralize_zscore_method():
    """neutralize의 method='zscore'(그룹 표준편차 정규화)가 프롬프트에 노출돼야 LLM이
    '섹터 중립 포트폴리오' 류 질문을 demean만으로 잘못 처리하지 않는다."""
    captured: dict = {}

    def fake_llm(prompt):
        captured["prompt"] = prompt
        return json.dumps({"pipeline": []})

    generate_backtest_steps("섹터 중립화한 모멘텀 z-score로 상위 20종목 뽑아줘", fake_llm)
    p = captured["prompt"]
    assert "method='zscore'" in p
    assert "method='demean'" in p


def test_pipeline_prompt_correlation_example_also_includes_scatter_combo():
    """실서버 재현 버그: '상관관계 구하고 산점도로, 분위별 평균은 막대그래프로' 같은 3종
    복합 요청에서 LLM이 correlation/quantile_bucket_means만 뽑고 scatter_data를 빠뜨렸다.
    프롬프트의 correlation+quantile_bucket_means 예시가 scatter_data까지 포함한 3종 조합
    예시여야, 비슷한 질문에서 LLM이 산점도 단계를 빠뜨리지 않는다."""
    captured: dict = {}

    def fake_llm(prompt):
        captured["prompt"] = prompt
        return json.dumps({"pipeline": []})

    generate_backtest_steps("PBR과 GPA 상관관계 구하고 산점도로 보여줘", fake_llm)
    p = captured["prompt"]
    # 기존 correlation+quantile_bucket_means 예시가 있던 자리에 scatter_data 단계도
    # 함께 있어야 한다(3개 산출물을 한 파이프라인에서 동시에 요청하는 패턴을 학습시킴).
    assert '"op": "correlation"' in p
    assert '"op": "quantile_bucket_means"' in p
    assert '"op": "scatter_data"' in p
    # 여러 out을 만드는 파이프라인이 전부 보존된다는 것도 명시해, LLM이 "마지막 하나만
    # 살아남는다"고 오해해 산출물을 하나로 줄이지 않게 한다.
    assert "여러 개의 최종 산출물" in p or "모두 보존" in p


# ── answer_backtest_question: steps 비어있고 llm_fn 있으면 자동 생성해 감사배선에 넘긴다 ──
def test_answer_backtest_question_auto_generates_steps_when_empty(tmp_path):
    conn = _writable_conn(tmp_path)
    generated = [{"op": "run_backtest", "params": {}, "out": "bt"}]
    gen_calls: list = []

    def spy_generate(question, llm_fn, today=None):
        gen_calls.append(question)
        return generated

    audit_captured: dict = {}

    def spy_audit(steps, c, question, run_pipeline_fn, llm_fn=None, market="KR"):
        audit_captured["steps"] = steps
        return {"blocked": False, "error": None, "result": dict(_BT_RESULT),
                "hard": [], "warnings": []}

    out = answer_backtest_question(
        "저PER 벨류 백테스트 알려줘", [], conn,
        llm_fn=lambda p: "무시됨(spy_generate가 대신 호출됨)",
        generate_steps_fn=spy_generate,
        run_audit_fn=spy_audit,
    )

    assert gen_calls == ["저PER 벨류 백테스트 알려줘"]
    assert audit_captured["steps"] == generated
    assert out["result"]["performance"]["cagr"] == 5.0


def test_answer_backtest_question_on_progress_includes_pipeline_detail(tmp_path):
    """자동 생성된 파이프라인 JSON이 on_progress의 detail로도 실려야 실시간 트리에서
    실제 op/params를 보여줄 수 있다(요약 한 줄 "N단계"만으로는 검증 불가능)."""
    conn = _writable_conn(tmp_path)
    generated = [{"op": "run_backtest", "params": {"n": 5}, "out": "bt"}]
    events: list = []

    def spy_audit(steps, c, question, run_pipeline_fn, llm_fn=None, market="KR", on_progress=None):
        return {"blocked": False, "error": None, "result": dict(_BT_RESULT), "hard": [], "warnings": []}

    answer_backtest_question(
        "저PER 벨류 백테스트 알려줘", [], conn,
        llm_fn=_pipeline_llm(generated),
        run_audit_fn=spy_audit,
        on_progress=lambda step, summary, detail=None: events.append((step, summary, detail)),
    )

    detail_events = [e for e in events if e[2] is not None]
    assert len(detail_events) == 1
    step, summary, detail = detail_events[0]
    assert detail["kind"] == "backtest_pipeline"
    assert detail["steps"] == generated


def test_answer_backtest_question_skips_generation_when_steps_already_given(tmp_path):
    """steps가 이미 채워져 있으면(기존 호출부 회귀 방지) 자동 생성을 시도하지 않는다."""
    conn = _writable_conn(tmp_path)
    gen_calls: list = []

    def spy_generate(question, llm_fn, today=None):
        gen_calls.append(question)
        return [{"op": "should_not_be_used"}]

    steps = [{"op": "run_backtest", "params": {}, "out": "bt"}]
    answer_backtest_question(
        "질문", steps, conn, llm_fn=lambda p: "x",
        generate_steps_fn=spy_generate,
        run_pipeline_fn=lambda s, conn=None: dict(_BT_RESULT),
    )

    assert gen_calls == []


def test_answer_backtest_question_skips_generation_when_no_llm_fn(tmp_path):
    """llm_fn이 없으면(LLM 미가용) 생성을 시도하지 않고 결정론적으로 빈 파이프라인 그대로 둔다."""
    conn = _writable_conn(tmp_path)
    gen_calls: list = []

    def spy_generate(question, llm_fn, today=None):
        gen_calls.append(question)
        return [{"op": "should_not_be_used"}]

    answer_backtest_question(
        "질문", [], conn, llm_fn=None,
        generate_steps_fn=spy_generate,
        run_pipeline_fn=lambda s, conn=None: dict(_BT_RESULT),
    )

    assert gen_calls == []


# ── on_progress — 감사배선(run_audit_fn)으로 그대로 전달되고, steps 자동생성 시에도
#    "실행계획 생성 중" 진행 이벤트를 낸다 ──────────────────────────────────────────
def test_on_progress_passed_through_to_run_audit_fn(tmp_path):
    conn = _writable_conn(tmp_path)
    captured: dict = {}

    def spy_audit(steps, c, question, run_pipeline_fn, llm_fn=None, market="KR", on_progress=None):
        captured["on_progress"] = on_progress
        return {"blocked": False, "error": None, "result": dict(_BT_RESULT),
                "hard": [], "warnings": []}

    sentinel = lambda step, summary: None
    answer_backtest_question(
        "저PER 백테스트", [{"op": "run_backtest", "params": {}, "out": "bt"}], conn,
        run_audit_fn=spy_audit, on_progress=sentinel,
    )

    assert captured["on_progress"] is sentinel


def test_on_progress_reports_step_generation_when_steps_auto_generated(tmp_path):
    conn = _writable_conn(tmp_path)
    events: list[tuple[str, str]] = []
    generated = [{"op": "run_backtest", "params": {}, "out": "bt"}]

    answer_backtest_question(
        "저PER 백테스트", [], conn,
        llm_fn=lambda p: "무시됨",
        generate_steps_fn=lambda question, llm_fn, today=None: generated,
        run_pipeline_fn=lambda s, conn=None: dict(_BT_RESULT),
        on_progress=lambda step, summary, detail=None: events.append((step, summary)),
    )

    summaries = [s for _, s in events]
    assert any("실행계획 생성" in s and "중" in s for s in summaries)
    assert any("실행계획 생성 완료" in s and "1" in s for s in summaries)


def test_on_progress_without_it_is_unaffected(tmp_path):
    """on_progress 생략 시(기본값 None) 기존과 완전히 동일하게 동작 — 회귀 방지."""
    conn = _writable_conn(tmp_path)
    out = answer_backtest_question(
        "저PER 백테스트", [{"op": "run_backtest", "params": {}, "out": "bt"}], conn,
        run_pipeline_fn=lambda s, conn=None: dict(_BT_RESULT),
    )
    assert out["blocked"] is False


# ── 파이프라인 사전검증(HA-9) — LLM이 지어낸 알 수 없는 연산/미정의 참조를 run_pipeline이
#    실제로 실행하기 전에 잡아낸다. 검증 없이는 20단계 중 뒷부분에서 문제가 생겨도 앞부분의
#    무거운 연산(DB 조회 등)이 이미 다 실행된 뒤에야 실패가 드러난다(실거래봇 상주 머신이라
#    낭비된 연산도 비용). ─────────────────────────────────────────────────────────────
def test_validate_pipeline_steps_returns_empty_for_valid_pipeline():
    steps = [
        {"op": "get_cross_section", "params": {"asof": "2026-07-15"}, "out": "xs"},
        {"op": "combine", "params": {"rows": {"$ref": "xs"}, "criteria": []}, "out": "picked"},
    ]
    assert validate_pipeline_steps(steps) == []


def test_validate_pipeline_steps_returns_empty_for_empty_list():
    assert validate_pipeline_steps([]) == []


def test_validate_pipeline_steps_detects_unknown_op():
    steps = [{"op": "delete_all_data", "params": {}, "out": "x"}]
    errors = validate_pipeline_steps(steps)
    assert len(errors) == 1
    assert "delete_all_data" in errors[0]


def test_validate_pipeline_steps_detects_undefined_ref():
    """앞 스텝이 정의하지 않은 이름을 참조하면(오타 등) 실행 전에 잡는다."""
    steps = [{"op": "combine", "params": {"rows": {"$ref": "존재안함"}, "criteria": []}, "out": "picked"}]
    errors = validate_pipeline_steps(steps)
    assert len(errors) == 1
    assert "존재안함" in errors[0]


def test_validate_pipeline_steps_allows_ref_to_earlier_step_out():
    steps = [
        {"op": "get_cross_section", "params": {"asof": "2026-07-15"}, "out": "xs"},
        {"op": "zscore", "params": {"rows": {"$ref": "xs"}, "field": "per"}, "out": "ranked"},
    ]
    assert validate_pipeline_steps(steps) == []


def test_validate_pipeline_steps_collects_multiple_errors():
    steps = [
        {"op": "unknown_op_1", "params": {}, "out": "a"},
        {"op": "combine", "params": {"rows": {"$ref": "없는참조"}, "criteria": []}, "out": "b"},
    ]
    errors = validate_pipeline_steps(steps)
    assert len(errors) == 2


def test_answer_backtest_question_blocks_and_skips_execution_on_unknown_op(tmp_path):
    """알 수 없는 연산이 있으면 run_pipeline_fn/run_audit_fn을 아예 호출하지 않는다(실행 낭비 방지)."""
    conn = _writable_conn(tmp_path)
    pipeline_calls: list = []
    audit_calls: list = []

    out = answer_backtest_question(
        "이상한 백테스트", [{"op": "존재하지않는연산", "params": {}, "out": "x"}], conn,
        run_pipeline_fn=lambda s, conn=None: pipeline_calls.append(s) or dict(_BT_RESULT),
        run_audit_fn=lambda *a, **k: audit_calls.append(a) or {
            "blocked": False, "error": None, "result": dict(_BT_RESULT), "hard": [], "warnings": [],
        },
    )

    assert pipeline_calls == []
    assert audit_calls == []
    assert out["blocked"] is True
    assert "존재하지않는연산" in out["error"]
    assert out["result"] is None


def test_answer_backtest_question_blocks_on_undefined_ref(tmp_path):
    conn = _writable_conn(tmp_path)
    steps = [{"op": "combine", "params": {"rows": {"$ref": "없음"}, "criteria": []}, "out": "picked"}]

    out = answer_backtest_question(
        "질문", steps, conn,
        run_pipeline_fn=lambda s, conn=None: dict(_BT_RESULT),
    )

    assert out["blocked"] is True
    assert "없음" in out["error"]


def test_answer_backtest_question_reports_progress_on_validation_failure(tmp_path):
    conn = _writable_conn(tmp_path)
    events: list[tuple[str, str]] = []

    answer_backtest_question(
        "질문", [{"op": "없는연산", "params": {}, "out": "x"}], conn,
        run_pipeline_fn=lambda s, conn=None: dict(_BT_RESULT),
        on_progress=lambda step, summary: events.append((step, summary)),
    )

    assert any("검증" in s for _, s in events)


def test_answer_backtest_question_still_runs_valid_pipeline_after_validation_added(tmp_path):
    """유효한 파이프라인은 검증 도입 후에도 그대로 실행된다(회귀 방지)."""
    conn = _writable_conn(tmp_path)
    out = answer_backtest_question(
        "저PER 백테스트", [{"op": "run_backtest", "params": {}, "out": "bt"}], conn,
        run_pipeline_fn=lambda s, conn=None: dict(_BT_RESULT),
    )
    assert out["blocked"] is False
    assert out["result"]["performance"]["cagr"] == 5.0


def test_answer_backtest_question_surfaces_top_n_for_list_result_from_combine_n():
    """실서버 재현: QVM 스크리닝처럼 combine의 n=20으로 20개를 뽑아도, backtest 도메인
    결과에 top_n이 없어 검증/종합결론 LLM 프롬프트가 앞 5개로 잘라버리고(supervisor.
    _truncate_for_prompt) "일부 종목만 있다"고 오판했다. steps의 마지막 n 파라미터를
    top_n으로 함께 반환해 kr/us 스크리닝과 동일한 축약 예외를 받도록 한다."""
    rows = [{"stock_code": f"S{i}", "qvm_score": float(i)} for i in range(20)]
    out = answer_backtest_question(
        "QVM 상위 20종목", [
            {"op": "get_cross_section_qvm", "params": {"asof": "2026-07-15"}, "out": "xs"},
            {"op": "compute_qvm_scores", "params": {"rows": {"$ref": "xs"}}, "out": "scored"},
            {
                "op": "combine",
                "params": {"rows": {"$ref": "scored"}, "criteria": [], "n": 20},
                "out": "picked",
            },
        ],
        conn=None,
        run_pipeline_fn=lambda s, conn=None: rows,
    )
    assert out["blocked"] is False
    assert out["result"] == rows
    assert out["top_n"] == 20


def test_answer_backtest_question_no_top_n_key_when_steps_lack_n_param():
    """회귀: n 파라미터가 없는 파이프라인(예: 상관관계 분석)은 top_n 키를 추가하지 않는다
    (기존 동작 그대로 head=5 축약 — 새 필드 도입이 무관한 경로에 영향을 주지 않는지 확인)."""
    out = answer_backtest_question(
        "PBR과 GPA 상관관계", [
            {"op": "get_cross_section", "params": {"asof": "2026-07-15"}, "out": "xs"},
            {"op": "correlation", "params": {"rows": {"$ref": "xs"}, "field_x": "pbr", "field_y": "gp_a"}, "out": "corr"},
        ],
        conn=None,
        run_pipeline_fn=lambda s, conn=None: {"r": 0.5, "n": 100},
    )
    assert out["blocked"] is False
    assert "top_n" not in out


# ── qvm_summary — QVM 스크리닝 결과에 asof/excluded_count/sector_distribution 부착 ──
# 사용자 요청 [출력] 절: "마지막에: 사용한 데이터 기준일, 결측/제외된 종목 수, 섹터 분포
# 요약"을 답하기 위한 배선. compute_qvm_scores가 각 row에 남기는 내부 마커
# '_qvm_excluded_count'(primitives.py)로 excluded_count를 복원하고, asof는 steps의
# get_cross_section_qvm 파라미터에서 정적으로 추출한다(_infer_requested_count와 동일 관례).
_QVM_STEPS_WITH_COMBINE = [
    {"op": "get_cross_section_qvm", "params": {"asof": "2026-07-10"}, "out": "xs"},
    {"op": "compute_qvm_scores", "params": {"rows": {"$ref": "xs"}}, "out": "scored"},
    {
        "op": "combine",
        "params": {"rows": {"$ref": "scored"}, "criteria": [], "n": 3},
        "out": "picked",
    },
]


def _qvm_scored_rows(sectors: list[str], excluded_count: int = 2) -> list[dict]:
    """compute_qvm_scores가 실제로 반환할 법한 모양(각 row에 _qvm_excluded_count 마커
    포함)의 fixture. run_pipeline_fn이 이 마커 붙은 rows를 그대로 돌려주도록 주입해
    실제 DB/LLM 없이 answer_backtest_question의 qvm_summary 배선만 검증한다."""
    return [
        {"stock_code": f"S{i}", "sector": sector, "qvm_score": float(i),
         "_qvm_excluded_count": excluded_count}
        for i, sector in enumerate(sectors)
    ]


def test_answer_backtest_question_builds_qvm_summary_with_top_n():
    """combine으로 top_n(3개)까지 걸러진 뒤에도 qvm_summary가 마커를 그대로 복원한다."""
    rows = _qvm_scored_rows(["화학", "화학", "금융"], excluded_count=2)
    out = answer_backtest_question(
        "퀄리티 밸류 모멘텀 상위 3종목", _QVM_STEPS_WITH_COMBINE,
        conn=None,
        run_pipeline_fn=lambda s, conn=None: rows,
    )
    assert out["blocked"] is False
    assert out["qvm_summary"] == {
        "asof": "2026-07-10",
        "excluded_count": 2,
        "sector_distribution": {"화학": 2, "금융": 1},
    }


def test_answer_backtest_question_builds_qvm_summary_without_top_n():
    """AC4: combine/n 없이 compute_qvm_scores 단계만 있는(전체 유니버스 요청) 파이프라인도
    qvm_summary가 자연스럽게 동작해야 한다(top_n 키는 붙지 않아도 된다)."""
    steps = [
        {"op": "get_cross_section_qvm", "params": {"asof": "2026-07-11"}, "out": "xs"},
        {"op": "compute_qvm_scores", "params": {"rows": {"$ref": "xs"}}, "out": "scored"},
    ]
    rows = _qvm_scored_rows(["화학", "IT", "IT"], excluded_count=0)
    out = answer_backtest_question(
        "퀄리티 밸류 모멘텀 전체 유니버스 점수", steps,
        conn=None,
        run_pipeline_fn=lambda s, conn=None: rows,
    )
    assert out["blocked"] is False
    assert "top_n" not in out  # combine/n 단계가 없으므로 top_n은 여전히 안 붙는다
    assert out["qvm_summary"] == {
        "asof": "2026-07-11",
        "excluded_count": 0,
        "sector_distribution": {"화학": 1, "IT": 2},
    }


def test_answer_backtest_question_no_qvm_summary_when_not_qvm_pipeline(tmp_path):
    """회귀: compute_qvm_scores 단계가 없는 일반 파이프라인은 qvm_summary 키 자체가
    없어야 한다(무관한 경로에 새 필드가 영향을 주지 않는지 확인 — top_n의 하위호환 관례와 동일)."""
    conn = _writable_conn(tmp_path)
    out = answer_backtest_question(
        "저PER 20개 종목 분기 리밸런싱 백테스트",
        [{"op": "run_backtest", "params": {}, "out": "bt"}],
        conn,
        run_pipeline_fn=lambda s, conn=None: dict(_BT_RESULT),
    )
    assert out["blocked"] is False
    assert "qvm_summary" not in out


# ── data_asof — get_cross_section(_qvm)의 요청 asof를 결과에 노출한다. 실서버 재현:
#    "코스피 전종목 pbr/gpa 상관관계 5분위 평균" 같은 correlation/quantile_bucket_means
#    파이프라인은 집계값만 반환해(result에 시점 정보가 전혀 안 남음) 사용자가 "어느 시점
#    데이터인지 검증할 수 없다"는 답변을 받았다. steps에서 asof를 정적으로 추출해
#    kr/us 도메인과 동일한 키({"price_date": ...})로 노출하면, supervisor.py의 종합결론
#    로직(도메인 무관하게 data_asof를 언급하도록 이미 배선돼 있음)이 그대로 재사용된다. ──
def test_answer_backtest_question_attaches_data_asof_for_correlation_pipeline():
    out = answer_backtest_question(
        "PBR과 GPA 상관관계", [
            {"op": "get_cross_section", "params": {"asof": "2026-07-15"}, "out": "xs"},
            {"op": "correlation", "params": {"rows": {"$ref": "xs"}, "field_x": "pbr", "field_y": "gp_a"}, "out": "corr"},
        ],
        conn=None,
        run_pipeline_fn=lambda s, conn=None: {"r": 0.5, "n": 100},
    )
    assert out["data_asof"] == {"price_date": "2026-07-15"}


def test_answer_backtest_question_attaches_data_asof_for_qvm_cross_section():
    rows = _qvm_scored_rows(["화학", "화학", "금융"], excluded_count=2)
    out = answer_backtest_question(
        "퀄리티 밸류 모멘텀 상위 3종목", _QVM_STEPS_WITH_COMBINE,
        conn=None,
        run_pipeline_fn=lambda s, conn=None: rows,
    )
    assert out["data_asof"] == {"price_date": "2026-07-10"}


def test_answer_backtest_question_data_asof_includes_financial_quarter_when_conn_given(tmp_path):
    """실사용 리포트: "가격날짜는 있는데 재무데이터 시점은 안 나온다" — conn이 주어지면
    financials 테이블에서 실제 재무 기준분기(최빈값)도 함께 붙어야 한다."""
    conn = _writable_conn(tmp_path)
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
        "VALUES (?,?,?,?,?)",
        ("000001", "2026Q1", "2026-05-15", "revenue", 100.0),
    )
    conn.commit()
    out = answer_backtest_question(
        "PBR과 GPA 상관관계", [
            {"op": "get_cross_section", "params": {"asof": "2026-07-18"}, "out": "xs"},
            {"op": "correlation", "params": {"rows": {"$ref": "xs"}, "field_x": "pbr", "field_y": "gp_a"}, "out": "corr"},
        ],
        conn,
        run_pipeline_fn=lambda s, conn=None: {"r": 0.5, "n": 100},
    )
    assert out["data_asof"] == {"price_date": "2026-07-18", "financial_quarter": "2026Q1"}


def test_answer_backtest_question_no_data_asof_when_no_cross_section_step(tmp_path):
    """회귀: get_cross_section을 쓰지 않는 파이프라인(run_backtest 등)은 data_asof 키
    자체가 없어야 한다(top_n/qvm_summary와 동일한 하위호환 관례)."""
    conn = _writable_conn(tmp_path)
    out = answer_backtest_question(
        "저PER 20개 종목 분기 리밸런싱 백테스트",
        [{"op": "run_backtest", "params": {}, "out": "bt"}],
        conn,
        run_pipeline_fn=lambda s, conn=None: dict(_BT_RESULT),
    )
    assert "data_asof" not in out


# ── rebalance_summary — 다중 리밸런싱 백테스트면 시점별 보유종목+구간수익률을 결정론적
#    텍스트로 결과 payload에 첨부한다(top_n/qvm_summary와 동일한 순수 추가 필드 관례).
#    "반기별이면 반기마다 어떤 종목이 있었는지와 반기별 수익률도 같이 항상 답한다"는 요구를
#    LLM 요약 재량이 아니라 결정론적으로 보장하기 위한 배선. ──────────────────────────
_BT_MULTI_REBALANCE = {
    "dates": ["2025-06-30", "2025-12-31", "2026-06-30"],
    "navs": [1.0, 1.05, 1.1],
    "benchmark": None,
    "performance": {"cagr": 5.0, "avg_turnover": 0.3},
    "holdings": [
        {"date": "2025-06-30", "codes": ["000001", "000002"], "period_return": 0.05},
        {"date": "2025-12-31", "codes": ["000003"], "period_return": -0.02},
    ],
}


def test_answer_backtest_question_builds_rebalance_summary_for_multi_rebalance(tmp_path):
    conn = _writable_conn(tmp_path)
    out = answer_backtest_question(
        "저PER 20개 종목 반기 리밸런싱 백테스트",
        [{"op": "run_backtest", "params": {}, "out": "bt"}],
        conn,
        run_pipeline_fn=lambda s, conn=None: dict(_BT_MULTI_REBALANCE),
    )
    assert out["blocked"] is False
    summary = out["rebalance_summary"]
    assert isinstance(summary, str)
    # 각 리밸런싱 시점의 종목과 구간수익률(부호·퍼센트)이 모두 결정론적으로 담겨야 한다.
    assert "2025-06-30" in summary and "2025-12-31" in summary
    assert "000001" in summary and "000002" in summary and "000003" in summary
    assert "+5.00%" in summary        # 0.05 → +5.00%
    assert "-2.00%" in summary        # -0.02 → -2.00%


def test_answer_backtest_question_no_rebalance_summary_for_single_holding(tmp_path):
    """단일 리밸런싱(buy&hold, holdings 1개)이면 구간별 서술이 의미 없으므로 필드 자체가 없다."""
    conn = _writable_conn(tmp_path)
    out = answer_backtest_question(
        "삼성전자 buy&hold 백테스트",
        [{"op": "run_backtest", "params": {}, "out": "bt"}],
        conn,
        run_pipeline_fn=lambda s, conn=None: dict(_BT_RESULT),  # holdings 1개
    )
    assert out["blocked"] is False
    assert "rebalance_summary" not in out


def test_answer_backtest_question_no_rebalance_summary_when_blocked(tmp_path):
    """하드차단(결과 폐기)이면 rebalance_summary도 붙지 않는다."""
    conn = _writable_conn(tmp_path)
    conn.execute("INSERT INTO delisting(stock_code, name, delisting_date) VALUES (?,?,?)",
                 ("000001", "죽은회사", "2020-01-01"))
    conn.commit()
    out = answer_backtest_question(
        "저PER 반기 리밸런싱 백테스트",
        [{"op": "run_backtest", "params": {}, "out": "bt"}],
        conn, llm_fn=_json_llm(True, "x"), market="KR",
        run_pipeline_fn=lambda s, conn=None: dict(_BT_MULTI_REBALANCE),
    )
    assert out["blocked"] is True
    assert "rebalance_summary" not in out
