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


def test_market_us_selects_us_snapshot_agent(tmp_path, monkeypatch):
    """market='US'이면 기본 스냅샷 에이전트로 get_price_snapshot_us를 선택한다."""
    import src.agents.domain_backtest as mod

    picked = {"kr": 0, "us": 0}
    monkeypatch.setattr(mod, "get_price_snapshot_kr",
                        lambda *a, **k: picked.__setitem__("kr", picked["kr"] + 1) or [])
    monkeypatch.setattr(mod, "get_price_snapshot_us",
                        lambda *a, **k: picked.__setitem__("us", picked["us"] + 1) or [])

    conn = _writable_conn(tmp_path)
    answer_backtest_question(
        "AAPL 백테스트", [{"op": "run_backtest", "params": {}, "out": "bt"}], conn,
        stock_codes="AAPL", market="US",
        run_pipeline_fn=lambda s, conn=None: dict(_BT_RESULT),
    )

    assert picked == {"kr": 0, "us": 1}


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
