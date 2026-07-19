"""자동 트리거 배선(AUD-7) + 심각도별 결과 분리(AUD-8) 테스트.

run_backtest op가 있는 파이프라인 실행 시에만 사전/사후 감사가 자동 개입하고,
- 하드차단이 발동하면 정상 결과(holdings/performance) 없이 에러만 반환(AC9),
- 소프트경고는 정상 결과에 audit_warnings로 첨부(AC10)됨을 검증한다.
"""
from __future__ import annotations

import sqlite3
from datetime import date

from src.legacy.graph.nodes import Deps, make_nodes
from tests.conftest import FakeLLM


def _deps(llm, db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return Deps(conn=conn, store=None, schema="(schema)", today=date(2026, 6, 22), llm=llm)


def _db_with_company(tmp_path):
    from src.db import init_db

    db = tmp_path / "wiring.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.execute("INSERT INTO company(stock_code, name, market, sector) VALUES (?,?,?,?)",
                 ("000001", "가나전자", "KOSPI", "기타"))
    conn.commit()
    conn.close()
    return str(db)


_BT_RESULT = {
    "dates": ["2025-09-30", "2025-12-31"],
    "navs": [1.0, 1.1],
    "benchmark": None,
    "performance": {"cagr": 5.0},
    "holdings": [{"date": "2025-12-31", "codes": ["000001"]}],
}


# --------------------------------------------------------------------------
# AUD-7: run_backtest op일 때만 사전/사후 훅이 각각 호출된다
# --------------------------------------------------------------------------
def test_audit_hooks_fire_for_run_backtest_pipeline(tmp_path, monkeypatch):
    from src.backtest import auditor
    from src.backtest import pipeline_exec as pe

    calls = {"pre": 0, "post": 0}
    monkeypatch.setattr(auditor, "pre_audit",
                        lambda steps, conn, run_pipeline_fn: calls.__setitem__("pre", calls["pre"] + 1) or None)
    monkeypatch.setattr(auditor, "post_audit",
                        lambda result, conn, question, llm_fn=None, **kwargs: (calls.__setitem__("post", calls["post"] + 1)
                                                                              or {"blocked": False, "hard": [], "soft": []}))
    monkeypatch.setattr(pe, "run_pipeline", lambda steps, conn=None: dict(_BT_RESULT))

    nodes = make_nodes(_deps(FakeLLM("{}"), _db_with_company(tmp_path)))
    state = {"route": "pipeline", "raw_question": "질문",
             "pipeline": [{"op": "run_backtest", "params": {}, "out": "bt"}]}
    nodes["execute_node"](state)
    assert calls == {"pre": 1, "post": 1}


def test_audit_hooks_skip_when_no_run_backtest(tmp_path, monkeypatch):
    from src.backtest import auditor
    from src.backtest import pipeline_exec as pe

    calls = {"pre": 0, "post": 0}
    monkeypatch.setattr(auditor, "pre_audit",
                        lambda *a, **k: calls.__setitem__("pre", calls["pre"] + 1) or None)
    monkeypatch.setattr(auditor, "post_audit",
                        lambda *a, **k: calls.__setitem__("post", calls["post"] + 1) or {})
    monkeypatch.setattr(pe, "run_pipeline", lambda steps, conn=None: [{"stock_code": "000001", "per": 5.0}])

    nodes = make_nodes(_deps(FakeLLM("{}"), _db_with_company(tmp_path)))
    state = {"route": "pipeline",
             "pipeline": [{"op": "get_cross_section", "params": {"asof": "2025-12-31"}, "out": "xs"}]}
    nodes["execute_node"](state)
    assert calls == {"pre": 0, "post": 0}  # 백테스트 아닌 파이프라인엔 개입 안 함(오버헤드 방지)


# --------------------------------------------------------------------------
# AUD-8: 하드차단 → 에러만 / 소프트경고 → 결과+경고 공존
# --------------------------------------------------------------------------
def test_pre_hard_block_returns_error_without_normal_result(tmp_path, monkeypatch):
    from src.backtest import auditor
    from src.backtest import pipeline_exec as pe

    ran = {"pipeline": False}

    def _run(steps, conn=None):
        ran["pipeline"] = True
        return dict(_BT_RESULT)

    monkeypatch.setattr(pe, "run_pipeline", _run)
    monkeypatch.setattr(auditor, "pre_audit", lambda steps, conn, run_pipeline_fn: {
        "sin": "short_positions", "blocked": True, "reason": "음수 비중",
        "evidence": [{"asset": "000002", "weight": -0.1}]})

    nodes = make_nodes(_deps(FakeLLM("{}"), _db_with_company(tmp_path)))
    state = {"route": "pipeline",
             "pipeline": [{"op": "run_backtest", "params": {"weights": {"$ref": "w"}}}]}
    out = nodes["execute_node"](state)
    assert out["error"] and "short_positions" in out["error"]
    assert out["rows"] == [] and out["row_count"] == 0  # 정상 결과(holdings/performance) 없음(AC9)
    assert ran["pipeline"] is False  # 사전차단이므로 run_backtest는 실행조차 안 됨


def test_post_hard_block_discards_result(tmp_path, monkeypatch):
    from src.backtest import auditor
    from src.backtest import pipeline_exec as pe

    monkeypatch.setattr(pe, "run_pipeline", lambda steps, conn=None: dict(_BT_RESULT))
    monkeypatch.setattr(auditor, "post_audit", lambda result, conn, question, llm_fn=None, **kwargs: {
        "blocked": True,
        "hard": [{"sin": "survivorship", "blocked": True, "reason": "상폐 종목 포함",
                  "evidence": [{"stock_code": "000009", "asof": "2020-01-01"}]}],
        "soft": []})

    nodes = make_nodes(_deps(FakeLLM("{}"), _db_with_company(tmp_path)))
    state = {"route": "pipeline",
             "pipeline": [{"op": "run_backtest", "params": {}, "out": "bt"}]}
    out = nodes["execute_node"](state)
    assert out["error"] and "survivorship" in out["error"]
    assert out["rows"] == []


def test_soft_warnings_attached_without_hiding_result(tmp_path, monkeypatch):
    from src.backtest import auditor
    from src.backtest import pipeline_exec as pe

    monkeypatch.setattr(pe, "run_pipeline", lambda steps, conn=None: dict(_BT_RESULT))
    monkeypatch.setattr(auditor, "post_audit", lambda result, conn, question, llm_fn=None, **kwargs: {
        "blocked": False,
        "hard": [],
        "soft": [{"sin": "snooping", "triggered": True, "message": "사후정당화 의심"},
                 {"sin": "outlier", "triggered": False, "message": ""}]})

    nodes = make_nodes(_deps(FakeLLM("{}"), _db_with_company(tmp_path)))
    state = {"route": "pipeline",
             "pipeline": [{"op": "run_backtest", "params": {}, "out": "bt"}]}
    out = nodes["execute_node"](state)
    assert out["error"] is None
    # 정상 결과는 그대로 유지
    assert out["rows"][0]["performance"] == {"cagr": 5.0}
    # triggered된 경고만 첨부(AC10)
    assert len(out["audit_warnings"]) == 1
    assert out["audit_warnings"][0]["sin"] == "snooping"


# --------------------------------------------------------------------------
# search_strategy(역백테스트) 파이프라인 — architect 검토(MAJOR) 반영:
# run_backtest가 아니라 search_strategy op일 때도 스누핑 소프트경고가 1회 첨부된다.
# --------------------------------------------------------------------------
_SEARCH_RESULT = [
    {"criteria": [{"key": "per", "direction": "low", "weight": 1.0}],
     "performance": {"sharpe": 2.0, "mdd": -5.0}, "holdings": [{"date": "2026-03-31", "codes": ["000001"]}]},
]


def test_search_strategy_pipeline_gets_snooping_soft_warning(tmp_path, monkeypatch):
    from src.backtest import auditor
    from src.backtest import pipeline_exec as pe

    monkeypatch.setattr(pe, "run_pipeline", lambda steps, conn=None: list(_SEARCH_RESULT))
    calls = {"n": 0}
    monkeypatch.setattr(
        auditor, "audit_search_strategy_result",
        lambda results, question, llm_fn: (calls.__setitem__("n", calls["n"] + 1)
                                           or [{"sin": "snooping", "triggered": True, "message": "의심"}]),
    )

    nodes = make_nodes(_deps(FakeLLM("{}"), _db_with_company(tmp_path)))
    state = {"route": "pipeline", "raw_question": "전략 찾아줘",
             "pipeline": [{"op": "search_strategy", "params": {}, "out": "found"}]}
    out = nodes["execute_node"](state)
    assert calls["n"] == 1
    assert out["error"] is None
    assert len(out["audit_warnings"]) == 1
    assert out["audit_warnings"][0]["sin"] == "snooping"


def test_search_strategy_pipeline_result_rows_preserved_with_warning(tmp_path, monkeypatch):
    from src.backtest import auditor
    from src.backtest import pipeline_exec as pe

    monkeypatch.setattr(pe, "run_pipeline", lambda steps, conn=None: list(_SEARCH_RESULT))
    monkeypatch.setattr(auditor, "audit_search_strategy_result", lambda *a, **k: [])

    nodes = make_nodes(_deps(FakeLLM("{}"), _db_with_company(tmp_path)))
    state = {"route": "pipeline",
             "pipeline": [{"op": "search_strategy", "params": {}, "out": "found"}]}
    out = nodes["execute_node"](state)
    assert out["error"] is None
    assert out["row_count"] == 1
    assert "audit_warnings" not in out  # triggered 없으면 첨부 안 함(기존 관례)
