"""refine_node가 절대금액 질문에 환율을 삽입하는지 (TDD, C-5 AC7 나머지 절반).

.omc/specs/brainstorming-us-nl-sql-integration.md 참고. "삼성전자와 애플 중
시가총액 큰 곳" 같은 절대금액 질문은 SQL 생성 LLM이 실시간 환율을 모르므로,
refine_node가 정제된 질문에 환율을 명시적으로 삽입해 전달한다.
"""
from __future__ import annotations

import sqlite3
from datetime import date

from src.db import init_db
from src.legacy.graph.nodes import Deps, make_nodes
from src.ingest.exchange_rate import needs_exchange_rate
from tests.conftest import FakeLLM


def _deps(tmp_path, llm):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return Deps(conn=conn, store=None, schema="(schema)", today=date(2026, 7, 12), llm=llm)


def test_needs_exchange_rate_true_for_absolute_amount_keywords():
    assert needs_exchange_rate("삼성전자와 애플 중 시가총액 큰 곳") is True
    assert needs_exchange_rate("애플 매출액 알려줘") is True


def test_needs_exchange_rate_false_for_ratio_questions():
    assert needs_exchange_rate("삼성전자와 애플 중 PER 낮은 곳") is False
    assert needs_exchange_rate("삼성전자 ROE 알려줘") is False


def test_refine_node_inserts_exchange_rate_for_absolute_amount_question(tmp_path, monkeypatch):
    import src.legacy.graph.nodes as nodes_module

    monkeypatch.setattr(nodes_module, "get_usdkrw_rate", lambda conn, on: 1503.4)
    llm = FakeLLM("삼성전자와 애플 중 시가총액 큰 곳")
    deps = _deps(tmp_path, llm)
    nodes = make_nodes(deps)

    out = nodes["refine_node"]({"raw_question": "삼성전자랑 애플 중 시총 큰데가 어디야"})

    assert "1503.4" in out["question"] or "1,503.4" in out["question"]


def test_refine_node_does_not_insert_exchange_rate_for_ratio_question(tmp_path, monkeypatch):
    import src.legacy.graph.nodes as nodes_module

    calls = {"n": 0}
    monkeypatch.setattr(nodes_module, "get_usdkrw_rate", lambda conn, on: calls.__setitem__("n", calls["n"] + 1) or 1503.4)
    llm = FakeLLM("삼성전자와 애플 중 PER 낮은 곳")
    deps = _deps(tmp_path, llm)
    nodes = make_nodes(deps)

    nodes["refine_node"]({"raw_question": "삼성전자랑 애플 중 PER 낮은데가 어디야"})

    assert calls["n"] == 0
