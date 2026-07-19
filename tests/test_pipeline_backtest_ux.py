"""SQL/Python 파이프라인 백테스트 UX 개선 테스트.

Fable5 3라운드 조사(계층형 멀티에이전트 대신 좁은 결함 3개)에서 발견한 갭:
1. PIPELINE_USER 프롬프트가 run_backtest(이미 실행기엔 등록됨)를 숨기고 있어
   자연어 백테스트 질문이 원천적으로 동작 안 함.
2. holdings(분기별 보유종목)가 종목코드만 있고 이름 변환이 파이프라인 경로엔 없음.
3. CLI가 백테스트 dict 결과(performance/holdings 중첩)를 일반 표로 찍어 가독성이 없음.
"""
from __future__ import annotations

import sqlite3
from datetime import date

from src.legacy.graph import prompts
from src.legacy.graph.nodes import Deps, _resolve_holdings_names, make_nodes
from tests.conftest import FakeLLM


def test_pipeline_user_prompt_includes_run_backtest():
    assert "run_backtest" in prompts.PIPELINE_USER


def test_pipeline_user_prompt_still_formats_without_error():
    text = prompts.PIPELINE_USER.format(schema="(스키마)", today="2026-07-12", question="테스트 질문")
    assert "테스트 질문" in text
    assert "2026-07-12" in text


def test_pipeline_user_prompt_documents_compute_technical_indicator():
    """compute_technical_indicator(9번째 프리미티브)가 실행기(PRIMITIVE_OPS)엔 등록됐지만
    PIPELINE_USER 프롬프트에 문서화가 빠지면, 자연어로 기술지표 질문을 해도 LLM이
    이 프리미티브를 생성할 방법이 없는 갭이 생긴다(run_backtest/market 파라미터 때와
    동일 패턴 재발 방지)."""
    assert "compute_technical_indicator" in prompts.PIPELINE_USER
    assert prompts.PIPELINE_USER.format(schema="(s)", today="2026-07-13", question="q")


def test_pipeline_user_prompt_documents_all_four_indicator_names():
    idx = prompts.PIPELINE_USER.find("compute_technical_indicator(")
    assert idx != -1
    section = prompts.PIPELINE_USER[idx:idx + 800]
    for name in ("sma", "ema", "rsi", "macd", "bollinger"):
        assert name in section.lower()


def test_pipeline_user_prompt_has_technical_indicator_example():
    idx = prompts.PIPELINE_USER.find("compute_technical_indicator")
    tail = prompts.PIPELINE_USER[idx:]
    assert '"op": "compute_technical_indicator"' in tail


def test_pipeline_user_prompt_documents_search_strategy():
    """search_strategy(10번째 프리미티브)가 실행기엔 등록됐지만 프롬프트에 문서화가
    빠지면, 자연어 전략탐색 질문을 해도 LLM이 이 프리미티브를 생성할 방법이 없다."""
    assert "search_strategy" in prompts.PIPELINE_USER
    assert prompts.PIPELINE_USER.format(schema="(s)", today="2026-07-13", question="q")


def test_pipeline_user_prompt_documents_search_strategy_constraints_and_rank_by():
    idx = prompts.PIPELINE_USER.find("search_strategy(")
    section = prompts.PIPELINE_USER[idx:idx + 900]
    assert "constraints" in section
    assert "rank_by" in section


def test_pipeline_user_prompt_has_search_strategy_example_with_constraints():
    idx = prompts.PIPELINE_USER.find("search_strategy")
    tail = prompts.PIPELINE_USER[idx:]
    assert '"op": "search_strategy"' in tail
    assert '"constraints"' in tail


# --------------------------------------------------------------------------
# _resolve_holdings_names (순수 함수, DB 없이 code_to_name dict 주입)
# --------------------------------------------------------------------------
def test_resolve_holdings_names_replaces_codes_with_names():
    result = {
        "dates": ["2026-01-01"],
        "navs": [1.0],
        "holdings": [{"date": "2026-01-01", "codes": ["000001", "999999"]}],
    }
    out = _resolve_holdings_names(result, {"000001": "가나전자"})
    assert out["holdings"][0]["names"] == ["가나전자", "999999"]
    # 원본 codes는 유지된다
    assert out["holdings"][0]["codes"] == ["000001", "999999"]


def test_resolve_holdings_names_noop_when_no_holdings_key():
    result = {"slope": 1.0, "se_slope": 0.2}
    out = _resolve_holdings_names(result, {"000001": "가나전자"})
    assert out == result


# --------------------------------------------------------------------------
# execute_node 배선: run_pipeline이 holdings(codes만)를 반환해도 이름이 채워져야 한다
# --------------------------------------------------------------------------
def _deps(llm, db_path=None):
    if db_path:
        conn = sqlite3.connect(db_path)
    else:
        conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return Deps(conn=conn, store=None, schema="(schema)", today=date(2026, 6, 22), llm=llm)


def test_pipeline_execute_wires_holdings_names_end_to_end(tmp_path, monkeypatch):
    from src.db import init_db

    db = tmp_path / "test_market.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO company(stock_code, name, market, sector) VALUES (?,?,?,?)",
        ("000001", "가나전자", "KOSPI", "기타"),
    )
    conn.commit()
    conn.close()

    from src.backtest import pipeline_exec as pipeline_exec_module

    monkeypatch.setattr(
        pipeline_exec_module, "run_pipeline",
        lambda steps, conn=None: {
            "dates": ["2026-01-01", "2026-03-31"],
            "navs": [1.0, 1.1],
            "benchmark": None,
            "performance": {"cagr": 5.0},
            "holdings": [{"date": "2026-01-01", "codes": ["000001", "999999"]}],
        },
    )
    nodes = make_nodes(_deps(FakeLLM("{}"), db_path=str(db)))
    state = {"route": "pipeline", "pipeline": [{"op": "run_backtest", "params": {}, "out": "bt"}]}
    out = nodes["execute_node"](state)

    assert out["error"] is None
    holdings = out["rows"][0]["holdings"]
    assert holdings[0]["names"] == ["가나전자", "999999"]  # 999999는 매핑 없어 코드 그대로 폴백
