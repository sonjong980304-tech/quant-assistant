"""신규 goldset 러너(src/eval/hierarchical_runner.py) 단위테스트 (HA-14, AC18/AC20).

판정 로직(judge)·호출 카운터·오케스트레이션을 mock LLM/주입 fixture 로 결정론적으로 검증한다
(실제 OpenAI API·실제 그래프 없이). 실제 goldset 대량 재실행은 scripts/eval_hierarchical_goldset.py
가 담당하고, 그 정확도/성능 수치는 .omc/research/ 에 문서화한다.
"""
from __future__ import annotations

from src.db import init_db
from src.eval.hierarchical_runner import (
    CountingLLM,
    judge_hierarchical_answer,
    run_hierarchical_eval,
)


# ---------- CountingLLM ----------
def test_counting_llm_counts_and_forwards():
    seen: list[str] = []
    c = CountingLLM(lambda p: seen.append(p) or "resp")
    assert c("a") == "resp"
    assert c("b") == "resp"
    assert c.count == 2
    assert seen == ["a", "b"]


def test_counting_llm_reset():
    c = CountingLLM(lambda p: "x")
    c("a")
    c("b")
    c.reset()
    assert c.count == 0


def test_counting_llm_none_fn_returns_empty_and_still_counts():
    c = CountingLLM(None)
    assert c("a") == ""
    assert c.count == 1


# ---------- judge ----------
def _hier(conclusion="저PER 상위 종목은 A, B, C 입니다.", **kw):
    base = {"conclusion": conclusion, "routes": ["kr"], "domain_results": {}, "uncertain": False}
    base.update(kw)
    return base


def test_judge_match_true():
    judge = lambda p: '{"match": true, "reason": "핵심 종목 일치"}'  # noqa: E731
    out = judge_hierarchical_answer("q", [{"name": "A"}], _hier(), judge)
    assert out["applicable"] is True
    assert out["match"] is True


def test_judge_match_false():
    judge = lambda p: '{"match": false, "reason": "순위 불일치"}'  # noqa: E731
    out = judge_hierarchical_answer("q", [{"name": "A"}], _hier(), judge)
    assert out["applicable"] is True
    assert out["match"] is False


def test_judge_unavailable_when_no_llm():
    out = judge_hierarchical_answer("q", [{"name": "A"}], _hier(), None)
    assert out["applicable"] is False
    assert out["match"] is None


def test_judge_malformed_response_is_not_applicable():
    judge = lambda p: "그냥 잡담, JSON 아님"  # noqa: E731
    out = judge_hierarchical_answer("q", [{"name": "A"}], _hier(), judge)
    assert out["applicable"] is False
    assert out["match"] is None


def test_judge_prompt_includes_uncertain_marker():
    """불확실 응답이면 judge 프롬프트에 [불확실] 이 담겨 판정 근거로 전달된다."""
    captured: list[str] = []

    def judge(p):
        captured.append(p)
        return '{"match": false, "reason": "불확실"}'

    judge_hierarchical_answer("q", [{"name": "A"}], _hier(uncertain=True, reason="데이터 없음"), judge)
    assert "[불확실]" in captured[0]


# ---------- 오케스트레이션(격리 사본 + 카운팅 + 판정) ----------
def _seed(db_path: str) -> None:
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("INSERT INTO company(stock_code, name, market, sector) VALUES(?,?,?,?)",
                     ("000001", "가나전자", "KOSPI", "전기·전자"))
        conn.commit()
    finally:
        conn.close()


def test_run_hierarchical_eval_counts_calls_and_aggregates(tmp_path):
    db = str(tmp_path / "m.db")
    init_db(db)
    _seed(db)

    items = [
        {"id": 1, "question": "PER 낮은 회사", "sql": "SELECT name FROM company", "tags": "PER"},
        {"id": 2, "question": "PBR 낮은 회사", "sql": "SELECT name FROM company", "tags": "PBR"},
    ]

    def fake_hier(question, conn, llm_fn=None):
        # 신규 구조가 llm_fn(counter)을 3회 부른다고 가정(route/verify/synthesize).
        llm_fn("route?")
        llm_fn("verify?")
        llm_fn("synth?")
        return {"conclusion": "결론", "routes": ["kr"], "domain_results": {}, "uncertain": False}

    # 첫 문항은 match, 둘째 문항은 no-match 로 판정하는 mock judge.
    verdicts = iter(['{"match": true, "reason": "ok"}', '{"match": false, "reason": "no"}'])
    judge = lambda p: next(verdicts)  # noqa: E731

    rep = run_hierarchical_eval(
        items, fake_hier, llm_fn=lambda p: "x", judge_llm_fn=judge, db_path=db
    )

    assert rep["n"] == 2
    assert rep["judged"] == 2
    assert rep["match"] == 1
    assert rep["accuracy_pct"] == 50.0
    assert rep["avg_llm_calls"] == 3.0        # 문항마다 counter.reset() 후 3회
    assert rep["rows"][0]["llm_calls"] == 3
    assert rep["rows"][0]["match"] is True
    assert rep["rows"][1]["match"] is False
    assert rep["avg_latency_s"] is not None    # 응답시간 실측 필드 존재


def test_run_hierarchical_eval_absorbs_graph_exception(tmp_path):
    """신규 구조 실행이 예외를 던져도 배치가 중단되지 않고 해당 문항을 불확실로 기록한다."""
    db = str(tmp_path / "m.db")
    init_db(db)
    _seed(db)

    items = [{"id": 1, "question": "q", "sql": "SELECT name FROM company", "tags": "x"}]

    def boom(question, conn, llm_fn=None):
        raise RuntimeError("그래프 폭발")

    judge = lambda p: '{"match": false, "reason": "실행실패"}'  # noqa: E731
    rep = run_hierarchical_eval(items, boom, llm_fn=None, judge_llm_fn=judge, db_path=db)

    assert rep["n"] == 1
    assert rep["rows"][0]["uncertain"] is True
