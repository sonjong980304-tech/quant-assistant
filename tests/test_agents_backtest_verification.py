"""HA-5: 백테스트 검증 배선(src/agents/backtest_verification.py) 통합테스트.

.omc/state/sessions/f6b79d90-b7b2-4b4f-b4f7-405a8355724e/prd.json의 HA-5 참고.

auditor.py 자체의 판정 로직(하드차단 3종/소프트경고 4종)은 이 스토리에서 전혀 건드리지
않는다 — 새 계층형 아키텍처의 백테스트 도메인 에이전트가 쓸 오케스트레이션 함수
run_backtest_with_audit이 pre_audit → 실행 콜백 → post_audit 순서로 올바르게 배선하고,
하드차단 시 정상 결과를 폐기(AC11)하며 통과 시 run_soft_inspectors(소프트경고 4종)가
호출되어 결과에 첨부(AC12)되는지만 검증한다. 기존 tests/test_backtest_auditor_wiring.py
(레거시 src/graph/nodes.py 경로)는 이 스토리로 수정하지 않으며 그대로 통과해야 한다.
"""
from __future__ import annotations

import json
import sqlite3

from src.agents.backtest_verification import run_backtest_with_audit
from src.backtest import auditor


def _seeded_conn(tmp_path):
    from src.db import init_db

    db = tmp_path / "verify.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


_BT_RESULT = {
    "dates": ["2025-09-30", "2025-12-31"],
    "navs": [1.0, 1.1],
    "benchmark": None,
    "performance": {"cagr": 5.0},
    "holdings": [{"date": "2025-12-31", "codes": ["000001"]}],
}


def _json_llm(triggered=True, message="경고"):
    return lambda prompt: json.dumps({"triggered": triggered, "message": message})


# --------------------------------------------------------------------------
# AC11: 하드차단 3종(check_survivorship/check_lookahead/check_short_positions)이
# 결정론적으로 호출된다(auditor.py 원본 로직 그대로, monkeypatch로 호출 여부만 감시).
# --------------------------------------------------------------------------
def test_hard_checks_all_three_called(tmp_path, monkeypatch):
    conn = _seeded_conn(tmp_path)
    calls = {"survivorship": 0, "lookahead": 0, "short_positions": 0}

    orig_survivorship = auditor.check_survivorship
    orig_lookahead = auditor.check_lookahead
    orig_short = auditor.check_short_positions

    def spy_survivorship(*a, **k):
        calls["survivorship"] += 1
        return orig_survivorship(*a, **k)

    def spy_lookahead(*a, **k):
        calls["lookahead"] += 1
        return orig_lookahead(*a, **k)

    def spy_short(*a, **k):
        calls["short_positions"] += 1
        return orig_short(*a, **k)

    monkeypatch.setattr(auditor, "check_survivorship", spy_survivorship)
    monkeypatch.setattr(auditor, "check_lookahead", spy_lookahead)
    monkeypatch.setattr(auditor, "check_short_positions", spy_short)

    steps = [{"op": "run_backtest",
              "params": {"weights": {"000001": 0.6, "000002": 0.4}}, "out": "bt"}]
    run_pipeline_fn = lambda s, conn=None: dict(_BT_RESULT)
    out = run_backtest_with_audit(steps, conn, "질문", run_pipeline_fn, llm_fn=None, market="KR")

    assert calls == {"survivorship": 1, "lookahead": 1, "short_positions": 1}
    assert out["blocked"] is False


# --------------------------------------------------------------------------
# AC12: 하드차단 통과 시 run_soft_inspectors(소프트경고 4종)가 호출되고 결과에 첨부된다.
# --------------------------------------------------------------------------
def test_soft_inspectors_called_when_hard_checks_pass(tmp_path, monkeypatch):
    conn = _seeded_conn(tmp_path)
    calls = {"n": 0}
    orig_run_soft = auditor.run_soft_inspectors

    def spy_run_soft(*a, **k):
        calls["n"] += 1
        return orig_run_soft(*a, **k)

    monkeypatch.setattr(auditor, "run_soft_inspectors", spy_run_soft)

    steps = [{"op": "run_backtest", "params": {}, "out": "bt"}]
    run_pipeline_fn = lambda s, conn=None: dict(_BT_RESULT)
    out = run_backtest_with_audit(steps, conn, "질문", run_pipeline_fn,
                                  llm_fn=_json_llm(True, "경고"), market="KR")

    assert calls["n"] == 1


def test_soft_inspectors_receive_actual_pipeline_steps(tmp_path):
    """실서버 재현: correlation/quantile_bucket_means 같은 순수 통계 파이프라인은
    result에 performance/holdings가 없어 소프트경고 검사관이 아무 정보 없이 판단해야
    했다(항상 경고). run_backtest_with_audit이 steps를 post_audit까지 실제로 전달해
    검사관이 실제 파이프라인을 보고 판단하는지 확인한다."""
    conn = _seeded_conn(tmp_path)
    steps = [
        {"op": "get_cross_section", "params": {"asof": "2026-07-18"}, "out": "xs"},
        {"op": "correlation", "params": {"rows": {"$ref": "xs"}, "field_x": "pbr", "field_y": "gp_a"}, "out": "corr"},
    ]
    seen = []
    run_pipeline_fn = lambda s, conn=None: {"r": 0.5, "n": 100}
    out = run_backtest_with_audit(
        steps, conn, "질문", run_pipeline_fn,
        llm_fn=lambda p: (seen.append(p) or "{}"), market="KR",
    )
    assert out["blocked"] is False
    assert len(seen) == 4
    assert all("correlation" in p for p in seen)


# --------------------------------------------------------------------------
# 사전 하드차단(공매도) → run_pipeline_fn 자체가 호출되지 않고 정상 결과 없이 에러만 반환
# --------------------------------------------------------------------------
def test_pre_hard_block_skips_execution_and_returns_error(tmp_path):
    conn = _seeded_conn(tmp_path)
    ran = {"pipeline": False}

    def run_pipeline_fn(s, conn=None):
        ran["pipeline"] = True
        return dict(_BT_RESULT)

    steps = [{"op": "run_backtest",
              "params": {"weights": {"000001": 0.6, "000002": -0.1}}, "out": "bt"}]
    out = run_backtest_with_audit(steps, conn, "질문", run_pipeline_fn, llm_fn=None, market="KR")

    assert out["blocked"] is True
    assert "short_positions" in out["error"]
    assert out["result"] is None
    assert ran["pipeline"] is False


# --------------------------------------------------------------------------
# 사후 하드차단(생존편향) → 파이프라인은 실행됐지만 결과는 폐기되고 에러만 반환
# --------------------------------------------------------------------------
def test_post_hard_block_discards_result_but_pipeline_ran(tmp_path):
    conn = _seeded_conn(tmp_path)
    conn.execute("INSERT INTO delisting(stock_code, name, delisting_date) VALUES (?,?,?)",
                 ("000001", "죽은회사", "2020-01-01"))
    conn.commit()
    ran = {"pipeline": False}

    def run_pipeline_fn(s, conn=None):
        ran["pipeline"] = True
        return dict(_BT_RESULT)

    steps = [{"op": "run_backtest", "params": {}, "out": "bt"}]
    out = run_backtest_with_audit(steps, conn, "질문", run_pipeline_fn,
                                  llm_fn=_json_llm(True, "x"), market="KR")

    assert ran["pipeline"] is True
    assert out["blocked"] is True
    assert "survivorship" in out["error"]
    assert out["result"] is None
    assert out["warnings"] == []


# --------------------------------------------------------------------------
# 소프트경고: triggered된 것만 첨부되고 정상 결과는 그대로 유지된다
# --------------------------------------------------------------------------
def test_soft_warnings_only_triggered_attached_result_preserved(tmp_path):
    conn = _seeded_conn(tmp_path)

    def llm_fn(prompt):
        if "데이터마이닝" in prompt or "스누핑" in prompt:
            return json.dumps({"triggered": True, "message": "사후정당화 의심"})
        return json.dumps({"triggered": False, "message": ""})

    steps = [{"op": "run_backtest", "params": {}, "out": "bt"}]
    run_pipeline_fn = lambda s, conn=None: dict(_BT_RESULT)
    out = run_backtest_with_audit(steps, conn, "질문", run_pipeline_fn, llm_fn=llm_fn, market="KR")

    assert out["blocked"] is False
    assert out["result"]["performance"] == {"cagr": 5.0}
    assert len(out["warnings"]) == 1
    assert out["warnings"][0]["sin"] == "snooping"


def test_llm_unavailable_skips_soft_but_keeps_result(tmp_path):
    conn = _seeded_conn(tmp_path)
    steps = [{"op": "run_backtest", "params": {}, "out": "bt"}]
    run_pipeline_fn = lambda s, conn=None: dict(_BT_RESULT)
    out = run_backtest_with_audit(steps, conn, "질문", run_pipeline_fn, llm_fn=None, market="KR")

    assert out["blocked"] is False
    assert out["result"] is not None
    assert out["warnings"] == []


# --------------------------------------------------------------------------
# US 시장: 생존편향 '검증불가' 경고가 소프트경고로 노출된다(하드차단도 통과도 아님)
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# on_progress — 검사 에이전트(하드3종+소프트4종) 각각의 실행/결과를 실시간 통지한다.
# auditor.py 자체는 건드리지 않는다 — 이미 반환된 verdict 리스트를 순회해 이벤트만 낸다.
# --------------------------------------------------------------------------
def test_on_progress_reports_each_hard_and_soft_check_when_passed(tmp_path):
    """weights를 포함해 사전(공매도) 검사도 실제로 의미 있게 수행되는 fixture로 검증한다
    (weights가 없으면 pre_audit이 비중을 해석 못해 사전검사 자체를 스킵한다 — 정상 동작)."""
    conn = _seeded_conn(tmp_path)
    events: list[tuple[str, str]] = []

    steps = [{"op": "run_backtest",
              "params": {"weights": {"000001": 0.6, "000002": 0.4}}, "out": "bt"}]
    run_pipeline_fn = lambda s, conn=None: dict(_BT_RESULT)
    run_backtest_with_audit(
        steps, conn, "질문", run_pipeline_fn, llm_fn=_json_llm(True, "경고"), market="KR",
        on_progress=lambda step, summary: events.append((step, summary)),
    )

    summaries = [s for _, s in events]
    # 사전(공매도) + 사후 하드 2종 + 소프트 4종 = 최소 7건
    assert len(events) >= 7
    assert any("공매도" in s for s in summaries)
    assert any("생존편향" in s and "통과" in s for s in summaries)
    assert any("미래참조" in s and "통과" in s for s in summaries)
    assert sum("스토리텔링" in s for s in summaries) == 1
    assert sum("데이터스누핑" in s for s in summaries) == 1
    assert sum("신호감소" in s for s in summaries) == 1
    assert sum("이상치" in s for s in summaries) == 1
    # 소프트경고 4종 모두 이번 fixture(triggered=True)에서는 경고 문구를 담아야 한다
    assert sum("경고" in s for s in summaries) == 4


def test_on_progress_reports_pre_block_and_skips_post_checks(tmp_path):
    """사전 하드차단(공매도) 시 실행 자체가 안 되므로 사후검사 이벤트는 나오지 않는다."""
    conn = _seeded_conn(tmp_path)
    events: list[tuple[str, str]] = []

    steps = [{"op": "run_backtest",
              "params": {"weights": {"000001": 0.6, "000002": -0.1}}, "out": "bt"}]
    run_pipeline_fn = lambda s, conn=None: dict(_BT_RESULT)
    run_backtest_with_audit(
        steps, conn, "질문", run_pipeline_fn, llm_fn=None, market="KR",
        on_progress=lambda step, summary: events.append((step, summary)),
    )

    summaries = [s for _, s in events]
    assert any("공매도" in s and ("차단" in s) for s in summaries)
    assert not any("생존편향" in s for s in summaries)
    assert not any("스토리텔링" in s for s in summaries)


def test_on_progress_reports_post_hard_block_and_skips_soft(tmp_path):
    """사후 하드차단(생존편향) 시 소프트경고 검사는 생략되므로 그 이벤트도 나오지 않는다."""
    conn = _seeded_conn(tmp_path)
    conn.execute("INSERT INTO delisting(stock_code, name, delisting_date) VALUES (?,?,?)",
                 ("000001", "죽은회사", "2020-01-01"))
    conn.commit()
    events: list[tuple[str, str]] = []

    steps = [{"op": "run_backtest", "params": {}, "out": "bt"}]
    run_pipeline_fn = lambda s, conn=None: dict(_BT_RESULT)
    run_backtest_with_audit(
        steps, conn, "질문", run_pipeline_fn, llm_fn=_json_llm(True, "x"), market="KR",
        on_progress=lambda step, summary: events.append((step, summary)),
    )

    summaries = [s for _, s in events]
    assert any("생존편향" in s and "차단" in s for s in summaries)
    assert not any("스토리텔링" in s for s in summaries)


def test_without_on_progress_is_unaffected(tmp_path):
    """on_progress 생략 시(기본값 None) 기존과 완전히 동일하게 동작 — 회귀 방지."""
    conn = _seeded_conn(tmp_path)
    steps = [{"op": "run_backtest", "params": {}, "out": "bt"}]
    run_pipeline_fn = lambda s, conn=None: dict(_BT_RESULT)
    out = run_backtest_with_audit(steps, conn, "질문", run_pipeline_fn,
                                  llm_fn=_json_llm(True, "경고"), market="KR")
    assert out["blocked"] is False
    assert len(out["warnings"]) == 4


# --------------------------------------------------------------------------
# fail-closed 안전원칙(SoT §3.4/§3.5, AC9/AC11): 하드차단 3종의 검사기 "자체가"
# 내부 예외로 죽어도, '차단 안 됨=통과'로 새지 않고 반드시 안전측(결과 폐기·차단)으로
# 귀결돼야 한다. 안전장치가 고장 났을 때 검증 안 된 결과를 통과시키면 하드차단의 존재
# 목적(방어적 안전망)과 정면 충돌한다.
# --------------------------------------------------------------------------
def _boom(*a, **k):
    raise RuntimeError("검사기 자체가 폭발(회귀/DB오류 시뮬)")


def test_pre_hard_check_exception_fails_closed(tmp_path, monkeypatch):
    """사전 하드검사(공매도)가 내부 예외로 죽으면 → 실행 자체를 막고 결과 폐기(fail-closed)."""
    conn = _seeded_conn(tmp_path)
    ran = {"pipeline": False}
    monkeypatch.setattr(auditor, "check_short_positions", _boom)

    def run_pipeline_fn(s, conn=None):
        ran["pipeline"] = True
        return dict(_BT_RESULT)

    steps = [{"op": "run_backtest",
              "params": {"weights": {"000001": 0.6, "000002": 0.4}}, "out": "bt"}]
    out = run_backtest_with_audit(steps, conn, "질문", run_pipeline_fn, llm_fn=None, market="KR")

    assert out["blocked"] is True
    assert out["result"] is None
    assert ran["pipeline"] is False  # 사전 차단이므로 파이프라인 실행 자체가 안 됨


def test_post_survivorship_check_exception_fails_closed(tmp_path, monkeypatch):
    """사후 하드검사(생존편향)가 내부 예외로 죽으면 → 정상 결과를 폐기하고 차단(fail-closed)."""
    conn = _seeded_conn(tmp_path)
    ran = {"pipeline": False}
    monkeypatch.setattr(auditor, "check_survivorship", _boom)

    def run_pipeline_fn(s, conn=None):
        ran["pipeline"] = True
        return dict(_BT_RESULT)

    steps = [{"op": "run_backtest", "params": {}, "out": "bt"}]
    out = run_backtest_with_audit(steps, conn, "질문", run_pipeline_fn,
                                  llm_fn=_json_llm(True, "x"), market="KR")

    assert ran["pipeline"] is True   # 실행은 됐지만
    assert out["blocked"] is True    # 검증 실패 → 결과 폐기
    assert out["result"] is None
    assert out["warnings"] == []


def test_post_lookahead_check_exception_fails_closed(tmp_path, monkeypatch):
    """사후 하드검사(미래참조)가 내부 예외로 죽어도 → 결과 폐기하고 차단(fail-closed)."""
    conn = _seeded_conn(tmp_path)
    monkeypatch.setattr(auditor, "check_lookahead", _boom)

    steps = [{"op": "run_backtest", "params": {}, "out": "bt"}]
    run_pipeline_fn = lambda s, conn=None: dict(_BT_RESULT)
    out = run_backtest_with_audit(steps, conn, "질문", run_pipeline_fn, llm_fn=None, market="KR")

    assert out["blocked"] is True
    assert out["result"] is None
    assert out["warnings"] == []
