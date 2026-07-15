"""백테스트 7대 죄악 자동 감사 레이어 단위 테스트 (TDD).

.omc/specs/brainstorming-backtest-auditor-7-sins.md 참고.
- 하드차단 3종(생존편향/미래참조편향/공매도비용): LLM 없이 결정론적 코드 검사.
- 소프트경고 4종(스토리텔링/스누핑/신호감소/이상치): LLM 판정(DI로 mock).

DB 접근이 필요한 하드검사는 임시 SQLite에 시딩해 사용자 DB와 완전 격리한다.
LLM 호출은 이 프로젝트 기존 DI 관례대로 주입 가능한 llm_fn으로 분리해 네트워크 없이 검증한다.
"""
from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.backtest import auditor


# --------------------------------------------------------------------------
# 공용 헬퍼: 임시 시딩 DB
# --------------------------------------------------------------------------
def _seeded_conn(tmp_path):
    from src.db import init_db

    db = tmp_path / "auditor.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


# --------------------------------------------------------------------------
# AUD-1: 생존편향 하드차단 (check_survivorship)
# --------------------------------------------------------------------------
def test_check_survivorship_blocks_when_holding_delisted_before_asof(tmp_path):
    conn = _seeded_conn(tmp_path)
    # 000002는 2025-06-30에 상폐 → asof 2025-12-31 시점엔 이미 죽은 종목
    conn.execute("INSERT INTO delisting(stock_code, name, delisting_date) VALUES (?,?,?)",
                 ("000002", "죽은회사", "2025-06-30"))
    conn.commit()
    holdings = [{"date": "2025-12-31", "codes": ["000001", "000002"]}]
    v = auditor.check_survivorship(conn, holdings)
    assert v["blocked"] is True
    assert v["sin"] == "survivorship"
    assert any(e["stock_code"] == "000002" for e in v["evidence"])
    assert v["reason"]  # 사유 텍스트가 존재


def test_check_survivorship_us_market_is_unverifiable_not_pass_or_block(tmp_path):
    # 미국은 상장폐지 추적 데이터가 없어 생존편향 검증이 원천적으로 불가능하다.
    # → 거짓 "통과"도 아니고 하드차단도 아닌, 별도 "검증불가" 상태를 반환해야 한다.
    conn = _seeded_conn(tmp_path)
    holdings = [{"date": "2025-12-31", "codes": ["AAPL", "MSFT"]}]
    v = auditor.check_survivorship(conn, holdings, market="US")
    assert v["sin"] == "survivorship"
    assert v["blocked"] is False          # 하드차단 아님
    assert v.get("unverifiable") is True  # 통과도 아님 — 검증불가로 명확히 구분
    assert v["reason"]                    # 사유 텍스트 존재


def test_check_survivorship_kr_market_stays_bool_semantics(tmp_path):
    # KR(기본)은 기존 bool 판정 그대로 — 검증불가 플래그가 붙지 않는다(회귀 방지).
    conn = _seeded_conn(tmp_path)
    holdings = [{"date": "2025-12-31", "codes": ["000001"]}]
    v = auditor.check_survivorship(conn, holdings)
    assert v["blocked"] is False
    assert v.get("unverifiable") in (None, False)


def test_check_survivorship_passes_when_all_alive(tmp_path):
    conn = _seeded_conn(tmp_path)
    conn.execute("INSERT INTO delisting(stock_code, name, delisting_date) VALUES (?,?,?)",
                 ("000002", "나중상폐", "2026-06-30"))  # asof보다 미래 상폐 → 그 시점엔 살아있음
    conn.commit()
    holdings = [{"date": "2025-12-31", "codes": ["000001", "000002"]}]
    v = auditor.check_survivorship(conn, holdings)
    assert v["blocked"] is False
    assert v["evidence"] == []


# --------------------------------------------------------------------------
# AUD-2: 미래참조편향 하드차단 (check_lookahead)
# --------------------------------------------------------------------------
def test_check_lookahead_blocks_when_disclosed_after_asof(tmp_path):
    conn = _seeded_conn(tmp_path)
    # 미래 공시일 재무 row: 2025Q4는 2026-02-15에 공시됐는데 asof는 2025-12-31
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
        "VALUES (?,?,?,?,?)",
        ("000001", "2025Q4", "2026-02-15", "revenue", 1000.0),
    )
    conn.commit()
    holdings = [{"date": "2025-12-31", "codes": ["000001"]}]
    # 가드가 깨진 상황을 모사: quarter_fn이 disclosed_date를 무시하고 미래 분기를 고른다
    v = auditor.check_lookahead(conn, holdings, quarter_fn=lambda c, code, asof: "2025Q4")
    assert v["blocked"] is True
    assert v["sin"] == "lookahead"
    assert any(e["stock_code"] == "000001" for e in v["evidence"])


def test_check_lookahead_passes_with_real_effective_quarter(tmp_path):
    conn = _seeded_conn(tmp_path)
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
        "VALUES (?,?,?,?,?)",
        ("000001", "2025Q3", "2025-11-15", "revenue", 1000.0),
    )
    conn.commit()
    holdings = [{"date": "2025-12-31", "codes": ["000001"]}]
    # 기본(effective_quarter_at)은 disclosed_date<=asof만 고르므로 위반이 없어야 한다
    v = auditor.check_lookahead(conn, holdings)
    assert v["blocked"] is False
    assert v["evidence"] == []


# --------------------------------------------------------------------------
# AUD-3: 공매도비용 하드차단 (check_short_positions)
# --------------------------------------------------------------------------
def test_check_short_positions_blocks_on_negative_weight():
    v = auditor.check_short_positions({"000001": 0.6, "000002": -0.1, "000003": 0.5})
    assert v["blocked"] is True
    assert v["sin"] == "short_positions"
    negs = {e["asset"]: e["weight"] for e in v["evidence"]}
    assert negs == {"000002": -0.1}


def test_check_short_positions_passes_all_nonneg():
    v = auditor.check_short_positions({"000001": 0.6, "000002": 0.4})
    assert v["blocked"] is False
    assert v["evidence"] == []


# --------------------------------------------------------------------------
# AUD-5: 소프트경고 검사관 4종
# --------------------------------------------------------------------------
_RESULT = {"performance": {"cagr": 30.0, "mdd": -5.0, "sharpe": 2.5},
           "holdings": [{"date": "2020-03-31", "codes": ["000001"]}]}


def _json_llm(triggered=True, message="위험"):
    import json
    return lambda prompt: json.dumps({"triggered": triggered, "message": message})


def test_each_inspector_returns_consistent_structure():
    for fn, name, extra in [
        (auditor.inspect_storytelling, "storytelling", ()),
        (auditor.inspect_signal_decay, "signal_decay", ()),
        (auditor.inspect_outlier, "outlier", ()),
    ]:
        v = fn(_RESULT, _json_llm(True, "경고문"))
        assert v["sin"] == name
        assert v["triggered"] is True
        assert v["message"] == "경고문"
    v = auditor.inspect_snooping(_RESULT, "질문", _json_llm(False, ""))
    assert v["sin"] == "snooping"
    assert v["triggered"] is False


def test_inspect_snooping_prompt_includes_question():
    seen = []
    auditor.inspect_snooping(_RESULT, "PER 낮은 종목 3년 백테스트", lambda p: (seen.append(p) or "{}"))
    assert len(seen) == 1
    assert "PER 낮은 종목 3년 백테스트" in seen[0]  # 사용자 원본 질문이 프롬프트에 포함(AC11)


def test_inspectors_use_distinct_prompts():
    seen = []
    recorder = lambda p: (seen.append(p) or "{}")
    auditor.inspect_storytelling(_RESULT, recorder)
    auditor.inspect_snooping(_RESULT, "질문", recorder)
    auditor.inspect_signal_decay(_RESULT, recorder)
    auditor.inspect_outlier(_RESULT, recorder)
    assert len(seen) == 4
    assert len(set(seen)) == 4  # 4개 검사관이 서로 다른 프롬프트를 쓴다


# --------------------------------------------------------------------------
# AUD-6: ThreadPoolExecutor 병렬 오케스트레이터 (run_soft_inspectors)
# --------------------------------------------------------------------------
class _SpyPool:
    def __init__(self, inner):
        self._inner = inner
        self.submit_calls = 0

    def submit(self, *a, **k):
        self.submit_calls += 1
        return self._inner.submit(*a, **k)

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *a):
        return self._inner.__exit__(*a)


def test_run_soft_inspectors_submits_four_tasks():
    made = {}

    def factory():
        pool = _SpyPool(ThreadPoolExecutor(max_workers=4))
        made["pool"] = pool
        return pool

    out = auditor.run_soft_inspectors(_RESULT, "질문", _json_llm(True, "m"), pool_factory=factory)
    assert made["pool"].submit_calls == 4  # 4개 검사관이 각각 별도로 동시 제출됨(AC8)
    assert len(out) == 4


def test_run_soft_inspectors_reuses_pipeline_max_timeout():
    import inspect

    from src.backtest.pipeline_exec import MAX_TIMEOUT

    # 새 상수를 만들지 말고 기존 MAX_TIMEOUT을 재사용한다(AC8/스펙 §3.3)
    sig = inspect.signature(auditor.run_soft_inspectors)
    assert sig.parameters["timeout_s"].default == MAX_TIMEOUT


def test_run_soft_inspectors_returns_all_sins():
    out = auditor.run_soft_inspectors(_RESULT, "질문", _json_llm(True, "m"))
    assert {v["sin"] for v in out} == {"storytelling", "snooping", "signal_decay", "outlier"}


# --------------------------------------------------------------------------
# search_strategy(역백테스트) 전용 감사 — architect 검토(MAJOR) 반영:
# search_strategy는 op=="run_backtest"가 아니라 post_audit의 자동 발동 대상에서 빠지므로,
# 후보 20개 전체를 감사하면 LLM 호출이 최대 80회로 치솟는 비용 문제 없이, 가장 위험한
# 편향(스누핑 — 사후적으로 잘 맞는 조합을 고르는 행위 자체가 가장 취약)만 최상위 결과
# 기준으로 1회 판단한다.
# --------------------------------------------------------------------------
_SEARCH_RESULTS = [
    {"criteria": [{"key": "per", "direction": "low", "weight": 1.0}],
     "performance": {"sharpe": 2.0, "mdd": -5.0}, "holdings": [{"date": "2026-03-31", "codes": ["000001"]}]},
    {"criteria": [{"key": "roe", "direction": "high", "weight": 1.0}],
     "performance": {"sharpe": 1.0, "mdd": -8.0}, "holdings": [{"date": "2026-03-31", "codes": ["000002"]}]},
]


def test_audit_search_strategy_result_calls_snooping_inspector_exactly_once():
    calls = []
    llm_fn = lambda p: (calls.append(p) or _json_llm(False, "")(p))
    auditor.audit_search_strategy_result(_SEARCH_RESULTS, "질문", llm_fn)
    assert len(calls) == 1  # 후보 2개인데도 LLM 호출은 1회만(비용 억제)


def test_audit_search_strategy_result_uses_top_candidate_performance():
    seen = []
    llm_fn = lambda p: (seen.append(p) or _json_llm(False, "")(p))
    auditor.audit_search_strategy_result(_SEARCH_RESULTS, "질문", llm_fn)
    assert "2.0" in seen[0]  # 최상위(0번째) 후보의 sharpe가 프롬프트에 포함


def test_audit_search_strategy_result_empty_when_no_results():
    assert auditor.audit_search_strategy_result([], "질문", _json_llm(True, "m")) == []


def test_audit_search_strategy_result_empty_when_llm_unavailable():
    assert auditor.audit_search_strategy_result(_SEARCH_RESULTS, "질문", None) == []


def test_audit_search_strategy_result_prefixes_candidate_count_when_triggered():
    out = auditor.audit_search_strategy_result(_SEARCH_RESULTS, "질문", _json_llm(True, "사후정당화 의심"))
    assert len(out) == 1
    assert out[0]["sin"] == "snooping"
    assert out[0]["triggered"] is True
    assert "2개 후보" in out[0]["message"]
    assert "사후정당화 의심" in out[0]["message"]


# --------------------------------------------------------------------------
# 사전검사(pre_audit): 파이프라인의 optimize_weights 결과 비중을 해석해 음수 차단
# --------------------------------------------------------------------------
def test_resolve_backtest_weights_from_ref():
    steps = [
        {"op": "optimize_weights", "params": {"returns": {}}, "out": "w"},
        {"op": "run_backtest", "params": {"weights": {"$ref": "w"}}},
    ]
    calls = []

    def fake_run_pipeline(s, conn=None):
        calls.append(s)
        return {"000001": 0.6, "000002": -0.1}  # optimize_weights 결과 모사

    w = auditor._resolve_backtest_weights(steps, conn=None, run_pipeline_fn=fake_run_pipeline)
    assert w == {"000001": 0.6, "000002": -0.1}
    # optimize_weights(out="w")까지의 접두 파이프라인만 실행했는지
    assert calls == [steps[:1]]


def test_pre_audit_blocks_on_negative_optimized_weights():
    steps = [
        {"op": "optimize_weights", "params": {"returns": {}}, "out": "w"},
        {"op": "run_backtest", "params": {"weights": {"$ref": "w"}}},
    ]
    v = auditor.pre_audit(steps, conn=None,
                          run_pipeline_fn=lambda s, conn=None: {"000001": 0.7, "000002": -0.2})
    assert v is not None and v["blocked"] is True and v["sin"] == "short_positions"


def test_pre_audit_none_when_no_weights():
    steps = [{"op": "run_backtest", "params": {"criteria": []}}]
    v = auditor.pre_audit(steps, conn=None, run_pipeline_fn=lambda *a, **k: None)
    assert v is None


# --------------------------------------------------------------------------
# post_audit 종합
# --------------------------------------------------------------------------
def test_post_audit_hard_block_skips_soft(tmp_path):
    conn = _seeded_conn(tmp_path)
    conn.execute("INSERT INTO delisting(stock_code, name, delisting_date) VALUES (?,?,?)",
                 ("000002", "죽은회사", "2019-06-30"))
    conn.commit()
    result = {"performance": {"cagr": 99.0},
              "holdings": [{"date": "2020-03-31", "codes": ["000002"]}]}
    llm_calls = []
    audit = auditor.post_audit(result, conn, "질문",
                               llm_fn=lambda p: (llm_calls.append(p) or "{}"))
    assert audit["blocked"] is True
    assert audit["soft"] == []       # 하드차단 시 소프트검사는 생략
    assert llm_calls == []           # LLM 호출도 없음


def test_post_audit_attaches_soft_when_not_blocked(tmp_path):
    conn = _seeded_conn(tmp_path)
    result = {"performance": {"cagr": 10.0},
              "holdings": [{"date": "2025-12-31", "codes": ["000001"]}]}
    audit = auditor.post_audit(result, conn, "질문", llm_fn=_json_llm(True, "경고"))
    assert audit["blocked"] is False
    assert len(audit["soft"]) == 4
    assert all(v["triggered"] for v in audit["soft"])


def test_post_audit_us_attaches_survivorship_unverifiable_warning(tmp_path):
    # US 백테스트는 LLM 미가용(소프트 검사 스킵)이어도 생존편향 '검증불가' 경고가 붙어야 한다.
    conn = _seeded_conn(tmp_path)
    result = {"performance": {"cagr": 10.0},
              "holdings": [{"date": "2025-12-31", "codes": ["AAPL"]}]}
    audit = auditor.post_audit(result, conn, "질문", llm_fn=None, market="US")
    assert audit["blocked"] is False
    surv = [v for v in audit["soft"] if v["sin"] == "survivorship"]
    assert len(surv) == 1
    assert surv[0]["triggered"] is True
    assert "검증불가" in surv[0]["message"] or "상장폐지" in surv[0]["message"]
