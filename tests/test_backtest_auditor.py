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


def _seed_us_delisting_ep(conn, code, episodes):
    """us_delisting에 (listing_date, delisting_date) 구간들을 시드한다."""
    for listing, delisting in episodes:
        conn.execute(
            "INSERT INTO us_delisting(stock_code, company_name, exchange, listing_date, delisting_date) "
            "VALUES (?,?,?,?,?)", (code, code, "NYSE", listing, delisting))
    conn.commit()


def test_check_survivorship_us_blocks_when_holding_delisted_before_asof(tmp_path):
    # AC5: 미국도 KR과 동일하게 실제 하드차단한다(더 이상 "unverifiable"이 아님).
    conn = _seeded_conn(tmp_path)
    _seed_us_delisting_ep(conn, "DEAD", [("2013-01-01", "2020-01-01")])
    holdings = [{"date": "2021-06-30", "codes": ["AAPL", "DEAD"]}]
    v = auditor.check_survivorship(conn, holdings, market="US")
    assert v["sin"] == "survivorship"
    assert v["blocked"] is True
    assert v.get("unverifiable") in (None, False)  # 검증불가 개념 제거
    assert any(e["stock_code"] == "DEAD" for e in v["evidence"])


def test_check_survivorship_us_passes_when_all_alive(tmp_path):
    conn = _seeded_conn(tmp_path)
    holdings = [{"date": "2025-12-31", "codes": ["AAPL", "MSFT"]}]
    v = auditor.check_survivorship(conn, holdings, market="US")
    assert v["blocked"] is False
    assert v["evidence"] == []
    assert v.get("unverifiable") in (None, False)


def test_check_survivorship_us_passes_after_relisting(tmp_path):
    # AC15: 티커 재사용 — 재상장 구간 안(2024)의 보유는 앞선 상폐(2020)에도 불구하고 통과.
    conn = _seeded_conn(tmp_path)
    _seed_us_delisting_ep(conn, "TWTR", [("2013-01-01", "2020-01-01"), ("2023-01-01", "2025-01-01")])
    holdings = [{"date": "2024-01-01", "codes": ["TWTR"]}]
    v = auditor.check_survivorship(conn, holdings, market="US")
    assert v["blocked"] is False


def test_check_survivorship_us_passes_reused_active_single_episode(tmp_path):
    # critic 실데이터 모양(AC15): 옛 상폐 구간 1개 + 활성 마커인 재사용 종목 보유는 잘못 하드차단되지 않는다.
    conn = _seeded_conn(tmp_path)
    _seed_us_delisting_ep(conn, "TWTR", [("2013-11-07", "2022-10-27"), ("", "")])  # 2번째=활성 마커
    holdings = [{"date": "2024-01-01", "codes": ["TWTR"]}]
    v = auditor.check_survivorship(conn, holdings, market="US")
    assert v["blocked"] is False


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


def test_inspect_outlier_prompt_reflects_actual_available_methods():
    """이상치검사 프롬프트가 실제 코드 상태와 어긋나면 안 된다(son-checker형 회귀 방지).

    combine(method='rank_sum')과 winsorize 프리미티브가 이미 존재하므로, "z-score만
    지원하며 순위변환·IQR 윈저화는 없다"는 낡은 문구 대신 실제로 어떤 방식이 있고 무엇을
    확인해야 하는지가 프롬프트에 담겨야 한다.
    """
    seen = []
    auditor.inspect_outlier(_RESULT, lambda p: (seen.append(p) or "{}"))
    prompt = seen[0]
    assert "z-score 정규화만 지원하며 순위변환" not in prompt  # 더 이상 사실이 아닌 낡은 주장
    assert "rank_sum" in prompt
    assert "winsorize" in prompt


# --------------------------------------------------------------------------
# 실서버 재현 버그: "코스피 전종목 pbr/gpa 상관관계 5분위" 같은 순수 횡단면 통계 파이프라인
# (get_cross_section→correlation/quantile_bucket_means, run_backtest 없음)은 result에
# performance/holdings가 전혀 없어 4개 검사관이 아무 정보 없이 판단해야 했다. inspect_outlier
# 프롬프트의 "모르면 경고로 남겨라" 지시 때문에 사용자가 뭘 해도 이상치 경고가 100% 뜨는
# 구조적 결함이었다 — steps(실제 파이프라인)를 검사관에게 보여줘 해결한다.
# --------------------------------------------------------------------------
def test_inspect_outlier_prompt_includes_actual_steps_used():
    seen = []
    steps = [
        {"op": "get_cross_section", "params": {"asof": "2026-07-18"}, "out": "xs"},
        {"op": "winsorize", "params": {"rows": {"$ref": "xs"}, "field": "pbr"}, "out": "xs_w"},
    ]
    auditor.inspect_outlier(_RESULT, lambda p: (seen.append(p) or "{}"), steps=steps)
    prompt = seen[0]
    assert "get_cross_section" in prompt
    assert "winsorize" in prompt


def test_inspect_storytelling_prompt_flags_non_portfolio_pipeline_when_no_run_backtest():
    """run_backtest/run_qvm_backtest가 없는 파이프라인(순수 통계분석)이면 '기간 다양성'
    개념 자체가 적용 안 되니 반드시 triggered=false로 판단하라는 지시가 프롬프트에 있어야 한다."""
    seen = []
    steps = [
        {"op": "get_cross_section", "params": {"asof": "2026-07-18"}, "out": "xs"},
        {"op": "correlation", "params": {"rows": {"$ref": "xs"}, "field_x": "pbr", "field_y": "gp_a"}, "out": "corr"},
    ]
    auditor.inspect_storytelling(_RESULT, lambda p: (seen.append(p) or "{}"), steps=steps)
    prompt = seen[0]
    assert "correlation" in prompt
    assert "triggered=false" in prompt or "해당사항 없음" in prompt


def test_inspect_signal_decay_prompt_flags_non_portfolio_pipeline_when_no_run_backtest():
    seen = []
    steps = [{"op": "get_cross_section", "params": {"asof": "2026-07-18"}, "out": "xs"}]
    auditor.inspect_signal_decay(_RESULT, lambda p: (seen.append(p) or "{}"), steps=steps)
    prompt = seen[0]
    assert "get_cross_section" in prompt
    assert "triggered=false" in prompt or "해당사항 없음" in prompt


def test_inspect_snooping_prompt_includes_steps_for_context():
    seen = []
    steps = [{"op": "get_cross_section", "params": {"asof": "2026-07-18"}, "out": "xs"}]
    auditor.inspect_snooping(_RESULT, "질문", lambda p: (seen.append(p) or "{}"), steps=steps)
    assert "get_cross_section" in seen[0]


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


def test_run_soft_inspectors_passes_steps_to_all_four_inspectors():
    """AUD-6 배선 회귀: steps를 넘기면 4개 검사관 프롬프트 전부에 실제 파이프라인 정보가
    반영돼야 한다(순서 무관하게 동시 실행되므로 스레드-세이프하게 각자 prompt를 기록)."""
    import threading
    seen = []
    lock = threading.Lock()

    def recorder(p):
        with lock:
            seen.append(p)
        return "{}"

    steps = [{"op": "get_cross_section", "params": {"asof": "2026-07-18"}, "out": "xs"}]
    auditor.run_soft_inspectors(_RESULT, "질문", recorder, steps=steps)
    assert len(seen) == 4
    assert all("get_cross_section" in p for p in seen)


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


def test_post_audit_passes_steps_to_soft_inspectors(tmp_path):
    conn = _seeded_conn(tmp_path)
    result = {"r": 0.5, "n": 100}  # correlation 파이프라인 결과 — performance/holdings 없음
    steps = [
        {"op": "get_cross_section", "params": {"asof": "2026-07-18"}, "out": "xs"},
        {"op": "correlation", "params": {"rows": {"$ref": "xs"}, "field_x": "pbr", "field_y": "gp_a"}, "out": "corr"},
    ]
    seen = []
    audit = auditor.post_audit(
        result, conn, "질문", llm_fn=lambda p: (seen.append(p) or "{}"), steps=steps,
    )
    assert audit["blocked"] is False
    assert len(seen) == 4
    assert all("correlation" in p for p in seen)


def test_post_audit_us_hard_blocks_delisted_holding(tmp_path):
    # AC5: US 백테스트도 상장폐지 종목을 보유하면 KR과 동일하게 하드차단된다(LLM 미가용이어도).
    conn = _seeded_conn(tmp_path)
    _seed_us_delisting_ep(conn, "DEAD", [("2013-01-01", "2020-01-01")])
    result = {"performance": {"cagr": 10.0},
              "holdings": [{"date": "2021-06-30", "codes": ["DEAD"]}]}
    audit = auditor.post_audit(result, conn, "질문", llm_fn=None, market="US")
    assert audit["blocked"] is True
    surv = [v for v in audit["hard"] if v["sin"] == "survivorship"]
    assert surv and surv[0]["blocked"] is True


def test_post_audit_us_passes_cleanly_when_alive(tmp_path):
    # 상폐 이력이 없으면 US도 KR처럼 조용히 통과한다(더 이상 '검증불가' 경고를 강제로 붙이지 않음).
    conn = _seeded_conn(tmp_path)
    result = {"performance": {"cagr": 10.0},
              "holdings": [{"date": "2025-12-31", "codes": ["AAPL"]}]}
    audit = auditor.post_audit(result, conn, "질문", llm_fn=None, market="US")
    assert audit["blocked"] is False
    assert [v for v in audit["soft"] if v["sin"] == "survivorship"] == []
