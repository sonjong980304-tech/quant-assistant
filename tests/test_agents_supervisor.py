"""총괄 에이전트(src/agents/supervisor.py) 단위 테스트 (TDD, HA-10).

계층형 멀티에이전트의 최상위 "총괄 로직"을 순수 함수로 검증한다. LangGraph
StateGraph 배선은 다음 스토리(HA-11)의 몫이므로 여기서는 배선 없이 4개 함수의
입출력 계약만 검증한다:

- route_question(question, llm_fn) -> list[str]:
  질문을 보고 도메인 리스트(kr/us/macro/backtest, 복수 가능)를 반환 — AC1.
- dispatch_domains(routes, question, conn, llm_fn, steps=None) -> dict:
  각 도메인 answer_*_question을 호출해 원본 결과를 가공 없이 도메인별 키로 보존.
- verify_answer(question, domain_results, llm_fn) -> dict:
  도메인 결과가 원 질문과 부합하는지 판정({"valid", "reason"}) — AC2.
- answer_with_verification(...) -> dict:
  route→dispatch→verify 순서, 검증 실패 시 정확히 3회까지만 재시도(무한루프 없음),
  실패 시 uncertain=True, 성공 시 종합결론 + 원본 domain_results 병기 — AC3/AC4.
"""
from __future__ import annotations

from src.agents.supervisor import (
    answer_with_verification,
    dispatch_domains,
    route_question,
    synthesize_conclusion,
    verify_answer,
)


# ── AC1: route_question — 도메인 라우팅(단수/복수, mock LLM 분기) ─────────────

def test_route_question_single_domain_kr():
    """"삼성전자 PER" → mock LLM이 'kr'만 응답 → ["kr"]만 반환."""
    def fake_llm(prompt: str) -> str:
        return "kr"

    assert route_question("삼성전자 PER", fake_llm) == ["kr"]


def test_route_question_multi_domain_kr_and_us():
    """"삼성전자 vs 애플 비교" → mock LLM이 'kr, us' 응답 → 둘 다 반환."""
    def fake_llm(prompt: str) -> str:
        return "kr, us"

    assert route_question("삼성전자 vs 애플 비교", fake_llm) == ["kr", "us"]


def test_route_question_normalizes_order_and_dedupes():
    """LLM이 순서를 뒤섞거나 중복을 내도 정규 순서(kr,us,macro,backtest)로 정리."""
    def fake_llm(prompt: str) -> str:
        return "us kr us macro"

    assert route_question("복합 질문", fake_llm) == ["kr", "us", "macro"]


def test_route_question_parses_json_list_response():
    """LLM이 JSON 리스트로 응답해도 도메인 토큰을 정확히 추출한다."""
    def fake_llm(prompt: str) -> str:
        return '["backtest"]'

    assert route_question("전략 백테스트", fake_llm) == ["backtest"]


def test_route_question_fallback_without_llm_detects_backtest():
    routes = route_question("이 전략 백테스트 돌려줘", None)
    assert "backtest" in routes


def test_route_question_fallback_without_llm_detects_macro():
    routes = route_question("지금 매크로 신호 어때?", None)
    assert "macro" in routes


def test_route_question_fallback_defaults_to_kr():
    """휴리스틱으로도 판단 불가한 한국주식형 질문은 안전하게 ['kr']로 폴백."""
    assert route_question("삼성전자 PER 알려줘", None) == ["kr"]


# ── dispatch_domains — 각 도메인 원본 결과를 가공 없이 보존 ────────────────────

def test_dispatch_domains_preserves_raw_results(monkeypatch):
    import src.agents.supervisor as sup

    monkeypatch.setattr(
        sup, "answer_kr_question",
        lambda question, conn, llm_fn=None: {"stock_code": "005930", "raw": "KR-DATA"},
    )
    monkeypatch.setattr(
        sup, "answer_us_question",
        lambda question, conn, llm_fn=None: {"stock_code": "AAPL", "raw": "US-DATA"},
    )

    out = dispatch_domains(["kr", "us"], "삼성전자 vs 애플", conn=None, llm_fn=None)

    assert out["kr"] == {"stock_code": "005930", "raw": "KR-DATA"}
    assert out["us"] == {"stock_code": "AAPL", "raw": "US-DATA"}
    # 라우팅되지 않은 도메인은 키가 없어야 한다(가공 없이 요청한 것만).
    assert "macro" not in out
    assert "backtest" not in out


def test_dispatch_domains_routes_macro_and_backtest(monkeypatch):
    import src.agents.supervisor as sup

    calls: list[str] = []
    monkeypatch.setattr(
        sup, "answer_macro_question",
        lambda question, conn, **k: calls.append("macro") or {"overall": "RED"},
    )
    monkeypatch.setattr(
        sup, "answer_backtest_question",
        lambda question, steps, conn, **k: calls.append("backtest")
        or {"blocked": False, "result": {"cagr": 0.1}},
    )

    out = dispatch_domains(
        ["macro", "backtest"], "매크로 신호로 백테스트", conn=None, llm_fn=None, steps=[{"op": "noop"}]
    )

    assert calls == ["macro", "backtest"]
    assert out["macro"] == {"overall": "RED"}
    assert out["backtest"] == {"blocked": False, "result": {"cagr": 0.1}}


def test_dispatch_domains_passes_on_progress_to_backtest_domain(monkeypatch):
    """백테스트 도메인 호출 시 on_progress가 그대로 전달돼, 감사 에이전트들의 실행 과정도
    실시간 트리에 나타난다(HA-12 확장)."""
    import src.agents.supervisor as sup

    captured: dict = {}

    def spy_backtest(question, steps, conn, llm_fn=None, on_progress=None):
        captured["on_progress"] = on_progress
        return {"blocked": False, "result": {"cagr": 0.1}}

    monkeypatch.setattr(sup, "answer_backtest_question", spy_backtest)

    sentinel = lambda step, summary: None
    dispatch_domains(
        ["backtest"], "백테스트", conn=None, llm_fn=None, steps=[{"op": "noop"}],
        on_progress=sentinel,
    )

    assert captured["on_progress"] is sentinel


def test_dispatch_domains_absorbs_domain_exception(monkeypatch):
    """도메인 함수가 예외를 던져도 dispatch가 전파하지 않고 error를 기록한다."""
    import src.agents.supervisor as sup

    def boom(question, conn, llm_fn=None):
        raise RuntimeError("도메인 폭발")

    monkeypatch.setattr(sup, "answer_kr_question", boom)

    out = dispatch_domains(["kr"], "삼성전자", conn=None, llm_fn=None)
    assert "kr" in out
    assert "error" in out["kr"]
    assert "도메인 폭발" in out["kr"]["error"] or "RuntimeError" in out["kr"]["error"]


# ── AC2: verify_answer — 도메인 결과와 원 질문의 정합성 판정 ───────────────────

def test_verify_answer_llm_reports_mismatch_is_invalid():
    """의도적 오답 fixture: fake LLM이 '불일치'라고 응답 → valid False."""
    def fake_llm(prompt: str) -> str:
        return "불일치"

    domain_results = {"kr": {"stock_code": "005930", "financial": {"value": 12.5}}}
    verdict = verify_answer("삼성전자 PER 알려줘", domain_results, fake_llm)

    assert verdict["valid"] is False
    assert verdict["reason"]


def test_verify_answer_llm_reports_match_is_valid():
    def fake_llm(prompt: str) -> str:
        return "일치"

    domain_results = {"kr": {"stock_code": "005930", "financial": {"value": 12.5}}}
    verdict = verify_answer("삼성전자 PER 알려줘", domain_results, fake_llm)

    assert verdict["valid"] is True


def test_verify_answer_llm_json_verdict_is_honored():
    def fake_llm(prompt: str) -> str:
        return '{"valid": false, "reason": "질문은 PER인데 결과가 주가뿐"}'

    verdict = verify_answer(
        "삼성전자 PER", {"kr": {"price": [{"close": 71000}]}}, fake_llm
    )
    assert verdict["valid"] is False
    assert "주가" in verdict["reason"]


def test_verify_answer_empty_results_is_invalid_deterministically():
    verdict = verify_answer("삼성전자 PER", {}, llm_fn=None)
    assert verdict["valid"] is False


def test_verify_answer_no_llm_with_data_passes_deterministically():
    verdict = verify_answer(
        "삼성전자 PER", {"kr": {"stock_code": "005930", "financial": {"value": 12.5}}}, llm_fn=None
    )
    assert verdict["valid"] is True


# ── search_signal_strategy(탐색형) 결과는 제약 미충족이어도 '정직한 답'이므로 유효 ──────
#    (실사용 회귀: "MDD·수익률 조건을 만족하는 전략 찾아줘"에 대해 후보를 다 시도해 가장
#    근접한 결과를 constraints_met=False로 돌려줬는데, 검증 LLM이 '숫자가 목표에 못 미침'을
#    '답변 실패'로 오판 → 3회 재시도 → uncertain=True 로 빠지던 문제. 이 탐색 결과는 제약
#    충족 여부와 무관하게 유효한 답으로 본다 — 판정을 LLM에 넘기지 않고 결정론적으로 통과.)
def _best_effort_search_domain_results(constraints_met: bool) -> dict:
    best = {
        "entry_rule": {"left": {"kind": "indicator", "name": "sma", "period": 20},
                       "op": "cross_above",
                       "right": {"kind": "indicator", "name": "sma", "period": 60}},
        "exit_rule": {"left": {"kind": "indicator", "name": "sma", "period": 20},
                      "op": "cross_below",
                      "right": {"kind": "indicator", "name": "sma", "period": 60}},
        "performance": {"total_return": 263.35, "mdd": -29.79, "sharpe": 1.376},
        "holdings": [{"date": "2024-01-02", "codes": ["005930"]}],
        "dates": ["2023-07-15", "2026-07-15"], "navs": [1.0, 3.63],
        "constraints_met": constraints_met,
    }
    search_result = {"constraints_met": constraints_met, "results": [best], "best": best}
    return {"backtest": {"blocked": False, "error": None, "result": search_result,
                         "hard": [], "warnings": [], "data": []}}


def test_verify_answer_best_effort_search_unmet_is_valid_without_calling_llm():
    def boom_llm(prompt: str) -> str:  # 호출되면 안 됨(결정론 단락)
        raise AssertionError("탐색형 결과는 LLM 판정을 거치지 않아야 한다")

    verdict = verify_answer(
        "삼성전자 MDD 5% 이내·누적수익률 500% 이상 전략 찾아줘",
        _best_effort_search_domain_results(constraints_met=False),
        boom_llm,
    )
    assert verdict["valid"] is True
    assert verdict["reason"]


def test_verify_answer_search_met_is_valid_without_calling_llm():
    def boom_llm(prompt: str) -> str:
        raise AssertionError("탐색형 결과는 LLM 판정을 거치지 않아야 한다")

    verdict = verify_answer(
        "삼성전자 MDD 30% 이내·누적수익률 35% 이상 전략 찾아줘",
        _best_effort_search_domain_results(constraints_met=True),
        boom_llm,
    )
    assert verdict["valid"] is True


def test_verify_answer_normal_backtest_result_still_delegates_to_llm():
    """탐색형이 아닌 일반 백테스트 결과(constraints_met 키 없음)는 기존대로 LLM 판정을 받는다."""
    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return "불일치"

    normal = {"backtest": {"blocked": False, "error": None,
                           "result": {"dates": ["2024-01-02"], "navs": [1.0, 1.1],
                                      "performance": {"total_return": 10.0}, "holdings": []},
                           "hard": [], "warnings": [], "data": []}}
    verdict = verify_answer("삼성전자 골든크로스 백테스트", normal, fake_llm)
    assert calls, "일반 백테스트 결과는 LLM 판정을 거쳐야 한다"
    assert verdict["valid"] is False


# ── 업종(sector) 구어체→KRX 실제 분류 매핑을 검증 LLM이 불일치로 오판하지 않도록
#    프롬프트에 명시적 가이드를 준다 (실사용 회귀: "반도체"→"전기·전자" 매핑을 검증
#    단계가 불일치로 판정해 정확한 스크리닝 결과도 uncertain 처리되던 문제) ──────────
def test_verify_prompt_includes_sector_substitution_guidance():
    seen_prompts = []

    def spy_llm(prompt: str) -> str:
        seen_prompts.append(prompt)
        return "일치"

    domain_results = {
        "kr": {"intent": "screening", "sectors": ["전기·전자"], "result": [{"name": "x"}]}
    }
    verify_answer("반도체 업종에서 PER 낮은 5개", domain_results, spy_llm)
    assert seen_prompts
    assert "업종" in seen_prompts[0] and "KRX" in seen_prompts[0]


# ── 실사용 재현 버그: 검증 LLM이 "오늘" 날짜를 몰라서 실제 오늘자 데이터를 자기 학습
#    시점 기준 "미래"로 오판해 valid=false를 주던 문제("오늘 삼성전자 종가" 질의가
#    price.date=오늘 날짜인데도 매번 불확실 처리됨) ────────────────────────────────
def test_verify_prompt_includes_todays_date_for_freshness_judgment():
    from datetime import date

    seen_prompts = []

    def spy_llm(prompt: str) -> str:
        seen_prompts.append(prompt)
        return "일치"

    # domain_results 자체의 날짜(2020-01-01)는 실제 오늘 날짜와 절대 같을 수 없게 만들어,
    # 프롬프트에 실제 오늘 날짜 문자열이 나타난다면 그건 domain_results 덤프가 아니라
    # 별도의 명시적 "오늘 날짜는 ..." 문구에서 온 것임을 확실히 한다.
    domain_results = {"kr": {"price": [{"date": "2020-01-01", "close": 263000.0}]}}
    verify_answer("오늘 삼성전자 종가 얼마야", domain_results, spy_llm)
    assert seen_prompts
    assert "오늘 날짜" in seen_prompts[0]
    assert date.today().isoformat() in seen_prompts[0]


# ── 검증/종합 프롬프트 비대화 방지 — 스크리닝 rows가 최대 1000행이라 통째로 프롬프트에
#    넣으면 텍스트가 폭발한다. 앞부분만 남기고 나머지는 개수 요약으로 축약해야 한다 ──────

def _big_screening_rows(n: int = 1000) -> list[dict]:
    return [{"stock_code": f"{i:06d}", "name": f"종목{i}", "per": 10.0} for i in range(n)]


def test_verify_prompt_truncates_long_screening_rows():
    seen_prompts = []

    def spy_llm(prompt: str) -> str:
        seen_prompts.append(prompt)
        return "일치"

    domain_results = {"kr": {"intent": "screening", "result": _big_screening_rows()}}
    verify_answer("코스피 저PER 1000개", domain_results, spy_llm)

    prompt = seen_prompts[0]
    assert "000999" not in prompt  # 뒷부분 행은 생략돼야 함
    assert "1000" in prompt  # 총 개수는 요약으로 남아야 함
    assert len(prompt) < 5000  # 1000행 전체(수만 자)보다 훨씬 짧아야 함


def test_synthesize_prompt_truncates_long_screening_rows():
    captured = {}

    def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "결론"

    domain_results = {"kr": {"intent": "screening", "result": _big_screening_rows()}}
    synthesize_conclusion("코스피 저PER 1000개", domain_results, fake_llm)

    prompt = captured["prompt"]
    assert "000999" not in prompt
    assert len(prompt) < 5000


def test_verify_prompt_uses_json_serialization_not_python_repr():
    """dict를 파이썬 repr(작은따옴표)이 아니라 표준 JSON(큰따옴표)으로 직렬화해야 한다."""
    seen_prompts = []

    def spy_llm(prompt: str) -> str:
        seen_prompts.append(prompt)
        return "일치"

    verify_answer(
        "삼성전자 PER",
        {"kr": {"stock_code": "005930", "financial": {"value": 12.5}}},
        spy_llm,
    )
    assert seen_prompts
    assert '"stock_code": "005930"' in seen_prompts[0]


# ── AC3: answer_with_verification — 정확히 3회 재시도 후 uncertain(무한루프 없음) ──

def test_answer_with_verification_retries_exactly_three_times_then_uncertain():
    """verify_fn이 항상 실패 → 정확히 3회만 시도(4번째 없음) → uncertain True, attempts 3."""
    verify_calls: list[int] = []
    dispatch_calls: list[int] = []
    synth_calls: list[int] = []

    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        dispatch_calls.append(1)
        return {"kr": {"attempt": len(dispatch_calls)}}

    def always_invalid_verify(question, domain_results, llm_fn):
        verify_calls.append(1)
        return {"valid": False, "reason": "항상 불일치(시뮬레이션)"}

    def stub_synth(question, domain_results, llm_fn):
        synth_calls.append(1)
        return "종합결론"

    res = answer_with_verification(
        "삼성전자 PER", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch,
        verify_fn=always_invalid_verify, synthesize_fn=stub_synth,
    )

    assert len(verify_calls) == 3   # 정확히 3회
    assert len(dispatch_calls) == 3  # 4번째 시도 없음(무한루프 없음)
    assert synth_calls == []          # 실패 경로에서는 종합결론을 만들지 않음
    assert res["uncertain"] is True
    assert res["attempts"] == 3
    assert res["reason"]


def test_answer_with_verification_respects_custom_max_retries():
    verify_calls: list[int] = []

    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {}}

    def always_invalid_verify(question, domain_results, llm_fn):
        verify_calls.append(1)
        return {"valid": False, "reason": "실패"}

    res = answer_with_verification(
        "q", conn=None, llm_fn=None, max_retries=2,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=always_invalid_verify,
    )
    assert len(verify_calls) == 2
    assert res["attempts"] == 2
    assert res["uncertain"] is True


# ── AC4: 복합 도메인(macro+kr) — 종합결론 + 원본 domain_results 병기 ───────────

def test_answer_with_verification_composite_macro_kr_preserves_raw_and_conclusion():
    macro_raw = {"available": True, "overall": "RED", "spread": {"value": -0.1, "regime": "역전"}}
    kr_raw = {"stock_code": "005930", "financial": {"value": 12.5, "source": "DART"}}

    def stub_route(question, llm_fn):
        return ["macro", "kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        assert routes == ["macro", "kr"]
        return {"macro": macro_raw, "kr": kr_raw}

    def valid_verify(question, domain_results, llm_fn):
        return {"valid": True, "reason": "일치"}

    res = answer_with_verification(
        "지금 매크로 신호 안 좋은데 삼성전자 PER 괜찮아?", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=valid_verify,
    )

    assert res["uncertain"] is False
    assert isinstance(res["conclusion"], str) and res["conclusion"]  # 종합결론 존재
    assert res["attempts"] == 1
    # 원본 데이터가 가공 없이 그대로 병기되어야 한다.
    assert res["domain_results"]["macro"] == macro_raw
    assert res["domain_results"]["kr"] == kr_raw


def test_answer_with_verification_succeeds_on_second_attempt():
    """1회 실패 후 2회차에 통과하면 attempts=2, uncertain False."""
    verdicts = iter([{"valid": False, "reason": "1차 실패"}, {"valid": True, "reason": "2차 통과"}])

    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {"stock_code": "005930"}}

    def flaky_verify(question, domain_results, llm_fn):
        return next(verdicts)

    res = answer_with_verification(
        "삼성전자 PER", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=flaky_verify,
    )
    assert res["uncertain"] is False
    assert res["attempts"] == 2


# ── synthesize_conclusion — 기본 구현(LLM 없이도 비어있지 않은 요약) ──────────

def test_synthesize_conclusion_without_llm_returns_nonempty_summary():
    domain_results = {
        "macro": {"available": True, "overall": "RED"},
        "kr": {"stock_code": "005930", "financial": {"value": 12.5}},
    }
    conclusion = synthesize_conclusion("복합 질문", domain_results, llm_fn=None)
    assert isinstance(conclusion, str) and conclusion.strip()


def test_synthesize_conclusion_uses_llm_when_available():
    def fake_llm(prompt: str) -> str:
        return "LLM이 생성한 종합결론"

    conclusion = synthesize_conclusion("질문", {"kr": {"stock_code": "005930"}}, fake_llm)
    assert conclusion == "LLM이 생성한 종합결론"


def test_synthesize_prompt_explains_same_company_label():
    """스크리닝 결과의 _same_company 라벨(GOOG/GOOGL 등 형제 티커 표시)이 원본 데이터에
    그대로 병기되는 것만으로는 사용자에게 의미가 전달되지 않는다 — LLM이 종합결론을 쓸 때
    이 라벨을 설명하도록 프롬프트에 안내 문구가 있어야 한다."""
    captured = {}

    def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "결론"

    domain_results = {
        "us": {"intent": "screening", "result": [
            {"stock_code": "GOOGL", "name": "Alphabet Inc. Class A Common Stock", "_same_company": False},
            {"stock_code": "GOOG", "name": "Alphabet Inc. Class C Capital Stock", "_same_company": True},
        ]},
    }
    synthesize_conclusion("미국 영업이익 상위 기업", domain_results, fake_llm)
    # 필드명이 프롬프트에 있다는 것만으로는 부족하다(그건 domain_results를 그대로 넣기만
    # 해도 항상 참이 되는 가짜 검증) — LLM이 실제로 무슨 뜻인지 알 수 있는 설명 문구가
    # 있어야 한다.
    assert "동일 회사" in captured["prompt"] or "같은 회사" in captured["prompt"]


# ── on_progress 진행 콜백 — 실시간 트리 상세화. 라우팅/도메인별/검증 시도별로 즉시
#    콜백을 호출한다(기존 "노드 완료 후 요약 1줄"과 달리 진행 중에 여러 번 호출됨).
#    콜백을 안 주면(on_progress=None, 기본값) 기존 모든 테스트처럼 완전히 동일하게 동작한다. ──

def test_route_question_calls_on_progress_with_routing_summary():
    events: list[tuple[str, str]] = []

    def fake_llm(prompt: str) -> str:
        return "kr, us"

    route_question("삼성전자 vs 애플", fake_llm, on_progress=lambda step, summary: events.append((step, summary)))

    assert events
    step, summary = events[0]
    assert step == "supervisor"
    assert "한국" in summary and "미국" in summary


def test_route_question_without_on_progress_is_unaffected():
    """on_progress 생략 시(기본값 None) 콜백 없이 기존과 동일하게 동작(회귀 방지)."""
    routes = route_question("삼성전자 PER", lambda p: "kr")
    assert routes == ["kr"]


def test_dispatch_domains_calls_on_progress_for_start_and_complete(monkeypatch):
    import src.agents.supervisor as sup

    monkeypatch.setattr(
        sup, "answer_kr_question",
        lambda question, conn, llm_fn=None: {"stock_code": "005930", "financial": {"value": 12.5}},
    )
    events: list[tuple[str, str]] = []

    dispatch_domains(
        ["kr"], "삼성전자", conn=None, llm_fn=None,
        on_progress=lambda step, summary: events.append((step, summary)),
    )

    assert len(events) == 2  # 시작 1건 + 완료 1건
    assert events[0][0] == "kr" and "조회 중" in events[0][1]
    assert events[1][0] == "kr" and "완료" in events[1][1]


def test_dispatch_domains_calls_on_progress_on_exception(monkeypatch):
    import src.agents.supervisor as sup

    def boom(question, conn, llm_fn=None):
        raise RuntimeError("도메인 폭발")

    monkeypatch.setattr(sup, "answer_kr_question", boom)
    events: list[tuple[str, str]] = []

    dispatch_domains(
        ["kr"], "삼성전자", conn=None, llm_fn=None,
        on_progress=lambda step, summary: events.append((step, summary)),
    )

    assert events[-1][0] == "kr"
    assert "오류" in events[-1][1]


def test_dispatch_domains_without_on_progress_is_unaffected(monkeypatch):
    import src.agents.supervisor as sup

    monkeypatch.setattr(
        sup, "answer_kr_question",
        lambda question, conn, llm_fn=None: {"stock_code": "005930"},
    )
    out = dispatch_domains(["kr"], "삼성전자", conn=None, llm_fn=None)
    assert out["kr"] == {"stock_code": "005930"}


def test_answer_with_verification_emits_progress_in_order():
    """route→dispatch(도메인 시작/완료)→verify 순서로 이벤트가 쌓인다."""
    events: list[tuple[str, str]] = []

    def stub_route(question, llm_fn, on_progress=None):
        if on_progress:
            on_progress("supervisor", "라우팅: kr")
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None, on_progress=None):
        if on_progress:
            on_progress("kr", "한국 도메인 조회 중…")
            on_progress("kr", "한국 도메인 완료")
        return {"kr": {"stock_code": "005930"}}

    def valid_verify(question, domain_results, llm_fn):
        return {"valid": True, "reason": "일치"}

    answer_with_verification(
        "삼성전자 PER", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=valid_verify,
        on_progress=lambda step, summary: events.append((step, summary)),
    )

    steps_order = [s for s, _ in events]
    assert steps_order[0] == "supervisor"          # 라우팅이 가장 먼저
    assert "kr" in steps_order                       # 도메인 진행이 중간에
    assert steps_order[-1] == "verify"                # 검증 결과가 마지막
    assert "통과" in events[-1][1]


def test_answer_with_verification_emits_retry_progress_on_failure():
    """검증 실패마다 verify 이벤트가 재시도 사유와 함께 나온다(최대 시도 횟수만큼)."""
    events: list[tuple[str, str]] = []

    def stub_route(question, llm_fn, on_progress=None):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None, on_progress=None):
        return {"kr": {}}

    def always_invalid_verify(question, domain_results, llm_fn):
        return {"valid": False, "reason": "항상 불일치"}

    answer_with_verification(
        "q", conn=None, llm_fn=None, max_retries=3,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=always_invalid_verify,
        on_progress=lambda step, summary: events.append((step, summary)),
    )

    verify_events = [s for step, s in events if step == "verify"]
    assert len(verify_events) == 3
    assert all("항상 불일치" in s for s in verify_events)


def test_answer_with_verification_feeds_failure_reason_into_retry():
    """검증 실패 시 다음 시도의 dispatch_fn에 넘기는 question에 실패 사유를 피드백으로
    덧붙인다 — 안 그러면 도메인 에이전트가 같은 실수(예: 잘못된 필드로 스크리닝)를
    재시도마다 반복해 토큰만 낭비한다(실사용 재현: "직전 12개월 수익률" 질문에
    revenue_growth로 잘못 스크리닝하는 게 3회 내내 동일하게 반복됨)."""
    dispatch_questions: list[str] = []

    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        dispatch_questions.append(question)
        return {"kr": {"stock_code": "005930"}}

    verdicts = iter([
        {"valid": False, "reason": "잘못된 지표(revenue_growth)로 스크리닝함"},
        {"valid": False, "reason": "여전히 잘못된 지표"},
        {"valid": True, "reason": "이번엔 맞음"},
    ])

    def flaky_verify(question, domain_results, llm_fn):
        return next(verdicts)

    answer_with_verification(
        "저PER 10개", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=flaky_verify,
    )

    assert dispatch_questions[0] == "저PER 10개"  # 1차는 원본 그대로
    assert "저PER 10개" in dispatch_questions[1]  # 원본 질문 맥락은 유지
    assert "잘못된 지표(revenue_growth)로 스크리닝함" in dispatch_questions[1]  # 2차는 직전 피드백 포함
    assert "여전히 잘못된 지표" in dispatch_questions[2]  # 3차는 최신 피드백으로 갱신


def test_answer_with_verification_verify_and_synthesize_use_original_question():
    """검증/종합결론은 피드백이 섞이지 않은 원본 질문 그대로 받는다 — 검증 판정과 최종
    답변 자체가 피드백 문구로 왜곡되면 안 된다(피드백은 도메인 재실행 유도용일 뿐)."""
    verify_questions: list[str] = []
    synth_questions: list[str] = []

    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {"stock_code": "005930"}}

    verdicts = iter([{"valid": False, "reason": "실패1"}, {"valid": True, "reason": "통과"}])

    def flaky_verify(question, domain_results, llm_fn):
        verify_questions.append(question)
        return next(verdicts)

    def stub_synth(question, domain_results, llm_fn):
        synth_questions.append(question)
        return "결론"

    answer_with_verification(
        "저PER 10개", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=flaky_verify,
        synthesize_fn=stub_synth,
    )

    assert verify_questions == ["저PER 10개", "저PER 10개"]  # 검증은 항상 원본
    assert synth_questions == ["저PER 10개"]  # 종합결론도 원본


def test_answer_with_verification_without_on_progress_is_unaffected():
    """기존 모든 호출부(on_progress 생략)는 완전히 동일하게 동작 — 회귀 방지."""
    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {"stock_code": "005930"}}

    def valid_verify(question, domain_results, llm_fn):
        return {"valid": True, "reason": "일치"}

    res = answer_with_verification(
        "삼성전자 PER", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=valid_verify,
    )
    assert res["uncertain"] is False
