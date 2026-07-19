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

import src.agents.supervisor as supervisor_mod
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


def test_route_question_normalizes_order_and_dedupes():
    """LLM이 순서를 뒤섞거나 중복을 내도 정규 순서(kr,macro,backtest)로 정리."""
    def fake_llm(prompt: str) -> str:
        return "macro kr backtest kr"

    assert route_question("복합 질문", fake_llm) == ["kr", "macro", "backtest"]


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
    """'삼성' 키워드가 실제로 kr에 매치되어 감지된다(폴백이 아니라 정상 키워드 매치)."""
    assert route_question("삼성전자 PER 알려줘", None) == ["kr"]


def test_route_question_fallback_without_llm_detects_correlation_as_backtest():
    """실사용 회귀: "상관관계"를 물으면 '백테스트'/'전략' 단어가 전혀 없어도 backtest로
    라우팅돼야 한다 — 팩터간 상관관계·분위수 분석은 backtest 도메인의 correlation/
    quantile_bucket_means 프리미티브(get_cross_section 기반, top_n 상한 없음)로만 계산
    가능한데, 예전 키워드 사전엔 '전략 백테스트' 관련 단어만 있어 kr 스크리닝으로 잘못
    빠졌었다(실서버 curl로 확인된 버그)."""
    routes = route_question("코스피 PBR과 GPA의 상관관계 구해줘", None)
    assert "backtest" in routes


def test_route_question_fallback_without_llm_detects_quantile_as_backtest():
    routes = route_question("PBR 5분위수별 평균 GPA 보여줘", None)
    assert "backtest" in routes


def test_route_prompt_mentions_correlation_and_quantile_for_llm_routing():
    """LLM 우선 라우팅 경로도 인식하도록 _route_prompt의 backtest 설명에 상관관계/분위수
    분석이 포함돼야 한다("전략 백테스트"로만 설명되면 LLM도 이 도메인을 놓친다)."""
    prompt = supervisor_mod._route_prompt("아무 질문")
    assert "상관관계" in prompt or "분위수" in prompt


def test_route_question_fallback_without_llm_detects_histogram_as_backtest():
    """히스토그램(분포도) 질문은 '백테스트'/'전략' 단어가 없어도 backtest로 라우팅돼야 한다 —
    histogram_buckets 프리미티브(get_cross_section 기반)로만 계산 가능하다. 시장 키워드가
    없는 질문은 backtest 단독, '코스피' 등 kr 키워드가 섞이면 kr+backtest 복합으로 간다."""
    # kr/us 시장 키워드가 없으면 backtest 단독 라우팅
    assert route_question("PBR 분포도 그려줘", None) == ["backtest"]
    assert route_question("show me the histogram of pbr", None) == ["backtest"]
    # '코스피'가 섞이면(기존 kr 키워드) kr도 함께 걸리되 backtest는 반드시 포함된다
    assert "backtest" in route_question("코스피 PBR을 히스토그램으로 보여줘", None)


def test_route_prompt_mentions_histogram_for_llm_routing():
    """LLM 우선 라우팅 경로도 인식하도록 _route_prompt의 backtest 설명에 히스토그램/분포가 포함."""
    prompt = supervisor_mod._route_prompt("아무 질문")
    assert "히스토그램" in prompt or "분포" in prompt


def test_route_question_fallback_without_llm_detects_qvm_as_backtest():
    """QVM(퀄리티·밸류·모멘텀) 멀티팩터 질문은 '백테스트'/'전략' 단어가 없어도 backtest로
    라우팅돼야 한다 — 실서버 재현: "너는 한국 주식 멀티팩터 스크리너다" 같은 QVM 질문이
    compute_qvm_scores(backtest 전용 프리미티브)를 쓸 수 없는 kr 도메인으로 잘못 라우팅돼
    3회 검증에 모두 실패하고 자유 코드 폴백까지 실패했다(라우팅 자체가 원인)."""
    assert route_question("한국 주식 멀티팩터 스크리너 전략을 돌려줘", None) == ["backtest"]
    assert "backtest" in route_question("퀄리티 밸류 모멘텀 전략으로 코스피 상위 20개 뽑아줘", None)
    assert route_question("QVM 전략으로 스크리닝해줘", None) == ["backtest"]


def test_route_prompt_mentions_qvm_for_llm_routing():
    """LLM 우선 라우팅 경로도 인식하도록 _route_prompt의 backtest 설명에 QVM/멀티팩터
    복합점수 스크리닝이 포함돼야 한다("전략 백테스트"로만 설명되면 LLM이 이 도메인을
    kr의 단일지표 스크리닝과 혼동해 놓친다)."""
    prompt = supervisor_mod._route_prompt("아무 질문")
    assert "qvm" in prompt.lower() or "복합" in prompt or "멀티팩터" in prompt


def test_route_question_fallback_without_llm_detects_common_kr_company_nicknames():
    """실사용 재현: LLM 라우팅이 일시 실패해 휴리스틱으로 폴백했을 때, '코스피'/'삼성' 같은
    시장 키워드 없이 개별 종목 약칭만 있는 질문("하이닉스 12개월 누적수익률")이 빈 라우팅으로
    끝나 "질문을 이해하지 못했습니다"가 나오던 버그."""
    assert "kr" in route_question("하이닉스 12개월 누적수익률 구해줘", None)
    assert "kr" in route_question("네이버 최근 실적 어때", None)


def test_route_question_fallback_returns_empty_for_completely_unrelated_question():
    """휴리스틱 키워드 사전에 전혀 안 걸리는 질문(LLM도 없을 때)은 무조건 ['kr']로
    폴백하지 않고 빈 리스트(unknown)를 반환해야 한다 — 무관한 질문에 한국주식을 억지로
    갖다붙이지 않기 위함(실측 확인: '삼성' 같은 실제 키워드가 있는 질문은 이 폴백이 아니라
    정상 매치 경로를 타므로 영향받지 않는다)."""
    assert route_question("완전히 무관한 아무말 대잔치 질문입니다", None) == []


# ── dispatch_domains — 각 도메인 원본 결과를 가공 없이 보존 ────────────────────

def test_dispatch_domains_preserves_raw_results(monkeypatch):
    import src.agents.supervisor as sup

    monkeypatch.setattr(
        sup, "answer_kr_question",
        lambda question, conn, llm_fn=None: {"stock_code": "005930", "raw": "KR-DATA"},
    )

    out = dispatch_domains(["kr"], "삼성전자", conn=None, llm_fn=None)

    assert out["kr"] == {"stock_code": "005930", "raw": "KR-DATA"}
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


def test_dispatch_domains_passes_on_progress_to_kr_domain(monkeypatch):
    """스크리닝 조건 JSON을 실시간으로 노출하려면 kr 도메인 호출에도 on_progress가
    backtest와 동일하게 전달돼야 한다(기존엔 backtest만 전달되고 있었음)."""
    import src.agents.supervisor as sup

    captured: dict = {}

    def spy_kr(question, conn, llm_fn=None, on_progress=None):
        captured["kr"] = on_progress
        return {"stock_code": "005930"}

    monkeypatch.setattr(sup, "answer_kr_question", spy_kr)

    sentinel = lambda step, summary, detail=None: None
    dispatch_domains(["kr"], "삼성전자", conn=None, llm_fn=None, on_progress=sentinel)

    assert captured["kr"] is sentinel


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


# ── 검증 불가(LLM 장애) vs 검증 실패 구분 — OpenAI 등 검증 LLM 호출 자체가 실패하면
#    "검증 실패"로 멀쩡한 답을 버리지 않고, 데이터 존재 확인만으로 통과시킨다 ──────────

def test_verify_answer_llm_failure_is_treated_as_unavailable_not_invalid():
    def boom_llm(prompt: str) -> str:
        raise RuntimeError("OpenAI 일시 장애")

    domain_results = {"kr": {"stock_code": "005930", "financial": {"value": 12.5}}}
    verdict = verify_answer("삼성전자 PER", domain_results, boom_llm)

    assert verdict["valid"] is True
    assert verdict.get("verification_unavailable") is True
    assert "장애" in verdict["reason"] or "불가" in verdict["reason"]


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


# ── 프롬프트 축약 상수(5)가 사용자가 실제 요청한 개수(top_n)보다 작으면, 검증/종합
#    LLM이 "N개 중 5개만 표시"라며 요청 개수를 못 채운 것처럼 오판하던 실사용 버그.
#    스크리닝 결과가 top_n을 알고 있으므로, 축약 기준을 고정 5가 아니라 top_n에 맞춰야 한다 ──
def test_verify_prompt_keeps_full_list_up_to_requested_top_n():
    seen_prompts = []

    def spy_llm(prompt: str) -> str:
        seen_prompts.append(prompt)
        return "일치"

    domain_results = {"kr": {"intent": "screening", "top_n": 10, "result": _big_screening_rows(10)}}
    verify_answer("코스피 수익률 상위 10개", domain_results, spy_llm)

    prompt = seen_prompts[0]
    domain_json = prompt.split("도메인 결과: ", 1)[1]  # 안내문 예시 텍스트("...(총...")는 제외하고 실제 데이터만 검사
    assert "종목9" in domain_json          # 10번째(마지막) 행까지 온전히 포함돼야 함
    assert "...(총" not in domain_json     # top_n 이내라 축약 마커 자체가 없어야 함


def test_synthesize_prompt_keeps_full_list_up_to_requested_top_n():
    captured = {}

    def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "결론"

    domain_results = {"kr": {"intent": "screening", "top_n": 20, "result": _big_screening_rows(20)}}
    synthesize_conclusion("코스피 수익률 상위 20개", domain_results, fake_llm)

    prompt = captured["prompt"]
    assert "종목19" in prompt
    assert "...(총" not in prompt


# ── 백테스트 리밸런싱 구간별 보유종목·구간수익률은 LLM 재량과 무관하게 최종 결론에
#    '항상' 포함돼야 한다(요구: "반기마다 어떤 종목이 있었는지·반기별 수익률도 같이 항상").
#    domain_backtest가 만든 결정론적 rebalance_summary 텍스트를 supervisor가 결론에 직접
#    덧붙인다 — LLM이 홀딩스를 언급하지 않아도 보장된다. ──────────────────────────────
_REBALANCE_SUMMARY = (
    "리밸런싱 구간별 보유종목·구간수익률:\n"
    "- 2025-06-30: 000001, 000002 (구간수익률 +5.00%)\n"
    "- 2025-12-31: 000003 (구간수익률 -2.00%)"
)


def test_synthesize_conclusion_always_appends_backtest_rebalance_summary():
    def terse_llm(prompt: str) -> str:
        # LLM이 보유종목/구간수익률을 전혀 언급하지 않는 요약을 내도(재량),
        return "백테스트를 수행했습니다."

    domain_results = {
        "backtest": {
            "blocked": False, "error": None,
            "result": {"performance": {"cagr": 5.0}, "holdings": [{"date": "x"}, {"date": "y"}]},
            "hard": [], "warnings": [], "data": [],
            "rebalance_summary": _REBALANCE_SUMMARY,
        }
    }
    conclusion = synthesize_conclusion("반기 리밸런싱 백테스트", domain_results, terse_llm)
    # 결정론적 리밸런싱 블록이 통째로 최종 결론에 포함돼야 한다.
    assert _REBALANCE_SUMMARY in conclusion
    assert "000001" in conclusion and "000003" in conclusion
    assert "+5.00%" in conclusion and "-2.00%" in conclusion


def test_synthesize_conclusion_no_rebalance_block_when_field_absent():
    """rebalance_summary가 없으면(단일 리밸런싱/비백테스트) 결론에 블록을 덧붙이지 않는다(회귀)."""
    domain_results = {
        "backtest": {"blocked": False, "result": {"performance": {"cagr": 5.0}}, "hard": [], "warnings": []}
    }
    conclusion = synthesize_conclusion("buy&hold 백테스트", domain_results, lambda p: "LLM결론")
    assert conclusion == "LLM결론"


def test_verify_prompt_still_truncates_rows_beyond_top_n():
    """top_n을 넘는 초과분(비정상 상황 방어)까지 무제한으로 다 보여주진 않는다 —
    top_n개까지만 보장하고, 그 이상은 여전히 축약 대상이다."""
    seen_prompts = []

    def spy_llm(prompt: str) -> str:
        seen_prompts.append(prompt)
        return "일치"

    domain_results = {"kr": {"intent": "screening", "top_n": 10, "result": _big_screening_rows(1000)}}
    verify_answer("코스피 수익률 상위 10개", domain_results, spy_llm)

    prompt = seen_prompts[0]
    assert "종목9" in prompt           # top_n(10)까지는 보장
    assert "000999" not in prompt      # 그 이상은 여전히 생략
    assert "...(총 1000개 중" in prompt


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


# ── 자유 실행 폴백: 3회 정형 검증 실패 후 마지막 안전망(exec_fallback.run_free_exec_fallback)
#    으로 escalate — 정형 어휘(스크리닝 criteria/top_n, 파이프라인 연산)의 표현력 한계로
#    반복 실패하는 질문(예: "시장별로 나눠 각각 상위 N개")을 위한 최후 수단. ──────────────

def test_answer_with_verification_falls_back_to_free_exec_when_retries_exhausted():
    fallback_calls: list[tuple] = []

    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {"attempt": "x"}}

    def always_invalid_verify(question, domain_results, llm_fn):
        return {"valid": False, "reason": "시장별 분리가 안 됨"}

    def stub_synth(question, domain_results, llm_fn):
        return f"종합결론(free_exec 포함={'free_exec' in domain_results})"

    def fake_fallback(question, conn, llm_fn, last_reason=None, **kwargs):
        fallback_calls.append((question, conn, llm_fn, last_reason))
        return {"ok": True, "result": {"KOSPI": ["005930"]}, "sql": "SELECT 1;", "code": "result=1"}

    res = answer_with_verification(
        "코스피 코스닥 각각 상위 10개", conn="conn", llm_fn="llm",
        route_fn=stub_route, dispatch_fn=stub_dispatch,
        verify_fn=always_invalid_verify, synthesize_fn=stub_synth,
        fallback_fn=fake_fallback,
    )

    assert len(fallback_calls) == 1  # 정확히 1회만(재시도 없음 — 최후 수단이므로)
    assert fallback_calls[0][3] == "시장별 분리가 안 됨"  # last_reason이 그대로 전달됨
    assert res["uncertain"] is False
    assert res["used_fallback"] is True
    assert res["domain_results"]["free_exec"]["result"] == {"KOSPI": ["005930"]}
    assert res["domain_results"]["free_exec"]["fallback_used"] is True
    assert "free_exec 포함=True" in res["conclusion"]


def test_answer_with_verification_stays_uncertain_when_fallback_also_fails():
    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {}}

    def always_invalid_verify(question, domain_results, llm_fn):
        return {"valid": False, "reason": "실패사유"}

    def fake_fallback(question, conn, llm_fn, last_reason=None, **kwargs):
        return {"ok": False, "result": None, "sql": None, "code": None, "error": "SQL도 못 만듦"}

    res = answer_with_verification(
        "질문", conn="conn", llm_fn="llm",
        route_fn=stub_route, dispatch_fn=stub_dispatch,
        verify_fn=always_invalid_verify, fallback_fn=fake_fallback,
    )

    assert res["uncertain"] is True
    assert res["attempts"] == 3
    assert "free_exec" not in res["domain_results"]
    assert "SQL도 못 만듦" in res["reason"]


def test_answer_with_verification_does_not_attempt_fallback_when_verification_passes():
    fallback_calls: list = []

    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {"ok": True}}

    def always_valid_verify(question, domain_results, llm_fn):
        return {"valid": True, "reason": "통과"}

    def fake_fallback(question, conn, llm_fn, last_reason=None, **kwargs):
        fallback_calls.append(1)
        return {"ok": True, "result": "should not be used"}

    res = answer_with_verification(
        "질문", conn="conn", llm_fn="llm",
        route_fn=stub_route, dispatch_fn=stub_dispatch,
        verify_fn=always_valid_verify, fallback_fn=fake_fallback,
    )

    assert fallback_calls == []
    assert res["uncertain"] is False
    assert "free_exec" not in res["domain_results"]


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
        return "kr, macro"

    route_question("삼성전자 매크로 신호", fake_llm, on_progress=lambda step, summary: events.append((step, summary)))

    assert events
    step, summary = events[0]
    assert step == "supervisor"
    assert "한국" in summary and "매크로" in summary


def test_route_question_without_on_progress_is_unaffected():
    """on_progress 생략 시(기본값 None) 콜백 없이 기존과 동일하게 동작(회귀 방지)."""
    routes = route_question("삼성전자 PER", lambda p: "kr")
    assert routes == ["kr"]


def test_dispatch_domains_calls_on_progress_for_start_and_complete(monkeypatch):
    import src.agents.supervisor as sup

    monkeypatch.setattr(
        sup, "answer_kr_question",
        lambda question, conn, llm_fn=None, on_progress=None: {"stock_code": "005930", "financial": {"value": 12.5}},
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


# ── 복합 도메인 부분 재시도 — 일부 도메인만 검증 실패하면 그 도메인만 재-dispatch하고
#    이미 통과한 도메인 결과는 유지한다(전체 재실행 낭비 방지) ───────────────────────

def test_answer_with_verification_retries_only_failed_domain_in_composite_question():
    # '실패한 도메인만 부분 재시도'하는 총괄 메커니즘은 도메인 종류와 무관하다. 미국 도메인이
    # 비활성화된 뒤에도 이 메커니즘은 그대로이므로, 여전히 활성인 kr+macro 복합으로 검증한다.
    dispatch_routes_seen: list[list[str]] = []

    def stub_route(question, llm_fn):
        return ["kr", "macro"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        dispatch_routes_seen.append(list(routes))
        result = {}
        if "kr" in routes:
            result["kr"] = {"stock_code": "005930", "financial": {"value": 12.5}}
        if "macro" in routes:
            result["macro"] = {"available": True, "overall": "GREEN"}
        return result

    per_domain_by_attempt = iter([
        {"kr": {"valid": True, "reason": "일치"}, "macro": {"valid": False, "reason": "매크로 불일치"}},
        {"macro": {"valid": True, "reason": "이제 일치"}},
    ])

    def stub_verify(question, domain_results, llm_fn):
        per_domain = next(per_domain_by_attempt)
        overall = all(v["valid"] for v in per_domain.values())
        return {
            "valid": overall,
            "reason": "전체 통과" if overall else "부분 실패",
            "per_domain": per_domain,
        }

    res = answer_with_verification(
        "삼성전자 PER이랑 매크로 신호 비교", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=stub_verify,
    )

    assert dispatch_routes_seen[0] == ["kr", "macro"]  # 1차는 전체 라우트
    assert dispatch_routes_seen[1] == ["macro"]  # 2차는 실패했던 macro만
    assert res["uncertain"] is False
    assert res["attempts"] == 2
    assert res["domain_results"]["kr"]["stock_code"] == "005930"  # kr은 1차 결과 그대로 유지
    assert res["domain_results"]["macro"]["overall"] == "GREEN"


def test_answer_with_verification_returns_uncertain_immediately_when_routes_empty():
    """라우팅 결과가 빈 리스트(unknown)면 dispatch/verify를 시도조차 하지 않고 즉시
    '질문을 이해하지 못했습니다' 불확실 응답을 반환한다(불필요한 재시도 낭비 방지)."""
    dispatch_calls: list[int] = []

    def stub_route(question, llm_fn):
        return []

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        dispatch_calls.append(1)
        return {}

    res = answer_with_verification(
        "완전히 무관한 질문", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch,
    )
    assert dispatch_calls == []  # dispatch 자체가 호출되면 안 됨
    assert res["uncertain"] is True
    assert res["routes"] == []
    assert res["attempts"] == 0


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


# ── kr 다중종목(named multi-entity) 결과의 데이터 존재 판정 ────────────────────
# 실서버 재현 버그: domain_kr.answer_kr_question이 "entities" 리스트로 다중종목 결과를
# 반환하는데(예: "삼성전자와 SK하이닉스 종가"), _domain_has_data는 최상위 financial/price
# 키만 보고 entities 안의 실제 데이터를 못 봐서 "데이터 없음"으로 오판 → verify_answer가
# LLM 판정까지 가지도 못하고 매 시도 즉시 실패(재시도해도 절대 안 고쳐짐, 원래 버그와
# 동일한 실패 패턴).

def test_domain_has_data_recognizes_multi_entity_kr_result_with_price():
    kr_result = {
        "stock_code": None, "stock_codes": ["005930", "000660"],
        "financial": None, "price": None,
        "entities": [
            {"stock_code": "005930", "financial": None,
             "price": [{"close": 71000.0}], "errors": []},
            {"stock_code": "000660", "financial": None,
             "price": [{"close": 210000.0}], "errors": []},
        ],
        "errors": [],
    }
    assert supervisor_mod._domain_has_data(kr_result) is True


def test_domain_has_data_recognizes_multi_entity_kr_result_with_financial():
    kr_result = {
        "stock_code": None, "stock_codes": ["005930", "000660"],
        "financial": None, "price": None,
        "entities": [
            {"stock_code": "005930", "financial": {"value": 12.5}, "price": None, "errors": []},
        ],
        "errors": [],
    }
    assert supervisor_mod._domain_has_data(kr_result) is True


def test_domain_has_data_returns_false_when_all_entities_empty():
    kr_result = {
        "stock_code": None, "stock_codes": ["999999"],
        "financial": None, "price": None,
        "entities": [
            {"stock_code": "999999", "financial": None, "price": None,
             "errors": ["재무 지표를 인식하지 못함"]},
        ],
        "errors": [],
    }
    assert supervisor_mod._domain_has_data(kr_result) is False


# ── kr 다중분기(multi-period) 결과의 데이터 존재 판정 ──────────────────────────────
# "하이닉스 25년과 26년 1분기 영업이익률"처럼 한 종목의 여러 분기를 조회하면 최상위
# financial은 None이고 실제 데이터는 periods 리스트에 기간별로 담긴다(다중종목 entities와
# 동일 관례). _domain_has_data가 이 구조도 인식해야 verify_answer가 정상 진행한다.

def test_domain_has_data_recognizes_multi_period_kr_result():
    kr_result = {
        "stock_code": "000660", "question": "...", "intent": "financial",
        "financial": None, "price": None,
        "periods": [
            {"period": "2025Q1", "financial": {"value": 21.0}, "errors": []},
            {"period": "2026Q1", "financial": {"value": 19.0}, "errors": []},
        ],
        "errors": [],
    }
    assert supervisor_mod._domain_has_data(kr_result) is True


def test_domain_has_data_returns_false_when_all_periods_empty():
    kr_result = {
        "stock_code": "000660", "question": "...", "intent": "financial",
        "financial": None, "price": None,
        "periods": [
            {"period": "2025Q1", "financial": None, "errors": ["재무데이터 조회 실패"]},
        ],
        "errors": [],
    }
    assert supervisor_mod._domain_has_data(kr_result) is False


# ── 복합 도메인(kr+backtest 등) 검증 — 도메인 하나가 전체 질문을 혼자 못 채운다는
#    이유만으로 부당하게 실패 처리되지 않아야 한다 ────────────────────────────────
# 실서버 재현 버그: "SK하이닉스 최근 10년 골든크로스 전략 수익률과 그래프"가 kr+backtest
# 두 도메인으로 라우팅됨. backtest 도메인은 실제로 10년치를 정확히 계산해 반환했지만,
# kr 도메인(단순 종가 조회, 자기 몫만 정상 수행)이 "전체 질문"(백테스트 수익률/그래프 포함)
# 을 혼자 못 채운다는 이유로 검증에 실패 처리됐다. per_domain 검증이 각 도메인을 "전체
# 질문에 혼자 답해야 한다"는 기준으로 판정했기 때문 — 도메인끼리 서로 다른 부분을 나눠
# 담당하는 복합 질문에서는 항상, 결정론적으로(재시도해도 안 고쳐짐) 실패한다.

def test_verify_answer_composite_domains_pass_when_combined_result_satisfies_question():
    """kr(종가)+backtest(수익률) 각자 자기 몫만 답해도, 합쳐서 질문을 만족하면 전체 유효해야 한다."""
    domain_results = {
        "kr": {"stock_code": "000660", "financial": None,
               "price": [{"close": 2082000.0}], "errors": []},
        "backtest": {"blocked": False, "error": None,
                     "result": {"dates": ["2016-07-18", "2026-07-15"], "navs": [1.0, 11.6],
                                "performance": {"cagr": 28.66, "total_return": 1058.8}},
                     "hard": [], "warnings": [], "data": []},
    }
    calls: list[dict] = []

    def fake_llm(prompt: str) -> str:
        # 도메인이 몇 개 언급됐는지로 "합산 검증" 호출인지 "도메인별 개별" 호출인지 구분한다.
        mentions = sum(1 for d in ("\"kr\"", "\"backtest\"") if d in prompt)
        calls.append({"mentions": mentions})
        if mentions == 2:
            return '{"valid": true, "reason": "kr은 종가, backtest는 10년 수익률을 각각 답해 합쳐서 질문을 만족함"}'
        # 도메인 하나만 놓고 보면(예전 버그 재현) 늘 불일치로 나온다고 가정.
        return '{"valid": false, "reason": "이 도메인 결과만으로는 질문 전체에 답하지 못함"}'

    verdict = verify_answer(
        "SK하이닉스 최근 10년간 10일선/20일선 교차 매매전략 수익률과 그래프 알려줘",
        domain_results, fake_llm,
    )

    assert verdict["valid"] is True
    assert len(calls) == 1  # 합산 검증 1회로 끝남 — 도메인별 개별 호출로 새지 않음
    assert all(v["valid"] for v in verdict["per_domain"].values())


def test_verify_answer_composite_domains_falls_back_to_per_domain_when_combined_check_fails():
    """합산 검증이 실제로 실패하면(진짜 문제) 기존처럼 도메인별로 세분화해 원인을 특정한다."""
    domain_results = {
        "kr": {"stock_code": "005930", "financial": {"value": 12.5}, "price": None, "errors": []},
        "us": {"stock_code": "WRONG", "financial": {"value": 999.0}, "price": None, "errors": []},
    }

    def fake_llm(prompt: str) -> str:
        mentions = sum(1 for d in ("\"kr\"", "\"us\"") if d in prompt)
        if mentions == 2:
            return '{"valid": false, "reason": "us 결과의 종목이 질문과 다름"}'
        if '"us"' in prompt:
            return '{"valid": false, "reason": "us: 종목 불일치"}'
        return '{"valid": true, "reason": "kr: 일치"}'

    verdict = verify_answer("삼성전자와 엔비디아 PER 비교", domain_results, fake_llm)

    assert verdict["valid"] is False
    assert verdict["per_domain"]["kr"]["valid"] is True
    assert verdict["per_domain"]["us"]["valid"] is False


def test_verify_answer_single_domain_question_calls_llm_exactly_once():
    """회귀 방지: 단일 도메인 질문은 합산 검증 도입 전과 동일하게 LLM 호출 1회만 한다."""
    calls: list[str] = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return "일치"

    domain_results = {"kr": {"stock_code": "005930", "financial": {"value": 12.5}}}
    verdict = verify_answer("삼성전자 PER 알려줘", domain_results, fake_llm)

    assert len(calls) == 1
    assert verdict["valid"] is True


def test_domain_has_data_recognizes_multi_entity_kr_result_with_price():
    """실제 버그 재현: entities 데이터가 있으면 llm_fn=None 결정론 경로에서 통과해야 한다."""
    domain_results = {
        "kr": {
            "stock_code": None, "stock_codes": ["005930", "000660"],
            "financial": None, "price": None,
            "entities": [
                {"stock_code": "005930", "financial": None,
                 "price": [{"close": 71000.0}], "errors": []},
                {"stock_code": "000660", "financial": None,
                 "price": [{"close": 210000.0}], "errors": []},
            ],
            "errors": [],
        }
    }
    verdict = verify_answer("삼성전자와 SK하이닉스 종가 알려줘", domain_results, llm_fn=None)
    assert verdict["valid"] is True


# ── data_asof(기간 미지정 시 실제 사용된 시점)가 종합결론/프롬프트까지 살아남는지 ──────
def test_synthesize_conclusion_deterministic_summary_mentions_data_asof():
    """LLM 없이도(결정론 요약) data_asof의 기준일/기준분기가 종합결론 텍스트에 나타난다."""
    domain_results = {
        "kr": {
            "stock_code": "005930", "intent": "financial",
            "financial": {"metric": "per", "value": 12.5, "period": "2025Q1"},
            "price": None,
            "data_asof": {"price_date": "2026-07-11", "financial_quarter": "2025Q1"},
            "errors": [],
        }
    }
    conclusion = synthesize_conclusion("삼성전자 PER 알려줘", domain_results, llm_fn=None)
    assert "2026-07-11" in conclusion
    assert "2025Q1" in conclusion


def test_synthesize_prompt_includes_data_asof_field():
    """합성 LLM 프롬프트에 data_asof가 축약 없이 그대로 전달된다(사용자가 답변에서 볼 수 있게)."""
    captured = {}

    def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "결론"

    domain_results = {
        "kr": {
            "stock_code": "005930", "intent": "financial",
            "financial": {"metric": "per", "value": 12.5, "period": "2025Q1"},
            "data_asof": {"price_date": "2026-07-11", "financial_quarter": "2025Q1"},
        }
    }
    synthesize_conclusion("삼성전자 PER 알려줘", domain_results, fake_llm)

    prompt = captured["prompt"]
    assert "data_asof" in prompt
    assert "2026-07-11" in prompt


def test_synthesize_conclusion_deterministic_summary_mentions_backtest_data_asof():
    """실서버 재현: '코스피 전종목 pbr/gpa 상관관계' 같은 backtest 도메인 질문은 result가
    집계값만 담아(시점 정보 없음) 사용자가 데이터 기준시점을 검증할 수 없었다. backtest도
    kr/us와 동일하게 data_asof가 있으면 결정론 요약에 그대로 드러나야 한다."""
    domain_results = {
        "backtest": {
            "blocked": False,
            "result": {"r": 0.24, "n": 656},
            "data_asof": {"price_date": "2026-07-15"},
        }
    }
    conclusion = synthesize_conclusion("코스피 pbr gpa 상관관계", domain_results, llm_fn=None)
    assert "2026-07-15" in conclusion


def test_synthesize_conclusion_deterministic_summary_mentions_backtest_financial_quarter():
    """실사용 리포트: "가격날짜는 있는데 재무데이터 시점은 안 나온다" — data_asof에
    financial_quarter가 있으면 결정론 요약(backtest 분기)에도 함께 나와야 한다."""
    domain_results = {
        "backtest": {
            "blocked": False,
            "result": {"r": 0.24, "n": 656},
            "data_asof": {"price_date": "2026-07-18", "financial_quarter": "2026Q1"},
        }
    }
    conclusion = synthesize_conclusion("코스피 pbr gpa 상관관계", domain_results, llm_fn=None)
    assert "2026-07-18" in conclusion
    assert "2026Q1" in conclusion
