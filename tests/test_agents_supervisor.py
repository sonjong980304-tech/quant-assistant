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


def test_route_question_fallback_without_llm_detects_fama_french_factor_as_macro():
    """README 설계상 파마프렌치 팩터 조회는 매크로 에이전트(domain_macro.py) 안에서 처리한다
    (실사용 버그: "SMB 팩터 최근 값 알려줘"가 엉뚱하게 kr로 라우팅돼 실패하던 것 재현)."""
    routes = route_question("SMB 팩터 최근 값 알려줘", None)
    assert "macro" in routes


def test_route_prompt_mentions_fama_french_for_llm_routing():
    """LLM 우선 라우팅 경로도 인식하도록 _route_prompt의 macro 설명에 파마프렌치가 포함돼야
    한다 — "매크로 신호/금리차"로만 설명되면 LLM도 이 도메인을 놓친다."""
    prompt = supervisor_mod._route_prompt("아무 질문")
    assert "파마프렌치" in prompt or "Fama-French" in prompt or "fama" in prompt.lower()


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


def test_route_prompt_mentions_single_indicator_screening_is_kr():
    """LLM 우선 라우팅 반례: 특정 지표 하나를 순서대로 나열/정렬해서 그래프로 보여달라는
    질문('PBR 오름차순 나열해서 그래프 그려줘')은 분포/히스토그램/상관관계 분석이 아니라
    단순 스크리닝이므로 kr이다 — _route_prompt가 이 반례를 명시해, LLM이 'PBR'·'그래프'
    단어만 보고 backtest로 오분류하지 않게 한다(실서버 재현: role=sql LLM이 ['backtest']로
    오분류했다). backtest 설명(분포/QVM 등)과 대칭으로 넣어 두 방향 구분이 모두 프롬프트에 있다."""
    prompt = supervisor_mod._route_prompt("아무 질문")
    assert "단순 스크리닝" in prompt
    assert "오름차순" in prompt


def test_route_question_fallback_without_llm_detects_bucket_grouping_as_backtest():
    """실사용 재현: "PER을 10구간으로 나눠서 각 구간의 평균을 구해줘"가 '분위수'/'5분위'
    같은 기존 키워드에 안 걸려 kr로 잘못 라우팅됐다(kr은 구간별 집계를 표현할 방법이
    없어 3회 재시도가 전부 헛돌았음). "구간"으로 나누는 집계 요청도 '분위수'와 동일하게
    backtest 전용(quantile_bucket_means)이므로 키워드로 잡아야 한다."""
    routes = route_question("코스피 전종목 PER을 오름차순으로 나열해서 10구간으로 나눈다음 각 구간의 평균을 구해줘", None)
    assert "backtest" in routes


def test_route_prompt_mentions_bucket_average_for_llm_routing():
    """LLM 우선 라우팅 경로도 인식하도록 _route_prompt의 backtest 설명에 '구간으로 나눠
    평균/집계'가 히스토그램·분포와 별개로 명시돼야 한다 — 기존 문구는 "구간으로 나눠
    히스토그램/분포를 그리는" 경우만 backtest로 못박아, 그림이 아니라 평균 계산을
    요청하면 LLM이 놓칠 위험이 있었다."""
    prompt = supervisor_mod._route_prompt("아무 질문")
    assert "구간별 평균" in prompt or "구간 평균" in prompt


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


def test_verify_answer_empty_llm_response_is_treated_as_unavailable_not_invalid():
    """web의 _build_llm_fn은 LLM 호출 실패(quota 소진 등)를 예외가 아니라 **빈 문자열**로
    전파한다(LLMClient.complete가 예외를 삼키고 text=""를 반환). 빈 응답을 '검증 실패'로
    보면 멀쩡한 데이터를 두고 3회 재시도를 전부 소모한 뒤 불확실 응답으로 끝난다
    (실사용 재현: OpenAI insufficient_quota 상태에서 "sk하이닉스 26년 1분기 영업이익"이
    'kr: 검증 응답을 해석하지 못함: ' 사유로 매 시도 실패). 예외와 동일하게 '검증 불가'로
    구분해 데이터 존재 확인만으로 통과시켜야 한다."""
    def empty_llm(prompt: str) -> str:
        return ""

    domain_results = {"kr": {"stock_code": "000660", "financial": {"value": 123.0}}}
    verdict = verify_answer("sk하이닉스 26년 1분기 영업이익", domain_results, empty_llm)

    assert verdict["valid"] is True
    assert verdict.get("verification_unavailable") is True
    assert "불가" in verdict["reason"]


def test_verify_answer_empty_llm_response_composite_domains_also_unavailable():
    """복합 도메인(2개 이상)의 합산 검증 경로에서도 빈 응답은 '검증 불가'로 통과해야 한다."""
    def empty_llm(prompt: str) -> str:
        return ""

    domain_results = {
        "kr": {"stock_code": "005930", "price": [{"close": 71000}]},
        "backtest": {"result": {"performance": {"total_return": 10.0}}},
    }
    verdict = verify_answer("삼성전자 골든크로스 백테스트", domain_results, empty_llm)

    assert verdict["valid"] is True
    assert verdict.get("verification_unavailable") is True


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


# ── top_n 특혜에는 상한(cap)이 있어야 한다. "코스피 전체"처럼 kr 에이전트가 top_n을
#    4000(사실상 무제한 매직넘버)으로 해석하면, 위 top_n 특혜가 KOSPI 전종목을 통째로
#    프롬프트에 넣어버려 verify/synthesize 프롬프트가 각각 약 77만자(31.7만 토큰)까지
#    불어나고 실제 gpt-5.5 API 호출로 약 $4.8가 소진되는 것까지 실측 확인됐다. 상한을
#    넘는 top_n은 상한개수까지만 보여주되, 상한 이내(예: 상위 10개/50개)는 기존처럼
#    하나도 안 잘리는 동작을 그대로 유지해야 한다(그게 이 top_n 특혜가 막던 원래 버그) ──
def test_truncate_for_prompt_caps_result_when_top_n_exceeds_cap():
    domain_results = {"top_n": 4000, "result": _big_screening_rows(900)}
    truncated = supervisor_mod._truncate_for_prompt(domain_results)

    cap = supervisor_mod._PROMPT_TOP_N_CAP
    result = truncated["result"]
    assert len(result) == cap + 1  # 상한개 행 + 요약 문자열 1개
    assert result[-1] == f"...(총 900개 중 {cap}개만 표시, 나머지 생략)"
    assert result[cap - 1]["name"] == f"종목{cap - 1}"  # 마지막으로 보존된 행은 상한 바로 앞


def test_truncate_for_prompt_keeps_full_list_when_top_n_within_cap():
    """top_n이 상한 이내면(예: 상위 10개) 지금처럼 하나도 안 잘리고 그대로 다 보여야
    한다 — 이게 원래 top_n 특혜 로직이 막던 버그(5개만 보여서 검증 LLM이 오판)이므로
    상한을 추가해도 절대 재발시키면 안 된다(회귀 방지)."""
    domain_results = {"top_n": 10, "result": _big_screening_rows(10)}
    truncated = supervisor_mod._truncate_for_prompt(domain_results)

    assert len(truncated["result"]) == 10
    assert truncated["result"][-1]["name"] == "종목9"


# ── free_exec 폴백(top_n 없는 결과)의 리스트도 흔한 구간분석 규모(10분위 등)는 안 잘려야
#    한다. 실측 회귀: "코스피 전종목 PER 10구간 평균" 요청이 kr 도메인 3회 실패 후
#    free_exec 폴백으로 정확히 10구간 평균을 계산했는데도, top_n 필드가 없어 기본
#    head(5)로 잘려 최종 답변 LLM이 "10개 중 5개만 있어 불완전하다"고 잘못 보고했다 ──
def test_truncate_for_prompt_keeps_free_exec_bucket_result_without_top_n():
    domain_results = {
        "free_exec": {"fallback_used": True, "sql": "...", "code": "...",
                       "result": _big_screening_rows(10)},
    }
    truncated = supervisor_mod._truncate_for_prompt(domain_results)

    assert len(truncated["free_exec"]["result"]) == 10
    assert truncated["free_exec"]["result"][-1]["name"] == "종목9"


def test_truncate_for_prompt_still_caps_list_without_top_n_beyond_new_head():
    """head를 30으로 올렸어도 무제한이 된 게 아니다 — top_n 없는 리스트가 새 상한(30)도
    넘으면 여전히 잘리고 요약문구가 붙어야 한다(원래 이 축약 로직이 막던 원본 리스트
    폭발 문제가 재발하지 않았음을 증명하는 회귀 방지)."""
    domain_results = {"free_exec": {"result": _big_screening_rows(31)}}
    truncated = supervisor_mod._truncate_for_prompt(domain_results)

    head = supervisor_mod._PROMPT_LIST_HEAD
    result = truncated["free_exec"]["result"]
    assert len(result) == head + 1  # head개 행 + 요약 문자열 1개
    assert result[-1] == f"...(총 31개 중 {head}개만 표시, 나머지 생략)"


_SCREENING_ROW_EXTRA_FIELDS = {
    "sector": "전기전자", "market": "KOSPI", "pbr": 1.2, "roe": 5.3, "eps": 512.0,
    "bps": 8123.0, "dividend_yield": 2.1, "market_cap": 1_234_567_890, "price": 71000,
    "volume": 1234567, "operating_margin": 10.5, "net_margin": 8.2, "debt_ratio": 45.3,
    "current_ratio": 152.1, "revenue": 1_000_000_000, "operating_income": 100_000_000,
    "net_income": 80_000_000, "total_assets": 5_000_000_000, "total_equity": 2_000_000_000,
    "shares_outstanding": 10_000_000, "foreign_ratio": 30.2, "beta": 1.05,
    "week52_high": 15000, "week52_low": 8000, "avg_volume_20d": 900000,
    "listing_date": "2000-01-01", "industry_code": "26", "quarter": "2025Q1",
}


def _huge_screening_rows(n: int) -> list[dict]:
    """실측 버그(약 900개 x 약 30필드)를 흉내내는 필드 많은 스크리닝 행 fixture."""
    return [
        {"stock_code": f"{i:06d}", "name": f"종목{i}", **_SCREENING_ROW_EXTRA_FIELDS}
        for i in range(n)
    ]


def test_verify_and_synthesize_prompt_length_drastically_reduced_for_full_market_screening():
    """'코스피 전체 종목을 pbr 오름차순으로' 같은 질문에서 kr 에이전트가 '전체'를
    top_n=4000으로 해석해도, verify/synthesize 프롬프트는 상한 덕분에 대폭 줄어야 한다
    (수정 전 실측: 약 77만자/31.7만 토큰 -> 수정 후: 10만자 이하)."""
    domain_results = {"kr": {"intent": "screening", "top_n": 4000, "result": _huge_screening_rows(900)}}

    verify_prompt = supervisor_mod._verify_prompt("코스피 전체 종목 pbr 오름차순", domain_results)
    synth_prompt = supervisor_mod._synthesize_prompt("코스피 전체 종목 pbr 오름차순", domain_results)

    assert len(verify_prompt) < 100_000
    assert len(synth_prompt) < 100_000


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


# ── AC3: answer_with_verification — 정확히 max_retries회 재시도 후 uncertain(무한루프 없음) ──

def test_answer_with_verification_retries_exactly_three_times_then_uncertain():
    """max_retries=3을 명시하면 정확히 3회만 시도(4번째 없음) → uncertain True, attempts 3.

    기본값 자체는 2로 바뀌었으므로(아래 default 전용 테스트가 검증), 이 테스트는 "정확히
    N회, N+1은 없다"는 상한 메커니즘 자체를 기본값과 무관하게 계속 검증한다."""
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
        "삼성전자 PER", conn=None, llm_fn=None, max_retries=3,
        route_fn=stub_route, dispatch_fn=stub_dispatch,
        verify_fn=always_invalid_verify, synthesize_fn=stub_synth,
    )

    # 정형 3회 재시도 + backtest 추가시도 1회(routes에 backtest가 없었으므로) = 4회.
    # backtest 추가시도도 always_invalid_verify라 실패하므로 그다음 free_exec 폴백으로
    # 넘어간다(fallback_fn 미주입 시 기본 run_free_exec_fallback, llm_fn=None이라 즉시 실패).
    assert len(verify_calls) == 4
    assert len(dispatch_calls) == 4  # 5번째 시도 없음(무한루프 없음)
    assert synth_calls == []          # 실패 경로에서는 종합결론을 만들지 않음
    assert res["uncertain"] is True
    assert res["attempts"] == 3       # attempts는 정형 재시도 루프만 센다(추가시도는 별도)
    assert res["reason"]


def test_answer_with_verification_default_max_retries_is_two():
    """max_retries를 안 넘기면 기본값은 3이 아니라 2다 — kr 같은 도메인은 표현력 자체가
    없어 같은 질문을 3번 반복해도 결과가 안 바뀌는 경우가 많아, 정형 재시도는 2회로
    줄이고 그 대신 backtest 추가시도/free_exec 폴백으로 더 빨리 넘어가는 게 낫다는
    사용자 판단(실사용: 3회 재시도가 매번 같은 틀린 답만 반복하는 걸 확인)."""
    verify_calls: list[int] = []

    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {}}

    def always_invalid_verify(question, domain_results, llm_fn):
        verify_calls.append(1)
        return {"valid": False, "reason": "실패"}

    res = answer_with_verification(
        "q", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=always_invalid_verify,
    )
    # 정형 2회 + backtest 추가시도 1회 = 3회.
    assert len(verify_calls) == 3
    assert res["attempts"] == 2


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
    # 정형 2회 + backtest 추가시도 1회(routes=["kr"]이라 backtest가 없었음) = 3회.
    assert len(verify_calls) == 3
    assert res["attempts"] == 2  # attempts는 정형 재시도 루프만 센다
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
    assert res["attempts"] == 2  # 기본 max_retries=2
    assert "free_exec" not in res["domain_results"]
    assert "SQL도 못 만듦" in res["reason"]


# ── backtest 추가시도(escalation): 정형 재시도(기본 2회) 실패 + 원래 라우팅에 backtest가
#    없었으면, free_exec으로 넘어가기 전에 backtest를 1회 더 시도한다. kr처럼 "구간별
#    집계" 같은 표현력이 없는 도메인은 피드백을 아무리 줘도 같은 종류의 답만 반복하므로,
#    라우팅이 놓친 경우를 위한 안전망이다(실사용 재현: "PER 10구간 평균"이 kr로 잘못
#    라우팅됨 — 정형 3회가 매번 같은 틀린 답만 반복해 재시도 자체가 무의미하다고 판단,
#    max_retries 기본값도 3→2로 줄였다). ──
def test_answer_with_verification_escalates_to_backtest_when_retries_exhausted():
    dispatch_calls: list[list[str]] = []
    fallback_calls: list[int] = []

    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        dispatch_calls.append(list(routes))
        if "backtest" in routes:
            return {"backtest": {"result": [{"bucket": 1, "mean_per": 5.0}]}}
        return {"kr": {"result": [{"stock_code": "005930"}]}}

    def verify_fn(question, domain_results, llm_fn):
        if "backtest" in domain_results:
            return {"valid": True}
        return {"valid": False, "reason": "kr 결과가 질문과 안 맞음(구간 집계 미지원)"}

    def stub_synth(question, domain_results, llm_fn):
        return "종합결론"

    def fake_fallback(question, conn, llm_fn, last_reason=None, **kwargs):
        fallback_calls.append(1)
        return {"ok": True, "result": "이건 쓰이면 안 됨"}

    res = answer_with_verification(
        "PER 10구간 평균", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch,
        verify_fn=verify_fn, synthesize_fn=stub_synth,
        fallback_fn=fake_fallback,
    )

    # 정형 2회(전부 kr) + backtest 추가시도 1회 = 3회. free_exec 폴백은 호출조차 안 된다
    # (backtest 추가시도가 성공했으므로 최후 수단까지 갈 필요가 없다).
    assert dispatch_calls == [["kr"], ["kr"], ["backtest"]]
    assert fallback_calls == []
    assert res["uncertain"] is False
    assert "backtest" in res["routes"]
    assert res["domain_results"]["backtest"]["result"] == [{"bucket": 1, "mean_per": 5.0}]
    assert res["used_backtest_escalation"] is True
    assert res["conclusion"] == "종합결론"


def test_answer_with_verification_skips_backtest_escalation_when_already_routed():
    """원래 라우팅에 이미 backtest가 포함돼 있었다면(예: kr+backtest 복합 질문), 추가시도로
    중복 호출하지 않고 곧장 기존 free_exec 폴백으로 넘어가야 한다(낭비 방지)."""
    dispatch_calls: list[list[str]] = []

    def stub_route(question, llm_fn):
        return ["kr", "backtest"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        dispatch_calls.append(list(routes))
        return {"kr": {}, "backtest": {}}

    def always_invalid_verify(question, domain_results, llm_fn):
        return {"valid": False, "reason": "실패"}

    def fake_fallback(question, conn, llm_fn, last_reason=None, **kwargs):
        return {"ok": False, "result": None, "sql": None, "code": None, "error": "실패"}

    res = answer_with_verification(
        "kr+backtest 복합 질문", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch,
        verify_fn=always_invalid_verify, fallback_fn=fake_fallback,
    )

    assert len(dispatch_calls) == 2  # 정형 2회뿐, backtest 추가시도 없음(이미 routes에 있었음)
    assert res["uncertain"] is True


# ── free_exec 폴백 재검증 안전망: 폴백은 "정형 검증이 반복 실패한 뒤의 최후 수단"이라
#    검증을 다시 거치지 않던 기존 설계에는 구멍이 있었다(실사용 재현: "PER z-score
#    히스토그램" 요청이 z-score 없이 원본 PER로만 나온 결과가 검증 없이 그대로 나감).
#    무한루프를 피하려고 폴백을 다시 시도하지는 않되(fallback_fn은 여전히 정확히 1회),
#    성공한 폴백 결과를 한 번 더 verify_fn에 통과시켜 실패하면 그 사실을
#    domain_results["free_exec"]["verification_warning"]에 남긴다 — 답 자체는 폐기하지
#    않고(최후 수단이라 대안이 없음) synthesize_fn이 최종 답변에 신뢰도 유보 문구를
#    붙일 수 있게 근거만 제공한다. ──────────────────────────────────────────────
def test_answer_with_verification_flags_fallback_result_when_reverification_fails():
    fallback_calls: list = []

    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {"attempt": "x"}}

    def verify_fn(question, domain_results, llm_fn):
        if "free_exec" in domain_results:
            return {"valid": False, "reason": "z-score 필드가 결과에 없습니다"}
        return {"valid": False, "reason": "정형 검증 실패"}

    def stub_synth(question, domain_results, llm_fn):
        return "종합결론"

    def fake_fallback(question, conn, llm_fn, last_reason=None, **kwargs):
        fallback_calls.append(1)
        return {"ok": True, "result": {"per": [1, 2, 3]}, "sql": "SELECT 1;", "code": "result=1"}

    res = answer_with_verification(
        "PER z-score 히스토그램", conn="conn", llm_fn="llm",
        route_fn=stub_route, dispatch_fn=stub_dispatch,
        verify_fn=verify_fn, synthesize_fn=stub_synth,
        fallback_fn=fake_fallback,
    )

    assert len(fallback_calls) == 1  # 재검증 실패해도 폴백을 다시 시도하지 않는다(무한루프 방지)
    assert res["uncertain"] is False  # 답 자체는 폐기하지 않는다(최후 수단이라 대안이 없음)
    assert res["used_fallback"] is True
    assert res["domain_results"]["free_exec"]["verification_warning"] == "z-score 필드가 결과에 없습니다"


def test_answer_with_verification_does_not_flag_fallback_result_when_reverification_passes():
    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {"attempt": "x"}}

    def verify_fn(question, domain_results, llm_fn):
        if "free_exec" in domain_results:
            return {"valid": True}
        return {"valid": False, "reason": "정형 검증 실패"}

    def stub_synth(question, domain_results, llm_fn):
        return "종합결론"

    def fake_fallback(question, conn, llm_fn, last_reason=None, **kwargs):
        return {"ok": True, "result": {"per": [1, 2, 3]}, "sql": "SELECT 1;", "code": "result=1"}

    res = answer_with_verification(
        "질문", conn="conn", llm_fn="llm",
        route_fn=stub_route, dispatch_fn=stub_dispatch,
        verify_fn=verify_fn, synthesize_fn=stub_synth,
        fallback_fn=fake_fallback,
    )

    assert res["uncertain"] is False
    assert "verification_warning" not in res["domain_results"]["free_exec"]


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


def test_synthesize_prompt_explains_free_exec_verification_warning():
    """free_exec.verification_warning이 domain_results에 그대로 병기되는 것만으로는 LLM이
    최종 답변에 신뢰도 유보 문구를 붙이지 않는다 — 이 필드가 무슨 뜻인지 알려주는 안내
    문구가 프롬프트에 있어야 한다(재검증 안전망이 실제로 사용자에게 전달되는지 확인)."""
    captured = {}

    def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "결론"

    domain_results = {
        "free_exec": {
            "fallback_used": True,
            "result": {"per": [1, 2, 3]},
            "verification_warning": "z-score 필드가 결과에 없습니다",
        },
    }
    synthesize_conclusion("PER z-score 히스토그램", domain_results, fake_llm)
    assert "verification_warning" in captured["prompt"]
    assert "재검증" in captured["prompt"] or "자동검증" in captured["prompt"]


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

    # 정형 3회 + backtest 추가시도 1회(routes=["kr"]이라 backtest가 없었음)의 검증 실패
    # 이벤트가 모두 나와야 한다.
    verify_events = [s for step, s in events if step == "verify"]
    assert len(verify_events) == 4
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
        "저PER 10개", conn=None, llm_fn=None, max_retries=3,
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


# ── kr 다중지표(multi-metric) 결과의 데이터 존재 판정 ──────────────────────────────
# "현대차 PER PBR PSR"처럼 한 종목의 여러 지표를 조회하면 최상위 financial은 None이고
# 실제 데이터는 metrics 리스트에 지표별로 담긴다(다중분기/다중종목과 동일 관례).
# _domain_has_data가 이 구조도 인식해야 verify_answer가 정상 진행한다.

def test_domain_has_data_recognizes_multi_metric_kr_result():
    kr_result = {
        "stock_code": "005380", "question": "...", "intent": "financial",
        "financial": None, "price": None,
        "metrics": [
            {"metric": "per", "financial": {"value": 12.5}, "errors": []},
            {"metric": "pbr", "financial": {"value": 1.8}, "errors": []},
        ],
        "errors": [],
    }
    assert supervisor_mod._domain_has_data(kr_result) is True


def test_domain_has_data_returns_false_when_all_metrics_empty():
    kr_result = {
        "stock_code": "005380", "question": "...", "intent": "financial",
        "financial": None, "price": None,
        "metrics": [
            {"metric": "per", "financial": None, "errors": ["재무데이터 조회 실패"]},
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


def test_verify_prompt_explains_domains_need_not_each_fully_answer_alone():
    """실서버 재현 버그: "코스피 전종목 PER을 z-score로 바꿔 10분위 히스토그램" 질문에서
    backtest 도메인이 이미 z-score/분위/그래프까지 완전한 답을 냈는데도, 곁다리로 함께
    라우팅된 kr(단순 정렬 목록만 가능)이 "혼자서는 질문 전체를 못 채운다"는 이유로 검증
    LLM이 전체를 invalid로 오판했다 — 여러 도메인이 있을 때 "종합해서 이미 충족되면
    된다"는 지침이 프롬프트에 없었기 때문이다. 이 지침을 명시해야 한다."""
    prompt = supervisor_mod._verify_prompt(
        "코스피 전종목 PER z-score 10분위 히스토그램",
        {"kr": {}, "backtest": {}},
    )
    assert "종합" in prompt or "합쳐" in prompt
    assert "개별적으로" in prompt or "혼자" in prompt


def test_verify_answer_composite_domains_pass_when_one_domain_already_fully_answers():
    """backtest가 이미 z-score/분위/그래프를 다 계산했으면, kr이 단순 목록만 추가로
    제공해도 합산 검증에서 valid=true여야 한다(실서버 재현: 기존엔 이 경우 개별판정
    AND 로직 때문에 kr이 "혼자서는 부족하다"고 잡혀 전체가 무효 처리됐다)."""
    domain_results = {
        "kr": {"result": [{"stock_code": "005930", "per": 10.0}]},
        "backtest": {"blocked": False, "error": None,
                     "result": {"hist": {"field": "per_neutral", "counts": [1, 2, 3]}}},
    }
    calls: list[str] = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if "종합" in prompt or "합쳐" in prompt:
            return '{"valid": true, "reason": "backtest에 이미 z-score/히스토그램이 있어 충족됨"}'
        return '{"valid": false, "reason": "이 도메인 결과 혼자만으로는 부족"}'

    verdict = verify_answer(
        "코스피 전종목 PER z-score 10분위 히스토그램", domain_results, fake_llm,
    )

    assert verdict["valid"] is True
    assert len(calls) == 1  # 합산 검증 1회로 끝남 — 도메인별 개별 호출로 안 샌다


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


def test_verify_answer_accepts_multi_entity_kr_result_deterministic():
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
