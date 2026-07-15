"""스크리닝(다중종목 랭킹) 도메인 경로 단위테스트 (HA-15).

배경: 신규 계층형 도메인 에이전트(answer_kr_question/answer_us_question)는 단일종목
조회 지향이라 "PER 낮은 5개사" 같은 랭킹/스크리닝 질문에 답하지 못했다(goldset PER
랭킹 0/3 회귀). 이 스토리는 기존 백테스트 크로스섹션 인프라(get_cross_section/combine)를
재사용해 스크리닝 경로를 추가한다.

검증 대상:
- is_screening_question: 결정론적 키워드로 스크리닝 의도를 감지(단일종목 질문 오탐 없음).
- answer_kr_screening / answer_us_screening: 구조화 JSON(criteria/top_n/...)만 LLM에게
  생성시키고, 파싱 → get_cross_section → combine 호출 → rows 를 가공 없이 반환한다.
- 필드 환각(존재하지 않는 지표명)은 조용한 빈 결과가 아니라 명시적 오류로 남는다.
- answer_kr_question/answer_us_question 이 스크리닝 의도면 단일종목 경로 대신 분기한다.
"""
from __future__ import annotations

import json

import src.agents.domain_kr as kr
import src.agents.domain_us as us
from src.agents.domain_kr import answer_kr_screening, is_screening_question
from src.agents.domain_us import answer_us_screening


# 크로스섹션 대체용 가짜 유니버스 행(실제 metrics_at 스키마의 부분집합).
def _fake_rows(_conn, _asof):
    return [
        {"stock_code": "A", "name": "저PER가", "sector": "반도체", "market": "KOSPI", "per": 5.0, "roe": 8.0},
        {"stock_code": "B", "name": "중간나", "sector": "반도체", "market": "KOSPI", "per": 10.0, "roe": 12.0},
        {"stock_code": "C", "name": "고PER다", "sector": "화학", "market": "KOSDAQ", "per": 30.0, "roe": 20.0},
    ]


def _json_llm(payload: dict):
    def _fn(_prompt: str) -> str:
        return json.dumps(payload)
    return _fn


# ── is_screening_question ────────────────────────────────────────────────────
def test_is_screening_question_true_for_goldset_ranking_questions():
    assert is_screening_question("PER이 가장 낮은 10개 회사를 알려줘")
    assert is_screening_question("PER이 가장 높은 5개 회사")
    assert is_screening_question("PER이 낮은 상위 3개 종목")


def test_is_screening_question_true_for_strong_rank_words_without_count():
    assert is_screening_question("저PER 종목 순위 매겨줘")
    assert is_screening_question("우량주 좀 골라줘")


def test_is_screening_question_false_for_single_stock_questions():
    assert not is_screening_question("삼성전자 PER 알려줘")
    assert not is_screening_question("삼성전자 PER이랑 주가 같이 알려줘")
    assert not is_screening_question("005930 주가 알려줘")
    assert not is_screening_question("AAPL PER 알려줘")
    assert not is_screening_question("애플 주가 알려줘")


# ── is_screening_question: LLM 우선 판단 (키워드 목록에 없는 표현 인식, HA15 후속) ──────────
# 배경: "저PER 5종목" 같은 표현은 _SCREENING_DIRECTION_WORDS(낮은/높은/…)에 정확히 매치되는
# 단어가 없어("저PER"는 "낮은"이 아니다) 결정론 키워드 휴리스틱이 스크리닝으로 인식하지 못했다
# (실서버 curl로 재현됨). 사용자 지시: 키워드를 더 추가하는 대신 route_question/classify_intent와
# 동일한 "LLM 우선 + 키워드 안전망 폴백" 패턴으로 고친다.

def test_is_screening_question_llm_first_recognizes_phrasing_missing_from_keyword_list():
    """키워드 휴리스틱은 놓치지만(아래 회귀 테스트로 별도 확인) LLM 판단(yes)이 있으면 인식한다."""
    yes_llm = lambda _prompt: "yes"
    assert is_screening_question("저PER 5종목", llm_fn=yes_llm)
    assert is_screening_question("저평가된 5개 추천", llm_fn=yes_llm)


def test_is_screening_question_llm_prompt_includes_question_text():
    seen = []

    def spy_llm(prompt: str) -> str:
        seen.append(prompt)
        return "yes"

    is_screening_question("저PER 5종목", llm_fn=spy_llm)
    assert seen
    assert "저PER 5종목" in seen[0]


def test_is_screening_question_llm_first_overrides_keyword_heuristic():
    """LLM 우선순위: 파싱에 성공하면 키워드 판단과 달라도 LLM 판단을 채택한다
    (classify_intent 와 동일한 우선순위 원칙)."""
    no_llm = lambda _prompt: "no"
    # 키워드 휴리스틱이라면 True(순위 강한단어 "순위")였을 질문이지만 LLM이 no라고 답하면 그걸 따른다.
    assert is_screening_question("저PER 종목 순위 매겨줘", llm_fn=no_llm) is False


def test_is_screening_question_llm_unparseable_falls_back_to_keyword_heuristic():
    """LLM 응답에서 yes/no 를 못 뽑으면(파싱 실패) 기존 키워드 휴리스틱으로 폴백한다."""
    unclear_llm = lambda _prompt: "글쎄요, 판단하기 어렵습니다"
    # 키워드 휴리스틱상 True인 질문 → 폴백 후 True 유지
    assert is_screening_question("PER이 가장 낮은 10개 회사를 알려줘", llm_fn=unclear_llm)
    # 키워드 휴리스틱상 False인 단일종목 질문 → 폴백 후 False 유지(오탐 없음)
    assert not is_screening_question("삼성전자 PER 알려줘", llm_fn=unclear_llm)


def test_is_screening_question_llm_exception_falls_back_to_keyword_heuristic():
    """llm_fn 이 예외를 던져도 전파하지 않고 키워드 휴리스틱으로 안전 폴백한다."""

    def boom(_prompt: str) -> str:
        raise RuntimeError("LLM 다운")

    assert is_screening_question("PER이 가장 낮은 10개 회사를 알려줘", llm_fn=boom)
    assert not is_screening_question("삼성전자 PER 알려줘", llm_fn=boom)


def test_is_screening_question_without_llm_fn_keyword_heuristic_unchanged():
    """회귀: llm_fn 미지정 시 기존 키워드 휴리스틱 그대로 동작(동작 변화 없음).

    '저PER 5종목' 은 llm_fn 없이는 여전히 인식되지 않는다(키워드 목록에 없음) — 이 한계 자체는
    이번 변경의 대상이 아니라, llm_fn 이 주어졌을 때만 LLM 판단으로 보완된다.
    """
    assert is_screening_question("PER이 가장 낮은 10개 회사를 알려줘")
    assert is_screening_question("저PER 종목 순위 매겨줘")
    assert not is_screening_question("삼성전자 PER 알려줘")
    assert not is_screening_question("저PER 5종목")


# ── answer_kr_question/answer_us_question: llm_fn 이 스크리닝 의도 판단까지 관통 배선 ──────
def test_answer_kr_question_routes_low_per_phrasing_via_llm_screening_intent(monkeypatch):
    """키워드 목록에 없는 '저PER' 표현도 llm_fn 이 있으면 스크리닝 경로로 들어간다(배선 확인)."""
    sentinel = {"intent": "screening", "result": [{"name": "x"}], "errors": []}
    called = []

    def spy_screening(question, conn, **kwargs):
        called.append(question)
        return sentinel

    monkeypatch.setattr(kr, "answer_kr_screening", spy_screening)

    result = kr.answer_kr_question("저PER 5종목", conn=None, llm_fn=lambda _p: "yes")
    assert result is sentinel
    assert called == ["저PER 5종목"]


def test_answer_us_question_routes_low_per_phrasing_via_llm_screening_intent(monkeypatch):
    sentinel = {"intent": "screening", "result": [{"name": "x"}], "errors": []}
    called = []

    def spy_screening(question, conn, **kwargs):
        called.append(question)
        return sentinel

    monkeypatch.setattr(us, "answer_us_screening", spy_screening)

    result = us.answer_us_question("저PER 5종목", conn=None, llm_fn=lambda _p: "yes")
    assert result is sentinel
    assert called == ["저PER 5종목"]


# ── answer_kr_screening: 정상 경로(LLM JSON) ─────────────────────────────────
def test_answer_kr_screening_llm_json_returns_ranked_rows():
    llm = _json_llm({"criteria": [{"key": "per", "direction": "low"}], "top_n": 2})
    result = answer_kr_screening(
        "PER이 가장 낮은 2개 회사", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["intent"] == "screening"
    assert result["errors"] == []
    assert result["result"] is not None
    names = [r["name"] for r in result["result"]]
    assert names == ["저PER가", "중간나"]  # per 오름차순 상위 2개


# ── answer_kr_screening: 결정론 휴리스틱 폴백(llm_fn 없음) ────────────────────
def test_answer_kr_screening_heuristic_fallback_without_llm():
    result = answer_kr_screening(
        "PER이 가장 낮은 2개 회사", conn=None, llm_fn=None,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["errors"] == []
    assert result["result"] is not None
    assert [r["name"] for r in result["result"]] == ["저PER가", "중간나"]


# ── answer_kr_screening: 필드 환각은 조용한 빈 결과가 아니라 명시적 오류 ──────
def test_answer_kr_screening_hallucinated_field_reports_explicit_error():
    llm = _json_llm({"criteria": [{"key": "forward_12m_return", "direction": "low"}], "top_n": 3})
    result = answer_kr_screening(
        "미래수익률 가장 낮은 3개 회사", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["result"] is None  # 조용한 빈 리스트가 아니라 None(데이터 없음)
    assert result["errors"]
    assert any("forward_12m_return" in e for e in result["errors"])


# ── answer_kr_screening: 조건 해석 실패(파싱 실패 + 휴리스틱도 실패) ──────────
def test_answer_kr_screening_unparseable_and_no_metric_reports_error():
    llm = _json_llm({"unexpected": "shape"})  # criteria 없음 → 파싱 실패
    result = answer_kr_screening(
        "그냥 좋은 거 뽑아줘", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["result"] is None
    assert result["errors"]


# ── answer_kr_screening: asof 조회 실패 시 명시적 오류 ───────────────────────
def test_answer_kr_screening_missing_asof_reports_error():
    llm = _json_llm({"criteria": [{"key": "per", "direction": "low"}], "top_n": 2})

    def empty_exec(_sql, _conn):
        return {"ok": True, "rows": [{"d": None}]}

    result = answer_kr_screening(
        "PER 낮은 2개", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, execute_sql_fn=empty_exec,
    )
    assert result["result"] is None
    assert result["errors"]


# ── answer_kr_screening: 업종(sector) 필터가 실제 DB 값과 안 맞으면 조용한 빈 결과가
#    아니라 명시적 오류(+실제 유효 업종 목록)로 남긴다 (KRX 분류엔 "반도체" 카테고리가
#    없고 "전기·전자"로 흡수돼 있음 — 실사용에서 발견된 회귀) ───────────────────────
def test_answer_kr_screening_unmatched_sector_reports_explicit_error():
    llm = _json_llm({"criteria": [{"key": "per", "direction": "low"}], "top_n": 5, "sectors": ["게임"]})
    result = answer_kr_screening(
        "게임 업종에서 PER 낮은 5개", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["result"] is None  # 조용한 빈 리스트가 아니라 None(데이터 없음)
    assert result["errors"]
    assert any("게임" in e for e in result["errors"])
    assert any("반도체" in e and "화학" in e for e in result["errors"])  # 실제 유효 업종 안내


def test_answer_kr_screening_matched_sector_filters_correctly():
    llm = _json_llm({"criteria": [{"key": "per", "direction": "low"}], "top_n": 5, "sectors": ["반도체"]})
    result = answer_kr_screening(
        "반도체 업종에서 PER 낮은 5개", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["errors"] == []
    assert result["result"] is not None
    assert [r["name"] for r in result["result"]] == ["저PER가", "중간나"]  # 화학(고PER다) 제외


def test_screening_prompt_includes_valid_sector_list_when_available():
    seen_prompts = []

    def spy_llm(prompt: str) -> str:
        seen_prompts.append(prompt)
        return json.dumps({"criteria": [{"key": "per", "direction": "low"}], "top_n": 5})

    answer_kr_screening(
        "반도체 업종에서 PER 낮은 5개", conn=None, llm_fn=spy_llm,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert seen_prompts
    assert "반도체" in seen_prompts[0] and "화학" in seen_prompts[0]


# ── answer_kr_question 이 스크리닝 질문을 스크리닝 경로로 분기 ────────────────
def test_answer_kr_question_routes_screening_question_to_screening(monkeypatch):
    sentinel = {"intent": "screening", "result": [{"name": "x"}], "errors": []}
    called = []

    def spy_screening(question, conn, **kwargs):
        called.append(question)
        return sentinel

    monkeypatch.setattr(kr, "answer_kr_screening", spy_screening)
    # find_stock_code 가 호출되면 스크리닝으로 분기하지 않은 것 → 감시
    fsc_calls = []
    real_fsc = kr.find_stock_code
    monkeypatch.setattr(
        kr, "find_stock_code",
        lambda *a, **k: fsc_calls.append(1) or real_fsc(*a, **k),
    )

    result = kr.answer_kr_question("PER이 가장 낮은 5개 회사", conn=None)
    assert result is sentinel
    assert called == ["PER이 가장 낮은 5개 회사"]
    assert fsc_calls == []  # 단일종목 경로(find_stock_code)로 새지 않음


def test_answer_kr_question_non_screening_does_not_use_screening_path(monkeypatch):
    called = []
    monkeypatch.setattr(
        kr, "answer_kr_screening",
        lambda *a, **k: called.append(1) or {"intent": "screening"},
    )
    # 종목을 못 찾게 하고(빈 DB 없이) find_stock_code/find_stock_codes 를 종목없음으로 스텁
    # (다중종목 경로가 먼저 find_stock_codes 로 개수를 확인하므로 이것도 함께 스텁해야
    # conn=None 인 이 테스트에서 실제 DB 조회가 시도되지 않는다)
    monkeypatch.setattr(kr, "find_stock_code", lambda *a, **k: None)
    monkeypatch.setattr(kr, "find_stock_codes", lambda *a, **k: [])
    result = kr.answer_kr_question("삼성전자 PER 알려줘", conn=None)
    assert called == []  # 스크리닝 경로 미사용
    assert result["intent"] != "screening"


# ── US 스크리닝 ──────────────────────────────────────────────────────────────
def test_answer_us_screening_llm_json_returns_ranked_rows():
    llm = _json_llm({"criteria": [{"key": "per", "direction": "high"}], "top_n": 2})
    result = answer_us_screening(
        "PER 가장 높은 2개 회사", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["intent"] == "screening"
    assert result["errors"] == []
    assert [r["name"] for r in result["result"]] == ["고PER다", "중간나"]  # per 내림차순 상위 2개


def test_answer_us_screening_hallucinated_field_reports_explicit_error():
    llm = _json_llm({"criteria": [{"key": "psr", "direction": "low"}], "top_n": 3})
    result = answer_us_screening(
        "PSR 낮은 3개", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["result"] is None
    assert result["errors"]


def test_answer_us_question_routes_screening_question_to_screening(monkeypatch):
    sentinel = {"intent": "screening", "result": [{"name": "x"}], "errors": []}
    called = []
    monkeypatch.setattr(
        us, "answer_us_screening",
        lambda question, conn, **kwargs: called.append(question) or sentinel,
    )
    tk_calls = []
    monkeypatch.setattr(us, "resolve_ticker_us", lambda *a, **k: tk_calls.append(1) or None)

    result = us.answer_us_question("PER이 가장 낮은 5개 회사", conn=None)
    assert result is sentinel
    assert called == ["PER이 가장 낮은 5개 회사"]
    assert tk_calls == []  # 단일종목(resolve_ticker_us) 경로로 새지 않음


# ── 미국 시장(거래소) 필터: 도메인별 스펙 분리 + exchanges 필터 (HA15 혼재질문 버그) ──────
# 실사용 재현 버그: "코스피와 나스닥 각각 PER 낮은 5종목씩" 처럼 한 문장에 한국+미국 시장을
# 동시에 언급하면, KR/US 두 호출이 같은 원본 텍스트를 같은 프롬프트로 받아 US 쪽 spec 의
# markets 가 "코스피"에 오염돼(=["KOSPI"]) US 결과(exchange 값만 있는 rows)가 깨졌다.
# 아래 테스트는 (1) 도메인별 프롬프트 스코프 분리, (2) US exchanges 필터 실제 적용,
# (3) 단일시장/미지정 회귀 무결을 가짜 llm_fn(DI)로 실제 LLM 없이 검증한다.

# US 유니버스 대체 행: metrics_at_us 는 exchange 값을 'market' 필드에 담는다(data_access_us.py).
def _fake_us_rows(_conn, _asof):
    return [
        {"stock_code": "NAS1", "name": "나스닥가", "sector": "Technology", "market": "NASDAQ", "per": 5.0, "roe": 8.0},
        {"stock_code": "NAS2", "name": "나스닥나", "sector": "Technology", "market": "NASDAQ", "per": 8.0, "roe": 10.0},
        {"stock_code": "NYS1", "name": "뉴욕가", "sector": "Financials", "market": "NYSE", "per": 6.0, "roe": 9.0},
    ]


def _smart_scope_llm(prompt: str) -> str:
    """도메인 스코프 지시를 따르는 LLM 흉내: US 프롬프트면 exchanges, KR 프롬프트면 markets 반환.

    혼재 질문에서도 프롬프트가 '너는 이 도메인만 담당'을 명시하므로, 실제 LLM이 스코프를
    지켜 각자 자기 시장만 채운다는 가정을 재현한다(오염 없음을 배선 레벨에서 검증).
    """
    if "미국 시장" in prompt:
        return json.dumps(
            {"criteria": [{"key": "per", "direction": "low"}], "top_n": 5, "exchanges": ["NASDAQ"]}
        )
    return json.dumps(
        {"criteria": [{"key": "per", "direction": "low"}], "top_n": 5, "markets": ["KOSPI"]}
    )


def test_mixed_question_kr_and_us_specs_do_not_cross_contaminate():
    q = "코스피와 나스닥 각각 PER 낮은 주식 5종목씩 알려줘"
    kr_res = answer_kr_screening(
        q, conn=None, llm_fn=_smart_scope_llm, cross_section_fn=_fake_rows, asof="2026-07-14"
    )
    us_res = answer_us_screening(
        q, conn=None, llm_fn=_smart_scope_llm, cross_section_fn=_fake_us_rows, asof="2026-07-14"
    )
    # KR 호출: markets=["KOSPI"], exchanges 오염 없음, 결과는 KOSPI 종목만
    assert kr_res["markets"] == ["KOSPI"]
    assert kr_res["exchanges"] is None
    assert kr_res["errors"] == []
    assert all(r["market"] == "KOSPI" for r in kr_res["result"])
    # US 호출: exchanges=["NASDAQ"], markets 오염 없음, 결과는 NASDAQ 종목만(NYSE 제외)
    assert us_res["exchanges"] == ["NASDAQ"]
    assert us_res["markets"] is None
    assert us_res["errors"] == []
    assert all(r["market"] == "NASDAQ" for r in us_res["result"])


def test_answer_us_screening_exchanges_filter_excludes_other_exchanges():
    llm = _json_llm(
        {"criteria": [{"key": "per", "direction": "low"}], "top_n": 5, "exchanges": ["NASDAQ"]}
    )
    result = answer_us_screening(
        "나스닥 저PER 5종목", conn=None, llm_fn=llm,
        cross_section_fn=_fake_us_rows, asof="2026-07-14",
    )
    assert result["errors"] == []
    assert result["exchanges"] == ["NASDAQ"]
    assert [r["name"] for r in result["result"]] == ["나스닥가", "나스닥나"]  # NYSE(뉴욕가) 제외


def test_answer_us_screening_no_exchange_returns_all_exchanges():
    llm = _json_llm({"criteria": [{"key": "per", "direction": "low"}], "top_n": 5})  # exchanges 미지정
    result = answer_us_screening(
        "미국 저PER 5종목", conn=None, llm_fn=llm,
        cross_section_fn=_fake_us_rows, asof="2026-07-14",
    )
    assert result["errors"] == []
    assert result["exchanges"] is None
    assert {r["name"] for r in result["result"]} == {"나스닥가", "나스닥나", "뉴욕가"}  # 전체 대상


def test_us_screening_prompt_scopes_to_us_market_only():
    seen = []

    def spy(prompt: str) -> str:
        seen.append(prompt)
        return json.dumps({"criteria": [{"key": "per", "direction": "low"}], "top_n": 5})

    answer_us_screening(
        "코스피와 나스닥 각각 저PER 5종목", conn=None, llm_fn=spy,
        cross_section_fn=_fake_us_rows, asof="2026-07-14",
    )
    assert seen
    assert "미국" in seen[0]
    assert "exchanges" in seen[0]
    assert "KOSPI" not in seen[0]  # 한국 전용 하드코딩 규칙이 US 프롬프트에 없어야 함


def test_kr_screening_prompt_scopes_to_kr_market_only():
    seen = []

    def spy(prompt: str) -> str:
        seen.append(prompt)
        return json.dumps({"criteria": [{"key": "per", "direction": "low"}], "top_n": 5})

    answer_kr_screening(
        "코스피와 나스닥 각각 저PER 5종목", conn=None, llm_fn=spy,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert seen
    assert "한국" in seen[0]
    assert "markets" in seen[0]
    assert "NASDAQ" not in seen[0]  # 미국 전용 규칙이 KR 프롬프트에 없어야 함


def test_answer_kr_screening_markets_filter_still_works():
    """회귀: KR markets 필터(KOSDAQ 지정)가 기존과 동일하게 동작한다."""
    llm = _json_llm(
        {"criteria": [{"key": "per", "direction": "low"}], "top_n": 5, "markets": ["KOSDAQ"]}
    )
    result = answer_kr_screening(
        "코스닥 저PER 5종목", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["errors"] == []
    assert result["markets"] == ["KOSDAQ"]
    assert [r["name"] for r in result["result"]] == ["고PER다"]  # KOSDAQ 종목만(A/B는 KOSPI라 제외)


def test_us_heuristic_fallback_uses_exchanges_not_markets():
    """llm_fn 없음 → 휴리스틱 폴백. '나스닥' 키워드는 exchanges 로 잡고 markets 는 오염되지 않는다."""
    result = answer_us_screening(
        "나스닥 PER 낮은 5개", conn=None, llm_fn=None,
        cross_section_fn=_fake_us_rows, asof="2026-07-14",
    )
    assert result["errors"] == []
    assert result["exchanges"] == ["NASDAQ"]
    assert result["markets"] is None
    assert [r["name"] for r in result["result"]] == ["나스닥가", "나스닥나"]


def test_us_heuristic_fallback_ignores_kospi_keyword():
    """혼재 질문 + llm 없음: US 휴리스틱은 '코스피'를 markets 로 오염시키지 않고 '나스닥'만 잡는다."""
    result = answer_us_screening(
        "코스피와 나스닥 각각 PER 낮은 5개", conn=None, llm_fn=None,
        cross_section_fn=_fake_us_rows, asof="2026-07-14",
    )
    assert result["markets"] is None
    assert result["exchanges"] == ["NASDAQ"]


# ── security_type 필터 배선 (증권종류 — 워런트/ADR 등 제외, HA15 후속(B)) ──────────────
# answer_us_screening 은 cross_section_fn 이 기본값이든 테스트 주입값이든 관계없이
# security_type 필터를 적용한다(top_n 선정 '이전'에 적용) — 아래는 그 배선이 실제로
# 동작함을 fixture 레벨에서 빠르게 검증한다(실제 DB 경로 end-to-end 는
# tests/test_agents_domain_us.py 에서 별도 검증).
def _fake_us_rows_with_security_type(_conn, _asof):
    return [
        {"stock_code": "COM1", "name": "Common One Inc.", "sector": "Technology",
         "market": "NASDAQ", "security_type": "common", "per": 10.0},
        {"stock_code": "WRNT", "name": "Warrant Corp Warrant", "sector": "Technology",
         "market": "NASDAQ", "security_type": "warrant", "per": 0.001},  # 왜곡된 극단적 저PER
        {"stock_code": "ADRX", "name": "Foreign Co American Depositary Shares", "sector": "Technology",
         "market": "NASDAQ", "security_type": None, "per": 0.002},  # 미분류 + 이름 키워드 폴백 제외 대상
        {"stock_code": "COM2", "name": "Common Two Inc.", "sector": "Technology",
         "market": "NASDAQ", "security_type": "common", "per": 15.0},
    ]


def test_answer_us_screening_excludes_warrant_and_adr_before_ranking():
    llm = _json_llm({"criteria": [{"key": "per", "direction": "low"}], "top_n": 2})
    result = answer_us_screening(
        "나스닥 저PER 2종목", conn=None, llm_fn=llm,
        cross_section_fn=_fake_us_rows_with_security_type, asof="2026-07-14",
    )
    assert result["errors"] == []
    codes = [r["stock_code"] for r in result["result"]]
    # 필터가 top_n 선정보다 먼저 적용되지 않았다면 WRNT/ADRX의 왜곡된 저PER이 상위를
    # 차지해 COM1/COM2가 밀려났을 것 — 정확히 보통주 2개만 남아야 한다.
    assert codes == ["COM1", "COM2"]
    assert "WRNT" not in codes
    assert "ADRX" not in codes
