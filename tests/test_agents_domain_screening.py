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
from src.agents.domain_kr import answer_kr_screening, is_screening_question


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


# ── answer_kr_screening: sector가 없는(None/누락) 종목은 "기타"로 채워서 반환한다
#    (실사용에서 발견: KRX 미분류 종목이 sector=None으로 나와 업종 필터/집계에서
#    조용히 누락됨) ────────────────────────────────────────────────────────────
def _fake_rows_with_missing_sector(_conn, _asof):
    return [
        {"stock_code": "A", "name": "저PER가", "sector": "반도체", "market": "KOSPI", "per": 5.0, "roe": 8.0},
        {"stock_code": "N", "name": "섹터없다", "sector": None, "market": "KOSPI", "per": 1.0, "roe": 1.0},
    ]


def test_answer_kr_screening_missing_sector_defaults_to_gita():
    llm = _json_llm({"criteria": [{"key": "per", "direction": "low"}], "top_n": 2})
    result = answer_kr_screening(
        "PER이 가장 낮은 2개 회사", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows_with_missing_sector, asof="2026-07-14",
    )
    assert result["errors"] == []
    by_code = {r["stock_code"]: r for r in result["result"]}
    assert by_code["N"]["sector"] == "기타"
    assert by_code["A"]["sector"] == "반도체"  # 기존 값이 있는 행은 그대로


def test_answer_kr_screening_gita_sector_is_filterable_like_any_other():
    llm = _json_llm({"criteria": [{"key": "per", "direction": "low"}], "top_n": 5, "sectors": ["기타"]})
    result = answer_kr_screening(
        "기타 업종에서 PER 낮은 5개", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows_with_missing_sector, asof="2026-07-14",
    )
    assert result["errors"] == []
    assert [r["stock_code"] for r in result["result"]] == ["N"]


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


# ── answer_kr_question: 스크리닝 분기 시 질문의 기간(연도/분기)이 asof로 반영돼야 한다
#    ('25년기준 코스피 pbr 최저 3개'가 항상 오늘 날짜로만 조회되던 회귀 재현) ──────────
def test_answer_kr_question_screening_with_year_resolves_asof_from_period(monkeypatch):
    captured = {}

    def spy_screening(question, conn, **kwargs):
        captured.update(kwargs)
        return {"intent": "screening", "result": [], "errors": []}

    monkeypatch.setattr(kr, "answer_kr_screening", spy_screening)

    def fake_exec(sql, _conn):
        assert "2025-12-31" in sql  # 연도만 있으면 그 해 말일 이하로 조회해야 함
        return {"ok": True, "rows": [{"d": "2025-12-30"}]}

    kr.answer_kr_question(
        "25년기준 코스피 종목중 pbr이 가장 낮은 종목 3개", conn=None, execute_sql_fn=fake_exec,
    )
    assert captured.get("asof") == "2025-12-30"


def test_answer_kr_question_screening_with_quarter_resolves_asof_from_quarter_end(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        kr, "answer_kr_screening",
        lambda question, conn, **kwargs: captured.update(kwargs) or {"intent": "screening", "result": [], "errors": []},
    )

    def fake_exec(sql, _conn):
        assert "2025-03-31" in sql  # 2025년 1분기 → 분기말(3/31) 이하로 조회
        return {"ok": True, "rows": [{"d": "2025-03-28"}]}

    kr.answer_kr_question(
        "2025년 1분기 기준 PER 낮은 3개", conn=None, execute_sql_fn=fake_exec,
    )
    assert captured.get("asof") == "2025-03-28"


def test_answer_kr_question_screening_without_period_leaves_asof_unresolved(monkeypatch):
    """기간 언급이 없으면 기존과 동일하게 asof=None 을 넘겨 최신 거래일 폴백을 그대로 따른다."""
    captured = {}
    monkeypatch.setattr(
        kr, "answer_kr_screening",
        lambda question, conn, **kwargs: captured.update(kwargs) or {"intent": "screening", "result": [], "errors": []},
    )

    def fail_exec(_sql, _conn):
        raise AssertionError("기간 미지정이면 asof 사전조회 SQL이 실행되면 안 된다")

    kr.answer_kr_question("PER 낮은 3개", conn=None, execute_sql_fn=fail_exec)
    assert captured.get("asof") is None


# ── 스크리닝 조건(criteria) JSON 실시간 통지 + 편집·재실행(override_spec) 지원 ────────
def test_answer_kr_screening_calls_on_progress_with_spec_detail():
    llm = _json_llm({"criteria": [{"key": "per", "direction": "low"}], "top_n": 2})
    events = []
    answer_kr_screening(
        "PER 낮은 2개", conn=None, llm_fn=llm, cross_section_fn=_fake_rows, asof="2026-07-14",
        on_progress=lambda step, summary, detail=None: events.append((step, summary, detail)),
    )
    assert len(events) == 1
    step, summary, detail = events[0]
    assert detail["kind"] == "screening_spec"
    assert detail["domain"] == "kr"
    assert detail["spec"]["criteria"] == [{"key": "per", "direction": "low"}]
    # 실사용 회귀: "2024년 수익률" 같은 기간 질문의 실제 기준일(asof)이 spec에만 있으면
    # 사용자가 실시간 트리에서 확인할 방법이 없다("이게 진짜 2024년 수익률인지 알 수 없다")
    # — detail에 함께 실어 조건 JSON에서 바로 검증 가능하게 한다.
    assert detail["asof"] == "2026-07-14"


def test_answer_kr_screening_override_spec_skips_llm_entirely():
    """재실행(human-in-the-loop): 사용자가 편집한 spec을 주면 LLM 추출 단계를 완전히 건너뛴다."""
    def boom(_prompt):
        raise AssertionError("override_spec이 있으면 LLM을 호출하면 안 된다")

    result = answer_kr_screening(
        "무시될 질문", conn=None, llm_fn=boom, cross_section_fn=_fake_rows, asof="2026-07-14",
        override_spec={"criteria": [{"key": "per", "direction": "low"}], "top_n": 2,
                        "sectors": None, "markets": None},
    )
    assert result["errors"] == []
    assert result["result"] is not None
    assert result["criteria"] == [{"key": "per", "direction": "low"}]


def test_answer_kr_screening_override_spec_still_validates_hallucinated_field():
    """편집 재실행도 기존 안전장치(존재하지 않는 지표명 거부)를 그대로 통과해야 한다."""
    result = answer_kr_screening(
        "무시될 질문", conn=None, llm_fn=None, cross_section_fn=_fake_rows, asof="2026-07-14",
        override_spec={"criteria": [{"key": "존재하지않는필드", "direction": "low"}], "top_n": 2,
                        "sectors": None, "markets": None},
    )
    assert result["result"] is None
    assert result["errors"]


def test_answer_kr_question_screening_threads_on_progress_to_screening(monkeypatch):
    captured = {}

    def spy_screening(question, conn, **kwargs):
        captured.update(kwargs)
        return {"intent": "screening", "result": [], "errors": []}

    monkeypatch.setattr(kr, "answer_kr_screening", spy_screening)
    sentinel = lambda *a, **k: None
    kr.answer_kr_question("PER 낮은 3개", conn=None, on_progress=sentinel)
    assert captured.get("on_progress") is sentinel


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


# ── 미국 시장(거래소) 필터: 도메인별 스펙 분리 + exchanges 필터 (HA15 혼재질문 버그) ──────
# 실사용 재현 버그: "코스피와 나스닥 각각 PER 낮은 5종목씩" 처럼 한 문장에 한국+미국 시장을
# 동시에 언급하면, KR/US 두 호출이 같은 원본 텍스트를 같은 프롬프트로 받아 US 쪽 spec 의
# markets 가 "코스피"에 오염돼(=["KOSPI"]) US 결과(exchange 값만 있는 rows)가 깨졌다.
# 아래 테스트는 (1) 도메인별 프롬프트 스코프 분리, (2) US exchanges 필터 실제 적용,
# (3) 단일시장/미지정 회귀 무결을 가짜 llm_fn(DI)로 실제 LLM 없이 검증한다.

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


# ── _coerce_top_n: 상한 완화(50→4000, pipeline_exec.MAX_SIZE와 동일) ─────────
def test_coerce_top_n_allows_values_above_old_fifty_cap():
    # 예전엔 min(50, n)이라 500이 50으로 잘렸다 — 이제는 그대로 통과해야 한다.
    assert kr._coerce_top_n(500) == 500


def test_coerce_top_n_still_clamps_at_new_hard_cap_of_4000():
    assert kr._coerce_top_n(10_000) == 4000


def test_coerce_top_n_still_clamps_lower_bound_to_one():
    assert kr._coerce_top_n(0) == 1
    assert kr._coerce_top_n(-5) == 1


# ── _heuristic_screening_spec: "전체/모든/모두" 질문은 top_n을 크게(4000) ────
def test_heuristic_screening_spec_all_stocks_phrase_sets_large_top_n():
    spec = kr._heuristic_screening_spec("코스피 전체 종목 PBR 낮은 순으로 보여줘")
    assert spec is not None
    assert spec["top_n"] == 4000


def test_heuristic_screening_spec_explicit_count_not_overridden_by_all_phrase():
    # "전체"가 있어도 명시적 숫자가 있으면 그 숫자를 우선한다.
    spec = kr._heuristic_screening_spec("전체 종목 중에서 PBR 낮은 10개 보여줘")
    assert spec is not None
    assert spec["top_n"] == 10


def test_heuristic_screening_spec_without_count_or_all_phrase_defaults_to_ten():
    spec = kr._heuristic_screening_spec("PBR 낮은 순으로 보여줘")
    assert spec is not None
    assert spec["top_n"] == 10


# ── _screening_prompt: LLM에게 "전체" 판단 기준(4000) 지시문이 포함되는지 ────
def test_screening_prompt_includes_all_stocks_guidance_with_new_cap():
    prompt = kr._screening_prompt("코스피 전체 종목 PBR 낮은 순", kr._KR_SCREEN_FIELDS, (), domain="KR")
    assert "전체" in prompt
    assert "4000" in prompt


# ── sector_neutral 배선: spec의 sector_neutral을 combine_fn에 그대로 전달 ──────
def test_answer_kr_screening_passes_sector_neutral_true_to_combine():
    llm = _json_llm({
        "criteria": [{"key": "per", "direction": "low"}], "top_n": 2, "sector_neutral": True,
    })
    captured = {}

    def fake_combine(rows, criteria, **kwargs):
        captured.update(kwargs)
        return []

    answer_kr_screening(
        "섹터 중립화해서 PER 낮은 2개", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, combine_fn=fake_combine, asof="2026-07-14",
    )
    assert captured.get("sector_neutral") is True


def test_answer_kr_screening_overrides_llm_true_when_question_lacks_keyword():
    """실사용 재현 버그: LLM이 '정렬 후 다시 확인' 같은 서술만 보고 sector_neutral=True로
    과잉추론해도, 질문 원문에 명시적 표현이 없으면 최종적으로 False로 강제된다."""
    llm = _json_llm({
        "criteria": [{"key": "per", "direction": "low"}], "top_n": 2, "sector_neutral": True,
    })
    captured = {}

    def fake_combine(rows, criteria, **kwargs):
        captured.update(kwargs)
        return []

    answer_kr_screening(
        "PER 낮은 2개 종목을 알려주고 그 결과를 다시 한번 확인해줘", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, combine_fn=fake_combine, asof="2026-07-14",
    )
    assert captured.get("sector_neutral") is False


def test_answer_kr_screening_defaults_sector_neutral_false_to_combine():
    llm = _json_llm({"criteria": [{"key": "per", "direction": "low"}], "top_n": 2})
    captured = {}

    def fake_combine(rows, criteria, **kwargs):
        captured.update(kwargs)
        return []

    answer_kr_screening(
        "PER 낮은 2개", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, combine_fn=fake_combine, asof="2026-07-14",
    )
    assert captured.get("sector_neutral") is False


# ── winsorize 기본 적용: 멀티팩터(2개 이상) z-score 스크리닝은 백테스트와 동일하게
#    winsorize_z=3.0을 자동 적용한다(실시간 스크리닝-백테스트 이상치 처리 일관성). 단일
#    criterion(단순 정렬)에는 적용하지 않는다(회귀 방지) ────────────────────────
def test_answer_kr_screening_applies_default_winsorize_for_multi_criteria():
    llm = _json_llm({
        "criteria": [{"key": "per", "direction": "low"}, {"key": "roe", "direction": "high"}],
        "top_n": 2,
    })
    captured = {}

    def fake_combine(rows, criteria, **kwargs):
        captured.update(kwargs)
        return []

    answer_kr_screening(
        "PER 낮고 ROE 높은 2개", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, combine_fn=fake_combine, asof="2026-07-14",
    )
    assert captured.get("winsorize_z") == 3.0


def test_answer_kr_screening_no_winsorize_for_single_criterion():
    llm = _json_llm({"criteria": [{"key": "per", "direction": "low"}], "top_n": 2})
    captured = {}

    def fake_combine(rows, criteria, **kwargs):
        captured.update(kwargs)
        return []

    answer_kr_screening(
        "PER 낮은 2개", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, combine_fn=fake_combine, asof="2026-07-14",
    )
    assert captured.get("winsorize_z") is None


# ── 극값(최댓값/최솟값/최고/최저) 질문 → 스크리닝 경로 인식·처리 ─────────────────────
# 배경: "국내 주식의 PBR의 최댓값과 최솟값을 알려줘"처럼 종목명 없이 지표의 극값만 묻는
# 질문은 (a) find_stock_code가 실패해 단일종목 경로가 막히고 (b) 개수/랭킹 신호가 없어
# 스크리닝 판정도 못 해 exec_fallback으로 새어 실패했다. "최댓값 1개"는 "그 지표 상위 1개
# (top_n=1) 스크리닝"과 동치이므로, 기존 스크리닝 인프라를 그대로 재사용해 처리한다.

def test_is_screening_question_heuristic_recognizes_extreme_value_questions():
    # 종목명이 없고 극값 표현만 있는 질문 자체가 스크리닝으로 판정돼야 한다.
    assert kr._is_screening_question_heuristic("국내 주식의 PBR의 최댓값과 최솟값을 알려줘")
    assert kr._is_screening_question_heuristic("PBR 최댓값 알려줘")
    assert kr._is_screening_question_heuristic("PBR 최솟값 알려줘")
    assert kr._is_screening_question_heuristic("ROE가 가장 높은 종목")


def test_is_screening_question_heuristic_extreme_expansion_no_false_positive_regression():
    # 회귀: 극값 표현이 없는 단일종목 질문은 여전히 스크리닝이 아니다(오탐 방지).
    assert not kr._is_screening_question_heuristic("삼성전자 PER 알려줘")
    assert not kr._is_screening_question_heuristic("삼성전자 PER이랑 주가 같이 알려줘")
    assert not kr._is_screening_question_heuristic("005930 주가 알려줘")


def test_heuristic_screening_spec_extreme_max_is_top1_high():
    spec = kr._heuristic_screening_spec("PBR 최댓값 알려줘", domain="KR")
    assert spec is not None
    assert spec["top_n"] == 1
    assert spec["criteria"][0]["direction"] == "high"
    assert spec["criteria"][0]["key"] == "pbr"
    assert spec.get("both_extremes") is False


def test_heuristic_screening_spec_extreme_min_is_top1_low():
    spec = kr._heuristic_screening_spec("PBR 최솟값 알려줘", domain="KR")
    assert spec is not None
    assert spec["top_n"] == 1
    assert spec["criteria"][0]["direction"] == "low"
    assert spec.get("both_extremes") is False


def test_heuristic_screening_spec_both_extremes_sets_flag_and_dual_criteria():
    spec = kr._heuristic_screening_spec("PBR의 최댓값과 최솟값을 알려줘", domain="KR")
    assert spec is not None
    assert spec["both_extremes"] is True
    assert spec["top_n"] == 1
    directions = {c["direction"] for c in spec["criteria"]}
    assert directions == {"high", "low"}
    assert all(c["key"] == "pbr" for c in spec["criteria"])


def test_heuristic_screening_spec_explicit_count_still_wins_over_extreme():
    # 회귀: 명시 숫자(10)가 있으면 극값 표현이 있어도 top_n=1로 덮어쓰지 않는다.
    spec = kr._heuristic_screening_spec("PBR 최댓값 상위 10개", domain="KR")
    assert spec is not None
    assert spec["top_n"] == 10


def test_screening_intent_prompt_mentions_extreme_value_guidance():
    prompt = kr._screening_intent_prompt("아무 질문")
    assert "최댓값" in prompt


def test_screening_prompt_mentions_both_extremes_and_top1_guidance():
    prompt = kr._screening_prompt("PBR 최댓값과 최솟값", kr._KR_SCREEN_FIELDS, (), domain="KR")
    assert "both_extremes" in prompt
    assert "최댓값" in prompt


def test_answer_kr_screening_both_extremes_returns_highest_and_lowest():
    """both_extremes 스펙이면 direction별 top_n=1 결과를 highest/lowest로 나란히 반환한다."""
    llm = _json_llm({
        "criteria": [{"key": "per", "direction": "high"}, {"key": "per", "direction": "low"}],
        "top_n": 1, "both_extremes": True,
    })
    result = answer_kr_screening(
        "PER의 최댓값과 최솟값을 알려줘", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["errors"] == []
    assert result["both_extremes"] is True
    assert isinstance(result["result"], dict)
    assert [r["name"] for r in result["result"]["highest"]] == ["고PER다"]  # per 최댓값
    assert [r["name"] for r in result["result"]["lowest"]] == ["저PER가"]   # per 최솟값


def test_answer_kr_screening_both_extremes_heuristic_fallback_without_llm():
    """LLM 없이도(휴리스틱 폴백) 최댓값+최솟값 질문이 highest/lowest로 처리된다."""
    result = answer_kr_screening(
        "PER의 최댓값과 최솟값을 알려줘", conn=None, llm_fn=None,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["errors"] == []
    assert result["both_extremes"] is True
    assert [r["name"] for r in result["result"]["highest"]] == ["고PER다"]
    assert [r["name"] for r in result["result"]["lowest"]] == ["저PER가"]


def test_answer_kr_screening_single_extreme_returns_flat_list_backward_compat():
    """하위호환: 최댓값만(양극단 아님) 요청이면 기존처럼 result는 평평한 list다."""
    result = answer_kr_screening(
        "PER 최댓값 알려줘", conn=None, llm_fn=None,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["errors"] == []
    assert result["both_extremes"] is False
    assert isinstance(result["result"], list)
    assert [r["name"] for r in result["result"]] == ["고PER다"]  # per 최댓값 1개


# ── 섹터중립 비교(sector_neutral_compare) — 섹터중립화 전/후를 한 번에 비교 ─────────────
# 배경: 섹터중립 스크리닝은 sector_neutral 이중 게이트(LLM/휴리스틱이 True여도 질문 원문에
# 실제 표현이 없으면 무시)로 과잉추론을 막는다. 사용자가 "섹터중립화 전후를 한 번에 비교"를
# 요청해, both_extremes(최고/최저 동시)와 같은 패턴으로 raw/sector_neutral 두 결과를 나란히
# 담는 sector_neutral_compare를 추가한다. 게이트: 섹터중립 키워드 + 비교 의도 둘 다 있을 때만 True.

def test_detect_sector_neutral_compare_keyword_gate():
    # 섹터중립 키워드 + 비교 의도 → True
    assert kr._detect_sector_neutral_compare_keyword("섹터중립화 전후 비교해서 보여줘")
    assert kr._detect_sector_neutral_compare_keyword("섹터중립화하고 안 하고 둘 다 보여줘")
    assert kr._detect_sector_neutral_compare_keyword("섹터중립 동시에 같이 보여줘")
    # 섹터중립 키워드만(비교 의도 없음) → False
    assert not kr._detect_sector_neutral_compare_keyword("섹터중립화해서 보여줘")
    # 비교 의도만(섹터중립 키워드 없음) → False
    assert not kr._detect_sector_neutral_compare_keyword("PBR 낮은 10개 비교해줘")


def test_screening_prompt_mentions_sector_neutral_compare():
    prompt = kr._screening_prompt("아무 질문", kr._KR_SCREEN_FIELDS, (), domain="KR")
    assert "sector_neutral_compare" in prompt


def test_parse_screening_json_extracts_sector_neutral_compare():
    spec = kr._parse_screening_json(json.dumps({
        "criteria": [{"key": "per", "direction": "low"}], "top_n": 5, "sector_neutral_compare": True,
    }), domain="KR")
    assert spec is not None
    assert spec.get("sector_neutral_compare") is True


def test_heuristic_screening_spec_sector_neutral_compare_gate():
    # 키워드+비교 의도 → True
    spec = kr._heuristic_screening_spec("PER 섹터중립화 전후 비교해줘", domain="KR")
    assert spec is not None
    assert spec.get("sector_neutral_compare") is True
    # 키워드만(비교 의도 없음) → False, sector_neutral만 True
    spec2 = kr._heuristic_screening_spec("PER 섹터중립화해서 보여줘", domain="KR")
    assert spec2 is not None
    assert spec2.get("sector_neutral_compare") is False
    assert spec2.get("sector_neutral") is True
    # 섹터중립 키워드 없음(비교만) → False
    spec3 = kr._heuristic_screening_spec("PBR 낮은 10개 비교해줘", domain="KR")
    assert spec3 is not None
    assert spec3.get("sector_neutral_compare") is False


def test_normalize_override_spec_trusts_sector_neutral_compare_ungated():
    # 사람이 명시적으로 재실행한 값이므로 게이트 없이 그대로 신뢰(sector_neutral/both_extremes 패턴).
    spec = kr._normalize_override_spec({
        "criteria": [{"key": "per", "direction": "low"}], "top_n": 5, "sector_neutral_compare": True,
    })
    assert spec is not None
    assert spec.get("sector_neutral_compare") is True


def test_answer_kr_screening_sector_neutral_compare_true_when_keyword_and_intent():
    llm = _json_llm({
        "criteria": [{"key": "per", "direction": "low"}], "top_n": 10, "sector_neutral_compare": True,
    })
    result = answer_kr_screening(
        "PER 낮은 10개를 섹터중립화 전후로 비교해서 보여줘", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["sector_neutral_compare"] is True


def test_answer_kr_screening_sector_neutral_compare_false_without_compare_intent():
    # 섹터중립 키워드만 있고 비교 의도 없음 → compare=False, sector_neutral만 True(과잉추론 방지).
    llm = _json_llm({
        "criteria": [{"key": "per", "direction": "low"}], "top_n": 10,
        "sector_neutral": True, "sector_neutral_compare": True,
    })
    result = answer_kr_screening(
        "PER 낮은 10개 섹터중립화해서 보여줘", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["sector_neutral_compare"] is False
    assert result["sector_neutral"] is True


def test_answer_kr_screening_sector_neutral_compare_false_without_sector_neutral_keyword():
    # "비교"라는 단어는 있지만 섹터중립 키워드 부재 → 게이트 차단으로 False.
    llm = _json_llm({
        "criteria": [{"key": "pbr", "direction": "low"}], "top_n": 10, "sector_neutral_compare": True,
    })
    result = answer_kr_screening(
        "PBR 낮은 10개 비교해줘", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["sector_neutral_compare"] is False


def test_answer_kr_screening_compare_returns_raw_and_sector_neutral():
    llm = _json_llm({
        "criteria": [{"key": "per", "direction": "low"}], "top_n": 2, "sector_neutral_compare": True,
    })
    result = answer_kr_screening(
        "PER 낮은 2개 섹터중립화 전후 비교", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["errors"] == []
    assert result["sector_neutral_compare"] is True
    assert isinstance(result["result"], dict)
    assert set(result["result"].keys()) == {"raw", "sector_neutral"}
    # raw(섹터중립 안 함)는 기존 단일 스크리닝과 동일한 결과여야 한다.
    assert [r["name"] for r in result["result"]["raw"]] == ["저PER가", "중간나"]
    assert isinstance(result["result"]["sector_neutral"], list)


def test_answer_kr_screening_compare_heuristic_fallback_without_llm():
    result = answer_kr_screening(
        "PER 낮은 2개 섹터중립화 전후 비교", conn=None, llm_fn=None,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["errors"] == []
    assert result["sector_neutral_compare"] is True
    assert set(result["result"].keys()) == {"raw", "sector_neutral"}
    assert [r["name"] for r in result["result"]["raw"]] == ["저PER가", "중간나"]


def test_answer_kr_screening_both_extremes_and_compare_four_way_nesting():
    llm = _json_llm({
        "criteria": [{"key": "per", "direction": "high"}, {"key": "per", "direction": "low"}],
        "top_n": 1, "both_extremes": True, "sector_neutral_compare": True,
    })
    result = answer_kr_screening(
        "PER의 최댓값과 최솟값을 섹터중립화 전후로 비교해서 보여줘", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["errors"] == []
    assert result["both_extremes"] is True
    assert result["sector_neutral_compare"] is True
    r = result["result"]
    assert set(r.keys()) == {"highest", "lowest"}
    assert set(r["highest"].keys()) == {"raw", "sector_neutral"}
    assert set(r["lowest"].keys()) == {"raw", "sector_neutral"}
    assert [x["name"] for x in r["highest"]["raw"]] == ["고PER다"]
    assert [x["name"] for x in r["lowest"]["raw"]] == ["저PER가"]


def test_answer_kr_screening_override_compare_trusted_without_keyword():
    # override_spec은 게이트 없이 그대로 신뢰 → 질문에 키워드가 없어도 compare 실행.
    result = answer_kr_screening(
        "PER 낮은 2개", conn=None, llm_fn=None,
        cross_section_fn=_fake_rows, asof="2026-07-14",
        override_spec={
            "criteria": [{"key": "per", "direction": "low"}], "top_n": 2,
            "sector_neutral_compare": True,
        },
    )
    assert result["errors"] == []
    assert result["sector_neutral_compare"] is True
    assert set(result["result"].keys()) == {"raw", "sector_neutral"}


def test_answer_kr_screening_non_compare_result_shape_unchanged():
    # 회귀: compare가 아니면 기존처럼 result는 평평한 list(구조 불변).
    llm = _json_llm({"criteria": [{"key": "per", "direction": "low"}], "top_n": 2})
    result = answer_kr_screening(
        "PER 낮은 2개", conn=None, llm_fn=llm,
        cross_section_fn=_fake_rows, asof="2026-07-14",
    )
    assert result["errors"] == []
    assert result["sector_neutral_compare"] is False
    assert isinstance(result["result"], list)

