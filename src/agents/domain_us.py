"""미국주식 도메인 에이전트 — 질문을 재무/주가 데이터 에이전트로 위임하고 종합 반환 (HA-7).

HA-6(한국주식 도메인 에이전트, src/agents/domain_kr.py)과 대칭 역할이지만 한 가지
구조적 차이가 있다: 한국은 재무데이터가 DART/FnGuide 두 출처로 나뉘어 있어
HA-2(src/agents/data_financial.py)가 지표별 소스판단(DART vs FnGuide)을 맡지만,
미국은 재무데이터 출처가 `us_financials` 테이블 하나뿐이다(HA-4 모듈 docstring과 동일
근거 — src/ingest/us_financials.py가 yfinance 단일 출처로 채움). 따라서 **이 파일에는
소스판단 로직 자체가 없다** — get_financials_us()는 항상 us_financials 하나만 본다.

이 파일이 담당하는 세 가지:
1. 티커/회사명 추출(resolve_ticker_us): 이미 티커 형태(AAPL 등)인 입력은 정규식으로
   즉시 인식하고, 아니면(예: "애플" 같은 한글 회사명) us_company 테이블(HA-4가 이미
   채운 이름↔티커 매핑, src/ingest/us_universe.py) 조회 또는 llm_fn 폴백에 위임한다.
   HA-2의 classify_source(metric, llm_fn)와 동일한 "규칙기반 우선 + LLM 폴백" DI 패턴.
2. 재무 vs 주가·기술지표 라우팅(_classify_intent): 질문 키워드로 financial/price/both를
   가른다.
3. 가까운 계층 재시도(_call_with_retry) + 결과 종합(answer_us_question): 데이터 에이전트
   호출이 예외를 던지거나 ok=False를 반환하면 도메인 에이전트(가까운 계층) 레벨에서
   즉시 1회 더 재시도하고, 그래도 실패하면 예외를 전파하지 않고 실패 사유를 반환값에
   담는다 — HA-6과 동일 원칙(총괄 에이전트까지 예외가 뚫고 올라가면 안 됨).

get_financials_us()는 새 SQL 병합 로직을 만들지 않는다 — metrics_at_us(이미 완성된
전종목 스냅샷, src/backtest/data_access_us.py)를 그대로 재사용하고 stock_code로
필터링할 뿐이다. asof 기준일을 새로 조회하는 SQL 한 조각(_default_asof_us)만
execute_sql(HA-1 실행기, src/agents/exec_runtime.py)을 경유해 실행한다(conn.execute()
직접 호출 금지 — conn은 connect_readonly() 읽기전용 연결이어야 함).
"""
from __future__ import annotations

import re
from typing import Callable

from src.agents.data_price_us import get_price_history_us, get_price_snapshot_us
from src.agents.domain_kr import (
    _intent_prompt,
    _parse_intent,
    _parse_period,
    _resolve_screening_asof,
    _run_screening,
    _strip_retry_feedback,
    _summarize_price_history,
    _wants_price_history,
    is_screening_question,
)
from src.agents.exec_runtime import execute_sql
from src.backtest.data_access_us import METRIC_FIELD_DESCRIPTIONS_US, metrics_at_us
from src.backtest.primitives import combine, get_cross_section

# 미국 유니버스 스크리닝 지표 key(get_cross_section(metrics_at_us) 출력 스키마 — KR 보다 좁다).
# METRIC_FIELD_DESCRIPTIONS_US(단일 정의처, src/backtest/data_access_us.py)에서 key만
# 파생한다 — 손으로 다시 베껴 적지 않는다(domain_kr.py의 _KR_SCREEN_FIELDS와 동일 패턴).
_US_SCREEN_FIELDS: tuple[str, ...] = tuple(METRIC_FIELD_DESCRIPTIONS_US.keys())

# 미국 티커 형식(data_price_us._TICKER_RE와 동일 정의 — 사설 심볼 임포트를 피하려고
# 이 작은 정규식만 로컬에 복제한다. 바뀌면 두 파일을 함께 갱신할 것).
_TICKER_RE = re.compile(r"^[A-Z]{1,6}(\.[A-Z]{1,3})?$")

_FINANCIAL_KEYWORDS = ("per", "pbr", "roe", "매출", "영업이익", "순이익", "재무", "이익률", "자기자본")
_PRICE_KEYWORDS = ("주가", "종가", "이동평균", "rsi", "macd", "볼린저", "거래량", "기술지표", "가격", "sma", "ema")

# 티커 후보에서 제외할 대문자 키워드(PER/RSI 등은 티커 형식(^[A-Z]{1,6}$)과 겹치지만
# 실제로는 지표/지표군 이름이다) — 위 키워드 목록에서 자동 도출해 중복 정의를 피한다.
_NON_TICKER_KEYWORDS = {
    k.upper() for k in _FINANCIAL_KEYWORDS + _PRICE_KEYWORDS if k.isascii() and k.isalpha()
}


def _classify_intent(question: str, llm_fn: Callable[[str], str] | None = None) -> str:
    """질문이 재무/주가/둘다 중 무엇을 원하는지 판단한다('financial' | 'price' | 'both').

    HA-6 classify_intent(domain_kr)와 동일하게 **LLM 우선**이다 — llm_fn 이 주어지면 먼저 LLM 에
    판단을 위임하고(공용 _intent_prompt/_parse_intent 재사용), 응답에서 판단을 못 뽑으면(파싱
    실패/예외) 아래 키워드 휴리스틱으로 폴백한다. 양쪽 키워드가 다 있거나 둘 다 없으면 'both'로
    안전 폴백한다 — 필요한 데이터를 빠뜨리는 것보다 과다 조회가 낫다는 원칙.
    """
    if llm_fn is not None:
        try:
            raw = llm_fn(_intent_prompt(question)) or ""
        except Exception:  # noqa: BLE001 — LLM 실패는 키워드 폴백으로 흡수
            raw = ""
        intent = _parse_intent(raw)
        if intent is not None:
            return intent
    q = question.lower()
    wants_financial = any(k in q for k in _FINANCIAL_KEYWORDS)
    wants_price = any(k in q for k in _PRICE_KEYWORDS)
    if wants_financial and not wants_price:
        return "financial"
    if wants_price and not wants_financial:
        return "price"
    return "both"


def _extract_ticker_token(question: str) -> str | None:
    """질문에 이미 티커 형태(AAPL, BRK.B 등)로 들어있는 토큰을 찾는다.

    PER/RSI처럼 티커 형식과 겹치는 지표 키워드는 _NON_TICKER_KEYWORDS로 제외한다.
    원본 토큰이 이미 전부 대문자일 때만 "사용자가 실제로 티커를 입력했다"고 인정한다 —
    소문자 회사명(예: "nvidia", 6글자)이 대문자 변환 후 티커 형식 정규식과 우연히
    길이가 맞아떨어져 즉시 채택돼버리면, 뒤에 있는 진짜 회사명→티커 변환 단계
    (resolve_ticker_us의 llm_fn/us_company 조회)를 건너뛰어 매번 결정론적으로
    실패한다(실서버 재현: "현재 nvidia 주식 주가" → stock_code="NVIDIA").
    """
    for token in re.findall(r"[A-Za-z][A-Za-z.]{0,6}", question):
        if not token.isupper():
            continue
        if token in _NON_TICKER_KEYWORDS:
            continue
        if _TICKER_RE.match(token):
            return token
    return None


def _lookup_ticker_by_name(conn, name_fragment: str, execute_sql_fn: Callable | None = None) -> str | None:
    """us_company.name에서 부분일치(대소문자 무시)로 티커를 찾는다.

    llm_fn이 정확한 티커가 아니라 회사명(예: 'Apple')을 돌려준 경우의 2차 폴백이다.
    """
    execute_sql_fn = execute_sql_fn or execute_sql
    escaped = name_fragment.strip().replace("'", "''")
    if not escaped:
        return None
    sql = f"SELECT stock_code FROM us_company WHERE UPPER(name) LIKE UPPER('%{escaped}%') LIMIT 1"
    result = execute_sql_fn(sql, conn)
    if not result.get("ok") or not result["rows"]:
        return None
    return result["rows"][0]["stock_code"]


def _ticker_prompt(question: str) -> str:
    """resolve_ticker_us의 llm_fn 폴백용 프롬프트. _intent_prompt(domain_kr)와 동일 관례.

    llm_fn(question)처럼 원본 질문을 그대로 넘기면 LLM이 챗봇처럼 장문으로 답해버려
    (실측 확인: role="sql" 모델에 "앤비디아 주식 주가"를 그대로 넣으면 "저는 실시간
    시세를 직접 조회할 수는 없습니다..." 같은 설명문이 돌아옴) 뒤에서 티커를 못 뽑아
    매번 실패했다. "티커만 답하라"는 지시를 명시해야 한다.
    """
    return (
        "다음 질문에서 언급된 회사의 미국 주식시장 티커 심볼(예: AAPL, NVDA, MSFT)만 답하세요.\n"
        "설명이나 다른 말은 절대 덧붙이지 말고, 티커 심볼 하나만 출력하세요.\n"
        "확실하지 않으면 'UNKNOWN'이라고만 답하세요.\n\n"
        f"질문: {question}\n티커:"
    )


def resolve_ticker_us(
    question: str,
    conn,
    llm_fn: Callable[[str], str] | None = None,
    execute_sql_fn: Callable | None = None,
) -> str | None:
    """질문에서 미국 티커를 추출한다.

    순서: (1) 질문에 이미 티커 형태 토큰이 있으면 즉시 채택 → (2) llm_fn(_ticker_prompt(...))
    폴백(한글 회사명 등 규칙기반으로 못 찾는 경우, HA-2 classify_source와 동일한 DI 패턴)
    → (2)의 결과가 티커 형식이면 그대로, 아니면 us_company.name으로 재조회 → 그래도
    못 찾으면(LLM이 프롬프트 지시를 어기고 부연설명을 붙인 경우) 응답 텍스트 안에서
    티커형 토큰을 뽑아 다시 회사명/형식 순으로 시도한다. 끝까지 못 찾으면 None(호출부가
    실패로 처리).
    """
    token = _extract_ticker_token(question)
    if token:
        return token
    if llm_fn is None:
        return None
    guess = llm_fn(_ticker_prompt(question))
    ticker = _resolve_from_llm_guess(guess, conn, execute_sql_fn=execute_sql_fn)
    if ticker:
        return ticker
    # 방어적 재시도: 프롬프트 지시를 어기고 부연설명이 붙은 응답(예: "티커: **NVDA**")에서
    # 티커형 토큰을 뽑아 같은 순서(회사명 조회 → 형식 매치)로 한 번 더 시도한다.
    candidate = _extract_ticker_token(guess or "")
    if candidate:
        return _resolve_from_llm_guess(candidate, conn, execute_sql_fn=execute_sql_fn)
    return None


def _lookup_exact_ticker(conn, code: str, execute_sql_fn: Callable | None = None) -> str | None:
    """guess가 us_company.stock_code와 '완전히 일치'하는 실제 티커인지 확인한다.

    회사명 부분일치(_lookup_ticker_by_name)와 달리 stock_code 완전일치만 본다. LLM이
    올바른 티커(예: "META")를 돌려줬는데도 그 문자열이 다른 회사명의 부분 문자열이라
    (예: "META" ⊂ "Aqua Metals Inc.") 회사명 조회가 엉뚱한 종목(AQMS)을 먼저 잡아채던
    실측 버그(experiment/us-domain-llm-flexible 비교에서 A/B 세 접근 모두 공통 실패)를
    막기 위해, 회사명 부분일치보다 먼저 이 완전일치를 확인한다.
    """
    execute_sql_fn = execute_sql_fn or execute_sql
    up = (code or "").strip().upper()
    if not _TICKER_RE.match(up):
        return None
    escaped = up.replace("'", "''")
    result = execute_sql_fn(f"SELECT stock_code FROM us_company WHERE stock_code='{escaped}'", conn)
    if not result.get("ok") or not result["rows"]:
        return None
    return result["rows"][0]["stock_code"]


def _resolve_from_llm_guess(
    guess: str | None, conn, execute_sql_fn: Callable | None = None
) -> str | None:
    """llm_fn 추측 하나를 (완전일치 티커 → 회사명 부분조회 → 티커 형식) 순으로 확정 시도한다."""
    if not guess:
        return None
    guess = guess.strip()
    if not guess:
        return None
    # (1) 완전일치 티커 우선: guess가 실제 us_company.stock_code면 그대로 채택한다. 회사명
    # 부분일치보다 먼저 봐야, 올바른 티커("META")가 다른 회사명("Aqua Metals")의 부분
    # 문자열이라 엉뚱한 종목(AQMS)으로 새는 것을 막는다(실측 재현: "메타 주가" → AQMS).
    exact = _lookup_exact_ticker(conn, guess, execute_sql_fn=execute_sql_fn)
    if exact:
        return exact
    # (2) 회사명 부분조회 — 짧은 회사명(예: "Apple")은 티커 형식 정규식과 우연히 겹칠 수
    # 있어(예: "APPLE"도 6자 이하 대문자), 실제 us_company에 존재하는지부터 확인하는 편이
    # 오탐(hallucination) 위험이 낮다. "APPLE"은 완전일치 티커가 아니므로 (1)을 지나 여기서
    # AAPL로 해석된다(회귀 없음). 못 찾으면 리터럴 티커로 폴백한다.
    by_name = _lookup_ticker_by_name(conn, guess, execute_sql_fn=execute_sql_fn)
    if by_name:
        return by_name
    guess_upper = guess.upper()
    if _TICKER_RE.match(guess_upper):
        return guess_upper
    return None


def _default_asof_us(conn, stock_code: str, execute_sql_fn: Callable | None = None) -> str | None:
    """asof 미지정 시 us_financials에 실제 존재하는 최신 disclosed_date(공시일 근사).

    실제 캘린더 오늘 날짜가 아니라 DB에 실제로 존재하는 시점을 쓴다(data_price_kr/us의
    asof 기본값 원칙과 동일).
    """
    execute_sql_fn = execute_sql_fn or execute_sql
    sql = (
        "SELECT MAX(disclosed_date) AS max_date FROM us_financials "
        f"WHERE stock_code='{stock_code}' AND disclosed_date IS NOT NULL"
    )
    result = execute_sql_fn(sql, conn)
    if not result.get("ok") or not result["rows"]:
        return None
    return result["rows"][0].get("max_date")


def get_financials_us(
    conn,
    stock_code: str,
    asof: str | None = None,
    metrics_fn: Callable | None = None,
    execute_sql_fn: Callable | None = None,
) -> dict | None:
    """단일 티커의 asof 시점 재무 스냅샷(PER/PBR/ROE/영업이익률/순이익률 등)을 반환한다.

    HA-2(DART/FnGuide)와 달리 미국은 us_financials 테이블 하나가 유일 출처이므로
    소스판단(source routing) 로직 자체가 필요 없다 — 반환 dict에 'source' 필드가 없다.
    metrics_at_us(이미 완성된 전종목 스냅샷, src/backtest/data_access_us.py)를 그대로
    재사용한다(신규 SQL 병합 로직을 만들지 않는다) — 전체 결과에서 stock_code만
    필터링해서 돌려준다. metrics_fn은 테스트 주입용(기본=metrics_at_us).

    티커 데이터가 없거나(형식 불일치 포함) 해당 시점에 유효 분기가 없으면 None.
    """
    metrics_fn = metrics_fn or metrics_at_us
    code = (stock_code or "").strip().upper()
    if not _TICKER_RE.match(code):
        return None
    resolved_asof = asof or _default_asof_us(conn, code, execute_sql_fn=execute_sql_fn)
    if not resolved_asof:
        return None
    rows = metrics_fn(conn, resolved_asof)
    for row in rows:
        if row.get("stock_code") == code:
            return row
    return None


def _is_failure_result(result) -> bool:
    return isinstance(result, dict) and result.get("ok") is False


def _call_with_retry(fn: Callable, *args, retries: int = 1, **kwargs) -> dict:
    """데이터 에이전트 호출을 최대 retries회(기본 1회) 재시도한다(HA-6과 동일 원칙).

    fn(*args, **kwargs) 호출이 예외를 던지거나 {"ok": False, ...} 형태(execute_sql
    계약)를 반환하면 실패로 간주해 즉시 재시도하고, 그래도 계속 실패하면 예외를
    전파하지 않고 실패 사유를 담아 반환한다 — 도메인 에이전트(가까운 계층)에서
    흡수하므로 총괄 에이전트까지 예외가 뚫고 올라가지 않는다.

    반환: {"ok": True, "result": ..., "error": None} 또는
          {"ok": False, "result": None, "error": str}.
    """
    last_error: str | None = None
    for _ in range(retries + 1):
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — 데이터 에이전트 예외를 여기서 흡수
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if _is_failure_result(result):
            last_error = result.get("error") or "data agent returned ok=False"
            continue
        return {"ok": True, "result": result, "error": None}
    return {"ok": False, "result": None, "error": last_error}


# ── security_type 필터 (증권종류 — 워런트/ADR 등 파생·특수 증권 제외, HA15 후속(B)) ──────────
# 배경: 실서버 curl 재현 — 나스닥 저PER 스크리닝 상위권에 SKHYV(ADR)/RNWWW·BNCWW(Warrant)
# 같은 파생·특수 증권이 섞여 나왔다(시총이 비정상적으로 작아 PER이 왜곡됨). us_company.name에
# 이미 "…Warrant"/"…American Depositary Shares" 처럼 판별 정보가 그대로 들어있으므로,
# 정규식/접미어 키워드 목록으로 판정하지 않고 회사명 전체 문자열을 LLM이 의미로 읽고
# 판단하게 한다(사용자 명시 지시). 다만 스크리닝 질문마다 매번 LLM을 부르면 비용/속도가
# 감당 안 되므로 "판단은 AI, 실행은 캐싱" 원칙을 따른다 — 분류(판단) 자체는
# scripts/backfill_us_security_type.py가 1회 배치로 미리 해서 us_company.security_type에
# 캐싱하고, 이 런타임 필터는 그 캐시를 읽기만 한다(매 요청마다 LLM 호출 없음).
#
# security_type이 아직 없는(NULL, 배치 미실행/신규상장) 종목만 route_question/classify_intent/
# is_screening_question과 동일한 "AI 판단(배치 캐시) 우선, 미분류 시 규칙 폴백" 원칙에 따라
# 회사명 키워드로 최소한의 안전망을 적용한다.
_SECURITY_TYPE_FALLBACK_KEYWORDS: tuple[str, ...] = (
    "warrant", "depositary", "depository", "preferred", "rights", "units", "unit",
)


def _is_common_stock_by_name_fallback(name: str) -> bool:
    """security_type 배치 분류가 아직 안 된(NULL) 종목의 최소 안전망.

    회사명에 파생·특수 증권을 뜻하는 접미어가 있으면 보통주가 아닌 것으로 본다. 이건
    어디까지나 미분류 종목을 위한 안전망 폴백일 뿐이며, 1차 판단은 항상 배치 캐시된
    LLM 분류(security_type)다.
    """
    n = (name or "").lower()
    return not any(kw in n for kw in _SECURITY_TYPE_FALLBACK_KEYWORDS)


def _filter_common_stock(rows: list[dict]) -> list[dict]:
    """US 스크리닝 후보에서 일반 보통주만 남긴다(top_n 선정 전에 적용, 순위 왜곡 방지).

    1차: us_company.security_type(배치 스크립트가 LLM으로 미리 분류해 캐싱한 값)이
    'common'인 행만 채택. security_type이 아직 없으면(None/키 없음) 회사명 키워드
    안전망(_is_common_stock_by_name_fallback)으로 최소한만 걸러낸다.
    """
    out = []
    for r in rows:
        st = r.get("security_type")
        if st is not None:
            if st == "common":
                out.append(r)
            continue
        if _is_common_stock_by_name_fallback(r.get("name") or ""):
            out.append(r)
    return out


def answer_us_screening(
    question: str,
    conn,
    llm_fn: Callable[[str], str] | None = None,
    cross_section_fn: Callable | None = None,
    combine_fn: Callable | None = None,
    asof: str | None = None,
    execute_sql_fn: Callable | None = None,
    security_filter_fn: Callable | None = None,
    override_spec: dict | None = None,
    on_progress: Callable[..., None] | None = None,
) -> dict:
    """미국주식 스크리닝: 조건에 맞는 종목을 순위 매겨 상위 N개를 반환한다(다중종목 경로).

    HA-6/HA-15 한국 스크리닝(_run_screening)과 대칭이며 유일한 차이는 데이터 소스다 —
    get_cross_section 의 metrics_fn 으로 metrics_at_us 를 주입하고(us_financials/us_prices),
    asof 는 us_prices 최신 거래일을 쓴다. LLM 은 구조화 JSON 만 생성하고, 존재하지 않는
    지표명(환각)/해석 실패는 조용한 빈 결과가 아니라 errors 사유로 남긴다(_run_screening 공용).

    security_filter_fn(기본=_filter_common_stock)을 cross_section_fn 결과에 항상 적용해
    워런트/ADR 등을 top_n 선정 '이전'에 제외한다 — cross_section_fn 이 기본값이든 테스트
    주입값이든 관계없이 적용된다(어느 한쪽만 걸러지는 구멍 방지).
    """
    base_cross_section_fn = cross_section_fn or (
        lambda c, a: get_cross_section(c, a, metrics_fn=metrics_at_us)
    )
    security_filter_fn = security_filter_fn or _filter_common_stock

    def filtered_cross_section_fn(c, a):
        return security_filter_fn(base_cross_section_fn(c, a))

    combine_fn = combine_fn or combine
    return _run_screening(
        question, conn, llm_fn, filtered_cross_section_fn, combine_fn,
        price_table="us_prices", fields=_US_SCREEN_FIELDS,
        asof=asof, execute_sql_fn=execute_sql_fn, domain="US",
        override_spec=override_spec, on_progress=on_progress,
    )


def answer_us_question(
    question: str,
    conn,
    llm_fn: Callable[[str], str] | None = None,
    price_fn: Callable | None = None,
    financial_fn: Callable | None = None,
    price_history_fn: Callable | None = None,
    execute_sql_fn: Callable | None = None,
    on_progress: Callable[..., None] | None = None,
) -> dict:
    """미국주식 질문에 답한다 — 티커 해석 → 재무/주가 위임 → 결과 종합.

    price_fn 기본값은 get_price_snapshot_us(HA-4), financial_fn 기본값은
    get_financials_us(이 파일). 둘 다 테스트 주입용(가까운 계층 재시도 검증 등).

    반환: {"ok": bool, "stock_code": str|None, "intent": str|None,
           "financial": dict|None, "price": list[dict]|None, "error": str|None,
           "price_history": dict|None}.
    티커를 못 찾거나 데이터 에이전트 호출이 (재시도 후에도) 실패해도 예외를 던지지
    않고 항상 이 dict 계약을 지킨다. price_history 는 HA-6(KR)과 대칭으로, 시계열/차트를
    원하는 주가 질문일 때만 get_price_history_us 로 최근 1년 종가를 담고 그 외엔 None.
    """
    price_fn = price_fn or get_price_snapshot_us
    financial_fn = financial_fn or get_financials_us
    price_history_fn = price_history_fn or get_price_history_us
    base_question = _strip_retry_feedback(question)
    period = _parse_period(base_question)

    # 스크리닝(다중종목 랭킹) 질문은 단일티커 조회 경로 대신 스크리닝 경로로 분기한다(HA-15).
    # is_screening_question 도 LLM 우선 판단이므로 llm_fn 을 관통시킨다(HA-6과 동일 배선 원칙).
    # period(질문의 연도/분기)를 asof로 확정해 넘긴다 — 안 하면 스크리닝이 항상 오늘 날짜
    # 기준으로만 계산돼 "2024년기준" 같은 조건이 무시되는 회귀(HA-6과 동일 버그).
    if is_screening_question(question, llm_fn=llm_fn):
        screening_asof = _resolve_screening_asof(period, conn, "us_prices", execute_sql_fn)
        return answer_us_screening(
            question, conn, llm_fn=llm_fn, execute_sql_fn=execute_sql_fn, asof=screening_asof,
            on_progress=on_progress,
        )

    ticker = resolve_ticker_us(question, conn, llm_fn=llm_fn)
    if not ticker:
        return {
            "ok": False,
            "stock_code": None,
            "intent": None,
            "financial": None,
            "price": None,
            "error": "질문에서 미국 티커/회사명을 해석할 수 없음",
        }

    intent = _classify_intent(question, llm_fn=llm_fn)
    financial_result = None
    price_result = None
    price_history = None
    errors: list[str] = []

    if intent in ("financial", "both"):
        outcome = _call_with_retry(financial_fn, conn, ticker)
        if outcome["ok"]:
            financial_result = outcome["result"]
        else:
            errors.append(f"financial: {outcome['error']}")

    if intent in ("price", "both"):
        outcome = _call_with_retry(price_fn, conn, ticker)
        if outcome["ok"]:
            price_result = outcome["result"]
            # 시계열/차트를 원하는 질문이면 최근 1년 종가 시계열도 담는다(버그A, KR과 대칭).
            if price_result and _wants_price_history(base_question):
                history = price_history_fn(conn, ticker)
                if history:
                    price_history = _summarize_price_history(history)
        else:
            errors.append(f"price: {outcome['error']}")

    return {
        "ok": not errors,
        "stock_code": ticker,
        "intent": intent,
        "financial": financial_result,
        "price": price_result,
        "error": "; ".join(errors) if errors else None,
        "price_history": price_history,
    }
