"""한국주식 도메인 에이전트 — 질문을 재무/주가 데이터 에이전트로 세부 위임한다 (HA-6).

계층: 총괄 에이전트(HA-10, 미구현) → **이 도메인 에이전트** → 데이터 에이전트
(HA-2 resolve_metric/src/agents/data_financial.py, HA-3 get_price_snapshot_kr/
src/agents/data_price_kr.py). 이미 "한국주식 도메인"으로 라우팅된 뒤의 세부 위임만
다루므로 판단 로직은 간단한 키워드 휴리스틱(+옵션 llm_fn)으로 충분하다.

핵심 책임:
1) 질문에서 종목명(예: "삼성전자") 또는 6자리 종목코드를 찾아 stock_code로 변환한다
   (company 테이블, execute_sql(HA-1 실행기) 경유 — conn.execute() 직접 호출 금지).
2) 질문이 재무데이터/주가·기술지표/둘 다 중 무엇을 요구하는지 판단해(classify_intent)
   해당 데이터 에이전트에 위임한다.
3) 결과를 종합해 반환한다 — 재무 결과(resolve_metric)와 주가 결과(get_price_snapshot_kr)
   를 각각 구분 가능한 키에 담는다.

"가까운 계층 재시도" (Constraints): 데이터 에이전트 호출이 실패(예외)하면 이 도메인
에이전트가 같은 호출을 즉시 1회 재시도하고, 그래도 실패하면 예외를 전파하지 않고
실패 사유를 반환값의 errors에 담는다 — 상위 총괄 에이전트까지 예외가 뚫고 올라가지
않게 하는 것이 목적이다.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from src.agents.data_financial import METRIC_SOURCE_MAP, resolve_metric
from src.agents.data_price_kr import get_price_history_kr, get_price_snapshot_kr
from src.agents.exec_runtime import execute_sql
from src.backtest.data_access import METRIC_FIELD_DESCRIPTIONS
from src.backtest.data_access_us import METRIC_FIELD_DESCRIPTIONS_US
from src.backtest.primitives import combine, get_cross_section
from src.llm import extract_json
from src.version import quarter_end_date

# _screening_prompt는 KR/US 공용 함수(domain_us.py가 그대로 import해 재사용)라 두 도메인의
# 지표 설명 단일 정의처(canonical source)를 모두 여기서 갖고 있다가 domain 인자로 고른다.

_STOCK_CODE_RE = re.compile(r"\d{6}")

# 지표명(한국어) → METRIC_SOURCE_MAP(src/agents/data_financial.py) 키. 질문에서 흔히
# 쓰이는 한국어 표현만 최소한으로 다룬다(복잡한 NL 파싱은 이 에이전트의 책임이 아님).
_METRIC_KO_ALIASES: dict[str, str] = {
    "매출액": "revenue",
    "매출": "revenue",
    "영업이익": "operating_profit",
    "당기순이익": "net_income",
    "순이익": "net_income",
    "자산총계": "total_assets",
    "부채총계": "total_liabilities",
    "자본총계": "total_equity",
    "배당": "dividend",
    "목표주가": "target_price",
    "투자의견": "analyst_opinion",
}

_PRICE_KEYWORDS: list[str] = [
    "주가", "가격", "시가", "고가", "저가", "종가", "거래량",
    "이동평균", "이평선", "rsi", "macd", "볼린저", "차트", "기술적",
]

_FINANCIAL_KEYWORDS: list[str] = (
    list(METRIC_SOURCE_MAP) + list(_METRIC_KO_ALIASES) + ["재무", "실적", "지표"]
)


def _escape_sql_literal(value: str) -> str:
    """SQL 문자열 리터럴에 안전하게 끼워 넣기 위해 작은따옴표를 이스케이프한다.

    execute_sql은 파라미터 바인딩을 지원하지 않으므로(sql 문자열 하나만 받음) 질문
    텍스트를 SQL에 직접 문자열로 끼워 넣기 전에 반드시 이 처리를 거친다(주입 방지).
    """
    return value.replace("'", "''")


def find_stock_code(
    conn,
    question: str,
    execute_sql_fn: Callable | None = None,
) -> str | None:
    """질문 텍스트에서 종목코드(6자리 숫자) 또는 종목명을 찾아 종목코드로 변환한다.

    1) 질문에 6자리 숫자가 있으면 그대로 종목코드로 쓴다(가장 신뢰도 높음).
    2) 없으면 company 테이블에서 "질문이 그 회사명을 포함하는" 행을 역방향 LIKE로
       찾는다(`question LIKE '%'||name||'%'`와 동등). 여러 회사명이 부분 포함될 수
       있어(예: "SK" vs "SK하이닉스") 가장 긴(구체적인) 이름을 우선한다.
    3) SQL은 execute_sql(HA-1 실행기)로만 실행한다(conn.execute() 직접 호출 금지).

    매치가 없으면 None을 반환한다.
    """
    execute_sql_fn = execute_sql_fn or execute_sql
    text = question or ""
    code_match = _STOCK_CODE_RE.search(text)
    if code_match:
        return code_match.group(0)
    if not text.strip():
        return None
    escaped = _escape_sql_literal(text)
    sql = (
        f"SELECT stock_code FROM company WHERE '{escaped}' LIKE '%' || name || '%' "
        "AND name IS NOT NULL AND name != '' ORDER BY LENGTH(name) DESC LIMIT 1"
    )
    result = execute_sql_fn(sql, conn)
    if not result.get("ok") or not result["rows"]:
        return None
    return result["rows"][0]["stock_code"]


def find_stock_codes(
    conn,
    question: str,
    execute_sql_fn: Callable | None = None,
) -> list[str]:
    """질문 텍스트에 언급된 모든 종목(복수)을 종목코드 리스트로 찾는다.

    "삼성전자와 SK하이닉스 종가 알려줘"처럼 한 질문이 이름으로 여러 종목을 지목하는
    경우를 위한 것이다(find_stock_code 단수는 회사명 매치가 여러 건이어도 가장 긴
    이름 하나만 골라 반환하므로, 이런 질문에서는 나머지 종목이 통째로 누락된다).

    1) 질문에 등장하는 모든 6자리 종목코드를 먼저 모은다(순서 보존, 중복 제거).
    2) company 테이블에서 질문에 이름이 부분 포함되는 모든 행을 이름 길이 내림차순으로
       조회한다. 이미 선택한(더 긴) 이름의 부분 문자열인 후보(예: "SK하이닉스"를 이미
       선택했다면 그 안에 포함된 "SK")는 건너뛴다 — find_stock_code의 "가장 구체적인
       이름 우선" 원칙을 "하나만 고르기"가 아니라 "겹치는 후보만 제거하기"로 확장한 것.
    3) 위 두 결과를 종목코드 기준으로 중복 제거해 합친다.

    매치가 없으면 빈 리스트. SQL은 execute_sql(HA-1 실행기)로만 실행한다.
    """
    execute_sql_fn = execute_sql_fn or execute_sql
    text = question or ""

    codes: list[str] = list(dict.fromkeys(_STOCK_CODE_RE.findall(text)))

    if text.strip():
        escaped = _escape_sql_literal(text)
        sql = (
            f"SELECT stock_code, name FROM company WHERE '{escaped}' LIKE '%' || name || '%' "
            "AND name IS NOT NULL AND name != '' ORDER BY LENGTH(name) DESC"
        )
        result = execute_sql_fn(sql, conn)
        if result.get("ok"):
            accepted_names: list[str] = []
            for row in result["rows"]:
                name = row["name"]
                if any(name in accepted for accepted in accepted_names):
                    continue
                accepted_names.append(name)
                if row["stock_code"] not in codes:
                    codes.append(row["stock_code"])

    return codes


# ── 스크리닝(다중종목 랭킹) 경로 (HA-15) ─────────────────────────────────────
# 실사용 핵심 기능 회귀 복구: "PER 낮은 5개사" 처럼 여러 종목을 조건으로 순위 매겨 뽑는
# 질문은 단일종목 조회(find_stock_code) 경로로는 답할 수 없다. 이미 검증된 백테스트
# 크로스섹션 인프라(get_cross_section/combine → select_stocks._validate_criteria_keys)를
# 그대로 재사용해 스크리닝 경로를 추가한다. LLM은 원시 SQL이 아니라 작은 구조화 JSON
# (criteria/top_n/sectors/markets)만 생성하므로 CASE WHEN 누락류 SQL 버그가 재도입될 수 없다.

# 스크리닝 의도 감지(결정론적 키워드) — src/legacy/graph/heuristic.py 의 방향/개수 키워드 관례를
# 따른다(오탐 0건으로 검증된 스타일). 단일종목 질문("삼성전자 PER 알려줘")은 개수/랭킹 신호가
# 없어 매치되지 않는다.
_SCREENING_COUNT_RE = re.compile(r"\d+\s*(개사|개|곳|위|종목|companies?|stocks?)")
_SCREENING_RANK_WORDS: tuple[str, ...] = (
    "가장", "제일", "상위", "하위", "순위", "랭킹", "top", "골라", "뽑아", "추천",
)
_SCREENING_STRONG_WORDS: tuple[str, ...] = ("순위", "랭킹", "골라", "뽑아", "추천")
_SCREENING_DIRECTION_WORDS: tuple[str, ...] = (
    "낮은", "높은", "작은", "큰", "많은", "적은", "최저", "최고", "저평가", "고평가",
)

# 스크리닝 지표 별칭(한국어/영문) → get_cross_section(metrics_at) 출력 필드명.
# 긴 키워드 우선 매칭(영업이익률 > 영업이익)을 위해 사용 시 길이 내림차순으로 순회한다.
_SCREEN_METRIC_ALIASES: dict[str, str] = {
    "per": "per", "주가수익비율": "per",
    "pbr": "pbr", "주가순자산비율": "pbr",
    "psr": "psr", "주가매출비율": "psr",
    "roe": "roe", "자기자본이익률": "roe",
    "roa": "roa", "총자산이익률": "roa",
    "영업이익률": "operating_margin",
    "순이익률": "net_margin",
    "부채비율": "debt_ratio",
    "매출성장": "revenue_growth", "매출증가": "revenue_growth",
    "영업이익성장": "op_growth",
    "순이익성장": "ni_growth",
    # 절대값(원화, 단일분기) 별칭. 위 비율/성장률 별칭보다 짧은 문자열이라
    # _SCREEN_METRIC_ORDER(길이 내림차순 + 첫 매치에서 break)가 항상 비율/성장률
    # 별칭을 먼저 매치시킨다 — "영업이익률"/"영업이익성장"이 "영업이익"보다 길어
    # 먼저 검사되므로 서로 안 섞인다(회귀 테스트로 확인, tests/test_absolute_financial_fields.py).
    "영업이익": "operating_profit",
    "매출액": "revenue",
    "매출": "revenue",
    "순이익": "net_income",
    "12개월수익률": "return_12m", "12개월 수익률": "return_12m",
    "가격수익률": "return_12m", "주가수익률": "return_12m", "모멘텀": "return_12m",
    # 마법공식(그린블라트)/GPA 팩터 — 스크리닝 경로는 LLM이 프롬프트의 필드설명을
    # 직접 읽어 인식하지만, 단일종목 조회 경로(_extract_metric)는 이 사전에 없으면
    # 절대 인식하지 못한다(실서버 재현 버그: "삼성전자 투하자본수익률" 결정론적 실패).
    "투하자본수익률": "roc", "roc": "roc",
    "이익수익률": "earnings_yield", "ey": "earnings_yield",
    "매출총이익률": "gp_a", "gpa": "gp_a", "gp/a": "gp_a",
}
_SCREEN_METRIC_ORDER: list[str] = sorted(_SCREEN_METRIC_ALIASES, key=len, reverse=True)

# LLM 프롬프트에 노출하는 한국 유니버스 지표 key(get_cross_section/metrics_at 출력 스키마).
# METRIC_FIELD_DESCRIPTIONS(단일 정의처)에서 key만 파생한다 — 손으로 다시 베껴 적지 않는다.
_KR_SCREEN_FIELDS: tuple[str, ...] = tuple(METRIC_FIELD_DESCRIPTIONS.keys())

# ── 단일종목 조회 경로에서 쓰는 "계산전용" 지표 (HA-6 return_12m 배선 복구) ────────────
# metrics_at()이 계산하는 지표 중 resolve_metric(DART financials/metrics 테이블 조회 전용,
# METRIC_SOURCE_MAP)로는 조회할 수 없는 것만 골라낸다. per/pbr/roe/operating_margin/
# debt_ratio는 이미 metrics 테이블에 사전계산돼 resolve_metric으로 정상 동작하므로 제외
# 한다(기존 라우팅을 그대로 유지 — 회귀 방지). "12개월 수익률"처럼 가격 시계열에서
# 즉석 계산해야만 하는 지표(return_12m 등)가 단일종목 질문에서 인식되지 않던 문제를 고친다.
_COMPUTED_ONLY_FIELDS: tuple[str, ...] = tuple(
    f for f in _KR_SCREEN_FIELDS if f not in METRIC_SOURCE_MAP
)
# 별도 별칭표를 새로 만들지 않고 스크리닝 경로의 _SCREEN_METRIC_ALIASES를 그대로 재사용해
# 위 계산전용 지표들의 한국어/영문 별칭만 추린다(중복 정의 금지).
_COMPUTED_KO_ALIASES: dict[str, str] = {
    ko: field for ko, field in _SCREEN_METRIC_ALIASES.items() if field in _COMPUTED_ONLY_FIELDS
}
# 긴 별칭 우선 매칭("매출성장"이 "매출"의 상위 문자열이라, 더 구체적인 계산지표 별칭이
# _METRIC_KO_ALIASES의 짧은 재무지표 별칭보다 먼저 매치돼야 한다) — _SCREEN_METRIC_ORDER와
# 동일한 관례.
_COMPUTED_METRIC_ORDER: list[str] = sorted(_COMPUTED_KO_ALIASES, key=len, reverse=True)


def _is_screening_question_heuristic(question: str) -> bool:
    """질문이 '여러 종목을 조건으로 순위 매겨 상위 N개를 뽑는' 스크리닝 의도인지 판정한다.

    결정론적 키워드 기반(LLM 불필요, 오탐 0건으로 검증된 기존 로직 — 재작성 금지). 강한
    랭킹어(순위/랭킹/골라/뽑아/추천)는 그 자체로, 또는 개수 신호(N개/종목/곳/위)와 방향·랭킹어가
    함께 있으면 스크리닝으로 본다. 단일종목 질문("삼성전자 PER 알려줘")은 개수/랭킹 신호가
    없어 매치되지 않는다(오탐 방지). is_screening_question 의 LLM 폴백 안전망으로 쓰인다.
    """
    q = (question or "").lower()
    if any(w in q for w in _SCREENING_STRONG_WORDS):
        return True
    has_count = bool(_SCREENING_COUNT_RE.search(q))
    has_rank = any(w in q for w in _SCREENING_RANK_WORDS)
    has_direction = any(w in q for w in _SCREENING_DIRECTION_WORDS)
    return has_count and (has_rank or has_direction)


def _screening_intent_prompt(question: str) -> str:
    """is_screening_question 의 LLM 판단 프롬프트. classify_intent/_intent_prompt 와 동일 관례.

    "저PER 5종목"처럼 키워드 목록(_SCREENING_DIRECTION_WORDS 등)에 정확히 없는 표현도 LLM이
    의미로 판단하게 한다 — 키워드를 계속 추가하는 대신, 판단 자체를 LLM에 맡기고 키워드는
    안전망(폴백)으로만 남긴다.
    """
    return (
        "다음 질문이 여러 종목을 특정 조건으로 순위 매겨 상위 N개를 뽑아달라는 "
        "'스크리닝(다중종목 랭킹)' 요청인지 판단하세요.\n"
        "예(스크리닝=yes): 'PER 낮은 5종목', '저PER 5종목', '저평가된 5개 추천', '순위 매겨줘'.\n"
        "예(스크리닝=no): 특정 한 종목(회사명/티커)의 정보를 묻는 질문(예: '삼성전자 PER 알려줘', "
        "'AAPL 주가 알려줘').\n"
        "yes 또는 no 로만 답하세요.\n\n"
        f"질문: {question}\n답:"
    )


_SCREENING_INTENT_YES_RE = re.compile(r"\b(yes|true)\b")
_SCREENING_INTENT_NO_RE = re.compile(r"\b(no|false)\b")


def _parse_screening_intent(raw: str | None) -> bool | None:
    """LLM 응답에서 스크리닝 여부(yes/no)를 판정한다. 불명확하면 None(→ 키워드 폴백).

    부정 신호(아니오/아니요/아닙니다/no/false)를 긍정 신호(예/네/맞습니다/yes/true)보다 먼저
    확인한다 — 한국어 부정 표현이 흔히 더 긴 문장에 섞여 나오는 관례를 고려한 순서.
    """
    t = (raw or "").strip().lower()
    if not t:
        return None
    negative = bool(_SCREENING_INTENT_NO_RE.search(t)) or any(
        w in t for w in ("아니오", "아니요", "아닙니다", "아님")
    )
    positive = bool(_SCREENING_INTENT_YES_RE.search(t)) or any(
        w in t for w in ("예", "네", "맞습니다", "맞음")
    )
    if negative and not positive:
        return False
    if positive and not negative:
        return True
    return None


def is_screening_question(question: str, llm_fn: Callable[[str], str] | None = None) -> bool:
    """질문이 '여러 종목을 조건으로 순위 매겨 상위 N개를 뽑는' 스크리닝 의도인지 판정한다.

    route_question(supervisor.py)/classify_intent 와 동일한 **LLM 우선 + 키워드 안전망 폴백**
    패턴이다 — llm_fn 이 주어지면 먼저 LLM 에 판단을 위임하고(_screening_intent_prompt), 응답을
    yes/no 로 파싱할 수 있으면 그 결과를 채택한다(키워드 판단과 달라도 LLM 판단이 우선).
    llm_fn 이 없거나 예외/파싱 실패면 결정론 키워드 휴리스틱(_is_screening_question_heuristic,
    오탐 0건으로 검증된 기존 로직)으로 폴백한다. "저PER 5종목"처럼 키워드 목록에 없는 표현도
    llm_fn 이 있으면 인식되지만, 키워드 자체를 늘리지는 않는다(사용자 명시 지시).
    """
    if llm_fn is not None:
        try:
            raw = llm_fn(_screening_intent_prompt(question)) or ""
        except Exception:  # noqa: BLE001 — LLM 실패는 키워드 폴백으로 흡수
            raw = ""
        verdict = _parse_screening_intent(raw)
        if verdict is not None:
            return verdict
    return _is_screening_question_heuristic(question)


def _screening_prompt(
    question: str,
    fields: tuple[str, ...],
    sectors: tuple[str, ...] = (),
    domain: str = "KR",
) -> str:
    """스크리닝 스펙 추출 프롬프트. domain 에 따라 시장 판단 범위를 도메인 전용으로 스코프한다.

    같은 원본 질문 텍스트를 KR/US 두 도메인이 각각 받더라도, 프롬프트 자체가 "너는 지금 이
    도메인만 담당한다"를 명시해 다른 나라 시장 언급을 무시하게 만든다("코스피와 나스닥 각각…"
    같은 혼재 질문에서 US 호출이 '코스피'에 오염되던 회귀 방지). 시장 구분은 하드코딩 키워드
    규칙이 아니라 LLM이 질문 의미를 읽고 판단하며, 값 후보만 도메인별로 제한한다.

    지표 목록은 "key만"이 아니라 "key: 한글설명"으로 나열한다(METRIC_FIELD_DESCRIPTIONS(_US)
    단일 정의처에서 조회) — LLM이 별도 한글→필드명 별칭사전 없이도 설명만 보고 "영업이익"
    질문을 operating_profit으로 스스로 매핑할 수 있게 한다. fields 는 노출할 key의 집합/순서만
    결정하고, 설명 텍스트는 domain 에 맞는 딕셔너리에서 가져온다(둘이 항상 같은 목록이도록
    _KR_SCREEN_FIELDS/_US_SCREEN_FIELDS 가 이 딕셔너리에서 파생되므로 자동 동기화된다).
    """
    descriptions = METRIC_FIELD_DESCRIPTIONS_US if domain == "US" else METRIC_FIELD_DESCRIPTIONS
    fields_block = "\n".join(f"  - {k}: {descriptions.get(k, k)}" for k in fields)
    sector_hint = (
        f"실제 업종(sector) 목록: {', '.join(sectors)}.\n"
        "질문의 업종 표현이 이 목록에 정확히 없으면 가장 가까운 실제 항목으로 매핑하세요"
        "(예: 반도체/전자부품→전기·전자, 게임→소프트웨어·게임, 자동차→운송장비·부품). "
        "매핑할 항목이 전혀 없으면 sectors를 생략(null)하세요.\n"
        if sectors else ""
    )
    if domain == "US":
        market_key = "exchanges"
        scope_line = (
            "이 질문에서 미국 시장(나스닥/뉴욕증권거래소) 관련 조건만 판단하세요. 질문에 한국 등 "
            "다른 나라 시장이 함께 언급돼 있어도 무시하고 미국 부분만 봅니다.\n"
        )
        market_rule = (
            'exchanges: 질문이 미국 시장 중 나스닥만 뜻하면 ["NASDAQ"], 뉴욕증권거래소(NYSE)만 '
            '뜻하면 ["NYSE"], 미국 시장 구분이 없거나 둘 다면 null.\n'
        )
    else:
        market_key = "markets"
        scope_line = (
            "이 질문에서 한국 시장(코스피/코스닥) 관련 조건만 판단하세요. 질문에 미국 등 다른 "
            "나라 시장이 함께 언급돼 있어도 무시하고 한국 부분만 봅니다.\n"
        )
        market_rule = (
            'markets: 질문이 한국 시장 중 코스피만 뜻하면 ["KOSPI"], 코스닥만 뜻하면 '
            '["KOSDAQ"], 한국 시장 구분이 없거나 둘 다면 null.\n'
        )
    return (
        "다음 질문은 여러 종목을 특정 지표로 순위 매겨 상위 N개를 뽑는 스크리닝 요청입니다.\n"
        "질문에서 조건만 추출해 JSON으로만 답하세요(설명/코드/SQL 금지).\n"
        '형식: {"criteria":[{"key":"<지표>","direction":"low|high"}],'
        f'"top_n":<정수>,"sectors":null,"{market_key}":null}}\n'
        f"사용 가능한 지표(key: 설명):\n{fields_block}\n"
        "direction: 낮을수록/저평가 우수면 low, 높을수록 우수면 high.\n"
        "top_n: 질문에 개수가 명시돼 있으면 그 숫자를, 명시돼 있지 않으면 10을, "
        "'전체'/'모든 종목'/'모두'처럼 개수 제한 없이 전부를 요구하면 4000을 쓰세요.\n"
        f"{scope_line}"
        f"{market_rule}"
        f"{sector_hint}\n"
        f"질문: {question}\n답:"
    )


def _normalize_criteria(raw_criteria: list) -> list[dict] | None:
    """LLM JSON 의 criteria 리스트를 combine/select_stocks 가 받는 형태로 정규화한다.

    각 항목은 최소 key(str)를 가져야 한다 — 없으면 파싱 실패(None)로 처리해 휴리스틱 폴백을
    유발한다. direction 은 low|high 로 강제하고(그 외/누락 시 low), weight 는 있으면 유지한다.
    (존재하지 않는 지표명 자체는 combine→_validate_criteria_keys 가 ValueError 로 잡는다.)
    """
    norm: list[dict] = []
    for c in raw_criteria:
        if not isinstance(c, dict) or "key" not in c:
            return None
        key = str(c["key"]).strip().lower()
        if not key:
            return None
        direction = str(c.get("direction", "low")).strip().lower()
        if direction not in ("low", "high"):
            direction = "low"
        entry = {"key": key, "direction": direction}
        if "weight" in c:
            try:
                entry["weight"] = float(c["weight"])
            except (TypeError, ValueError):
                pass
        norm.append(entry)
    return norm or None


def _parse_screening_json(raw: str, domain: str = "KR") -> dict | None:
    """LLM 응답에서 스크리닝 스펙(criteria/top_n/sectors/markets|exchanges)을 추출한다.

    criteria 를 정상 파싱하지 못하면 None(→ 결정론 휴리스틱 폴백). 존재하지 않는 필드명이
    섞여 있어도 여기서는 통과시키고, 실제 필드 검증은 combine 이 수행한다(조용한 빈 결과 방지).

    시장 필터는 도메인 전용으로 읽는다 — KR 은 markets(코스피/코스닥), US 는 exchanges
    (나스닥/뉴욕). 프롬프트가 도메인 스코프를 명시하므로 자기 도메인 키만 읽어 혼재 질문에서
    다른 도메인 값이 섞이지 않게 한다(항상 두 키를 spec 에 두되 반대 도메인 값은 None).
    """
    data = extract_json(raw)
    if not isinstance(data, dict):
        return None
    raw_criteria = data.get("criteria")
    if not isinstance(raw_criteria, list) or not raw_criteria:
        return None
    criteria = _normalize_criteria(raw_criteria)
    if criteria is None:
        return None
    spec = {
        "criteria": criteria,
        "top_n": _coerce_top_n(data.get("top_n")),
        "sectors": data.get("sectors") or None,
        "markets": None,
        "exchanges": None,
    }
    if domain == "US":
        spec["exchanges"] = data.get("exchanges") or None
    else:
        spec["markets"] = data.get("markets") or None
    return spec


def _coerce_top_n(value, default: int = 10) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    # 하드 상한은 pipeline_exec.MAX_SIZE(전체 종목 안전 상한)와 동일하게 맞춘다 —
    # "전체 종목" 요청이 top_n=4000까지는 잘리지 않고 그대로 통과해야 하기 때문.
    return max(1, min(4000, n))


def _heuristic_screening_spec(question: str, domain: str = "KR") -> dict | None:
    """LLM 없이(또는 JSON 파싱 실패 시) 결정론적 키워드로 스크리닝 스펙을 만든다.

    heuristic.py 의 지표/방향/개수 감지 관례를 스크리닝 지표(per/pbr/roe/…)에 맞춰 재현한다.
    지표를 하나도 못 찾으면 None(→ 호출부가 명시적 오류를 남김).

    시장 필터도 도메인 스코프를 지킨다 — KR 은 코스피/코스닥만 markets 로, US 는 나스닥/뉴욕만
    exchanges 로 잡는다. 혼재 질문("코스피와 나스닥…")의 US 폴백 경로에서 '코스피'가 markets 를
    오염시키던 문제(LLM 프롬프트와 동일한 회귀)를 폴백 경로에서도 막는다.
    """
    q = (question or "").lower()
    metric = None
    for kw in _SCREEN_METRIC_ORDER:
        if kw in q:
            metric = _SCREEN_METRIC_ALIASES[kw]
            break
    if metric is None:
        return None

    high_words = ("높은", "높게", "큰", "많은", "최고", "고평가")
    direction = "high" if any(w in q for w in high_words) else "low"

    count_match = re.search(r"(\d+)", q)
    if count_match:
        top_n = _coerce_top_n(count_match.group(1))
    elif any(w in q for w in ("전체", "모든", "모두")):
        top_n = _coerce_top_n(4000)  # 개수 제한 없이 전부 요구 → 상한까지 채운다
    else:
        top_n = 10

    markets = None
    exchanges = None
    if domain == "US":
        if "나스닥" in q or "nasdaq" in q:
            exchanges = ["NASDAQ"]
        elif "뉴욕" in q or "nyse" in q:
            exchanges = ["NYSE"]
    else:
        if "코스피" in q or "kospi" in q:
            markets = ["KOSPI"]
        elif "코스닥" in q or "kosdaq" in q:
            markets = ["KOSDAQ"]

    return {
        "criteria": [{"key": metric, "direction": direction, "weight": 1.0}],
        "top_n": top_n,
        "sectors": None,
        "markets": markets,
        "exchanges": exchanges,
    }


def _extract_screening_spec(
    question: str,
    llm_fn: Callable[[str], str] | None,
    fields: tuple[str, ...] = _KR_SCREEN_FIELDS,
    sectors: tuple[str, ...] = (),
    domain: str = "KR",
) -> tuple[dict | None, str | None]:
    """스크리닝 스펙을 (spec, error) 로 반환한다. LLM JSON 우선, 실패 시 결정론 휴리스틱 폴백.

    LLM 이 파싱 가능한 criteria 를 주면 그대로 채택한다(존재하지 않는 필드명이어도 combine 이
    잡도록 통과) — 조용히 휴리스틱으로 바꿔치지 않는다. LLM 이 없거나 파싱 실패면 휴리스틱으로
    폴백하고, 그것도 지표를 못 찾으면 (None, 사유) 를 돌려준다.

    sectors 는 실제 DB 에 존재하는 업종 목록(실사용에서 발견: KRX 분류엔 "반도체" 카테고리가
    없고 "전기·전자"로 흡수돼 있어, 이 목록 없이는 LLM 이 구어체 업종명을 그대로 돌려줘
    조용히 빈 결과로 이어졌다) — 프롬프트에 포함시켜 LLM 이 실제 항목으로 매핑하게 한다.

    domain 은 시장 판단 스코프(KR=코스피/코스닥→markets, US=나스닥/뉴욕→exchanges)를 프롬프트/
    파싱/휴리스틱 전 경로에 일관되게 전달한다 — 혼재 질문의 도메인 간 오염 방지 핵심.
    """
    if llm_fn is not None:
        try:
            raw = llm_fn(_screening_prompt(question, fields, sectors, domain=domain)) or ""
        except Exception:  # noqa: BLE001 — LLM 실패는 휴리스틱 폴백으로 흡수
            raw = ""
        spec = _parse_screening_json(raw, domain=domain)
        if spec is not None:
            return spec, None
    spec = _heuristic_screening_spec(question, domain=domain)
    if spec is not None:
        return spec, None
    return None, "질문에서 스크리닝 지표/방향을 해석하지 못함"


def _default_screening_asof(
    conn,
    table: str,
    execute_sql_fn: Callable | None = None,
) -> str | None:
    """스크리닝 기준시점(asof) = 해당 가격 테이블의 최신 거래일(실제 존재하는 시점).

    캘린더 오늘이 아니라 DB 에 실제로 있는 최신 날짜를 쓴다(data_price_kr/us 의 asof 원칙 동일).
    conn.execute 직접 호출 대신 HA-1 실행기(execute_sql)를 경유한다(도메인 에이전트 관례).
    """
    execute_sql_fn = execute_sql_fn or execute_sql
    result = execute_sql_fn(f"SELECT MAX(date) AS d FROM {table}", conn)
    if not result.get("ok") or not result["rows"]:
        return None
    return result["rows"][0].get("d")


def _resolve_screening_asof(
    period: dict | None,
    conn,
    table: str,
    execute_sql_fn: Callable | None = None,
) -> str | None:
    """스크리닝 질문의 기간(period)을 실제 asof 날짜로 확정한다.

    period가 없으면(질문에 연도/분기 언급이 없으면) None을 그대로 반환한다 — 호출부가
    이 None을 answer_kr_screening/answer_us_screening의 asof에 그대로 넘기면 기존과 동일하게
    _default_screening_asof(전체 최신 거래일)로 폴백한다. period가 있으면 그 기간의 말일
    (연도→12/31, 분기→quarter_end_date) '이하' 최신 거래일을 찾는다 — 스크리닝은 combine이
    바로 쓸 수 있는 구체적 날짜(asof) 하나가 필요하므로, resolve_metric처럼 quarter/year를
    그대로 넘길 수 없어 이 변환이 필요하다. "이하"인 이유: 요청 기간 말일이 휴장일/미래일
    수도 있으므로, 그 시점까지 실제로 존재하는 가장 최근 거래일을 찾는다(_default_screening_asof
    와 동일한 "캘린더가 아니라 DB에 실재하는 날짜" 원칙).
    """
    if period is None:
        return None
    execute_sql_fn = execute_sql_fn or execute_sql
    if period.get("kind") == "quarter":
        bound = quarter_end_date(period["quarter"]).isoformat()
    else:
        bound = f"{period['year']}-12-31"
    result = execute_sql_fn(f"SELECT MAX(date) AS d FROM {table} WHERE date <= '{bound}'", conn)
    if not result.get("ok") or not result["rows"]:
        return None
    return result["rows"][0].get("d")


def _normalize_override_spec(raw: dict) -> dict | None:
    """사용자가 편집해 재실행으로 넘긴 spec을 정규화한다(휴먼인더루프).

    _parse_screening_json과 동일한 정규화(_normalize_criteria/_coerce_top_n)를 거치므로,
    LLM이 생성했든 사용자가 직접 편집했든 이후 처리는 완전히 같은 경로를 탄다 — 검증을
    우회하는 별도 경로가 아니다. criteria가 없거나 형식이 어긋나면 None(→ 호출부가 기존과
    동일한 "스크리닝 조건 해석 실패"로 처리).
    """
    if not isinstance(raw, dict):
        return None
    criteria = _normalize_criteria(raw.get("criteria") or [])
    if criteria is None:
        return None
    return {
        "criteria": criteria,
        "top_n": _coerce_top_n(raw.get("top_n")),
        "sectors": raw.get("sectors") or None,
        "markets": raw.get("markets") or None,
        "exchanges": raw.get("exchanges") or None,
    }


def _run_screening(
    question: str,
    conn,
    llm_fn: Callable[[str], str] | None,
    cross_section_fn: Callable,
    combine_fn: Callable,
    price_table: str,
    fields: tuple[str, ...],
    asof: str | None,
    execute_sql_fn: Callable | None,
    domain: str = "KR",
    override_spec: dict | None = None,
    on_progress: Callable[..., None] | None = None,
) -> dict:
    """스크리닝 공용 실행부(KR/US 대칭). 스펙추출 → asof해석 → 크로스섹션 → combine → rows 반환.

    실패는 조용히 빈 결과가 아니라 result=None + errors 사유로 남긴다(이번 세션 초반 '필드 환각이
    조용히 빈 결과로 둔갑' 버그 재발 방지). combine 이 던지는 ValueError(존재하지 않는 필드 등)를
    명시적으로 잡아 사유를 기록한다.

    domain 은 시장 필터 스코프를 가른다 — KR 은 spec["markets"](코스피/코스닥), US 는
    spec["exchanges"](나스닥/뉴욕). 두 경우 모두 rows 의 'market' 필드로 필터링된다
    (metrics_at_us 가 exchange 값을 'market' 에 담으므로 combine 의 markets= 인자를 그대로 재사용).

    override_spec(휴먼인더루프 재실행용): 주어지면 LLM/휴리스틱 추출(_extract_screening_spec)을
    완전히 건너뛰고 이 값을 그대로 spec으로 쓴다 — 사용자가 실시간 트리에서 본 조건 JSON을
    직접 고쳐 재실행할 때, LLM을 다시 부르지 않고 고친 값 그대로 실행하기 위함이다. 이렇게
    받은 spec도 뒤 이은 업종검증/combine(존재하지 않는 지표명 거부 등)은 그대로 다 통과해야
    한다 — 안전장치를 우회하는 별도 경로가 아니라 "스펙의 출처만 다른" 같은 실행 경로다.

    on_progress(step, summary, detail=None)를 주면 spec이 확정되는 즉시(LLM 추출이든
    override든) 1건 통지한다 — detail에 확정된 spec 전체를 실어, 실시간 트리에서 "AI가
    이렇게 이해했다"를 사용자가 바로 확인/편집할 수 있게 한다(HA-12 확장).
    """
    result: dict = {
        "question": question, "intent": "screening",
        "criteria": None, "top_n": None, "sectors": None, "markets": None, "exchanges": None,
        "asof": None, "result": None, "errors": [],
    }

    resolved_asof = asof or _default_screening_asof(conn, price_table, execute_sql_fn)
    result["asof"] = resolved_asof
    if not resolved_asof:
        result["errors"].append(f"스크리닝 기준시점(asof)을 찾을 수 없습니다: {price_table} 데이터 없음")
        return result

    try:
        rows = cross_section_fn(conn, resolved_asof)
    except Exception as exc:  # noqa: BLE001 — 크로스섹션 조회 실패도 사유로 흡수
        result["errors"].append(f"횡단면(cross-section) 조회 실패: {type(exc).__name__}: {exc}")
        return result

    # KRX 미분류 등으로 sector가 비어있는 종목은 "기타"로 채운다 — 채우지 않으면 valid_sectors
    # 집계·업종 필터·화면 표시에서 조용히 누락된다(사용자가 실사용에서 발견한 회귀).
    for row in rows:
        if not row.get("sector"):
            row["sector"] = "기타"

    # 실제 존재하는 업종 목록(질문 해석 시 LLM에게 알려주고, 매칭 실패를 사후 검증하는 데도 쓴다).
    valid_sectors = sorted({r["sector"] for r in rows if r.get("sector")})

    if override_spec is not None:
        spec, err = _normalize_override_spec(override_spec), None
    else:
        spec, err = _extract_screening_spec(
            question, llm_fn, fields, sectors=tuple(valid_sectors), domain=domain
        )
    if spec is None:
        result["errors"].append(f"스크리닝 조건 해석 실패: {err}")
        return result
    result["criteria"] = spec["criteria"]
    result["top_n"] = spec["top_n"]
    result["sectors"] = spec["sectors"]
    result["markets"] = spec.get("markets")
    result["exchanges"] = spec.get("exchanges")

    if on_progress:
        on_progress(
            "code", f"{domain} 스크리닝 조건 생성 완료",
            detail={
                "kind": "screening_spec", "domain": domain.lower(), "spec": spec,
                "asof": resolved_asof,  # 기간 질문("2024년 수익률" 등)의 실제 기준일을 눈으로 확인 가능하게
            },
        )

    requested_sectors = spec["sectors"]
    if isinstance(requested_sectors, str):
        requested_sectors = [requested_sectors]
    if requested_sectors and valid_sectors and not (set(requested_sectors) & set(valid_sectors)):
        result["errors"].append(
            f"요청한 업종({', '.join(requested_sectors)})이 실제 데이터에 없습니다. "
            f"실제 업종 목록: {', '.join(valid_sectors)}"
        )
        return result

    # 시장/거래소 필터: KR 은 markets(코스피/코스닥), US 는 exchanges(나스닥/뉴욕)를 rows 의
    # 'market' 필드에 대해 적용한다(select_stocks 의 markets= 인자 재사용 — 대칭 구현).
    market_filter = spec.get("exchanges") if domain == "US" else spec.get("markets")

    try:
        selected = combine_fn(
            rows, spec["criteria"], method="zscore",
            n=spec["top_n"], sectors=requested_sectors, markets=market_filter,
        )
    except ValueError as exc:  # _validate_criteria_keys 등: 존재하지 않는 필드명(환각) 포함
        result["errors"].append(f"스크리닝 조건 오류: {exc}")
        return result
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"스크리닝 실행 실패: {type(exc).__name__}: {exc}")
        return result

    result["result"] = selected
    return result


def answer_kr_screening(
    question: str,
    conn,
    llm_fn: Callable[[str], str] | None = None,
    cross_section_fn: Callable | None = None,
    combine_fn: Callable | None = None,
    asof: str | None = None,
    execute_sql_fn: Callable | None = None,
    override_spec: dict | None = None,
    on_progress: Callable[..., None] | None = None,
) -> dict:
    """한국주식 스크리닝: 조건에 맞는 종목을 순위 매겨 상위 N개를 반환한다(다중종목 경로).

    LLM 에게 구조화 JSON(criteria/top_n/sectors/markets)만 생성시키고, 파싱 →
    get_cross_section(metrics_at, KR) → combine(=select_stocks) 를 호출한다. rows 는 가공
    없이 그대로 result 에 담는다. 존재하지 않는 지표명(환각)/조건 해석 실패는 조용한 빈 결과가
    아니라 errors 에 사유로 남긴다.

    cross_section_fn/combine_fn 은 테스트 주입용(기본=get_cross_section/combine). asof 미지정 시
    prices 테이블의 최신 거래일을 쓴다. override_spec/on_progress는 _run_screening 참고
    (휴먼인더루프 재실행 / 실시간 조건 JSON 통지).
    """
    cross_section_fn = cross_section_fn or (lambda c, a: get_cross_section(c, a))
    combine_fn = combine_fn or combine
    return _run_screening(
        question, conn, llm_fn, cross_section_fn, combine_fn,
        price_table="prices", fields=_KR_SCREEN_FIELDS,
        asof=asof, execute_sql_fn=execute_sql_fn,
        override_spec=override_spec, on_progress=on_progress,
    )


# ── 기간(period) 파싱 + 재시도 피드백 분리 (버그B/버그A 공용) ──────────────────
# supervisor.answer_with_verification 이 재시도 때 원본 질문 뒤에 붙이는 실패 피드백 마커.
# 기간 파싱/차트 의도 판단은 재시도 피드백이 아니라 **원본 질문**으로 해야 하므로
# (wants_chart 가 원본 question 으로 판단하는 것과 동일 원칙), 이 마커 앞부분만 떼어 쓴다.
_RETRY_FEEDBACK_MARKER = "\n\n[이전 시도 실패 피드백]"

# 연도 표현: 4자리(2025/2025년) 또는 2자리+년(25년). 6자리 종목코드/큰 숫자(200000)를
# 연도로 오인하지 않도록 4자리 연도는 앞뒤에 다른 숫자가 붙지 않은 경우만 인정한다.
_YEAR_FULL_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)\s*년?")
_YEAR_SHORT_RE = re.compile(r"(?<!\d)(\d{2})\s*년")
_QUARTER_KO_RE = re.compile(r"([1-4])\s*분기")
_QUARTER_Q_RE = re.compile(r"[qQ]\s*([1-4])")


def _strip_retry_feedback(question: str) -> str:
    """재시도 시 덧붙는 실패 피드백을 떼어내 원본 질문만 돌려준다(마커 없으면 그대로)."""
    return (question or "").split(_RETRY_FEEDBACK_MARKER, 1)[0]


def _parse_period(question: str) -> dict | None:
    """질문에서 조회 기간을 파싱한다. resolve_metric 의 period 인자 형식으로 반환한다.

    - 연도+분기("26년 1분기") → {"kind":"quarter","quarter":"2026Q1"}.
    - 연도만("25년 전체"/"2025년") → {"kind":"annual","year":2025}.
    - 기간 언급이 전혀 없거나 분기만 있고 연도가 없으면 → None(현행 최신분기 유지, 회귀 금지).

    호출부는 재시도 피드백이 섞이지 않도록 _strip_retry_feedback 을 거친 원본 질문을 넘긴다.
    """
    q = question or ""
    year: int | None = None
    m = _YEAR_FULL_RE.search(q)
    if m:
        year = int(m.group(1))
    else:
        m = _YEAR_SHORT_RE.search(q)
        if m:
            year = 2000 + int(m.group(1))
    if year is None:
        return None  # 연도가 없으면 기간 미지정으로 본다(분기 단독은 현행 유지)

    quarter_n: int | None = None
    m = _QUARTER_KO_RE.search(q)
    if m:
        quarter_n = int(m.group(1))
    else:
        m = _QUARTER_Q_RE.search(q)
        if m:
            quarter_n = int(m.group(1))

    if quarter_n is not None:
        return {"kind": "quarter", "quarter": f"{year}Q{quarter_n}"}
    return {"kind": "annual", "year": year}


# ── 주가 시계열(price_history) 첨부 (버그A) ───────────────────────────────────
# "최근 1년 주가 그래프"처럼 시계열/차트를 원하는 단일종목 가격 질문은, 단일 시점 스냅샷만
# 담아 보내면 검증기(verify_answer)가 "1년 기간을 충족 못 함"이라며 정당하게 반려한다.
# 그래서 이런 질문일 때만 get_price_history_kr(차트용으로 supervisor 가 이미 쓰는 함수)로
# 최근 1년 종가 시계열을 조회해 응답에 요약+시계열을 함께 담는다. 판단은 wants_chart 와
# 동일하게 원본 질문 기준(부분일치 키워드)이며, 일반 단발 시세 질문에는 붙이지 않는다.
_PRICE_HISTORY_KEYWORDS: tuple[str, ...] = (
    "그래프", "차트", "그려", "시각화", "plot", "chart",
    "추이", "추세", "시계열", "1년", "일년", "최근 1년", "히스토리", "history",
)


def _wants_price_history(question: str) -> bool:
    q = (question or "").lower()
    return any(kw.lower() in q for kw in _PRICE_HISTORY_KEYWORDS)


def _summarize_price_history(history: list[dict]) -> dict:
    """get_price_history_* 시계열(과거→최신)을 검증/프론트가 쓸 수 있는 요약+배열로 담는다."""
    return {
        "days": 365,
        "count": len(history),
        "start_date": history[0].get("date"),
        "end_date": history[-1].get("date"),
        "series": [{"date": r.get("date"), "close": r.get("close")} for r in history],
    }


def _extract_metric(question: str) -> str | None:
    """질문에서 재무 지표명을 인식한다.

    계산전용 지표 별칭(_COMPUTED_KO_ALIASES, 예: "매출성장"→revenue_growth)을 가장 먼저
    확인한다 — 더 구체적인(긴) 별칭이 그 상위 문자열인 짧은 재무지표 별칭(예: "매출"→revenue)
    보다 먼저 매치돼야 하기 때문이다. 그다음 METRIC_SOURCE_MAP 키(영문, 대소문자 무시),
    마지막으로 한국어 별칭(_METRIC_KO_ALIASES) 순으로 먼저 매치되는 것을 쓴다.
    """
    q = question.lower()
    for ko in _COMPUTED_METRIC_ORDER:
        if ko in q:
            return _COMPUTED_KO_ALIASES[ko]
    for key in METRIC_SOURCE_MAP:
        if key in q:
            return key
    for ko, key in _METRIC_KO_ALIASES.items():
        if ko in question:
            return key
    return None


def resolve_computed_metric(
    conn,
    stock_code: str,
    metric: str,
    asof: str | None = None,
    execute_sql_fn: Callable | None = None,
    cross_section_fn: Callable | None = None,
) -> dict:
    """metrics_at()이 계산하는 가격파생 지표(_COMPUTED_ONLY_FIELDS)를 단일 종목에 대해 조회한다.

    resolve_metric(DART financials/metrics 테이블 전용)로는 다룰 수 없는 return_12m 등의
    계산 지표를 위해 존재한다. get_cross_section(=metrics_at 래핑, 스크리닝 경로와 동일
    인프라)으로 특정 기준시점(prices 테이블) 스냅샷을 계산한 뒤 해당 종목 행 하나만
    골라낸다 — 새 SQL/계산 로직을 추가하지 않고 기존 인프라를 그대로 재사용한다.

    asof를 주면(예: "삼성전자 25년 투하자본수익률"처럼 질문에 연도/분기가 명시된 경우,
    호출부가 _resolve_screening_asof로 확정해 넘김) 그 시점 그대로 쓴다. 생략하면(기존
    호출부 하위호환) _default_screening_asof(prices 테이블 최신 거래일)로 폴백한다 — 이
    폴백이 없으면 "질문은 2025년을 물었는데 결과는 오늘 날짜로 계산됨"이라는 실서버
    재현 버그(검증이 매번 결정론적으로 실패)가 재발한다.

    반환 형식은 resolve_metric()과 동일한 계약에 "estimated"를 더한다: {"stock_code",
    "metric","value","source","period","estimated"}. source="computed"는 재무제표가 아니라
    가격 시계열에서 즉석 계산된 값임을 표시하고, period는 계산 기준시점(asof)이다.
    estimated는 행에 '{metric}_estimated' 컴패니언 필드(예: roc_estimated — 감가상각비
    데이터가 없어 0으로 근사했는지)가 있으면 그 값을, 없으면 None을 담는다 — metrics_at()이
    계산한 근사 여부가 단일종목 조회에서도 조용히 사라지지 않게 한다.
    """
    execute_sql_fn = execute_sql_fn or execute_sql
    cross_section_fn = cross_section_fn or (lambda c, a: get_cross_section(c, a))
    asof = asof or _default_screening_asof(conn, "prices", execute_sql_fn)
    value = None
    estimated = None
    if asof:
        row = next(
            (r for r in cross_section_fn(conn, asof) if r.get("stock_code") == stock_code),
            None,
        )
        if row is not None:
            value = row.get(metric)
            estimated = row.get(f"{metric}_estimated")
    return {
        "stock_code": stock_code, "metric": metric,
        "value": value, "source": "computed", "period": asof, "estimated": estimated,
    }


def _intent_prompt(question: str) -> str:
    """classify_intent 용 LLM 판단 프롬프트(재무/주가/둘다). route_question 의 _route_prompt 와 동일 관례.

    시장 구분(markets/exchanges)과 마찬가지로, 재무 vs 주가 구분도 하드코딩 키워드가 아니라
    LLM 이 질문 의미를 읽고 판단하게 한다. LLM 은 코드가 아니라 정해진 셋(financial/price/both)
    중 하나만 고른다.
    """
    return (
        "다음 질문이 어떤 데이터를 원하는지 판단하세요.\n"
        "- financial: 재무데이터(매출/영업이익/순이익/PER/PBR/ROE 등). 매출성장률·12개월 수익률·"
        "모멘텀처럼 값으로 계산되는 '지표' 요청도 financial 로 봅니다(단순 시세 조회가 아님).\n"
        "- price: 주가·기술지표(종가/시가/거래량/이동평균/RSI/MACD/볼린저 등)\n"
        "- both: 재무와 주가를 모두 원함\n"
        "financial, price, both 중 하나만 답하세요.\n\n"
        f"질문: {question}\n답:"
    )


_INTENT_TOKEN_RE = re.compile(r"\b(financial|price|both)\b")


def _parse_intent(raw: str | None) -> str | None:
    r"""LLM 응답에서 financial/price/both 를 뽑는다. 명확한 신호가 없으면 None(→ 키워드 폴백).

    단어경계(\b) 매칭이라 'financials' 같은 부분문자열에 오탐하지 않는다. 한국어 응답
    (재무/주가/둘)도 관대하게 수용한다. 재무·주가가 함께 잡히면 안전하게 both 로 본다.
    """
    t = (raw or "").lower()
    found = set(_INTENT_TOKEN_RE.findall(t))
    if "재무" in t:
        found.add("financial")
    if "주가" in t:
        found.add("price")
    if "둘" in t:
        found.add("both")
    if not found:
        return None
    if "both" in found or ("financial" in found and "price" in found):
        return "both"
    if "financial" in found:
        return "financial"
    return "price"


def _classify_intent_heuristic(question: str) -> str:
    """키워드 기반 intent 판단(LLM 미가용/불명확 시 안전망). 둘 다/불명확이면 both 로 폴백한다.

    기존 classify_intent 의 결정론 판단부를 그대로 분리한 것 — return_12m/'수익률' 같은 계산전용
    재무지표 키워드(_COMPUTED_KO_ALIASES)를 financial 로 보호하던 회귀 방지 로직도 여기 유지된다.
    """
    q = question.lower()
    needs_financial = any(k in q for k in _FINANCIAL_KEYWORDS) or any(
        k in q for k in _COMPUTED_KO_ALIASES
    )
    needs_price = any(k in q for k in _PRICE_KEYWORDS)
    if needs_financial and needs_price:
        return "both"
    if needs_financial:
        return "financial"
    if needs_price:
        return "price"
    return "both"


def classify_intent(question: str, llm_fn: Callable[[str], str] | None = None) -> str:
    """질문이 재무데이터/주가데이터/둘 다 중 무엇을 요구하는지 판단한다.

    반환: "financial" | "price" | "both".
    route_question(총괄 라우팅)과 동일하게 **LLM 우선**이다 — llm_fn 이 주어지면 먼저 LLM 에
    판단을 위임하고(_intent_prompt), 응답에서 financial/price/both 를 뽑는다. LLM 이 없거나
    응답에서 판단을 못 뽑으면(파싱 실패/예외) 키워드 휴리스틱(_classify_intent_heuristic)으로
    폴백한다 — 안전망은 유지한다(예: return_12m/'수익률' 은 폴백에서 financial 로 보호). 끝까지
    불명확하면 놓치는 것보다 과다조회가 안전하므로 both 로 폴백한다.
    """
    if llm_fn is not None:
        try:
            raw = llm_fn(_intent_prompt(question)) or ""
        except Exception:  # noqa: BLE001 — LLM 실패는 키워드 폴백으로 흡수
            raw = ""
        intent = _parse_intent(raw)
        if intent is not None:
            return intent
    return _classify_intent_heuristic(question)


def _call_with_retry(fn: Callable[[], Any]) -> tuple[Any, str | None]:
    """fn()을 실행하고 실패(예외)하면 "가까운 계층"에서 즉시 1회 재시도한다.

    재시도까지 실패하면 예외를 전파하지 않고 (None, 실패사유문자열)을 반환해 상위
    (총괄 에이전트, HA-10)까지 예외가 뚫고 올라가지 않게 한다.
    """
    try:
        return fn(), None
    except Exception:  # noqa: BLE001 — 하위 데이터 에이전트가 어떤 예외든 던질 수 있음
        try:
            return fn(), None
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"


def _answer_kr_multi_entity_question(
    question: str,
    conn,
    stock_codes: list[str],
    llm_fn: Callable[[str], str] | None,
    resolve_metric_fn: Callable,
    price_snapshot_fn: Callable,
    execute_sql_fn: Callable,
    indicators: list[dict] | None,
    computed_metric_fn: Callable,
) -> dict:
    """한 질문이 이름으로 지목한 여러 종목(예: "삼성전자와 SK하이닉스 종가") 각각에 답한다.

    find_stock_codes가 서로 다른 종목을 2개 이상 찾았을 때만 이 경로로 온다 — "PER 낮은
    5개사"처럼 조건으로 종목을 "고르는" 스크리닝과 다르다(그건 answer_kr_screening이
    이미 먼저 가로챈다). intent/metric/period는 질문 전체에서 한 번만 판단해 모든 종목에
    동일하게 적용한다(같은 질문 안에서 종목마다 다른 지표를 묻는 경우는 범위 밖).
    """
    base_question = _strip_retry_feedback(question)
    period = _parse_period(base_question)
    intent = classify_intent(question, llm_fn=llm_fn)
    metric = _extract_metric(question) if intent in ("financial", "both") else None

    entities: list[dict] = []
    for code in stock_codes:
        entity: dict = {"stock_code": code, "financial": None, "price": None, "errors": []}

        if intent in ("financial", "both"):
            if metric is None:
                entity["errors"].append("재무 지표를 인식하지 못함")
            elif metric in _COMPUTED_ONLY_FIELDS:
                computed_asof = _resolve_screening_asof(period, conn, "prices", execute_sql_fn)
                financial, err = _call_with_retry(
                    lambda c=code: computed_metric_fn(
                        conn, c, metric, asof=computed_asof, execute_sql_fn=execute_sql_fn,
                    )
                )
                if err:
                    entity["errors"].append(f"계산 지표 조회 실패: {err}")
                else:
                    entity["financial"] = financial
            else:
                period_kwargs = {"period": period} if period is not None else {}
                financial, err = _call_with_retry(
                    lambda c=code: resolve_metric_fn(conn, c, metric, llm_fn=llm_fn, **period_kwargs)
                )
                if err:
                    entity["errors"].append(f"재무데이터 조회 실패: {err}")
                else:
                    entity["financial"] = financial

        if intent in ("price", "both"):
            price, err = _call_with_retry(
                lambda c=code: price_snapshot_fn(conn, c, indicators=indicators)
            )
            if err:
                entity["errors"].append(f"주가데이터 조회 실패: {err}")
            else:
                entity["price"] = price

        entities.append(entity)

    return {
        "stock_code": None,
        "stock_codes": stock_codes,
        "question": question,
        "intent": intent,
        "financial": None,
        "price": None,
        "entities": entities,
        "errors": [],
    }


def answer_kr_question(
    question: str,
    conn,
    llm_fn: Callable[[str], str] | None = None,
    resolve_metric_fn: Callable | None = None,
    price_snapshot_fn: Callable | None = None,
    execute_sql_fn: Callable | None = None,
    indicators: list[dict] | None = None,
    computed_metric_fn: Callable | None = None,
    price_history_fn: Callable | None = None,
    on_progress: Callable[..., None] | None = None,
) -> dict:
    """한국주식 도메인 에이전트 진입점.

    질문에서 종목(stock_code)을 찾고, 필요한 데이터 종류(classify_intent)에 따라
    HA-2(resolve_metric)·HA-3(get_price_snapshot_kr)에 위임한 뒤 결과를 종합한다.
    하위 데이터 에이전트 호출은 "가까운 계층 재시도"(_call_with_retry)를 거친다 —
    실패해도 예외를 전파하지 않고 errors에 사유를 담는다.

    인식된 지표가 _COMPUTED_ONLY_FIELDS(예: return_12m — 재무제표가 아니라 가격 시계열에서
    즉석 계산해야 하는 지표)에 해당하면 resolve_metric_fn 대신 computed_metric_fn
    (기본=resolve_computed_metric)을 호출한다. 그 외 지표(PER 등 기존 재무지표)는 기존과
    동일하게 resolve_metric_fn으로 조회한다(회귀 없음).

    Returns:
        {
            "stock_code": str | None,
            "question": str,
            "intent": "financial" | "price" | "both" | None,
            "financial": dict | None,     # resolve_metric() 또는 resolve_computed_metric() 결과
            "price": list[dict] | None,   # get_price_snapshot_kr() 결과
            "errors": list[str],
        }
    """
    resolve_metric_fn = resolve_metric_fn or resolve_metric
    price_snapshot_fn = price_snapshot_fn or get_price_snapshot_kr
    execute_sql_fn = execute_sql_fn or execute_sql
    computed_metric_fn = computed_metric_fn or resolve_computed_metric
    price_history_fn = price_history_fn or get_price_history_kr

    # 기간/차트 의도는 재시도 피드백이 아니라 원본 질문 기준으로 판단한다(wants_chart 원칙).
    base_question = _strip_retry_feedback(question)
    period = _parse_period(base_question)

    # 스크리닝(다중종목 랭킹) 질문은 단일종목 조회 경로 대신 스크리닝 경로로 분기한다(HA-15).
    # is_screening_question 도 LLM 우선 판단이므로 llm_fn 을 관통시킨다(키워드 목록에 없는
    # "저PER 5종목" 같은 표현도 LLM 판단으로 인식되게 하기 위함).
    if is_screening_question(question, llm_fn=llm_fn):
        screening_asof = _resolve_screening_asof(period, conn, "prices", execute_sql_fn)
        return answer_kr_screening(
            question, conn, llm_fn=llm_fn, execute_sql_fn=execute_sql_fn, asof=screening_asof,
            on_progress=on_progress,
        )

    result: dict = {
        "stock_code": None,
        "question": question,
        "intent": None,
        "financial": None,
        "price": None,
        "errors": [],
    }

    # 다중종목(named multi-entity) 경로: 질문이 이름으로 2개 이상의 서로 다른 종목을
    # 지목하면(예: "삼성전자와 SK하이닉스 종가") 단일종목 경로(find_stock_code, 하나만
    # 고름)로는 나머지 종목이 통째로 누락된다. find_stock_codes로 먼저 개수를 확인하고,
    # 2개 이상이면 종목별 개별 조회 경로로 분기한다(0/1개는 기존 단일종목 경로 그대로).
    multi_codes = find_stock_codes(conn, question, execute_sql_fn=execute_sql_fn)
    if len(multi_codes) >= 2:
        return _answer_kr_multi_entity_question(
            question, conn, multi_codes, llm_fn=llm_fn,
            resolve_metric_fn=resolve_metric_fn, price_snapshot_fn=price_snapshot_fn,
            execute_sql_fn=execute_sql_fn, indicators=indicators,
            computed_metric_fn=computed_metric_fn,
        )

    stock_code = find_stock_code(conn, question, execute_sql_fn=execute_sql_fn)
    result["stock_code"] = stock_code
    if stock_code is None:
        result["errors"].append("종목을 찾을 수 없습니다: 질문에서 종목명/종목코드를 인식하지 못함")
        return result

    intent = classify_intent(question, llm_fn=llm_fn)
    result["intent"] = intent

    if intent in ("financial", "both"):
        metric = _extract_metric(question)
        if metric is None:
            result["errors"].append("재무 지표를 인식하지 못함")
        elif metric in _COMPUTED_ONLY_FIELDS:
            computed_asof = _resolve_screening_asof(period, conn, "prices", execute_sql_fn)
            financial, err = _call_with_retry(
                lambda: computed_metric_fn(
                    conn, stock_code, metric, asof=computed_asof, execute_sql_fn=execute_sql_fn,
                )
            )
            if err:
                result["errors"].append(f"계산 지표 조회 실패: {err}")
            else:
                result["financial"] = financial
        else:
            # 기간이 파싱된 경우에만 period 인자를 넘긴다 — 미지정이면 기존 fake/시그니처
            # (conn, stock_code, metric, llm_fn)을 그대로 존중해 회귀를 막는다.
            period_kwargs = {"period": period} if period is not None else {}
            financial, err = _call_with_retry(
                lambda: resolve_metric_fn(conn, stock_code, metric, llm_fn=llm_fn, **period_kwargs)
            )
            if err:
                result["errors"].append(f"재무데이터 조회 실패: {err}")
            else:
                result["financial"] = financial

    if intent in ("price", "both"):
        price, err = _call_with_retry(
            lambda: price_snapshot_fn(conn, stock_code, indicators=indicators)
        )
        if err:
            result["errors"].append(f"주가데이터 조회 실패: {err}")
        else:
            result["price"] = price
            # 시계열/차트를 원하는 질문이면 최근 1년 종가 시계열도 함께 담아, 검증기가
            # "1년 기간을 충족하는 데이터가 실제로 있다"를 확인할 수 있게 한다(버그A).
            if price and _wants_price_history(base_question):
                history = price_history_fn(conn, stock_code)
                if history:
                    result["price_history"] = _summarize_price_history(history)

    return result
