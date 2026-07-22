"""한국주식 도메인 에이전트 — 질문을 재무/주가 데이터 에이전트로 세부 위임한다 (HA-6).

계층: 총괄 에이전트(HA-10, 미구현) → **이 도메인 에이전트** → 데이터 에이전트
(HA-2 resolve_metric/src/agents/data_financial.py, HA-3 get_price_snapshot_kr/
src/agents/data_price_kr.py). 이미 "한국주식 도메인"으로 라우팅된 뒤의 세부 위임만
다루므로 판단 로직은 간단한 키워드 휴리스틱(+옵션 llm_fn)으로 충분하다.

핵심 책임:
1) 질문에서 종목명(예: "삼성전자") 또는 6자리 종목코드를 찾아 stock_code로 변환한다
   (company 테이블, execute_sql(HA-1 실행기) 경유 — conn.execute() 직접 호출 금지).
2) 질문이 재무데이터/순수 시세/기술지표 중 무엇을(여러 개 동시 가능) 요구하는지 판단해(classify_intent)
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
from datetime import date
from typing import Any, Callable

from src.agents.data_financial import _METRICS_TABLE_COLS, METRIC_SOURCE_MAP, resolve_metric
from src.agents.data_price_kr import get_price_history_kr, get_price_snapshot_kr
from src.agents.exec_runtime import execute_sql
from src.backtest.data_access import METRIC_FIELD_DESCRIPTIONS, price_return_over_months
from src.backtest.primitives import combine, get_cross_section
from src.llm import extract_json
from src.version import quarter_end_date

# _screening_prompt는 국내 스크리닝 스펙 추출 프롬프트를 만들고, 지표 설명 단일 정의처
# (canonical source)인 METRIC_FIELD_DESCRIPTIONS 에서 key→한글설명을 조회한다.

# 종목코드는 정확히 6자리 숫자 하나(더 긴 숫자열의 일부가 아님) — 앞뒤에 다른 숫자가
# 붙지 않은 경우만 인정한다("1200000원"처럼 7자리 금액에서 6자리 부분을 코드로 오려내지 않게).
_STOCK_CODE_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")
# 바로 뒤에 통화 단위 '원'이 붙은 6자리 숫자는 주가/금액이지 종목코드가 아니다
# ("삼성전자 200000원 돌파했어?"의 200000). 종목코드를 "005930원"처럼 쓰는 경우는 없다.
_PRICE_WON_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)\s*원")


def _stock_code_candidates(text: str) -> list[str]:
    """질문에서 종목코드로 볼 수 있는 6자리 숫자만 순서대로 추린다(주가 금액 '…원'은 제외).

    6자리 숫자를 무조건 종목코드로 취급하면 "200000원 돌파"의 금액을 엉뚱한 종목코드로
    오인한다 — 통화 단위 '원'이 붙은 6자리 숫자는 후보에서 뺀다. 코드와 금액이 섞여 있으면
    ("005930 200000원 돌파?") 금액만 빠지고 진짜 코드는 남는다.
    """
    price_amounts = set(_PRICE_WON_RE.findall(text))
    return [m for m in _STOCK_CODE_RE.findall(text) if m not in price_amounts]

# 지표명(한국어) → METRIC_SOURCE_MAP(src/agents/data_financial.py) 키. 질문에서 흔히
# 쓰이는 한국어 표현만 최소한으로 다룬다(복잡한 NL 파싱은 이 에이전트의 책임이 아님).
_METRIC_KO_ALIASES: dict[str, str] = {
    # "주가매출비율"(PSR)은 "매출"을 부분문자열로 포함하므로 반드시 "매출"(revenue) 별칭보다
    # 먼저 와야 한다(같은 이유로 딕셔너리 맨 앞에 둔다 — 매출 계열 별칭 전체보다도 먼저).
    # psr/pcr/ev_ebitda는 per/pbr과 동일하게 METRIC_SOURCE_MAP(metrics 테이블 direct-WHERE
    # 조회)에 등록돼 있어(_COMPUTED_KO_ALIASES가 아니라 여기서) 한국어 별칭을 등록해야 인식된다.
    # 영문 약어(PSR/PCR/PEG)는 METRIC_SOURCE_MAP 원본 키 매치(tier 2)로 이미 인식되지만,
    # "EV/EBITDA"는 실제 키가 "ev_ebitda"(밑줄)라 슬래시 표기를 별도로 등록해야 한다.
    "주가매출비율": "psr",
    "주가현금흐름비율": "pcr",
    "ev/ebitda": "ev_ebitda", "ev-ebitda": "ev_ebitda",
    # "매출총이익률"/"매출원가율"/"매출성장률"(비율·성장률)은 반드시 "매출액"/"매출"(원본
    # 계정 금액)보다 먼저 와야 한다 — 셋 다 "매출"을 부분문자열로 포함해서, 뒤에 두면 raw
    # 계정(revenue)으로 오매핑된다(실서버 재현 버그: "SK하이닉스 26년 1분기 매출총이익률"이
    # gross_margin이 아니라 revenue 원본 금액으로 조회됨). gross_margin/cogs_ratio/
    # revenue_growth는 METRIC_SOURCE_MAP에 등록돼 있어(operating_margin과 동일하게 EAV
    # 즉석계산) resolve_metric으로 조회 가능하다.
    "매출총이익률": "gross_margin",
    "매출원가율": "cogs_ratio",
    "매출성장률": "revenue_growth", "매출성장": "revenue_growth", "매출증가율": "revenue_growth",
    "매출액": "revenue",
    "매출": "revenue",
    # "영업이익률"/"영업이익성장률"(비율·성장률)은 반드시 "영업이익"(원본 계정 금액)보다 먼저
    # 와야 한다 — _extract_metric이 이 딕셔너리를 삽입 순서대로 순회하며 첫 부분일치를
    # 채택하므로, 더 긴 별칭을 앞에 두지 않으면 부분문자열 "영업이익"이 먼저 매치돼
    # operating_profit으로 오매핑된다(실서버 재현 버그). operating_margin은 metrics 테이블에
    # 사전계산돼 resolve_metric으로 조회 가능하다(computed-only가 아니라 _COMPUTED_KO_ALIASES
    # 경로로는 안 잡혀 여기서 별칭을 명시해야 인식된다).
    "영업이익률": "operating_margin",
    "영업이익성장률": "op_growth", "영업이익성장": "op_growth",
    "영업이익": "operating_profit",
    # "순이익률"/"순이익성장률"(비율·성장률)도 동일한 이유로 "당기순이익"/"순이익"(원본
    # 계정)보다 먼저 와야 한다.
    "순이익률": "net_margin",
    "순이익성장률": "ni_growth", "순이익성장": "ni_growth",
    "당기순이익": "net_income",
    "순이익": "net_income",
    "자산총계": "total_assets",
    "부채총계": "total_liabilities",
    "자본총계": "total_equity",
    "배당": "dividend",
    "목표주가": "target_price",
    "투자의견": "analyst_opinion",
    # ── 여기부터 이번 세션에 새로 WHERE(직접 quarter 매치) 경로로 옮긴 지표들 ──────────
    # 전부 가격/시가총액이 필요 없는 순수 재무비율이라 look-ahead asof 간접참조 없이 지목된
    # 분기를 그대로 쓸 수 있다(roa/gp_a/interest_coverage/current_ratio는 _RATIO_TTM_ACCOUNTS,
    # revenue_growth/op_growth/ni_growth는 위에서 이미 등록한 _YOY_GROWTH_ACCOUNTS 경로).
    "총자산이익률": "roa",
    "매출총이익/총자산": "gp_a", "gpa": "gp_a", "gp/a": "gp_a",
    "이자보상배율": "interest_coverage",
    "유동비율": "current_ratio",
}

# 순수 시세(종가/시가/거래량 등) 키워드 — get_price_snapshot_kr(indicators=None) 경로.
_PRICE_KEYWORDS: list[str] = [
    "주가", "가격", "시가", "고가", "저가", "종가", "거래량", "차트",
]

# 기술지표 키워드 — 순수 시세와 별개 축이다. TA-Lib 기반 compute_technical_indicator
# (src/backtest/primitives.py, 이동평균/RSI/MACD/볼린저밴드)로 가격 시계열에서 파생 계산된다.
# get_price_snapshot_kr(indicators=[...]) 경로로 흐른다.
_TECHNICAL_KEYWORDS: list[str] = [
    "이동평균", "이평선", "이평", "골든크로스", "데드크로스", "기술적", "기술지표",
    "rsi", "macd", "볼린저", "sma", "ema", "bollinger",
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


# ── 그룹명 생략 구어체 종목명 보강 (SK하이닉스="하이닉스", LG이노텍="이노텍") ──────────────
# 한국어는 대기업 계열사를 부를 때 그룹·지주 접두어를 흔히 생략한다("하이닉스"=SK하이닉스,
# "이노텍"=LG이노텍). 역방향 LIKE(질문이 회사 공식명을 통째로 포함하는가)만 쓰면 접두어가
# 빠진 구어체는 매칭에 실패하고, 하필 "이닉스"(452400)처럼 무관한 짧은 회사명이 "하이닉스"의
# 부분문자열로 우연히 걸려 조용히 틀린 종목을 반환하는 정합성 버그가 난다. 아래 접두어 목록은
# company 테이블에서 실제로 계열사가 많은(SK 21·현대 41·한국 79 등) 그룹·지주 접두어다.
_GROUP_PREFIXES: tuple[str, ...] = (
    "미래에셋", "포스코", "삼성", "현대", "한화", "신한", "두산", "롯데", "한국", "하나",
    "SK", "LG", "CJ", "GS", "KB", "NH", "HD", "DB",
)
# 접두어를 뗀 나머지(remainder)가 이 글자 수 이상일 때만 구어체 매칭으로 인정한다. "전자"/"화재"
# 같은 2글자 일반어가 접두어 제거로 무관한 질문에 오탐되는 것을 막는다(하이닉스=4, 이노텍=3).
_MIN_STRIPPED_REMAINDER = 3


def _name_match_sql(escaped_text: str) -> str:
    """질문 텍스트에서 종목명 매칭 후보를 한 번의 SQL로 뽑는 쿼리를 만든다.

    두 종류의 매칭을 UNION 으로 합친다(단일 execute_sql 호출 유지):
    - direct: 질문이 회사 공식명을 통째로 포함(기존 역방향 LIKE). matched_text=회사명.
    - stripped: 그룹·지주 접두어(_GROUP_PREFIXES)를 뗀 나머지가 질문에 포함되는 실재 회사.
      matched_text=접두어를 뗀 나머지. company 테이블에 실제로 있는 회사만 후보가 되므로
      "접두어+매칭텍스트가 실존할 때만 확장"이 자동으로 보장된다(무조건 확장 아님).

    정렬: 매칭된 텍스트 길이 DESC → direct 우선 → 회사명 길이 DESC. "하이닉스"(stripped,4)가
    "이닉스"(direct,3)를 이겨 그룹명 생략 구어체를 정확한 종목으로 resolve하고, 무관한 짧은
    이름의 오탐을 억제한다("가장 구체적인 매치 우선" 원칙을 두 매칭 방식에 걸쳐 확장한 것).
    """
    when_clauses = "\n".join(
        f"        WHEN name LIKE '{p.replace(chr(39), chr(39) * 2)}%' "
        f"AND LENGTH(name) >= {len(p) + _MIN_STRIPPED_REMAINDER} "
        f"THEN SUBSTR(name, {len(p) + 1})"
        for p in _GROUP_PREFIXES
    )
    return (
        "SELECT stock_code, name, matched_text, is_direct FROM (\n"
        "  SELECT stock_code, name, name AS matched_text, 1 AS is_direct FROM company\n"
        f"    WHERE name IS NOT NULL AND name != '' AND '{escaped_text}' LIKE '%' || name || '%'\n"
        "  UNION ALL\n"
        "  SELECT stock_code, name, matched_text, 0 AS is_direct FROM (\n"
        "    SELECT stock_code, name, CASE\n"
        f"{when_clauses}\n"
        "      ELSE NULL END AS matched_text\n"
        "    FROM company WHERE name IS NOT NULL AND name != ''\n"
        f"  ) WHERE matched_text IS NOT NULL AND '{escaped_text}' LIKE '%' || matched_text || '%'\n"
        ") ORDER BY LENGTH(matched_text) DESC, is_direct DESC, LENGTH(name) DESC"
    )


def _match_company_candidates(conn, text: str, execute_sql_fn: Callable) -> list[dict]:
    """종목명 매칭 후보를 우선순위(가장 구체적인 매치 먼저) 정렬된 리스트로 반환한다.

    각 항목은 {stock_code, name, matched_text} — matched_text 는 질문에서 실제로 이 후보를
    지목한 문자열(direct 는 회사명, stripped 는 접두어를 뗀 나머지)이며, find_stock_codes 의
    "겹치는 후보 제거"(더 긴 매칭텍스트의 부분문자열이면 스킵)에 쓰인다. find_stock_code 와
    find_stock_codes 가 동일한 이 헬퍼를 공유해 매칭 규칙을 한 곳으로 일원화한다.
    """
    result = execute_sql_fn(_name_match_sql(_escape_sql_literal(text)), conn)
    if not result.get("ok"):
        return []
    return [
        {"stock_code": r["stock_code"], "name": r["name"], "matched_text": r["matched_text"]}
        for r in result["rows"]
    ]


def _former_name_match_sql(escaped_text: str) -> str:
    """예전 사명(kr_stock_changes.name_before/name_after)이 질문에 포함되는 종목을 뽑는 SQL.

    회사가 사명을 바꾸면 예전 이름은 company.name(현재 사명)에 없어 매칭에 실패한다 — 이 이력
    테이블에서 변경전/후 이름을 모두 후보로 삼는다(중간 사명은 한 행의 변경후이자 다음 행의
    변경전이라, 양쪽을 모두 매칭하면 사명 변경 체인의 모든 과거 이름을 커버한다). 매칭 방식은
    현재 사명 매칭(_name_match_sql)과 동일한 역방향 LIKE(질문이 예전 이름을 통째로 포함하는가)이며,
    가장 구체적인 매치(가장 긴 예전 이름)를 우선한다. execute_sql(읽기전용) 경유로만 실행한다.
    """
    return (
        "SELECT stock_code, matched_text FROM (\n"
        "  SELECT stock_code, name_before AS matched_text FROM kr_stock_changes\n"
        f"    WHERE name_before IS NOT NULL AND name_before != '' "
        f"AND '{escaped_text}' LIKE '%' || name_before || '%'\n"
        "  UNION\n"
        "  SELECT stock_code, name_after AS matched_text FROM kr_stock_changes\n"
        f"    WHERE name_after IS NOT NULL AND name_after != '' "
        f"AND '{escaped_text}' LIKE '%' || name_after || '%'\n"
        ") ORDER BY LENGTH(matched_text) DESC"
    )


def _match_former_name_candidates(conn, text: str, execute_sql_fn: Callable) -> list[dict]:
    """예전 사명 매칭 후보를 우선순위(가장 긴 예전 이름 먼저) 정렬 리스트로 반환한다.

    각 항목은 {stock_code, matched_text}. kr_stock_changes 테이블이 없거나(구 DB) 조회 실패면
    빈 리스트를 돌려 현재 사명 매칭만으로도 안전하게 동작한다(폴백이 크래시를 유발하지 않음).
    """
    result = execute_sql_fn(_former_name_match_sql(_escape_sql_literal(text)), conn)
    if not result.get("ok"):
        return []
    return [
        {"stock_code": r["stock_code"], "matched_text": r["matched_text"]}
        for r in result["rows"]
    ]


def find_stock_code(
    conn,
    question: str,
    execute_sql_fn: Callable | None = None,
) -> str | None:
    """질문 텍스트에서 종목코드(6자리 숫자) 또는 종목명을 찾아 종목코드로 변환한다.

    1) 질문에 6자리 숫자가 있으면 그대로 종목코드로 쓴다(가장 신뢰도 높음).
    2) 없으면 company 테이블에서 종목명을 매칭한다(_match_company_candidates 공유 헬퍼):
       질문이 회사 공식명을 통째로 포함하거나(역방향 LIKE), 그룹·지주 접두어(SK/LG/…)를
       뗀 나머지가 질문에 포함되는(실재 회사) 경우를 모두 후보로 삼아, 가장 구체적인
       매치를 우선한다("하이닉스"→SK하이닉스, 무관한 짧은 이름 "이닉스" 오탐 억제).
    3) 현재 사명(company)으로 못 찾으면 예전 사명(kr_stock_changes.name_before/name_after)으로
       폴백한다 — 회사가 이름을 바꾸면 옛 이름은 company 에 없어 매칭에 실패하므로, 사명 변경
       이력에서 과거 이름으로도 종목코드를 찾는다("옛이름전자"→현재 종목코드).
    4) SQL은 execute_sql(HA-1 실행기)로만 실행한다(conn.execute() 직접 호출 금지).

    매치가 없으면 None을 반환한다.
    """
    execute_sql_fn = execute_sql_fn or execute_sql
    text = question or ""
    candidates = _stock_code_candidates(text)
    if candidates:
        return candidates[0]
    if not text.strip():
        return None
    matches = _match_company_candidates(conn, text, execute_sql_fn)
    if matches:
        return matches[0]["stock_code"]
    # 현재 사명으로 못 찾으면 예전 사명(사명 변경 이력)으로 폴백한다.
    former = _match_former_name_candidates(conn, text, execute_sql_fn)
    return former[0]["stock_code"] if former else None


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
    2) company 테이블에서 종목명을 매칭한다(_match_company_candidates 공유 헬퍼 — 역방향
       LIKE + 그룹명 생략 보강, find_stock_code와 완전히 동일한 규칙). 후보를 가장 구체적인
       매치부터 순회하며, 이미 선택한(더 긴) 매칭텍스트의 부분 문자열인 후보(예: "SK하이닉스"를
       이미 선택했다면 그 안에 포함된 "SK"나 "하이닉스", "이닉스")는 건너뛴다 — find_stock_code의
       "가장 구체적인 매치 우선" 원칙을 "하나만 고르기"가 아니라 "겹치는 후보만 제거하기"로 확장.
    3) 위 두 결과를 종목코드 기준으로 중복 제거해 합친다.

    매치가 없으면 빈 리스트. SQL은 execute_sql(HA-1 실행기)로만 실행한다.
    """
    execute_sql_fn = execute_sql_fn or execute_sql
    text = question or ""

    codes: list[str] = list(dict.fromkeys(_stock_code_candidates(text)))

    if text.strip():
        accepted_texts: list[str] = []
        # 현재 사명 매칭 + 예전 사명(사명 변경 이력) 폴백을 이어 순회한다. 현재 사명을 먼저
        # 처리해 겹치는(부분문자열) 예전 이름은 자동으로 걸러진다(find_stock_code 단수와 동일한
        # 폴백 규칙을 복수 경로로 확장 — 옛 이름으로 지목된 종목도 누락되지 않게).
        candidates = (
            _match_company_candidates(conn, text, execute_sql_fn)
            + _match_former_name_candidates(conn, text, execute_sql_fn)
        )
        for cand in candidates:
            matched_text = cand["matched_text"]
            if any(matched_text in accepted for accepted in accepted_texts):
                continue
            accepted_texts.append(matched_text)
            if cand["stock_code"] not in codes:
                codes.append(cand["stock_code"])

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
# 극값(최댓값/최솟값/최고/최저/가장 높은·낮은) 표현. "그 지표로 상위/하위 1개(top_n=1)"를
# 요구하는 것과 동치이므로 그 자체로 스크리닝(랭킹) 신호로 본다 — 종목명 없이 "PBR 최댓값"류
# 질문이 단일종목 조회 경로로 새서 실패하던 문제를 막는다. high/low를 나눠 두어 방향(direction)
# 판정과 양극단("최댓값과 최솟값 둘 다") 판정(both_extremes)에 함께 재사용한다.
_SCREENING_EXTREME_HIGH: tuple[str, ...] = (
    "최댓값", "최댓치", "최고값", "최고치", "최고", "가장 높은",
)
_SCREENING_EXTREME_LOW: tuple[str, ...] = (
    "최솟값", "최솟치", "최저값", "최저치", "최저", "가장 낮은",
)
_SCREENING_EXTREME_WORDS: tuple[str, ...] = _SCREENING_EXTREME_HIGH + _SCREENING_EXTREME_LOW

# 스크리닝 지표 별칭(한국어/영문) → get_cross_section(metrics_at) 출력 필드명.
# 긴 키워드 우선 매칭(영업이익률 > 영업이익)을 위해 사용 시 길이 내림차순으로 순회한다.
_SCREEN_METRIC_ALIASES: dict[str, str] = {
    "per": "per", "주가수익비율": "per",
    "pbr": "pbr", "주가순자산비율": "pbr",
    "psr": "psr", "주가매출비율": "psr",
    "pcr": "pcr", "주가현금흐름비율": "pcr",
    "ev/ebitda": "ev_ebitda", "ev_ebitda": "ev_ebitda", "ev-ebitda": "ev_ebitda",
    "peg": "peg",
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
    # 매출총이익률(=매출총이익÷매출액)은 GPA(gp_a=매출총이익÷총자산)와 분모가 다른 별개 지표다.
    # 과거엔 '매출총이익률'을 gp_a로 잘못 매핑했으나, 이제 별도 gross_margin 지표가 생겨 교정한다.
    # GPA 자체는 'gpa'/'gp/a' 별칭으로만 지목한다(한국어 '매출총이익률'은 gross_margin으로).
    "매출총이익률": "gross_margin",
    "매출원가율": "cogs_ratio",
    "gpa": "gp_a", "gp/a": "gp_a",
    # "시가총액 상위 N개" 스크리닝이 LLM 없이(휴리스틱 폴백) 동작할 때도 인식되게 한다
    # (실서버 재현 버그: market_cap이 METRIC_FIELD_DESCRIPTIONS에 없어 "총자산"으로 잘못 매핑됨).
    "시가총액": "market_cap", "시총": "market_cap",
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
    # 지표의 극값(최댓값/최솟값/최고/최저/가장 높은·낮은)을 묻는 질문은 "그 지표 상위/하위
    # 1개" 조회와 동치이므로 개수/랭킹 신호 없이도 그 자체로 스크리닝으로 본다.
    if any(w in q for w in _SCREENING_EXTREME_WORDS):
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
        "특정 회사명 없이 어떤 지표의 극값(최댓값/최솟값/최고/최저/가장 높은·낮은)을 묻는 "
        "질문도 '그 지표 기준 1위 조회'이므로 스크리닝(yes)으로 봅니다.\n"
        "예(스크리닝=yes): 'PER 낮은 5종목', '저PER 5종목', '저평가된 5개 추천', '순위 매겨줘', "
        "'PBR 최댓값과 최솟값', 'ROE가 가장 높은 종목'.\n"
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
    """스크리닝 스펙 추출 프롬프트. 시장 판단 범위를 국내(코스피/코스닥)로 스코프한다.

    프롬프트 자체가 "너는 지금 한국 시장만 담당한다"를 명시해 다른 나라 시장 언급을
    무시하게 만든다. 시장 구분은 하드코딩 키워드 규칙이 아니라 LLM이 질문 의미를 읽고
    판단하며, 값 후보만 코스피/코스닥으로 제한한다.

    지표 목록은 "key만"이 아니라 "key: 한글설명"으로 나열한다(METRIC_FIELD_DESCRIPTIONS
    단일 정의처에서 조회) — LLM이 별도 한글→필드명 별칭사전 없이도 설명만 보고 "영업이익"
    질문을 operating_profit으로 스스로 매핑할 수 있게 한다. fields 는 노출할 key의 집합/순서만
    결정하고, 설명 텍스트는 이 딕셔너리에서 가져온다(_KR_SCREEN_FIELDS 가 이 딕셔너리에서
    파생되므로 자동 동기화된다).
    """
    descriptions = METRIC_FIELD_DESCRIPTIONS
    fields_block = "\n".join(f"  - {k}: {descriptions.get(k, k)}" for k in fields)
    sector_hint = (
        f"실제 업종(sector) 목록: {', '.join(sectors)}.\n"
        "질문의 업종 표현이 이 목록에 정확히 없으면 가장 가까운 실제 항목으로 매핑하세요"
        "(예: 반도체/전자부품→전기·전자, 게임→소프트웨어·게임, 자동차→운송장비·부품). "
        "매핑할 항목이 전혀 없으면 sectors를 생략(null)하세요.\n"
        if sectors else ""
    )
    market_key = "markets"
    scope_line = (
        "이 질문에서 한국 시장(코스피/코스닥) 관련 조건만 판단하세요. 질문에 다른 "
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
        f'"top_n":<정수>,"sectors":null,"{market_key}":null,"sector_neutral":false,'
        '"both_extremes":false,"sector_neutral_compare":false}\n'
        f"사용 가능한 지표(key: 설명):\n{fields_block}\n"
        "direction: 낮을수록/저평가 우수면 low, 높을수록 우수면 high.\n"
        "top_n: 질문에 개수가 명시돼 있으면 그 숫자를, 명시돼 있지 않으면 10을, "
        "'전체'/'모든 종목'/'모두'처럼 개수 제한 없이 전부를 요구하면 4000을 쓰세요. "
        "'최댓값'/'최솟값'/'최고'/'최저'/'가장 높은·낮은'처럼 지표의 극값 1개만 물으면 top_n=1을 쓰세요.\n"
        "both_extremes: 질문이 한 지표의 '최댓값과 최솟값'(또는 최고와 최저)을 둘 다 요구하면 "
        "true로 설정하고, 이때 criteria에는 같은 지표를 direction=high와 direction=low 두 개로 "
        "넣고 top_n=1로 하세요. 한쪽(최댓값만/최솟값만)만 물으면 false로 두세요.\n"
        "sector_neutral: 질문이 '섹터 중립화'/'업종 중립'/'섹터별로 정규화'/'섹터 편중을 없애고' "
        "처럼 섹터(업종) 간 비교 왜곡을 제거해 달라고 명시적으로 요구하면 true로, 그렇지 않으면 "
        "false로 설정하세요.\n"
        "sector_neutral_compare: 질문이 '섹터중립화 전후를 비교'/'섹터중립 하고 안 하고 둘 다' "
        "처럼 섹터중립화 전(원본)과 후(섹터중립) 결과를 한 번에 비교해 보여달라고 요구하면 "
        "true로, 그렇지 않으면 false로 설정하세요.\n"
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

    시장 필터는 markets(코스피/코스닥)로 읽는다. 프롬프트가 국내 도메인 스코프를 명시하므로
    질문에 다른 나라 시장이 섞여 있어도 한국 부분만 markets 로 반영한다.
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
        "sector_neutral": _coerce_sector_neutral(data.get("sector_neutral")),
        # _coerce_sector_neutral은 이름과 달리 순수 bool 강제 변환("true"/1/yes 관대 수용)이라
        # both_extremes/sector_neutral_compare 플래그에도 그대로 재사용한다(중복 정의 방지).
        "both_extremes": _coerce_sector_neutral(data.get("both_extremes")),
        "sector_neutral_compare": _coerce_sector_neutral(data.get("sector_neutral_compare")),
    }
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


# ── 섹터 중립화(sector_neutral) 감지 ────────────────────────────────────────
# "섹터 중립화"/"업종 중립"류 표현을 결정론적으로 감지하는 키워드(휴리스틱 폴백용). 주경로는
# LLM이 프롬프트 지시문을 읽고 판단하지만, LLM 없을 때/파싱 실패 시 이 키워드로 폴백한다.
_SECTOR_NEUTRAL_PHRASES: tuple[str, ...] = (
    "섹터중립", "섹터 중립", "업종중립", "업종 중립",
    "섹터별 정규화", "섹터별로 정규화", "업종별 정규화", "업종별로 정규화",
    "섹터 편중", "업종 편중", "섹터 쏠림", "업종 쏠림",
    "sector neutral", "sector-neutral",
)


def _coerce_sector_neutral(value) -> bool:
    """sector_neutral 값을 안전하게 bool로 변환한다(누락/오타/이상값은 False).

    LLM JSON은 보통 실제 bool(true/false)을 주지만, 문자열 "true"/"1"/"yes"나 숫자 1도
    관대하게 True로 수용한다. 그 외("maybe"/None/빈값 등)는 모두 False로 안전하게 처리한다.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y")
    return False


def _detect_sector_neutral_keyword(question: str) -> bool:
    """질문 텍스트에 섹터 중립화 요구 표현이 있으면 True(결정론 휴리스틱 폴백용)."""
    q = (question or "").lower()
    return any(p in q for p in _SECTOR_NEUTRAL_PHRASES)


# "섹터중립화 전/후를 한 번에 비교"(sector_neutral_compare) 감지용 비교 의도 표현.
# 게이트: 섹터중립 키워드(_detect_sector_neutral_keyword)와 이 비교 의도 표현이 둘 다 있을
# 때만 compare로 본다 — sector_neutral 자체의 과잉추론 방지 철학을 그대로 따라, 섹터중립
# 키워드만 있고 비교 의도가 없으면(그냥 "섹터중립화해서 보여줘") compare는 False로 둔다.
_SECTOR_NEUTRAL_COMPARE_PHRASES: tuple[str, ...] = (
    "비교", "전후", "둘 다", "둘다", "동시에", "함께 보여", "같이 보여",
)


def _detect_sector_neutral_compare_keyword(question: str) -> bool:
    """섹터중립 키워드 + 비교 의도 표현이 둘 다 있을 때만 True(결정론 이중 게이트)."""
    q = (question or "").lower()
    return _detect_sector_neutral_keyword(q) and any(
        p in q for p in _SECTOR_NEUTRAL_COMPARE_PHRASES
    )


def _heuristic_screening_spec(question: str, domain: str = "KR") -> dict | None:
    """LLM 없이(또는 JSON 파싱 실패 시) 결정론적 키워드로 스크리닝 스펙을 만든다.

    heuristic.py 의 지표/방향/개수 감지 관례를 스크리닝 지표(per/pbr/roe/…)에 맞춰 재현한다.
    지표를 하나도 못 찾으면 None(→ 호출부가 명시적 오류를 남김).

    시장 필터는 코스피/코스닥만 markets 로 잡는다(질문에 다른 나라 시장이 섞여 있어도
    한국 부분만 본다 — LLM 프롬프트와 동일한 스코프를 폴백 경로에서도 유지).
    """
    q = (question or "").lower()
    metric = None
    for kw in _SCREEN_METRIC_ORDER:
        if kw in q:
            metric = _SCREEN_METRIC_ALIASES[kw]
            break
    if metric is None:
        return None

    # 극값 표현 감지: high/low 양쪽이 다 있으면 "최댓값과 최솟값 둘 다"(both_extremes)로 본다.
    has_high_extreme = any(w in q for w in _SCREENING_EXTREME_HIGH)
    has_low_extreme = any(w in q for w in _SCREENING_EXTREME_LOW)
    both_extremes = has_high_extreme and has_low_extreme

    high_words = ("높은", "높게", "큰", "많은", "고평가") + _SCREENING_EXTREME_HIGH
    direction = "high" if any(w in q for w in high_words) else "low"

    count_match = re.search(r"(\d+)", q)
    if count_match:
        top_n = _coerce_top_n(count_match.group(1))
    elif has_high_extreme or has_low_extreme:
        top_n = 1  # "최댓값"/"최고"처럼 극값 1개만 요구 → 그 지표 상위/하위 1개와 동치
    elif any(w in q for w in ("전체", "모든", "모두")):
        top_n = _coerce_top_n(4000)  # 개수 제한 없이 전부 요구 → 상한까지 채운다
    else:
        top_n = 10

    markets = None
    exchanges = None
    if "코스피" in q or "kospi" in q:
        markets = ["KOSPI"]
    elif "코스닥" in q or "kosdaq" in q:
        markets = ["KOSDAQ"]

    if both_extremes:
        criteria = [
            {"key": metric, "direction": "high", "weight": 1.0},
            {"key": metric, "direction": "low", "weight": 1.0},
        ]
    else:
        criteria = [{"key": metric, "direction": direction, "weight": 1.0}]

    return {
        "criteria": criteria,
        "top_n": top_n,
        "sectors": None,
        "markets": markets,
        "exchanges": exchanges,
        "sector_neutral": _detect_sector_neutral_keyword(q),
        "both_extremes": both_extremes,
        "sector_neutral_compare": _detect_sector_neutral_compare_keyword(q),
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

    domain 은 시장 판단 스코프(코스피/코스닥→markets)를 프롬프트/파싱/휴리스틱 전 경로에
    일관되게 전달한다(현재 국내 도메인 전용).
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
    이 None을 answer_kr_screening의 asof에 그대로 넘기면 기존과 동일하게
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


def _computed_metric_period_mismatch_note(period: dict | None, financial: dict | None) -> str | None:
    """계산전용 지표(_COMPUTED_ONLY_FIELDS)가 질문이 지목한 분기와 다른 분기로 계산됐으면
    그 사실을 사용자에게 알릴 경고문을 만든다. 다르지 않거나 판단할 정보가 없으면 None.

    실서버 재현 버그: "SK하이닉스 26년 1분기 매출총이익률"이 아직 공시 전인 26년 1분기 대신
    25년 4분기 값을 아무 표시 없이 반환했다. _COMPUTED_ONLY_FIELDS 지표는 look-ahead 방지를
    위해 asof(분기 말일 이하 최신 거래일)까지 공시된 최신 분기를 쓰는데(get_cross_section/
    effective_quarter_at, 백테스트와 공유하는 안전장치 — 수정 금지), 분기 말일과 실제 공시일
    사이엔 DART 정기공시 특성상 항상 수십일의 시차가 있어 지목한 분기 자체가 매번 조용히
    그 이전 분기로 대체된다. operating_margin 등 resolve_metric(DART financials 직접 quarter
    매치) 경로는 이 간접참조를 안 거쳐 영향받지 않는다(같은 분기 질문에 정상 응답하는 이유).
    안전장치 자체는 그대로 두고, resolve_computed_metric이 이미 넘겨주는 data_quarter(실제
    사용된 분기)가 요청 분기와 다르면 조용한 대체를 정직한 대체로 바꾼다.

    return_12m(가격모멘텀)은 예외다 — get_cross_section의 모든 행이 공통으로 "quarter"
    필드를 달고 나오지만, return_12m 자체의 값(1년 전 종가 대비 수익률)은 재무제표 분기와
    전혀 무관하게 순수 가격만으로 계산된다. 그대로 두면 실측 재현된 오탐(실제로는 정확한
    최신 시세인데 "아직 공시 안 됨" 경고가 붙음)이 발생하므로 이 지표만 건너뛴다.
    """
    if period is None or not financial or financial.get("metric") == "return_12m":
        return None
    target = period["quarter"] if period.get("kind") == "quarter" else f"{period['year']}Q4"
    actual = financial.get("data_quarter")
    if actual is None or actual == target:
        return None
    return (
        f"요청하신 {target} 데이터는 아직 공시되지 않아, 대신 공시된 가장 최근 분기"
        f"({actual}) 기준으로 계산되었습니다."
    )


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
        "sector_neutral": _coerce_sector_neutral(raw.get("sector_neutral")),
        "both_extremes": _coerce_sector_neutral(raw.get("both_extremes")),
        "sector_neutral_compare": _coerce_sector_neutral(raw.get("sector_neutral_compare")),
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
    """스크리닝 실행부(국내 도메인). 스펙추출 → asof해석 → 크로스섹션 → combine → rows 반환.

    실패는 조용히 빈 결과가 아니라 result=None + errors 사유로 남긴다(이번 세션 초반 '필드 환각이
    조용히 빈 결과로 둔갑' 버그 재발 방지). combine 이 던지는 ValueError(존재하지 않는 필드 등)를
    명시적으로 잡아 사유를 기록한다.

    시장 필터는 spec["markets"](코스피/코스닥)를 rows 의 'market' 필드로 적용한다
    (combine 의 markets= 인자를 그대로 재사용).

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
        "sector_neutral": False, "both_extremes": False, "sector_neutral_compare": False,
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
    result["both_extremes"] = bool(spec.get("both_extremes", False))
    # LLM이 "정렬 후 다시 확인" 같은 2단계 서술만 보고도 섹터중립화를 스스로 추론해 true로
    # 켜버리는 과잉판단이 실사용에서 재현됐다(사용자가 명시적으로 요구하지 않았는데도 항상
    # 켜짐). override_spec(사용자가 직접 편집한 값)은 그대로 신뢰하되, LLM/휴리스틱 추출
    # 경로는 질문 원문에 실제로 그 표현(_detect_sector_neutral_keyword)이 있을 때만 최종
    # true로 인정한다 — "명시적으로 요청했을 때만" 켜지도록 결정론적으로 이중 게이트.
    if override_spec is not None:
        spec["sector_neutral"] = spec.get("sector_neutral", False)
    else:
        spec["sector_neutral"] = (
            spec.get("sector_neutral", False) and _detect_sector_neutral_keyword(question)
        )
    # spec 자체를 게이트된 값으로 갱신해둔다 — 아래 combine_fn 호출(spec["sector_neutral"])과
    # on_progress detail(spec 그대로 노출)이 항상 이 최종값 하나만 보게 만들어, 두 곳이
    # 서로 다른(게이트 전/후) 값을 따로 참조하는 재발을 막는다.
    result["sector_neutral"] = spec["sector_neutral"]
    # sector_neutral_compare도 동일한 이중 게이트를 적용한다 — override(사람이 직접 편집)는
    # 그대로 신뢰하되, LLM/휴리스틱 추출 경로는 질문 원문에 섹터중립 키워드 + 비교 의도가
    # 둘 다 실제로 있을 때만(_detect_sector_neutral_compare_keyword) 최종 true로 인정한다.
    if override_spec is not None:
        spec["sector_neutral_compare"] = spec.get("sector_neutral_compare", False)
    else:
        spec["sector_neutral_compare"] = (
            spec.get("sector_neutral_compare", False)
            and _detect_sector_neutral_compare_keyword(question)
        )
    result["sector_neutral_compare"] = spec["sector_neutral_compare"]

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

    # 시장 필터: markets(코스피/코스닥)를 rows 의 'market' 필드에 대해 적용한다
    # (select_stocks 의 markets= 인자 재사용).
    market_filter = spec.get("markets")

    sector_neutral = spec.get("sector_neutral", False)
    compare = result["sector_neutral_compare"]

    def _combine(criteria: list[dict], sn: bool):
        # 멀티팩터(2개 이상) z-score 조합 스크리닝에는 백테스트 엔진과 동일하게 z-score 이상치
        # 완화(winsorize_z=3.0)를 기본 적용한다 — 실시간 매수후보 스크리닝과 백테스트가 똑같은
        # 이상치 처리를 받아야 결과가 일관되기 때문. 단일 criterion(단순 정렬)은 z-score 클리핑이
        # 순위에 영향을 주지 않으므로 미적용(None)해 기존 동작을 보존한다(회귀 방지).
        winsorize_z = 3.0 if len(criteria) >= 2 else None
        return combine_fn(
            rows, criteria, method="zscore",
            n=spec["top_n"], sectors=requested_sectors, markets=market_filter,
            sector_neutral=sn, winsorize_z=winsorize_z,
        )

    def _extreme_criteria() -> tuple[list[dict], list[dict]]:
        # both_extremes: criteria가 한쪽 direction만 담겨 와도 key 기준으로 high/low를 재조립한다.
        keys: list[str] = []
        for c in spec["criteria"]:
            if c["key"] not in keys:
                keys.append(c["key"])
        return (
            [{"key": k, "direction": "high", "weight": 1.0} for k in keys],
            [{"key": k, "direction": "low", "weight": 1.0} for k in keys],
        )

    try:
        if result["both_extremes"] and compare:
            # 4-way 중첩: (최댓값/최솟값) × (섹터중립 전=raw / 후=sector_neutral)를 모두 계산한다.
            high_criteria, low_criteria = _extreme_criteria()
            selected = {
                "highest": {
                    "raw": _combine(high_criteria, False),
                    "sector_neutral": _combine(high_criteria, True),
                },
                "lowest": {
                    "raw": _combine(low_criteria, False),
                    "sector_neutral": _combine(low_criteria, True),
                },
            }
        elif compare:
            # 섹터중립화 전(raw=원본)과 후(sector_neutral)를 한 번에 나란히 담아 비교한다.
            selected = {
                "raw": _combine(spec["criteria"], False),
                "sector_neutral": _combine(spec["criteria"], True),
            }
        elif result["both_extremes"]:
            # "최댓값과 최솟값 둘 다": 같은 지표를 가중합으로 섞으면(combine의 zscore) 의도와
            # 달라지므로, 지표 key만 뽑아 direction=high(최댓값)와 low(최솟값)로 각각 top_n=1
            # combine을 따로 호출해 highest/lowest로 나란히 담는다. criteria가 한쪽 direction만
            # 담겨 와도 양쪽 다 재구성되도록 key 기준으로 재조립한다(LLM 응답 견고성).
            high_criteria, low_criteria = _extreme_criteria()
            highest = _combine(high_criteria, sector_neutral)
            lowest = _combine(low_criteria, sector_neutral)
            selected = {"highest": highest, "lowest": lowest}
        else:
            selected = _combine(spec["criteria"], sector_neutral)
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


_CONJUNCTION_TOKENS: tuple[str, ...] = ("그리고", "이랑", "랑", "와", "과", "및", ",", "·", "&", " ")


def _is_gap_pure_conjunction(
    question: str, start: int, end: int, quarter_events: list[tuple[int, int, int]],
) -> bool:
    """[start:end) 구간 원문이 접속사(와/과/및/그리고 등)만으로 이루어졌는지 확인한다.

    구간에 포함된 분기 토큰(quarter_events, 예: "1분기")은 검사에서 제외한다 — 그건
    이미 파싱된 분기 자체이지 "지표명 반복" 같은 실질 단어가 아니기 때문이다. 나머지
    텍스트에서 접속사 토큰을 모두 제거했을 때 빈 문자열이면 순수 접속사로 판단한다.
    """
    pieces: list[str] = []
    cursor = start
    for qs, qe, _val in sorted(e for e in quarter_events if e[0] >= start and e[1] <= end):
        pieces.append(question[cursor:qs])
        cursor = qe
    pieces.append(question[cursor:end])
    remainder = "".join(pieces)
    for tok in _CONJUNCTION_TOKENS:
        remainder = remainder.replace(tok, "")
    return remainder == ""


def _parse_periods(question: str) -> list[dict]:
    """질문에서 여러 조회 기간을 순서대로 파싱한다(다중분기 지원). _parse_period의 리스트판.

    _parse_period는 첫 연도/첫 분기 하나만 잡아 "하이닉스 25년과 26년 1분기 영업이익률"의
    둘째 분기(2026Q1)를 통째로 버렸다. 이 함수는 언급된 각 기간을 위치 순서대로 담아 반환한다.
    원소 형식은 _parse_period와 동일하다({"kind":"quarter","quarter":"YYYYQn"} 또는
    {"kind":"annual","year":YYYY}).

    파싱 규칙(연도와 분기가 어떻게 짝지어지는지):
    - 연도 토큰(4자리/2자리+년)마다 새 "기간 슬롯"을 연다. 분기 토큰은 바로 앞의 아직 분기가
      비어 있는 슬롯을 채우고, 이미 찬 슬롯 뒤에 오면 같은 연도로 새 슬롯을 연다
      (예: "2025년 1분기와 2분기" → 2025Q1, 2025Q2 — 한 연도가 두 분기를 가짐).
    - 각 연도가 자기 분기를 하나씩 명시하면 그대로 짝짓는다
      (예: "2025년 1분기와 2026년 1분기" → 2025Q1, 2026Q1).
    - "공유 분기" 규칙: 분기가 딱 하나만 명시됐는데 분기 없는(bare) 연도 슬롯이 남아 있으면,
      그 단일 분기를 bare 연도들에도 적용한다. "25년과 26년 1분기"에서 앞의 "25년"은 뒤의
      "1분기"를 공유해 2025Q1이 되는 것이 한국어의 자연스러운 비교 표현이기 때문이다
      (이 경우 → 2025Q1, 2026Q1). 분기가 2개 이상 서로 다르게 명시되면 공유하지 않고 bare
      연도는 그 해 연간(사업보고서)으로 남긴다(혼재 질문의 합리적 해석).
    - 공유는 인접한 연도 슬롯 사이의 원문이 "순수 접속사"(과/와/및/그리고/쉼표 등, 그
      구간에 속한 분기 토큰 자체는 제외)로만 이루어졌을 때만 적용한다. "25년과 26년
      1분기"는 두 연도 사이가 "과 "뿐이라 공유되지만, "25년 영업이익률과 26년 1분기
      영업이익률"처럼 연도 사이에 지표명 등 실질적인 단어가 끼어 있으면(=그 해에 대해
      이미 완결된 절이 있다는 뜻) 공유하지 않고 bare 연도는 연간으로 남는다(실서버 재현:
      "SK하이닉스 25년 영업이익률과 26년 1분기 영업이익률"이 [2025Q1,2026Q1]으로 잘못
      해석되던 버그). 인접 연도가 없는 고립된 bare 슬롯(orphan quarter 케이스 등)은
      기존처럼 공유를 허용한다(회귀 방지, 흔치 않은 순서).
    - 연도가 하나도 없으면(분기 단독 포함) 빈 리스트를 반환한다 — _parse_period가 None을
      돌려 "현행 최신분기 유지"로 흐르던 것과 같은 회귀-안전 동작이다.

    호출부는 _strip_retry_feedback을 거친 원본 질문을 넘긴다(_parse_period와 동일 원칙).
    """
    q = question or ""

    # 연도 토큰 수집(4자리/2자리+년). 두 정규식은 lookbehind로 서로 겹치지 않는다
    # (2자리 연도는 앞에 숫자가 붙으면 매치 안 되므로 "2025" 안의 "25"는 잡히지 않음).
    year_events: list[tuple[int, int, int]] = []  # (start, end, year)
    for m in _YEAR_FULL_RE.finditer(q):
        year_events.append((m.start(), m.end(), int(m.group(1))))
    for m in _YEAR_SHORT_RE.finditer(q):
        year_events.append((m.start(), m.end(), 2000 + int(m.group(1))))

    # 분기 토큰 수집(한글 "N분기" / "qN"). 같은 위치를 두 형식이 동시에 잡는 일은 없다.
    quarter_events: list[tuple[int, int, int]] = []  # (start, end, quarter_n)
    for m in _QUARTER_KO_RE.finditer(q):
        quarter_events.append((m.start(), m.end(), int(m.group(1))))
    for m in _QUARTER_Q_RE.finditer(q):
        quarter_events.append((m.start(), m.end(), int(m.group(1))))

    if not year_events:
        return []  # 연도 없으면 기간 미지정(분기 단독은 현행 유지)

    # 연도/분기 이벤트를 위치 순서로 병합해 슬롯을 만든다.
    events = [(s, e, "year", val) for s, e, val in year_events] + \
             [(s, e, "quarter", val) for s, e, val in quarter_events]
    events.sort(key=lambda ev: ev[0])

    slots: list[dict] = []  # 각 원소 {"year", "quarter", "year_start", "year_end"}
    current_year: int | None = None
    orphan_quarters: list[tuple[int, int, int]] = []  # 연도 없이 먼저 온 분기(start,end,val)
    for s, e, kind, val in events:
        if kind == "year":
            current_year = val
            slots.append({"year": val, "quarter": None, "year_start": s, "year_end": e})
        else:  # quarter
            if slots and slots[-1]["quarter"] is None:
                slots[-1]["quarter"] = val
            elif current_year is not None:
                slots.append({"year": current_year, "quarter": val, "year_start": s, "year_end": e})
            else:
                # 연도가 아직 없는데 분기가 먼저 왔다(부자연스러운 "1분기 2025년" 순서).
                # 버리지 않고 모아 뒤의 bare 연도에 공유 분기로 붙인다 — _parse_period가
                # 순서 무관하게 첫 연도+첫 분기를 짝짓는 것과 동작을 일치시킨다(회귀-안전).
                orphan_quarters.append((s, e, val))

    # 공유 분기 백필: bare 연도 슬롯이 남았고 명시된 분기(고아 포함)가 정확히 하나면,
    # "인접 연도 슬롯 사이의 원문이 순수 접속사뿐인" bare 슬롯에만 공유한다(위 docstring
    # 참고 — 연도 사이에 지표명 등 실질 단어가 끼면 그 해는 이미 완결된 절이므로 공유 안 함).
    distinct_quarters = {s["quarter"] for s in slots if s["quarter"] is not None}
    distinct_quarters |= {v for _s, _e, v in orphan_quarters}
    has_bare = any(s["quarter"] is None for s in slots)
    if has_bare and len(distinct_quarters) == 1:
        shared = next(iter(distinct_quarters))
        for i, s in enumerate(slots):
            if s["quarter"] is not None:
                continue
            neighbors = []
            if i + 1 < len(slots):
                neighbors.append((s["year_end"], slots[i + 1]["year_start"]))
            if i - 1 >= 0:
                neighbors.append((slots[i - 1]["year_end"], s["year_start"]))
            if not neighbors:
                # 인접 연도가 없는 고립된 bare 슬롯(예: 고아 분기만 있는 경우) — 기존처럼 공유.
                s["quarter"] = shared
                continue
            if any(_is_gap_pure_conjunction(q, gs, ge, quarter_events) for gs, ge in neighbors):
                s["quarter"] = shared

    periods: list[dict] = []
    for s in slots:
        if s["quarter"] is not None:
            periods.append({"kind": "quarter", "quarter": f"{s['year']}Q{s['quarter']}"})
        else:
            periods.append({"kind": "annual", "year": s["year"]})

    # 동일 기간이 중복 언급돼도 한 번만(순서 보존) — 불필요한 중복 조회 방지.
    deduped: list[dict] = []
    for p in periods:
        if p not in deduped:
            deduped.append(p)
    return deduped


# ── "최근 N개월/N년 수익률" 기간 수익률 파서 (실서버 재현 버그 수정) ─────────────────
# "삼성전자 최근 3개월 수익률"처럼 12가 아닌 임의 개월수는 _extract_metric(리터럴 "12개월"
# 별칭만 인식)이 못 잡아 free_exec LLM 코드생성으로 폴백 → LLM이 8년 전 데이터로 계산해
# 틀린 답을 내던 문제를 고친다. 여기서 개월수를 뽑아 결정론 함수(price_return_over_months)로
# 라우팅한다.
#
# 회귀 안전장치:
# - 월 형태는 "최근/지난" 접두 또는 "간/동안" 접미가 있을 때만 매치한다 →
#   "직전 12개월 수익률"/"12개월 수익률"(bare)/"모멘텀"은 여기서 None을 돌려 기존
#   _COMPUTED_KO_ALIASES(return_12m) 경로가 그대로 처리하게 한다(기존 테스트/동작 유지).
# - 년 형태는 "최근/지난" 접두 또는 "간/동안" 접미가 있거나 한 자리 수(1~9)일 때만
#   매치한다 → "2024년 수익률"/"25년 수익률" 같은 캘린더 연도(_parse_period가 처리)를
#   "N년 전 기간"으로 오인하지 않는다.
_RECENT_RETURN_MONTHS_RE = re.compile(
    r"(?:최근|지난)\s*(\d+)\s*개월"          # 최근/지난 N개월 …
    r"|(\d+)\s*개월\s*(?:간|동안)"           # N개월간 …
)
_RECENT_RETURN_YEARS_RE = re.compile(
    r"(?:최근|지난)\s*(\d+)\s*년"            # 최근/지난 N년 …
    r"|(\d+)\s*년\s*(?:간|동안)"             # N년간 …
    r"|(?<!\d)([1-9])\s*년(?!\d)"           # 한 자리 N년 (캘린더 연도가 아님)
)


def _parse_recent_return_months(question: str) -> int | None:
    """질문에서 "최근 N개월/N년 수익률" 류 기간을 개월수로 변환한다("년"은 ×12). 없으면 None.

    "수익률"이 없으면 곧바로 None(주가/재무 다른 질문 오탐 방지). 월 형태를 년 형태보다
    먼저 확인해 "최근 12개월 수익률"을 12로 잡는다. 기존 리터럴 "12개월" 별칭 경로와
    겹쳐도 이 경로가 우선 반환하므로(호출부에서 먼저 분기) 순서가 명확하다.
    """
    q = question or ""
    if "수익률" not in q:
        return None
    m = _RECENT_RETURN_MONTHS_RE.search(q)
    if m:
        return int(m.group(1) or m.group(2))
    m = _RECENT_RETURN_YEARS_RE.search(q)
    if m:
        return int(m.group(1) or m.group(2) or m.group(3)) * 12
    return None


# ── 특정일자(연도 없는 월-일 포함) 시세 조회 (실서버 재현 버그) ──────────────────────
# "하이닉스 6월 18일 주가정보"(연도 미명시)가 데이터가 있는 가장 오래된 연도(2015-06-18)를
# 골라 답하던 버그를 고친다. _parse_period는 연도/분기만 인식해 "월-일"은 통째로 무시했고,
# 정형 경로가 특정일자를 반영하지 못해 검증에 실패 → exec_fallback 자유코드로 새어
# 날짜필터 없는 전체이력 조회 → 가장 오래된 행(2015)을 골랐다. domain_backtest의 "종료연도
# 미명시 시 오늘이 속한 연도 기본값" 공통 규칙과 동일 원칙을 여기에도 적용한다: 연도가 없는
# 월-일은 오늘이 속한 연도를 기본값으로 쓰고(연도가 명시됐으면 그 연도), 실제 거래일 폴백은
# get_latest_price_kr의 "date <= asof 중 최신"이 자연히 담당한다(미래/휴장일이면 그 시점
# 이하 가장 가까운 과거 거래일).
#
# "N월 D일" 형태만 특정일자로 본다("D일"이 반드시 있어야 함) — "6월 매출"(월만)이나
# "3개월"(개월, '월' 앞이 숫자가 아님)에는 매치되지 않는다(오탐 방지).
_MONTH_DAY_RE = re.compile(r"(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일")


def _parse_price_target_date(question: str, today: date | None = None) -> str | None:
    """질문에서 "특정일자(연도 없는 월-일 포함)"를 'YYYY-MM-DD' 문자열로 파싱한다. 없으면 None.

    - "6월 18일"(연도 미명시) → 오늘이 속한 연도의 6월 18일(domain_backtest의 "오늘이 속한
      연도 기본값" 규칙과 동일). 그 시점 이하 가장 가까운 과거 거래일 폴백은 조회 단계
      (get_latest_price_kr, date<=asof)가 담당하므로 여기서는 달력 날짜만 만든다.
    - "2025년 6월 18일" / "25년 6월 18일"(연도 명시) → 그 연도의 6월 18일.
    - "월-일"이 없으면(순수 시세 "주가 알려줘" 등) None → 호출부가 기존 최신값 경로 유지.
    - 존재하지 않는 날짜(예: 2월 30일)는 None(달력 검증).

    호출부는 재시도 피드백이 섞이지 않도록 _strip_retry_feedback을 거친 원본 질문을 넘긴다.
    """
    q = question or ""
    m = _MONTH_DAY_RE.search(q)
    if m is None:
        return None
    month, day = int(m.group(1)), int(m.group(2))
    today = today or date.today()
    ym = _YEAR_FULL_RE.search(q)
    if ym:
        year = int(ym.group(1))
    else:
        ys = _YEAR_SHORT_RE.search(q)
        year = 2000 + int(ys.group(1)) if ys else today.year
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None  # 존재하지 않는 날짜(2월 30일 등)


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
        # q(소문자화)로 비교한다 — 한글 별칭은 대소문자 개념이 없어 영향이 없고, "gpa"/"gp/a"
        # 처럼 영문 별칭이 섞여 있는 항목은 대문자 입력("GPA")도 인식돼야 한다(회귀 재현:
        # "삼성전자 GPA 알려줘"가 원본 question과 대조돼 소문자 별칭과 매치 못 하던 버그).
        if ko in q:
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

    data_quarter는 metrics_at()이 그 행 계산에 실제로 사용한 재무 분기(row["quarter"],
    look-ahead 방지로 asof 시점까지 공시된 분기)다. period(asof, 가격 기준일)와는 별개다 —
    질문이 특정 분기를 지목했는데 아직 그 분기가 공시 전이면 asof 기준으로 더 이전 분기가
    조용히 선택될 수 있고, 호출부가 이 필드로 그 대체를 사용자에게 알릴 수 있게 한다.
    """
    execute_sql_fn = execute_sql_fn or execute_sql
    cross_section_fn = cross_section_fn or (lambda c, a: get_cross_section(c, a))
    asof = asof or _default_screening_asof(conn, "prices", execute_sql_fn)
    value = None
    estimated = None
    data_quarter = None
    if asof:
        row = next(
            (r for r in cross_section_fn(conn, asof) if r.get("stock_code") == stock_code),
            None,
        )
        if row is not None:
            value = row.get(metric)
            estimated = row.get(f"{metric}_estimated")
            data_quarter = row.get("quarter")
    return {
        "stock_code": stock_code, "metric": metric,
        "value": value, "source": "computed", "period": asof, "estimated": estimated,
        "data_quarter": data_quarter,
    }


# ── 질문 → 기술지표 스펙 추출 (technical 축이 켜졌을 때만 사용) ────────────────────
# classify_intent 가 technical 로 판단하면 어떤 기술지표를 원하는지 뽑아
# compute_technical_indicator(src/backtest/primitives.py, TA-Lib) 스펙 리스트로 만든다.
# 지원 지표명은 primitives._SUPPORTED_INDICATORS(sma/ema/rsi/macd/bollinger)와 일치한다.
_INDICATOR_KEYWORD_SPECS: list[tuple[tuple[str, ...], dict]] = [
    (("골든크로스", "데드크로스", "이동평균", "이평선", "이평", "단순이동평균", "sma"), {"name": "sma"}),
    (("지수이동평균", "ema"), {"name": "ema"}),
    (("rsi",), {"name": "rsi"}),
    (("macd",), {"name": "macd"}),
    (("볼린저", "bollinger", "bband"), {"name": "bollinger"}),
]


def _extract_indicators(question: str) -> list[dict]:
    """질문에서 요청된 기술지표를 뽑아 compute_technical_indicator 용 스펙 리스트로 반환한다.

    classify_intent 가 technical 로 판단했을 때만 호출된다(그 외에는 순수 시세). 키워드로
    특정 지표를 못 집으면(예: LLM 이 키워드 없이 technical 로만 판단) 대표 지표인 이동평균(sma)
    으로 안전 폴백한다 — technical 축이 켜졌으면 최소 하나의 지표는 계산해야 하기 때문이다.
    """
    q = question.lower()
    specs: list[dict] = []
    for keywords, spec in _INDICATOR_KEYWORD_SPECS:
        if any(k in q for k in keywords):
            specs.append(dict(spec))
    return specs or [{"name": "sma"}]


def _intent_prompt(question: str) -> str:
    """classify_intent 용 LLM 판단 프롬프트(재무/주가/기술지표). route_question 의 _route_prompt 와 동일 관례.

    시장 구분(markets/exchanges)과 마찬가지로, 재무 vs 주가 vs 기술지표 구분도 하드코딩
    키워드가 아니라 LLM 이 질문 의미를 읽고 판단하게 한다. LLM 은 정해진 셋
    (financial/price/technical) 중 **필요한 것을 모두** 고른다(단독 또는 조합).
    """
    return (
        "다음 질문이 어떤 데이터를 원하는지 판단하세요. 해당하는 것을 모두 고르세요(여러 개 가능).\n"
        "- financial: 재무데이터(매출/영업이익/순이익/PER/PBR/ROE 등). 매출성장률·12개월 수익률·"
        "모멘텀처럼 값으로 계산되는 '지표' 요청도 financial 로 봅니다(단순 시세 조회가 아님).\n"
        "- price: 순수 시세(종가/시가/고가/저가/거래량 등)\n"
        "- technical: 기술지표(이동평균/이평선/RSI/MACD/볼린저밴드 등 가격 시계열로 계산하는 지표)\n"
        "financial, price, technical 중 필요한 것을 모두 쉼표로 구분해 답하세요(예: 'financial, price').\n\n"
        f"질문: {question}\n답:"
    )


_INTENT_TOKEN_RE = re.compile(r"\b(financial|price|technical)\b")


def _parse_intent(raw: str | None) -> tuple[str, ...] | None:
    r"""LLM 응답에서 financial/price/technical 을 모두 뽑아 정렬 튜플로 돌려준다.

    여러 축이 잡히면 모두 담는다(여러 서브에이전트를 함께 부르기 위함). 명확한 신호가
    전혀 없으면 None(→ 키워드 폴백). 단어경계(\b) 매칭이라 'financials' 같은 부분문자열에
    오탐하지 않는다. 한국어 응답(재무/시세·주가/기술지표)도 관대하게 수용한다.
    """
    t = (raw or "").lower()
    found = set(_INTENT_TOKEN_RE.findall(t))
    if "재무" in t:
        found.add("financial")
    if "시세" in t or "주가" in t:
        found.add("price")
    if "기술" in t:
        found.add("technical")
    if not found:
        return None
    return tuple(sorted(found))


def _classify_intent_heuristic(question: str) -> tuple[str, ...]:
    """키워드 기반 intent 판단(LLM 미가용/불명확 시 안전망). 재무/주가/기술지표 세 축을
    독립적으로 판단해, 해당하는 축을 모두 담은 정렬 튜플을 돌려준다.

    아무 축도 안 잡히면(불명확) 놓치는 것보다 과다조회가 안전하므로 재무+주가로 폴백한다
    — 기술지표는 무거운 TA-Lib 계산이라 명시적 신호가 있을 때만 켠다. return_12m/'수익률'
    같은 계산전용 재무지표 키워드(_COMPUTED_KO_ALIASES)를 financial 로 보호하던 회귀 방지
    로직도 여기 유지된다.
    """
    q = question.lower()
    needs_financial = any(k in q for k in _FINANCIAL_KEYWORDS) or any(
        k in q for k in _COMPUTED_KO_ALIASES
    )
    needs_price = any(k in q for k in _PRICE_KEYWORDS)
    needs_technical = any(k in q for k in _TECHNICAL_KEYWORDS)
    found: set[str] = set()
    if needs_financial:
        found.add("financial")
    if needs_price:
        found.add("price")
    if needs_technical:
        found.add("technical")
    if not found:
        return ("financial", "price")
    return tuple(sorted(found))


def classify_intent(
    question: str, llm_fn: Callable[[str], str] | None = None
) -> tuple[str, ...]:
    """질문이 재무/주가/기술지표 중 무엇을 요구하는지 판단한다(여러 축 동시 가능).

    반환: {"financial", "price", "technical"} 의 정렬된 부분집합 튜플(항상 비어있지 않음).
    - 하나만 필요하면 그 축만 담긴 튜플 → 단독 서브에이전트 호출
    - 여러 개 필요하면 여러 축 → 여러 서브에이전트(함수) 를 함께 호출
    route_question(총괄 라우팅)과 동일하게 **LLM 우선**이다 — llm_fn 이 주어지면 먼저 LLM 에
    판단을 위임하고(_intent_prompt), 응답에서 축을 모두 뽑는다. LLM 이 없거나 응답에서 판단을
    못 뽑으면(파싱 실패/예외) 키워드 휴리스틱(_classify_intent_heuristic)으로 폴백한다 —
    안전망은 유지한다(예: return_12m/'수익률' 은 폴백에서 financial 로 보호). 끝까지 불명확하면
    놓치는 것보다 과다조회가 안전하므로 재무+주가로 폴백한다(무거운 기술지표는 제외).
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


# ── 기간 미지정 시 "실제 사용된 데이터 시점" 라벨링(data_asof) ─────────────────────
# 질문에 연도/분기가 없으면 시스템이 자동으로 데이터 시점을 정한다(주가 기반 지표=종가
# 기준일, 재무제표 기반 지표=기준분기). 그 자동 결정된 시점을 사용자가 검증할 수 있도록,
# 이미 resolve된 값만 재사용해(effective_* 재호출/재계산 금지) 결과 dict에 순수 추가한다.
# roc_estimated/top_n/qvm_summary 등에서 써온 "결과 dict에 필드만 얹는" 관례를 그대로 따른다.
_QUARTER_LABEL_RE = re.compile(r"\d{4}\s*Q[1-4]", re.IGNORECASE)


def _is_quarter_label(label) -> bool:
    """'2025Q1'/'2025 연간' 같은 재무 기준분기 라벨인지(아니면 날짜형=가격 기준일으로 간주)."""
    return isinstance(label, str) and (bool(_QUARTER_LABEL_RE.search(label)) or "연간" in label)


def _collect_data_asof(financials: list, price_rows: list) -> dict | None:
    """재무 결과들 + 주가 스냅샷 행들에서 실제 사용된 시점을 추출한다(이미 계산된 값 재사용).

    - 주가 기반 재무지표(per/pbr/시총)는 resolve_metric이 담아준 price_date를 가격 기준일로,
    - 그 외 재무지표는 period(분기/연간 라벨)를 재무 기준분기로,
    - 계산전용 지표(return_12m 등)·FnGuide 스냅샷은 날짜형 period를 가격 기준일로,
    - 주가 스냅샷 행은 각 행의 date(prices 테이블 최신 거래일)를 가격 기준일로 쓴다.

    반환: {"price_date": ..., "financial_quarter": ...}(값이 있는 키만). 시점이 전혀 없으면 None.
    """
    asof: dict = {}
    for fin in financials:
        if not isinstance(fin, dict):
            continue
        pd = fin.get("price_date")
        if pd:
            asof.setdefault("price_date", pd)
        period = fin.get("period")
        if period:
            if _is_quarter_label(period):
                asof.setdefault("financial_quarter", period)
            else:
                asof.setdefault("price_date", period)
    dates = [r.get("date") for r in price_rows if isinstance(r, dict) and r.get("date")]
    if dates:
        asof["price_date"] = max(dates)
    return asof or None


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
    price_return_fn: Callable,
    today: date | None = None,
) -> dict:
    """한 질문이 이름으로 지목한 여러 종목(예: "삼성전자와 SK하이닉스 종가") 각각에 답한다.

    find_stock_codes가 서로 다른 종목을 2개 이상 찾았을 때만 이 경로로 온다 — "PER 낮은
    5개사"처럼 조건으로 종목을 "고르는" 스크리닝과 다르다(그건 answer_kr_screening이
    이미 먼저 가로챈다). intent/metric/period는 질문 전체에서 한 번만 판단해 모든 종목에
    동일하게 적용한다(같은 질문 안에서 종목마다 다른 지표를 묻는 경우는 범위 밖).
    """
    base_question = _strip_retry_feedback(question)
    period = _parse_period(base_question)

    # "최근 N개월/N년 수익률"은 단일종목 경로(answer_kr_question)와 동일하게 결정론 함수
    # (price_return_fn)로 종목별 계산한다. 이 처리가 다중종목 경로에 빠져 있으면 _extract_metric이
    # 12가 아닌 임의 개월수를 못 잡아 종목마다 "재무 지표를 인식하지 못함"으로 실패하고,
    # 상위(총괄)에서 LLM 자유 코드생성 폴백으로 새어 8년 전 데이터로 틀린 답을 내던 회귀를 막는다.
    recent_months = _parse_recent_return_months(base_question)
    if recent_months is not None:
        asof = _default_screening_asof(conn, "prices", execute_sql_fn)
        entities = []
        for code in stock_codes:
            entity: dict = {"stock_code": code, "financial": None, "price": None, "errors": []}
            financial, err = _call_with_retry(
                lambda c=code: price_return_fn(conn, c, asof, recent_months)
            )
            if err:
                entity["errors"].append(f"기간 수익률 계산 실패: {err}")
            else:
                entity["financial"] = financial
            entities.append(entity)
        result = {
            "stock_code": None,
            "stock_codes": stock_codes,
            "question": question,
            "intent": ("financial",),
            "financial": None,
            "price": None,
            "entities": entities,
            "errors": [],
        }
        if period is None and asof:
            result["data_asof"] = {"price_date": asof}
        return result

    intent = classify_intent(question, llm_fn=llm_fn)
    metric = _extract_metric(question) if "financial" in intent else None
    # technical 축이 켜졌으면(기술지표 요청) 질문에서 지표 스펙을 뽑아 채운다. 호출부가
    # indicators 를 명시했으면 그것을 우선한다(테스트 주입/명시 조회). 순수 시세면 None.
    active_indicators = indicators
    if "technical" in intent and not active_indicators:
        active_indicators = _extract_indicators(base_question)

    entities: list[dict] = []
    for code in stock_codes:
        entity: dict = {"stock_code": code, "financial": None, "price": None, "errors": []}

        if "financial" in intent:
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
                # 재무제표/사전계산 지표: resolve_metric 우선, per/pbr/roe/operating_margin/
                # debt_ratio/market_cap이 과거 시점 metrics 부재로 None이면 계산 폴백(문제 A,
                # 다중종목·단일기간). period가 None이면 폴백 미시도 → 기존 최신값 경로 무영향.
                financial, err = _call_with_retry(
                    lambda c=code: _resolve_metric_with_fallback(
                        conn, c, metric, period, llm_fn,
                        resolve_metric_fn, computed_metric_fn, execute_sql_fn,
                    )
                )
                if err:
                    entity["errors"].append(f"재무데이터 조회 실패: {err}")
                else:
                    entity["financial"] = financial

        if "price" in intent or "technical" in intent:
            # period가 명시됐으면(예: "25년 기준") 그 시점 이하 최근 거래일 종가를 가져온다
            # (_resolve_screening_asof가 quarter/annual→날짜로 변환) — 재무 쪽만 과거 시점을
            # 반영하고 가격은 항상 오늘 최신값을 보여줘 서로 다른 기준시점이 뒤섞이던 버그
            # 수정. period가 None이면 asof=None → get_price_snapshot_kr 기존 최신값 경로.
            # technical 이면 active_indicators 가 채워져 지표까지 계산되고, 순수 시세면 None.
            price_asof = _resolve_screening_asof(period, conn, "prices", execute_sql_fn)
            # "6월 18일" 특정일자(연도 없는 월-일 포함) 지목 시 그 날짜를 asof로 우선한다
            # (단일종목 경로와 동일 — 연도 미명시면 오늘이 속한 연도 기본값).
            target_date = _parse_price_target_date(base_question, today=today)
            if target_date is not None:
                price_asof = target_date
            price, err = _call_with_retry(
                lambda c=code: price_snapshot_fn(
                    conn, c, asof=price_asof, indicators=active_indicators
                )
            )
            if err:
                entity["errors"].append(f"주가데이터 조회 실패: {err}")
            else:
                entity["price"] = price

        entities.append(entity)

    result = {
        "stock_code": None,
        "stock_codes": stock_codes,
        "question": question,
        "intent": intent,
        "financial": None,
        "price": None,
        "entities": entities,
        "errors": [],
    }
    # 기간 미지정 시 실제 사용된 시점을 라벨링한다 — intent/metric/period가 모든 종목에
    # 동일하게 적용되므로 종목별 결과에서 모아 최상위 data_asof 하나로 담는다.
    if period is None:
        data_asof = _collect_data_asof(
            [e.get("financial") for e in entities],
            [row for e in entities for row in (e.get("price") or [])],
        )
        if data_asof:
            result["data_asof"] = data_asof
    return result


def _normalize_computed_result(computed: dict) -> dict:
    """resolve_computed_metric 결과를 resolve_metric과 동일한 키 집합으로 정규화한다(문제 A 폴백용).

    per/pbr/roe/operating_margin/debt_ratio/market_cap이 과거 시점 metrics 스냅샷 부재로
    None이라 metrics_at 기반 계산 폴백이 발동했을 때만 쓴다. 호출부(_collect_data_asof/
    _summarize_one/합성 프롬프트/web 계층)가 resolve_metric 형식(dart_*/fnguide_*/price_date)을
    가정하므로, computed 결과의 누락 키를 None으로 채워 형식 불일치로 UI/합성이 깨지지 않게
    한다. value/source/period는 computed 값을 그대로 쓰고, computed 고유의 estimated(근사 여부)도
    보존한다(다른 계산전용 경로 resolve_computed_metric과 동일 계약).
    """
    return {
        "stock_code": computed.get("stock_code"),
        "metric": computed.get("metric"),
        "value": computed.get("value"),
        "source": computed.get("source"),
        "period": computed.get("period"),
        "dart_value": None,
        "dart_period": None,
        "fnguide_value": None,
        "fnguide_period": None,
        "price_date": None,
        "estimated": computed.get("estimated"),
    }


def _resolve_metric_with_fallback(
    conn,
    stock_code: str,
    metric: str,
    period: dict | None,
    llm_fn: Callable[[str], str] | None,
    resolve_metric_fn: Callable,
    computed_metric_fn: Callable,
    execute_sql_fn: Callable,
) -> dict:
    """재무제표/사전계산 지표(계산전용 아님)를 resolve_metric으로 조회하고, 필요 시 계산 폴백을 덧댄다.

    우선순위는 항상 "resolve_metric 결과 우선 → 그래도 None이면 metrics_at 기반 계산 폴백"이다
    (기존 정상 경로 회귀 방지 — 사전계산값을 절대 덮어쓰지 않는다). 문제 A: metrics 사전계산
    테이블은 최신 한 분기 스냅샷만 유지해, 과거 시점을 지목한 per/pbr/roe/operating_margin/
    debt_ratio/market_cap(_METRICS_TABLE_COLS) 질문은 그 분기 metrics 행이 없어 resolve_metric이
    None으로 빠진다 — 원본 재무제표+주가(metrics_at)엔 그 시점 값이 있는데도. 그 경우 기간이
    명시(period is not None)됐으면 그 기간의 asof로 computed_metric_fn을 재시도한다. period가
    None(기간 미지정)이면 폴백을 아예 시도하지 않아 기존 최신값 경로에 무영향이다.
    """
    period_kwargs = {"period": period} if period is not None else {}
    result = resolve_metric_fn(conn, stock_code, metric, llm_fn=llm_fn, **period_kwargs)
    if (
        result.get("value") is None
        and period is not None
        and metric in _METRICS_TABLE_COLS
    ):
        asof = _resolve_screening_asof(period, conn, "prices", execute_sql_fn)
        computed = computed_metric_fn(
            conn, stock_code, metric, asof=asof, execute_sql_fn=execute_sql_fn,
        )
        if computed.get("value") is not None:
            return _normalize_computed_result(computed)
    return result


def _resolve_metric_over_periods(
    resolve_metric_fn: Callable,
    conn,
    stock_code: str,
    metric: str,
    periods: list[dict],
    llm_fn: Callable[[str], str] | None,
    computed_metric_fn: Callable,
    execute_sql_fn: Callable,
) -> list[dict]:
    """여러 기간(분기/연간) 각각에 지표를 조회해 기간별 결과 리스트로 담는다.

    반환 원소: {"period": <라벨>, "financial": <조회 결과 or None>, "errors": [...]}.
    라벨은 파싱된 기간(quarter="2026Q1" 또는 annual→"YYYY 연간")이라 프런트/검증이 어떤
    분기의 값인지 명확히 구분할 수 있다. 다중종목 경로의 entities 리스트와 동일한
    '리스트-of-딕셔너리' 관례를 따른다. 여기 오는 period는 항상 명시적이므로 period 인자를
    그대로 넘긴다(단일/무기간 경로는 호출부에서 이미 분기해 이 함수를 타지 않는다).

    문제 B: 지표 종류에 따라 기간별로 올바른 경로를 고른다(지표는 모든 기간에서 동일).
    - 계산전용 지표(_COMPUTED_ONLY_FIELDS: psr/roa/roc/ey/gpa/cfo_ratio/gross_margin/
      net_margin/cogs_ratio/return_12m/성장률 등)는 그 기간 asof로 computed_metric_fn
      (metrics_at 기반)을 호출한다.
    - 그 외 지표는 resolve_metric_fn으로 조회하되, per/pbr/roe/operating_margin/debt_ratio/
      market_cap이 과거 시점 metrics 부재로 None이면 계산 폴백을 덧댄다(문제 A와 동일 폴백).
    """
    is_computed = metric in _COMPUTED_ONLY_FIELDS
    out: list[dict] = []
    for p in periods:
        label = p["quarter"] if p["kind"] == "quarter" else f"{p['year']} 연간"
        entry: dict = {"period": label, "financial": None, "errors": []}
        if is_computed:
            financial, err = _call_with_retry(
                lambda p=p: computed_metric_fn(
                    conn, stock_code, metric,
                    asof=_resolve_screening_asof(p, conn, "prices", execute_sql_fn),
                    execute_sql_fn=execute_sql_fn,
                )
            )
        else:
            financial, err = _call_with_retry(
                lambda p=p: _resolve_metric_with_fallback(
                    conn, stock_code, metric, p, llm_fn,
                    resolve_metric_fn, computed_metric_fn, execute_sql_fn,
                )
            )
        if err:
            entry["errors"].append(f"재무데이터 조회 실패: {err}")
        else:
            entry["financial"] = financial
        out.append(entry)
    return out


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
    price_return_fn: Callable | None = None,
    on_progress: Callable[..., None] | None = None,
    today: date | None = None,
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
            "intent": tuple[str, ...] | None,   # {"financial","price","technical"} 의 부분집합
            "financial": dict | None,     # resolve_metric() 또는 resolve_computed_metric() 결과
            "price": list[dict] | None,   # get_price_snapshot_kr() 결과(technical 이면 지표 포함)
            "errors": list[str],
        }
    질문이 서로 다른 분기/연도를 2개 이상 지목하면(예: "25년과 26년 1분기") financial 대신
    "periods": [{"period": "2025Q1", "financial": {...}, "errors": []}, ...] 를 담는다
    (기간별로 명확히 구분; 다중종목 entities와 동일한 리스트 관례). 단일/무기간 질문은 기존
    그대로 financial 하나만 담고 periods 키는 없다(회귀 없음).
    """
    resolve_metric_fn = resolve_metric_fn or resolve_metric
    price_snapshot_fn = price_snapshot_fn or get_price_snapshot_kr
    execute_sql_fn = execute_sql_fn or execute_sql
    computed_metric_fn = computed_metric_fn or resolve_computed_metric
    price_history_fn = price_history_fn or get_price_history_kr
    price_return_fn = price_return_fn or price_return_over_months

    # 기간/차트 의도는 재시도 피드백이 아니라 원본 질문 기준으로 판단한다(wants_chart 원칙).
    base_question = _strip_retry_feedback(question)
    period = _parse_period(base_question)
    # 다중분기("25년과 26년 1분기")를 위한 기간 리스트. 0/1개면 기존 단일 period 경로가
    # 그대로 동작하고(회귀 없음), 2개 이상일 때만 아래 재무 조회에서 분기별로 순회한다.
    periods = _parse_periods(base_question)

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
            computed_metric_fn=computed_metric_fn, price_return_fn=price_return_fn,
            today=today,
        )

    stock_code = find_stock_code(conn, question, execute_sql_fn=execute_sql_fn)
    result["stock_code"] = stock_code
    if stock_code is None:
        result["errors"].append("종목을 찾을 수 없습니다: 질문에서 종목명/종목코드를 인식하지 못함")
        return result

    # "최근 N개월/N년 수익률"은 개월수가 12가 아니면 _extract_metric이 못 잡아 상위(총괄)에서
    # free_exec LLM 코드생성으로 폴백 → 8년 전 데이터로 계산해 틀린 답을 내던 실서버 버그.
    # LLM에 맡기지 않고 결정론 함수(price_return_over_months, prices 테이블 최신 거래일 기준
    # N개월 전 종가 대비 수익률)로 계산한다. intent 분류/_extract_metric을 건너뛰고 조기 반환한다.
    recent_months = _parse_recent_return_months(base_question)
    if recent_months is not None:
        result["intent"] = ("financial",)
        asof = _default_screening_asof(conn, "prices", execute_sql_fn)
        financial, err = _call_with_retry(
            lambda: price_return_fn(conn, stock_code, asof, recent_months)
        )
        if err:
            result["errors"].append(f"기간 수익률 계산 실패: {err}")
        else:
            result["financial"] = financial
        if period is None and asof:
            result["data_asof"] = {"price_date": asof}
        return result

    intent = classify_intent(question, llm_fn=llm_fn)
    result["intent"] = intent
    # technical 축이 켜졌으면 질문에서 지표 스펙을 뽑아 채운다(호출부 명시값 우선). 순수 시세면 None.
    active_indicators = indicators
    if "technical" in intent and not active_indicators:
        active_indicators = _extract_indicators(base_question)

    if "financial" in intent:
        metric = _extract_metric(question)
        if metric is None:
            result["errors"].append("재무 지표를 인식하지 못함")
        elif len(periods) >= 2:
            # 다중분기("25년과 26년 1분기"): 각 분기를 개별 조회해 기간별로 구분해 담는다.
            # 최상위 financial은 None으로 두고(다중종목 entities와 동일 관례) periods에 담는다.
            # 문제 C: 이 분기를 계산전용(_COMPUTED_ONLY_FIELDS) 검사보다 먼저 둬, 계산전용
            # 지표(PSR 등)도 다중분기면 여기로 위임된다. 계산전용 라우팅과 6개 지표(문제 A)
            # 폴백은 _resolve_metric_over_periods가 기간별로 모두 처리한다.
            result["periods"] = _resolve_metric_over_periods(
                resolve_metric_fn, conn, stock_code, metric, periods, llm_fn,
                computed_metric_fn=computed_metric_fn, execute_sql_fn=execute_sql_fn,
            )
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
                mismatch_note = _computed_metric_period_mismatch_note(period, financial)
                if mismatch_note:
                    result["errors"].append(mismatch_note)
        else:
            # 재무제표/사전계산 지표: resolve_metric 우선, per/pbr/roe/operating_margin/
            # debt_ratio/market_cap이 과거 시점 metrics 부재로 None이면 계산 폴백(문제 A).
            # period가 None이면 폴백을 시도하지 않아 기존 최신값 경로에 무영향(회귀 없음).
            financial, err = _call_with_retry(
                lambda: _resolve_metric_with_fallback(
                    conn, stock_code, metric, period, llm_fn,
                    resolve_metric_fn, computed_metric_fn, execute_sql_fn,
                )
            )
            if err:
                result["errors"].append(f"재무데이터 조회 실패: {err}")
            else:
                result["financial"] = financial

    if "price" in intent or "technical" in intent:
        # period가 명시됐으면 그 시점 이하 최근 거래일 종가를 가져온다(다중종목 경로와
        # 동일 원칙 — 재무 쪽만 과거 시점 반영하고 가격은 항상 오늘 최신값이라 서로 다른
        # 기준시점이 뒤섞이던 버그 수정). period가 None이면 asof=None → 기존 최신값 경로.
        # technical 이면 active_indicators 가 채워져 지표까지 계산되고, 순수 시세면 None.
        price_asof = _resolve_screening_asof(period, conn, "prices", execute_sql_fn)
        # "6월 18일"처럼 특정일자(연도 없는 월-일 포함)를 지목하면 그 날짜를 asof로 쓴다
        # (연도 미명시 시 오늘이 속한 연도 기본값). period 기반 asof(연간=말일)보다 우선한다 —
        # get_price_snapshot_kr의 date<=asof가 그 시점 이하 가장 가까운 과거 거래일을 고른다.
        target_date = _parse_price_target_date(base_question, today=today)
        if target_date is not None:
            price_asof = target_date
        price, err = _call_with_retry(
            lambda: price_snapshot_fn(
                conn, stock_code, asof=price_asof, indicators=active_indicators
            )
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

    # 기간 미지정 시 실제 사용된 데이터 시점(가격 기준일/재무 기준분기)을 라벨링한다.
    # 이미 resolve된 값(financial의 period/price_date, price 스냅샷의 date)만 재사용한다.
    if period is None:
        data_asof = _collect_data_asof([result.get("financial")], result.get("price") or [])
        if data_asof:
            result["data_asof"] = data_asof

    return result
