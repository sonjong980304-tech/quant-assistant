"""실제 시스템(run_hierarchical) 연동 어댑터 (US-8).

지금까지의 5개 도메인 factcheck 모듈(financials/price/chart/screening/backtest)은 전부
llm_fn/vision_fn을 단위테스트에서 모킹했다. 이 모듈은 그 자리에 **진짜로 시스템을 호출하는
어댑터**를 제공한다 — src/agents/graph.run_hierarchical 을 호출해 그 결과의 구조화된
domain_results 에서 필요한 값을 직접 꺼낸다(자연어 답변 파싱보다 구조화 값이 정확·안정적).

.omc/specs/brainstorming-factcheck-eval.md 참고. 설계 원칙:
- 자연어 파싱을 피하고 domain_results(구조화 dict)의 값을 직접 사용한다.
- 시스템 호출이 실패하거나 기대 값이 없으면 예외를 밖으로 던지지 않는다 — 숫자 도메인은
  float("nan")(비교에서 fail로 처리되되 within_pct_tolerance/exact_match가 예외로 죽지
  않는 값), 리스트 도메인은 빈 리스트, 차트는 chart_base64=None 으로 정규화한다. 상위
  스크립트가 이를 fail/측정불가로 안전하게 집계한다.
- run_fn 은 테스트 주입용이다(기본은 실제 run_hierarchical). 단위테스트는 run_fn을
  가짜로 주입해 어댑터의 '추출/정규화' 로직만 결정론적으로 검증한다.

vision_fn 은 src/llm.py의 LLMClient.complete_vision(role="sql", gpt-5.4-mini 재사용)을 감싸
판정 텍스트를 bool로 정규화한다(AC3).
"""
from __future__ import annotations

import math
from typing import Any, Callable

from src.agents.graph import run_hierarchical

# 재무 지표(account_key/metrics 필드) → 시스템에 물어볼 한국어 표현. run_hierarchical 의
# kr 도메인이 이 표현을 resolve_metric 별칭으로 인식한다(src/agents/domain_kr.py).
_METRIC_KO: dict[str, str] = {
    "operating_profit": "영업이익",
    "net_income": "당기순이익",
    "revenue": "매출액",
    "total_assets": "자산총계",
    "total_liabilities": "부채총계",
    "total_equity": "자본총계",
    "per": "PER",
    "pbr": "PBR",
    "roe": "ROE",
}

_VISION_PROMPT = (
    "아래 차트 이미지가 다음 질문의 데이터를 시각적으로 올바르게(축·막대/점의 크기·순서가"
    " 데이터와 일치하게) 표현했는지 판정하세요. 이미지가 비어 있거나 깨졌거나 데이터를"
    " 잘못 표현했으면 부적절입니다.\n"
    "질문: {question}\n"
    "'적절'(PASS) 또는 '부적절'(FAIL) 중 하나로만 시작해서 답하세요."
)


# --------------------------------------------------------------------------
# 추출/정규화 순수 헬퍼 (단위테스트 대상)
# --------------------------------------------------------------------------
def _kr_domain(result: dict) -> dict | None:
    """run_hierarchical 결과에서 kr 도메인 원본 결과를 꺼낸다(없으면 None)."""
    if not isinstance(result, dict):
        return None
    dr = result.get("domain_results")
    if not isinstance(dr, dict):
        return None
    kr = dr.get("kr")
    return kr if isinstance(kr, dict) else None


def _extract_financial_value(result: dict) -> float | None:
    """domain_results.kr.financial.value 를 float로 뽑는다. 없으면 None."""
    kr = _kr_domain(result)
    if kr is None:
        return None
    fin = kr.get("financial")
    if not isinstance(fin, dict):
        return None
    val = fin.get("value")
    if val is None:
        # 최근수익률 경로(return_pct)만 별도로 담기는 경우도 방어적으로 지원.
        val = fin.get("return_pct")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _extract_price_close(result: dict) -> float | None:
    """domain_results.kr.price[0].close 를 float로 뽑는다. 없으면 None."""
    kr = _kr_domain(result)
    if kr is None:
        return None
    price = kr.get("price")
    if not isinstance(price, list) or not price:
        return None
    first = price[0]
    if not isinstance(first, dict):
        return None
    close = first.get("close")
    if close is None:
        return None
    try:
        return float(close)
    except (TypeError, ValueError):
        return None


def _extract_screening_codes(result: dict, top_n: int | None = None) -> list[str]:
    """domain_results.kr.result(랭킹 리스트)에서 stock_code를 순서대로 뽑는다.

    result가 리스트가 아니면(both_extremes/sector_neutral_compare 등 중첩 dict) 빈 리스트를
    반환한다 — 이 경우 상위 비교(exact_match)에서 불일치(fail)로 안전하게 처리된다.
    top_n이 주어지면 앞에서 그만큼만 자른다.
    """
    kr = _kr_domain(result)
    if kr is None:
        return []
    rows = kr.get("result")
    if not isinstance(rows, list):
        return []
    codes = [r.get("stock_code") for r in rows if isinstance(r, dict) and r.get("stock_code")]
    if top_n is not None:
        codes = codes[:top_n]
    return codes


def _extract_chart(result: dict) -> dict:
    """차트 결과를 {"chart_base64", "actual_data"}로 정규화한다.

    chart_base64는 run_hierarchical 최상위 결과 키(supervisor가 붙임). actual_data는 차트가
    근거로 삼은 도메인 랭킹 데이터(stock_code 리스트) — 스크리닝 백트 차트 기준. 리스트가
    아니면 원본 payload를 그대로 둔다(측정불가/불일치로 처리됨).
    """
    chart_base64 = result.get("chart_base64") if isinstance(result, dict) else None
    actual_data: Any = _extract_screening_codes(result)
    if not actual_data:
        # 스크리닝 랭킹이 아니면(집계형 차트 등) 랭킹 코드가 비므로 원본 payload를 남긴다.
        kr = _kr_domain(result)
        if kr is not None and kr.get("result") is not None:
            actual_data = kr.get("result")
    return {"chart_base64": chart_base64, "actual_data": actual_data}


def _parse_vision_verdict(text: str | None) -> bool | None:
    """vision 응답 텍스트에서 pass/fail을 bool로 정규화한다. 불명확하면 None.

    '부적절'이 '적절'을 부분문자열로 포함하므로 부정 마커를 먼저 검사한다(supervisor의
    _INVALID_MARKERS 관례와 동일).
    """
    t = (text or "").strip().lower()
    if not t:
        return None
    negatives = ("부적절", "부적합", "잘못", "틀림", "fail", "no", "false")
    positives = ("적절", "적합", "올바", "정확", "맞음", "pass", "yes", "true")
    if any(m in t for m in negatives):
        return False
    if any(m in t for m in positives):
        return True
    return None


# --------------------------------------------------------------------------
# 어댑터 팩토리 (스크립트에서 각 도메인 모듈에 넘겨줄 실제 llm_fn/vision_fn)
# --------------------------------------------------------------------------
def _financial_question(item: dict) -> str:
    name = item.get("name", item["stock_code"])
    metric_ko = _METRIC_KO.get(item.get("metric", ""), item.get("metric", ""))
    return f"{name}({item['stock_code']})의 {metric_ko} 알려줘"


def make_financial_llm_fn(conn, llm_fn, run_fn: Callable = run_hierarchical) -> Callable[[dict], float]:
    """run_financials_check(items, llm_fn, dart_api_key)의 llm_fn(item)->숫자 어댑터.

    시스템이 값을 못 내면 float("nan")을 돌려준다 — within_pct_tolerance가 None으로
    죽지 않고 fail(nan 비교는 항상 False)로 처리되게 하는 안전값이다.
    """
    def _fn(item: dict) -> float:
        try:
            result = run_fn(_financial_question(item), conn, llm_fn=llm_fn)
        except Exception:  # noqa: BLE001 — 시스템 호출 실패는 nan(측정 실패)로 흡수
            return float("nan")
        val = _extract_financial_value(result)
        return val if val is not None else float("nan")

    return _fn


def make_price_llm_fn(conn, llm_fn, run_fn: Callable = run_hierarchical) -> Callable[[str], float]:
    """run_price_check(items, llm_fn)의 llm_fn(question)->종가 어댑터. 실패 시 nan."""
    def _fn(question: str) -> float:
        try:
            result = run_fn(question, conn, llm_fn=llm_fn)
        except Exception:  # noqa: BLE001
            return float("nan")
        close = _extract_price_close(result)
        return close if close is not None else float("nan")

    return _fn


def make_screening_llm_fn(conn, llm_fn, run_fn: Callable = run_hierarchical) -> Callable[[dict], list]:
    """run_screening_check(items, llm_fn, conn)의 llm_fn(item)->[stock_code] 어댑터.

    실패/구조 불일치 시 빈 리스트 → 재계산값과 exact_match 불일치(fail)로 안전 처리.
    """
    def _fn(item: dict) -> list:
        try:
            result = run_fn(item["question"], conn, llm_fn=llm_fn)
        except Exception:  # noqa: BLE001
            return []
        return _extract_screening_codes(result, top_n=item.get("top_n"))

    return _fn


def make_chart_llm_fn(conn, llm_fn, run_fn: Callable = run_hierarchical) -> Callable[[str], dict]:
    """run_chart_check(items, llm_fn, vision_fn)의 llm_fn(question)->{chart_base64,actual_data}."""
    def _fn(question: str) -> dict:
        try:
            result = run_fn(question, conn, llm_fn=llm_fn)
        except Exception:  # noqa: BLE001
            return {"chart_base64": None, "actual_data": []}
        return _extract_chart(result)

    return _fn


def make_vision_fn(vision_client=None, model: str | None = None) -> Callable[[str, str], bool]:
    """run_chart_check의 vision_fn(question, chart_base64)->bool 어댑터.

    src/llm.py의 LLMClient.complete_vision(role="sql")을 감싸 판정 텍스트를 bool로 정규화한다.
    이미지가 없거나(None) 판정을 파싱할 수 없으면 예외를 던져 chart.py가 pass=None(측정불가)로
    기록하게 한다(fail로 단정하지 않는다).
    """
    if vision_client is None:
        from src.llm import LLMClient

        vision_client = LLMClient(model=model) if model else LLMClient()

    def _fn(question: str, chart_base64: str | None) -> bool:
        if not chart_base64:
            raise ValueError("차트 이미지(base64)가 없어 vision 판정 불가")
        res = vision_client.complete_vision(
            _VISION_PROMPT.format(question=question), chart_base64, role="sql"
        )
        text = getattr(res, "text", None)
        if getattr(res, "ok", True) is False or not (text or "").strip():
            raise ValueError(f"vision 응답 없음/실패: {getattr(res, 'error', None)}")
        verdict = _parse_vision_verdict(text)
        if verdict is None:
            raise ValueError(f"vision 판정 파싱 실패: {text[:120]}")
        return verdict

    return _fn


def build_system_llm_fn(role: str = "sql"):
    """실제 시스템 텍스트 LLM(Callable[[str],str]). 미가용(키 없음) 시 None.

    scripts/eval_hierarchical_goldset._build_llm_fn 과 동일 규약 — run_hierarchical 에
    주입하는 SQL 생성/판정용 llm_fn 이다.
    """
    from src.llm import LLMClient

    client = LLMClient()
    if not client.available:
        return None
    return lambda prompt: (client.complete(prompt, role=role).text or "")


def build_vision_client():
    """vision 호출용 LLMClient(gpt-5.4-mini). 미가용 시 None."""
    from src.llm import LLMClient

    client = LLMClient()
    return client if client.available else None
