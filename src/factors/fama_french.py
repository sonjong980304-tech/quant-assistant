"""파마프렌치(Fama-French) 팩터 온디맨드 조회.

.omc/specs/brainstorming-fama-french-factor-lookup.md 참고.

이 프로젝트의 다른 크롤러(naver_prices.py 등)와 달리 배치 스케줄/저장이 전혀 없다 —
질문이 들어올 때마다 LLM으로 팩터 질문 여부를 판단하고, 사용자 확인(y/n) 후
Ken French Data Library를 즉시 조회해 바로 응답한다(캐시 없음, 매번 새로 조회).
"""
from __future__ import annotations

from typing import Callable

from ..llm import LLM, extract_json

FACTOR_INTENT_SYSTEM = (
    "당신은 사용자 질문이 파마프렌치(Fama-French) 팩터 데이터 질문인지 판단하는 분류기입니다. "
    "오직 JSON 하나만 출력합니다."
)

FACTOR_INTENT_USER = """다음 질문이 파마프렌치 팩터(미국 시장) 데이터를 묻는 질문인지 판단하세요.

지원 데이터셋: 5팩터(Mkt-RF/SMB/HML/RMW/CMA/RF), 모멘텀(Mom). 미국 시장 데이터만 지원합니다.
한국 주식/재무/PER/PBR/ROE 등 일반 질문은 팩터 질문이 아닙니다.

질문: {question}

JSON으로만 답하세요:
- 파마프렌치 팩터 질문이 아니면: {{"is_factor_question": false}}
- 파마프렌치 팩터 질문이면: {{"is_factor_question": true, "dataset": "5factor"|"momentum", "frequency": "daily"|"monthly", "latest_only": true|false, "start": "YYYY-MM-DD"|null, "end": "YYYY-MM-DD"|null}}
"""

_DATASET_NAMES = {
    ("5factor", "monthly"): "F-F_Research_Data_5_Factors_2x3",
    ("5factor", "daily"): "F-F_Research_Data_5_Factors_2x3_daily",
    ("momentum", "monthly"): "F-F_Momentum_Factor",
    ("momentum", "daily"): "F-F_Momentum_Factor_daily",
}

_CONFIRM_PROMPT = "파마프렌치 팩터 질문으로 보입니다. Ken French Data Library에서 조회할까요? (y/n): "


def classify_factor_intent(question: str, llm=None) -> dict | None:
    """질문이 파마프렌치 팩터 질문이면 조회 파라미터 dict를, 아니면 None을 반환한다."""
    llm = llm or LLM
    if not llm.available:
        return None
    result = llm.complete(
        FACTOR_INTENT_USER.format(question=question),
        system=FACTOR_INTENT_SYSTEM,
        role="factor_intent",
    )
    if not result.ok:
        return None
    data = extract_json(result.text)
    if not data.get("is_factor_question"):
        return None
    return data


def _fetch_famafrench_table(dataset_name: str, start=None, end=None):
    import pandas_datareader.data as pdr  # 지연 import — us_prices.py의 yfinance 패턴과 동일

    tables = pdr.DataReader(dataset_name, "famafrench", start, end)
    return tables[0]


def fetch_factor_data(
    dataset: str,
    frequency: str,
    start: str | None = None,
    end: str | None = None,
    latest_only: bool = True,
    fetch_fn: Callable | None = None,
) -> list[dict]:
    """Ken French Data Library에서 팩터 데이터를 조회한다. 캐시 없음 — 매 호출마다 새로 조회."""
    key = (dataset, frequency)
    if key not in _DATASET_NAMES:
        raise ValueError(f"지원하지 않는 데이터셋/주기 조합: {key}")
    fetch_fn = fetch_fn or _fetch_famafrench_table
    df = fetch_fn(_DATASET_NAMES[key], start, end)
    rows = [
        {"period": str(idx), **{col: float(row[col]) for col in df.columns}}
        for idx, row in df.iterrows()
    ]
    return rows[-1:] if latest_only else rows


def format_factor_result(intent: dict, rows: list[dict]) -> str:
    lines = [f"[{intent['dataset']} / {intent['frequency']}]"]
    for row in rows:
        parts = ", ".join(f"{k}={v}" for k, v in row.items() if k != "period")
        lines.append(f"  {row.get('period')}: {parts}")
    return "\n".join(lines)


def handle_query(
    question: str,
    ask_confirm: Callable[[str], str] = input,
    classify: Callable[[str], dict | None] | None = None,
    fetch: Callable | None = None,
) -> str | None:
    """파마프렌치 팩터 질문이면 결과 문자열을, 아니면 None(호출자가 일반 SQL 경로로 진행)을 반환한다."""
    classify = classify or classify_factor_intent
    intent = classify(question)
    if not intent:
        return None
    answer = ask_confirm(_CONFIRM_PROMPT)
    if str(answer).strip().lower() != "y":
        return None
    fetch = fetch or fetch_factor_data
    try:
        rows = fetch(
            intent["dataset"],
            intent["frequency"],
            start=intent.get("start"),
            end=intent.get("end"),
            latest_only=intent.get("latest_only", True),
        )
    except Exception as exc:  # 접속 실패/데이터 없음 — 에러만 출력하고 정상 종료(재시도 없음)
        return f"파마프렌치 팩터 조회 실패: {exc}"
    return format_factor_result(intent, rows)
