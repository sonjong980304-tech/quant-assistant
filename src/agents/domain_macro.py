"""매크로 도메인 에이전트 — 기존 신호판정 로직을 감싸기(wrap)만 한다 (HA-8).

이 파일이 절대 하지 않는 것: src/ingest/macro_signal.py(classify_spread_regime/
classify_cnn_band/classify_vix_band/regime_to_overall/compute_signal)와
src/ingest/macro_pipeline.py(run_macro_pipeline)의 판정 로직을 재계산하거나 수정하지
않는다. 그 로직은 매일 07:40 launchd가 run_macro_pipeline()으로 이미 실행해 macro_signal
테이블에 새 판정 1행을 append해 둔다(무거운 작업 — 실시간 질의응답에서 다시 돌리면 안 됨).

이 에이전트는 web/app.py의 api_macro_signal() 핸들러(GET /api/macro/signal)와 동일한
SQL 패턴으로 macro_signal의 최신 1행을 조회만 한다. 다만 SQL은 conn.execute()를 직접
부르지 않고 반드시 HA-1 실행기(src/agents/exec_runtime.execute_sql)를 경유한다(다른
데이터 에이전트들의 관례와 동일 — src/agents/data_price_kr.py 참고).

가장 중요한 사실(사용자에게 설명 가능해야 함): 종합신호(overall)는 오직 금리차 레짐
(spread_regime)에서만 결정된다. CNN 공포탐욕지수·VIX는 참고용 밴드 표시일 뿐 종합신호
계산에 전혀 관여하지 않는다 — 두 지표가 아무리 극단값이어도 overall은 바뀌지 않는다
(macro_signal.py 모듈 docstring의 '절대 불변' 규칙 그대로). 반환 dict의 explanation
필드에 이 사실을 항상 담아, 왜 CNN/VIX가 극단이어도 신호가 그대로인지 설명할 수 있게 한다.
"""
from __future__ import annotations

from typing import Callable

from src.agents.exec_runtime import execute_sql
from src.factors.fama_french import classify_factor_intent, fetch_factor_data

# web/app.py의 api_macro_signal()과 동일한 SQL(그 핸들러의 conn.execute() 패턴을 그대로
# 재사용하되, 여기서는 HA-1 실행기(execute_sql)로 실행한다).
_LATEST_SIGNAL_SQL = (
    "SELECT as_of, spread, spread_regime, cnn_value, cnn_band, "
    "vix_value, vix_band, overall, created_at "
    "FROM macro_signal ORDER BY id DESC LIMIT 1"
)

# web/app.py의 api_macro_history()와 동일한 정렬/뒤집기 패턴(그 핸들러의 conn.execute()
# 대신 HA-1 실행기(execute_sql)로 실행). 차트는 spread(장단기금리차) 시계열 하나만 그린다.
_HISTORY_SQL = (
    "SELECT as_of, spread FROM macro_signal ORDER BY as_of DESC, id DESC LIMIT {limit}"
)

_EXPLANATION = (
    "종합신호(overall)는 장단기 금리차 레짐(spread_regime: 정상→GREEN/평탄화→YELLOW/"
    "역전→RED)에서만 결정됩니다. CNN 공포탐욕지수와 VIX는 참고용 밴드 표시일 뿐 종합신호"
    "계산에 전혀 관여하지 않으므로, 두 지표가 극단값이어도 overall은 바뀌지 않습니다."
)


def _macro_payload(question: str, row: dict | None, error: str | None = None) -> dict:
    """macro_signal 최신 행(dict 또는 None)을 응답 dict로 변환.

    web/app.py의 _signal_payload(row)와 같은 정신(이력 없으면 available=False + 전
    필드 None)이되, 이 에이전트 고유의 question/explanation 필드를 더한다.

    error: DB 조회 자체가 실패한 경우(execute_sql이 ok=False 반환)에만 채워진다.
    row=None이라도 error가 없으면 '진짜로 아직 신호가 계산 안 됨'(빈 테이블)이고,
    error가 있으면 '조회가 실패해서 알 수 없음'이다 — 둘을 같은 available=False로
    뭉뚱그리되 error 필드로 구분해 사용자에게 조회 실패임을 숨기지 않는다.
    """
    if row is None:
        return {
            "available": False,
            "question": question,
            "as_of": None,
            "overall": None,
            "spread": {"value": None, "regime": None},
            "cnn": {"value": None, "band": None},
            "vix": {"value": None, "band": None},
            "created_at": None,
            "explanation": _EXPLANATION,
            "error": error,
        }
    return {
        "available": True,
        "question": question,
        "as_of": row["as_of"],
        "overall": row["overall"],
        "spread": {"value": row["spread"], "regime": row["spread_regime"]},
        "cnn": {"value": row["cnn_value"], "band": row["cnn_band"]},
        "vix": {"value": row["vix_value"], "band": row["vix_band"]},
        "created_at": row["created_at"],
        "explanation": _EXPLANATION,
        "error": None,
    }


def _factor_payload(question: str, intent: dict, fetch_factor_fn: Callable) -> dict:
    """파마프렌치 팩터 질문 응답 — macro_signal 관련 필드는 전부 None, factor 필드에 결과를 담는다.

    README 설계(웹: 계층형 멀티에이전트 아키텍처)상 파마프렌치 팩터 조회는 매크로 에이전트
    안에서 처리한다. src/factors/fama_french.py의 handle_query()는 CLI용 input() 기반
    y/n 확인을 전제로 해 웹 요청-응답 흐름에 맞지 않으므로, 여기서는 확인 없이 바로 조회해
    답한다(사용자가 이미 명시적으로 질문했으므로 재확인 불필요 — 다른 도메인 에이전트와
    동일하게 즉시 응답).
    """
    try:
        rows = fetch_factor_fn(
            intent["dataset"], intent["frequency"],
            start=intent.get("start"), end=intent.get("end"),
            latest_only=intent.get("latest_only", True),
        )
    except Exception as exc:  # noqa: BLE001 — 접속 실패/데이터 없음을 에러로만 알리고 정상 종료
        return {
            "available": False,
            "question": question,
            "as_of": None,
            "overall": None,
            "spread": {"value": None, "regime": None},
            "cnn": {"value": None, "band": None},
            "vix": {"value": None, "band": None},
            "created_at": None,
            "factor": {"dataset": intent.get("dataset"), "frequency": intent.get("frequency"), "rows": None},
            "explanation": "파마프렌치 팩터 조회 실패",
            "error": str(exc),
        }
    return {
        "available": True,
        "question": question,
        "as_of": None,
        "overall": None,
        "spread": {"value": None, "regime": None},
        "cnn": {"value": None, "band": None},
        "vix": {"value": None, "band": None},
        "created_at": None,
        "factor": {"dataset": intent["dataset"], "frequency": intent["frequency"], "rows": rows},
        "explanation": (
            f"Ken French Data Library에서 {intent['dataset']}({intent['frequency']}) "
            "팩터 데이터를 조회했습니다(캐시 없음, 매 요청마다 새로 조회)."
        ),
        "error": None,
    }


def answer_macro_question(
    question: str,
    conn,
    execute_sql_fn: Callable | None = None,
    classify_factor_fn: Callable | None = None,
    fetch_factor_fn: Callable | None = None,
) -> dict:
    """매크로 질문에 macro_signal 테이블의 최신 판정으로 답한다.

    질문이 파마프렌치(Fama-French) 팩터 조회면(classify_factor_fn이 판단) macro_signal을
    전혀 거치지 않고 Ken French Data Library에서 직접 조회해 답한다(README 계층형 아키텍처
    설계상 매크로 에이전트의 세 번째 데이터 소스 — 환율·해외지수·파마프렌치 팩터 중 하나).

    팩터 질문이 아니면 기존대로 판정을 새로 계산하지 않는다 — 이미 저장된 최신 1행을
    읽기만 한다(모듈 docstring 참고). conn은 connect_readonly() 읽기전용 연결을 넘겨야 하며
    (다른 데이터 에이전트 관례와 동일), SQL은 execute_sql_fn(기본=execute_sql, HA-1 실행기)을
    경유한다.

    macro_signal이 비어 있으면(파이프라인이 아직 한 번도 안 돈 경우) 예외를 던지지 않고
    available=False인 명시적 상태를 반환한다.

    execute_sql(HA-1 실행기)은 SQL 오류(스키마 오류, 타임아웃, 잠김 등)를 예외로 던지지
    않고 {"ok": False, "error": ...}로 반환한다. 이를 "아직 신호 없음"(빈 테이블)과 같은
    available=False로 뭉뚱그리면 진짜 조회 실패를 조용히 숨기게 되므로, ok=False인 경우는
    별도로 판별해 error 필드에 실패 사유를 담는다(row 유무와 무관하게 조회 자체가 실패했다는
    뜻).

    반환: {"available", "question", "as_of", "overall", "spread", "cnn", "vix",
           "created_at", "explanation", "error"}. error는 조회 실패 시에만 문자열이고,
           그 외(정상 조회 — 신호 있음/빈 테이블 모두 포함)에는 None이다. HA-10(총괄
           에이전트)이 이 함수를 answer_macro_question(question, conn) 형태로 호출한다.
    """
    classify_factor_fn = classify_factor_fn or classify_factor_intent
    intent = classify_factor_fn(question)
    if intent:
        fetch_factor_fn = fetch_factor_fn or fetch_factor_data
        return _factor_payload(question, intent, fetch_factor_fn)

    execute_sql_fn = execute_sql_fn or execute_sql
    result = execute_sql_fn(_LATEST_SIGNAL_SQL, conn)
    if not result.get("ok"):
        return _macro_payload(question, None, error=result.get("error") or "매크로 신호 조회 실패")
    row = result["rows"][0] if result.get("rows") else None
    return _macro_payload(question, row)


def get_macro_history(
    conn,
    days: int = 90,
    execute_sql_fn: Callable | None = None,
) -> list[dict]:
    """macro_signal에서 최근 days일의 (as_of, spread) 시계열을 과거→최신 순으로 반환한다(차트용).

    web/app.py의 api_macro_history()와 동일한 정렬/뒤집기 패턴이되, conn.execute() 대신
    HA-1 실행기(execute_sql)를 경유한다(도메인 에이전트 관례). 매크로 차트는 spread
    (장단기금리차) 시계열 하나만 그리므로 여기서도 그 두 컬럼만 조회한다.

    execute_sql은 파라미터 바인딩이 없어 days를 SQL에 직접 넣으므로 int로 강제한다(주입 방지).
    이력이 없으면 빈 리스트를 반환한다.
    """
    execute_sql_fn = execute_sql_fn or execute_sql
    try:
        limit = int(days)
    except (TypeError, ValueError):
        limit = 90
    if limit <= 0:
        return []
    result = execute_sql_fn(_HISTORY_SQL.format(limit=limit), conn)
    if not result.get("ok"):
        return []
    return list(reversed(result["rows"]))  # 최신순 조회 → 뒤집어 과거→최신(그리기 순서)
