"""스크리닝(PER/PBR/ROE 등) top-N 계산값 재검증(factcheck) — 의도 파싱과 무관 (US-6).

.omc/specs/brainstorming-factcheck-eval.md AC4 / Round4 참고: 기존 goldset.py가 "질문
의도를 올바른 SQL로 파싱했는가"를 보는 것과 달리, 이 컴포넌트는 top-N '계산값' 자체를
DB 원본 재계산과 대조해 독립적으로 재검증한다(둘은 서로 다른 실패 지점을 잡는 별개
컴포넌트 — Round4에서 중복이 아니라고 확정됨). 완전일치(오차 허용 없음, 결정론적
SQL 계산이므로).

재계산은 src.backtest.data_access.metrics_at(기존 PER/PBR/ROE 등 지표 계산 로직,
백테스트 크로스섹션 조회에 이미 쓰이는 SoT)을 그대로 재사용한다 — 값 계산 자체를
새로 발명하지 않고, 그 위에서 top_n개를 뽑는 정렬/선택만 이 모듈이 독립적으로
수행한다(스크리닝 실행 경로인 select_stocks 등은 타지 않는다 — 그래야 "계산값 자체"의
독립 재검증이 된다).
"""
from __future__ import annotations

from typing import Callable

from ...backtest.data_access import metrics_at
from .tolerance import exact_match


def _recompute_top_n(conn, item: dict) -> list[str]:
    """DB 원본에서 item의 metric 기준 top-N 종목코드를 독립적으로 재계산한다.

    item: {"metric":..., "top_n":..., "ascending": bool(선택, 기본 True=낮은 순이 우수),
           "asof": str(선택, 기본 prices 최신 거래일)}
    metric 값이 없는(None) 종목은 정렬 대상에서 제외한다.
    """
    metric = item["metric"]
    top_n = item["top_n"]
    ascending = item.get("ascending", True)

    asof = item.get("asof")
    if asof is None:
        row = conn.execute("SELECT MAX(date) AS d FROM prices").fetchone()
        asof = row["d"] if row else None

    rows = metrics_at(conn, asof)
    valid = [r for r in rows if r.get(metric) is not None]
    valid.sort(key=lambda r: r[metric], reverse=not ascending)
    return [r["stock_code"] for r in valid[:top_n]]


def run_screening_check(items: list[dict], llm_fn: Callable[[dict], list], conn) -> list[dict]:
    """각 item(question/top_n/metric)에 대해 시스템 스크리닝 결과와 DB 재계산을 대조한다.

    llm_fn(item): 시스템이 답한 top-N 종목코드 리스트(순서 포함)를 반환하는 함수.
    conn: _recompute_top_n이 원본 재계산에 쓰는 DB 커넥션(자매 모듈 financials.py의
    dart_api_key와 동일하게, 원본 재조회에 필요한 외부 리소스로서 인자로 받는다).

    반환: [{"question":..., "pass":bool, "note":str}] — note는 일치 시 "", 불일치 시
    시스템/재계산 리스트를 함께 남겨 리포트에서 바로 원인을 볼 수 있게 한다.
    """
    results: list[dict] = []
    for item in items:
        system_result = list(llm_fn(item))
        recomputed = _recompute_top_n(conn, item)
        ok = exact_match(recomputed, system_result)
        results.append({
            "question": item["question"],
            "pass": ok,
            "note": "" if ok else f"시스템={system_result}, 재계산={recomputed}",
        })
    return results
