"""재무제표 사실확인(factcheck) — 시스템 응답값을 DART 원문 재조회값과 대조 (US-3).

.omc/specs/brainstorming-factcheck-eval.md AC1 참고: 시가총액 상위 종목의 재무 지표
질의 응답값이 OpenDART API 원문 재조회값과 ±1% 이내인지 자동 판정한다.

DART 재조회는 ``src/ingest/dart.py``의 기존 함수(get_corp_codes, fetch_all_accounts,
_parse_all, REPRT_CODE)를 그대로 재사용한다 — 새 API 스펙을 추측하지 않는다.
"""
from __future__ import annotations

from datetime import date

from src.eval.factcheck.tolerance import within_pct_tolerance
from src.ingest.dart import REPRT_CODE, _parse_all, fetch_all_accounts, get_corp_codes
from src.version import estimate_available_quarter

_PCT_TOLERANCE = 0.01
_UNMEASURABLE_NOTE = "측정불가"


def _fetch_dart_original(dart_api_key: str, stock_code: str, metric: str) -> float | None:
    """DART API를 그 자리에서 재조회해 stock_code의 최신 공시분기 metric 원문값을 반환한다.

    ingest_dart의 연결(CFS) 우선 → 별도(OFS) 폴백 패턴을 그대로 따른다.
    corp_code 미상, 원문에 metric 없음, 네트워크/한도초과 등 어떤 이유로든 값을
    확정할 수 없으면 None을 반환한다(예외는 호출부인 run_financials_check가 처리).
    """
    corp_map = get_corp_codes(dart_api_key)
    corp = corp_map.get(stock_code)
    if not corp:
        return None

    latest_q = estimate_available_quarter(date.today())
    year = int(latest_q[:4])
    reprt = REPRT_CODE[int(latest_q[5])]

    rows = fetch_all_accounts(dart_api_key, corp, year, reprt, "CFS")
    data, _rcept = _parse_all(rows)
    if metric not in data:
        rows = fetch_all_accounts(dart_api_key, corp, year, reprt, "OFS")
        data, _rcept = _parse_all(rows)

    if metric not in data:
        return None
    return data[metric][0]


def run_financials_check(items: list[dict], llm_fn, dart_api_key: str | None) -> list[dict]:
    """각 item(stock_code/name/metric)에 대해 시스템 응답값과 DART 원문을 대조한다.

    반환: [{"stock_code":..., "expected":dart원문, "actual":시스템응답, "pass":bool|None,
            "note":str}, ...]

    DART 재조회가 실패(예외, None, 한도초과 등)하면 그 항목은 pass=None,
    note="측정불가"로 기록하고 예외를 상위로 전파하지 않는다(전체 함수가 죽지 않음).
    """
    results: list[dict] = []
    for item in items:
        stock_code = item["stock_code"]
        metric = item["metric"]
        actual = llm_fn(item)

        expected: float | None = None
        if dart_api_key:
            try:
                expected = _fetch_dart_original(dart_api_key, stock_code, metric)
            except Exception:  # noqa: BLE001 — DART 재조회 실패(한도초과 등) 격리
                expected = None

        if expected is None:
            results.append({
                "stock_code": stock_code,
                "expected": None,
                "actual": actual,
                "pass": None,
                "note": _UNMEASURABLE_NOTE,
            })
            continue

        results.append({
            "stock_code": stock_code,
            "expected": expected,
            "actual": actual,
            "pass": within_pct_tolerance(expected, actual, _PCT_TOLERANCE),
            "note": "",
        })
    return results
