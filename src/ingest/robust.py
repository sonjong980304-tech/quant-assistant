"""수집 안정성 헬퍼: 지수 백오프 재시도 + 종목별 로깅.

- request_with_retry: requests 실패/비200/JSON status!='000' 시 CONFIG.max_retries회,
  1·2·4초(지수 백오프) 대기 후 재시도. fetch_all_accounts·get_corp_codes 등에서 사용.
- log_ingest: 종목별 성공/실패를 data/ingest_log.jsonl에 append (+ print).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from ..config import CONFIG

ROOT = Path(__file__).resolve().parent.parent.parent
INGEST_LOG_PATH = ROOT / "data" / "ingest_log.jsonl"


class DartQuotaError(RuntimeError):
    """DART 일일 사용한도 초과(status=020). 재시도해도 회복 불가하므로 즉시 전파해 백필을 멈춘다."""


def request_with_retry(
    url: str,
    params: Optional[dict] = None,
    timeout: int = 30,
    max_retries: Optional[int] = None,
    expect_json: bool = True,
    ok_status: str = "000",
) -> Optional[requests.Response]:
    """지수 백오프 재시도로 GET 요청.

    실패 조건: 예외 발생 / 비200 응답 / (expect_json이면) JSON status != ok_status.
    재시도: max_retries회(기본 CONFIG.max_retries), 대기 1·2·4…초 (2^attempt).
    반환: 성공 시 Response, 모두 실패하면 None.
    """
    retries = CONFIG.max_retries if max_retries is None else max_retries
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code != 200:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            if expect_json:
                d = r.json()
                status = d.get("status")
                # status 020 = 일일 사용한도 초과: 재시도/계속 무의미 → 즉시 전파해 백필 중단.
                if status == "020":
                    raise DartQuotaError("DART 일일 사용한도 초과(020)")
                # status 013 = 조회 데이터 없음(정상 빈 결과). 재시도 불필요.
                if status not in (ok_status, "013"):
                    raise ValueError(f"DART status={status}")
            return r
        except DartQuotaError:
            raise  # 한도 초과는 재시도하지 않고 즉시 위로 전파
        except Exception as exc:  # noqa: BLE001 — 네트워크/파싱 전 케이스 격리
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1, 2, 4, ...
    if last_exc is not None:
        print(f"[robust] 요청 최종 실패({retries}회): {url} — {last_exc}")
    return None


def call_with_retry(
    fn: Callable[[], Any],
    max_retries: Optional[int] = None,
    label: str = "",
) -> Any:
    """임의 함수를 지수 백오프로 재시도. 예외/빈 결과(None, 빈 list) 시 재시도.

    반환: 첫 성공 결과. 모두 실패하면 마지막 결과(또는 None).
    """
    retries = CONFIG.max_retries if max_retries is None else max_retries
    result: Any = None
    for attempt in range(retries):
        try:
            result = fn()
            if result:  # None/빈 list/빈 dict가 아니면 성공
                return result
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"[robust] {label or 'call'} 최종 실패({retries}회): {exc}")
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return result


def log_ingest(record: dict, to_file: bool = True) -> None:
    """종목별 수집 결과를 stdout + data/ingest_log.jsonl(append)에 기록."""
    line = json.dumps(record, ensure_ascii=False)
    print(f"[ingest] {line}")
    if not to_file:
        return
    try:
        INGEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with INGEST_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:  # noqa: BLE001 — 로깅 실패가 수집을 막지 않도록
        print(f"[ingest] 로그 파일 기록 실패: {exc}")
