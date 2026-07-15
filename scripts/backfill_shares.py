"""기존 종목의 분기별 상장주식수 백필 (무인 실행용).

재무(ingest_dart_full)는 resume이 "financials가 있는 종목"을 통째로 스킵하므로,
이미 재무가 적재된 종목의 분기별 상장주식수(shares_outstanding)는 자동으로 수집되지
않는다. 이 스크립트는 그 빠진 (종목, 분기)에 대해서만 stockTotqySttus를 호출해
financials에 account_key='shares_outstanding'으로 채워 넣는다.

대상
----
- financials에서 account_key != 'shares_outstanding'인 (stock_code, quarter, disclosed_date)
  DISTINCT 가운데, 아직 shares_outstanding이 없는 (종목, 분기).
- resume: 이미 shares_outstanding이 있는 (stock_code, quarter)는 스킵 → 중단 후 다음 실행에서
  이어받는다. DART 분기당 1호출이라 한도(2만/일) 안에서 며칠에 걸쳐 진행된다.

한도/중단
--------
- DartQuotaError(일일 한도 초과 020) 발생 시 즉시 중단한다. 이미 받은 (종목,분기)는
  스킵되므로 다음 실행이 그 다음부터 이어받는다.
- 그 외 예외는 (종목,분기) 단위로 격리(로그)해 한 건 실패가 전체를 막지 않게 한다.

마무리
------
- 한 실행이 끝나면(또는 중단되면) 그동안 받은 주식수로 prices.market_cap을 갱신하기 위해
  backfill_marketcap()을 호출한다(idempotent라 다시 돌려도 안전).

실행: python3 scripts/backfill_shares.py   ← DART 일일 한도를 소모한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.db import connect, init_db
from src.ingest.dart import _ingest_shares, get_corp_codes
from src.ingest.robust import DartQuotaError, log_ingest


def _targets(conn) -> list:
    """백필 대상 (stock_code, quarter, disclosed_date) 목록을 반환.

    재무(shares_outstanding 제외)가 있는 (종목, 분기) 중, 아직 shares_outstanding이
    없는 것만 고른다. disclosed_date는 같은 (종목,분기)에서 가장 최근 값을 쓴다.
    quarter 오름차순(과거→최근)으로 정렬해 진행 상황을 예측 가능하게 한다.
    """
    rows = conn.execute(
        """SELECT f.stock_code AS stock_code,
                  f.quarter    AS quarter,
                  MAX(f.disclosed_date) AS disclosed_date
             FROM financials AS f
            WHERE f.account_key != 'shares_outstanding'
              AND NOT EXISTS (
                    SELECT 1 FROM financials AS s
                     WHERE s.stock_code = f.stock_code
                       AND s.quarter    = f.quarter
                       AND s.account_key = 'shares_outstanding'
              )
            GROUP BY f.stock_code, f.quarter
            ORDER BY f.stock_code, f.quarter"""
    ).fetchall()
    return [(r["stock_code"], r["quarter"], r["disclosed_date"]) for r in rows]


def backfill_shares(db_path: str | None = None, sleep: float | None = None) -> dict:
    """빠진 (종목, 분기)의 상장주식수를 분기별로 수집·적재한다.

    각 대상에 대해 year=int(quarter[:4]), qn=int(quarter[5])로 _ingest_shares를 호출해
    그 분기 보고서의 상장주식수를 financials(account_key='shares_outstanding')에 UPSERT한다.
    DartQuotaError 발생 시 즉시 중단하고, 그 외 예외는 격리한다.
    """
    if not CONFIG.has_dart_key:
        raise RuntimeError("DART_API_KEY 없음 — 백필할 수 없습니다.")

    sleep = CONFIG.api_call_delay if sleep is None else sleep
    api_key = CONFIG.dart_api_key
    init_db(db_path)
    conn = connect(db_path)

    ok = 0          # shares_outstanding을 실제로 적재한 (종목,분기) 수
    empty = 0       # 호출은 했으나 주식수가 없던 (종목,분기) 수
    failed = 0      # 격리된 예외 발생 수
    quota_exceeded = False
    try:
        # 종목코드(6자리) → DART corp_code(8자리) 매핑. 실패하면(주로 한도 초과 시 비-zip 응답)
        # 빈 dict가 되어 모든 대상이 no_corp_code로 빠지므로, 그 전에 명시적으로 중단 처리한다.
        try:
            corp_map = get_corp_codes(api_key)
        except DartQuotaError:
            quota_exceeded = True
            log_ingest({"status": "QUOTA_EXCEEDED", "stage": "get_corp_codes"})
            corp_map = {}
        except Exception as exc:  # noqa: BLE001 — corpCode 다운로드 실패 격리(한도 초과 추정)
            log_ingest({"status": "ABORT", "stage": "get_corp_codes", "error": str(exc)})
            corp_map = {}

        targets = _targets(conn) if corp_map else []
        total = len(targets)
        no_corp = 0
        processed = 0

        for code, quarter, disclosed in targets:
            corp = corp_map.get(code)
            if not corp:
                no_corp += 1
                continue
            year = int(quarter[:4])
            qn = int(quarter[5])
            try:
                # _ingest_shares는 내부에서 fetch_shares 호출 후 time.sleep(sleep)을 수행한다.
                # DartQuotaError는 그대로 전파되어 아래 except에서 잡혀 백필을 즉시 중단한다.
                rows = _ingest_shares(conn, api_key, code, corp, year, qn, quarter, disclosed, sleep)
                conn.commit()
                if rows:
                    ok += 1
                else:
                    empty += 1
            except DartQuotaError:
                quota_exceeded = True
                log_ingest({"stock_code": code, "quarter": quarter, "status": "QUOTA_EXCEEDED"})
                break
            except Exception as exc:  # noqa: BLE001 — (종목,분기) 단위 실패 격리
                failed += 1
                log_ingest(
                    {"stock_code": code, "quarter": quarter, "status": "error", "error": str(exc)}
                )
            processed += 1

        report = {
            "total": total,
            "ok": ok,
            "empty": empty,
            "failed": failed,
            "no_corp_code": no_corp,
            "processed": processed,
            "skipped": max(total - processed, 0),
            "quota_exceeded": quota_exceeded,
        }
        log_ingest({"stage": "backfill_shares_done", **report})
        return report
    finally:
        conn.close()


def main() -> None:
    """백필 1회 실행 후, 새로 받은 주식수로 prices.market_cap을 갱신한다."""
    report = backfill_shares()
    print(f"[shares] 완료: {report}")
    # 이번 실행에서 받은 주식수를 시가총액에 반영(idempotent). import 순환을 피하려고 지연 import.
    try:
        from scripts.backfill_marketcap import backfill_marketcap

        cap_report = backfill_marketcap()
        print(f"[shares] 시가총액 갱신: {cap_report}")
    except Exception as exc:  # noqa: BLE001 — 시총 갱신 실패가 주식수 백필 성과를 무효화하지 않도록 격리
        print(f"[shares] 시가총액 갱신 실패(주식수는 정상 적재됨): {exc}")


if __name__ == "__main__":
    main()
