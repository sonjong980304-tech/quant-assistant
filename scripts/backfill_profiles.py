"""회사개황 보완 백필: financials는 있는데 market/sector가 빈 종목만 채운다.

이미 재무(financials)가 적재됐지만 company.market/sector가 ""(빈값)인 종목에 한해
DART 회사개황(company.json)을 종목별 1회 호출해 표준산업분류로 채우는 1회용 스크립트.

동작:
- 체크포인트: market/sector가 이미 채워진 종목은 스킵.
- 종목별 1호출 + time.sleep(0.4)로 한도/속도 보호.
- 회사개황 실패는 try/except로 격리(다른 종목 진행을 막지 않음).
- DartQuotaError(일일 한도 초과) 발생 시 즉시 중단(다음 실행에서 이어서).

주의: DART 일일 한도가 회복된 후 실행할 것. 한도 소진 상태에서 실행하면 곧바로 중단된다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.db import connect
from src.ingest.dart import fetch_company_profile, get_corp_codes
from src.ingest.ksic import market_from_corp_cls, sector_from_induty
from src.ingest.robust import DartQuotaError, log_ingest


def main() -> None:
    if not CONFIG.has_dart_key:
        print("DART_API_KEY 없음 — 중단", flush=True)
        return

    api_key = CONFIG.dart_api_key
    conn = connect()
    try:
        corp_map = get_corp_codes(api_key)
        # financials가 있는데 market 또는 sector가 빈 종목만 대상.
        rows = conn.execute(
            """
            SELECT c.stock_code
              FROM company c
             WHERE (c.market = '' OR c.sector = '' OR c.market IS NULL OR c.sector IS NULL)
               AND EXISTS (SELECT 1 FROM financials f WHERE f.stock_code = c.stock_code)
            """
        ).fetchall()
        targets = [r[0] for r in rows]
        print(f"대상 종목 {len(targets)}개 (재무 있음 + market/sector 빈값)", flush=True)

        ok, skipped, failed = 0, 0, 0
        for code in targets:
            corp = corp_map.get(code)
            if not corp:
                skipped += 1  # corp_code 없음 → 회사개황 조회 불가
                continue
            try:
                profile = fetch_company_profile(api_key, corp)
            except DartQuotaError:
                log_ingest({"stock_code": code, "status": "QUOTA_EXCEEDED"})
                print("일일 한도 초과 — 중단(다음 실행에서 이어서)", flush=True)
                break
            except Exception as exc:  # noqa: BLE001 — 종목 실패 격리
                failed += 1
                log_ingest({"stock_code": code, "status": "profile_error", "error": str(exc)})
                time.sleep(0.4)
                continue
            time.sleep(0.4)
            if not profile:
                failed += 1
                continue
            market = market_from_corp_cls(profile.get("corp_cls"))
            sector = sector_from_induty(profile.get("induty_code"))
            conn.execute(
                "UPDATE company SET market = ?, sector = ? WHERE stock_code = ?",
                (market, sector, code),
            )
            conn.commit()
            ok += 1
            log_ingest({"stock_code": code, "status": "profile_ok", "market": market, "sector": sector})

        print(f"DONE ok={ok} skipped={skipped} failed={failed}", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
