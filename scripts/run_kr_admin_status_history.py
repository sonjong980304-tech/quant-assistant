"""DART 공시목록 기반 관리종목/매매거래정지 '진짜 과거 이력' 전종목 수집 스크립트 (백필).

DART OpenDART list.json 은 회사별 과거 공시목록을 소급 조회할 수 있어(bgn_de=20150101 정상),
kr_trading_status(KRX '오늘 스냅샷'만 → 미래 전용)와 달리 지나간 관리종목 지정/해제·매매거래정지
시작/해제 이력을 과거까지 복원한다(src/ingest/kr_admin_status_history.py 참고).

종목별로 corp_code 매핑 → list.json 페이지네이션 조회 → report_nm 순수 분류 → 구간화 → upsert 한다.
종목별 실패는 격리(스킵+Slack)하고 다음 종목으로 계속한다. UNIQUE upsert 로 재실행 멱등이며, 열린
구간이 나중 실행에서 해제 공시로 닫히면 end_date 만 갱신된다.

⚠️ DART 일일 호출 한도 주의: 종목당 (공시 페이지 수)회 호출한다. 전체 유니버스 백필은 한도를
쉽게 넘길 수 있으므로 --codes 로 소수 종목씩 나눠 돌리거나, 한도 초과(DartQuotaError) 시 다음날
재개하는 방식으로 운영한다. 기본 대상은 kr_trading_status 의 현재 지정/정지 종목(end_date IS NULL)
이다 — 실제 과거 이력이 있을 법한 종목만 우선 조사해 호출을 아낀다.

실행 예: python3 scripts/run_kr_admin_status_history.py --codes 023440,016790
        python3 scripts/run_kr_admin_status_history.py            # kr_trading_status 현재 지정 종목 전체
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import connect  # noqa: E402
from src.ingest.kr_admin_status_history import ingest_admin_status_history  # noqa: E402


def _current_flagged_codes() -> list[str]:
    """kr_trading_status 에서 현재 관리종목/거래정지(end_date IS NULL)인 종목코드(중복 제거)."""
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT DISTINCT stock_code FROM kr_trading_status WHERE end_date IS NULL ORDER BY stock_code"
        ).fetchall()
    finally:
        conn.close()
    return [r["stock_code"] for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", help="쉼표구분 종목코드(6자리). 미지정 시 kr_trading_status 현재 지정 종목")
    args = ap.parse_args()

    codes = args.codes.split(",") if args.codes else _current_flagged_codes()
    result = ingest_admin_status_history(codes=codes)
    print(
        f"[kr_admin_status_history] tickers={result['tickers']}/{result['total_codes']} "
        f"intervals_stored={result['intervals_stored']} review={result['review_count']} "
        f"failed={len(result['failed'])}"
    )
    if result["failed"]:
        print(f"[kr_admin_status_history] 실패 종목: {result['failed']}")


if __name__ == "__main__":
    main()
