"""FnGuide 백필(2026-07-16, background task bx29xkk7n) 중 ConnectionResetError로
실패한 87개 종목만 재시도하는 1회성 스크립트. ingest_fnguide_metrics()와 동일한
수집/저장 로직을 재사용하되, company 테이블 전체 대신 실패 목록만 대상으로 한다.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, ".")

from src.db import connect, init_db  # noqa: E402
from src.ingest.fnguide_metrics import (  # noqa: E402
    extract_snapshot_json,
    fetch_snapshot_html,
    parse_financial_highlights,
    parse_target_chart,
)
from src.ingest.http_fetch import ThrottledFetcher  # noqa: E402
from src.ingest.notify import send_slack_alert  # noqa: E402
from src.ingest.robust import log_ingest  # noqa: E402

FAILED_CODES = [
    '038390', '017800', '054450', '033130', '214180', '017480', '214430', '003800',
    '002420', '163730', '048430', '004360', '108860', '032540', '208370', '143540',
    '050760', '001380', '092230', '274090', '241590', '120030', '058730', '259960',
    '417500', '950140', '013870', '092730', '378340', '017860', '469880', '188260',
    '088280', '005950', '067990', '263020', '069140', '330730', '242040', '093640',
    '009770', '348210', '340440', '034310', '090430', '389030', '405920', '020150',
    '002990', '089890', '108230', '417790', '076610', '471050', '363260', '023810',
    '097520', '064820', '024800', '105760', '035200', '043910', '440290', '241770',
    '272110', '472230', '452430', '035600', '417200', '417840', '356860', '090460',
    '009970', '034730', '429270', '467930', '036010', '336680', '456040', '065450',
    '000020', '001470', '060150', '058860', '094940', '048770', '038870',
]


def main() -> None:
    db_path = "data/market.db"
    init_db(db_path)
    conn = connect(db_path)
    fetcher = ThrottledFetcher()
    collected_at = datetime.now(timezone.utc).isoformat()

    succeeded = 0
    failed: list[str] = []
    metric_rows = 0
    start = time.monotonic()
    for code in FAILED_CODES:
        try:
            html = fetch_snapshot_html(code, fetcher=fetcher)
            rows = parse_financial_highlights(
                extract_snapshot_json(html, "snpFinancial")
            ) + parse_target_chart(extract_snapshot_json(html, "snpTargetChart"))
            if not rows:
                raise ValueError("빈 응답(필드 누락)")
        except Exception as exc:  # noqa: BLE001 — 종목별 실패를 격리해 다음 종목으로 계속
            failed.append(code)
            log_ingest({"source": "fnguide_metrics", "stock_code": code, "status": "fail", "error": str(exc)})
            send_slack_alert(f"[fnguide_metrics retry] {code} 재수집 실패: {exc}")
            continue
        for row in rows:
            conn.execute(
                "INSERT OR REPLACE INTO fnguide_metrics"
                "(stock_code, as_of_date, metric_key, metric_value, source, collected_at) "
                "VALUES (?,?,?,?,?,?)",
                (code, row["as_of_date"], row["metric_key"], row["metric_value"], "fnguide", collected_at),
            )
            metric_rows += 1
        succeeded += 1
    conn.commit()
    conn.close()

    elapsed = time.monotonic() - start
    print(f"재시도 완료: {succeeded}/{len(FAILED_CODES)} 성공, "
          f"실패 {len(failed)}개, 총 {metric_rows}행, {elapsed:.0f}초 소요")
    if failed:
        print("여전히 실패한 종목:", failed)


if __name__ == "__main__":
    main()
