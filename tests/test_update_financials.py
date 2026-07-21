"""scripts/update_financials.py(매일 자동 실행되는 증분 재무 갱신)가 dart.py의
_write_reports_year와 동일하게 financials_revision(정정이력, look-ahead 방지용)에도
기록하는지 검증한다.

배경: _write_reports_year(라이브 대량수집이 쓰는 함수)는 INSERT OR REPLACE INTO
financials 직후 항상 _insert_revision을 호출해 정정 전/후 이력을 보존하는데,
update_financials.py는 같은 INSERT 로직을 별도로 복제해두고도 _insert_revision
호출이 빠져 있어, 매일 도는 증분 갱신에서는 이력이 전혀 쌓이지 않았다.
"""
from __future__ import annotations

from datetime import date

import scripts.update_financials as update_financials_mod
from src.db import connect, init_db


def test_update_financials_writes_revision_history(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    monkeypatch.setattr(update_financials_mod.CONFIG, "dart_api_key", "fake-key")
    monkeypatch.setattr(update_financials_mod, "_codes", lambda: ["000001"])
    monkeypatch.setattr(
        update_financials_mod, "get_corp_codes", lambda api_key: {"000001": "corp001"}
    )
    monkeypatch.setattr(update_financials_mod, "fetch_all_accounts", lambda *a, **k: [])
    monkeypatch.setattr(
        update_financials_mod,
        "_parse_all",
        lambda rows: ({"revenue": (123.0, "매출액")}, "20240515123456"),
    )
    monkeypatch.setattr(
        update_financials_mod,
        "_ingest_shares",
        lambda conn, api_key, code, corp, year, qn, quarter, disclosed, sleep: 0,
    )

    report = update_financials_mod.update_financials(
        db_path=db_path, today=date(2024, 10, 1)
    )
    assert report["financials_rows"] > 0  # 실제로 뭔가 적재됐어야 다음 assert가 의미 있음

    conn = connect(db_path)
    fin_rows = conn.execute(
        "SELECT quarter, amount FROM financials "
        "WHERE stock_code='000001' AND account_key='revenue'"
    ).fetchall()
    rev_rows = conn.execute(
        "SELECT quarter, amount FROM financials_revision "
        "WHERE stock_code='000001' AND account_key='revenue'"
    ).fetchall()

    assert len(fin_rows) > 0
    # financials에 새로 적재된 (분기,금액)마다 financials_revision에도 같은 값이 있어야 한다.
    fin_set = {(r["quarter"], r["amount"]) for r in fin_rows}
    rev_set = {(r["quarter"], r["amount"]) for r in rev_rows}
    assert fin_set == rev_set
