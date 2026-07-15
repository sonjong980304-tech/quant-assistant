"""DART 회사개황이 company.sector를 더 이상 덮어쓰지 않는지 확인.

sector의 유일한 최종 출처는 scripts/backfill_sector_krx.py(KRX 정보데이터시스템)로
통일한다. DART 회사개황(induty_code/KSIC)은 market만 갱신하고 sector는 건드리지 않는다
(기존에 있던 값을 그대로 둔다 — 최초 시딩값이든 KRX 백필값이든 보존).
"""
from __future__ import annotations

import src.ingest.dart as dart_mod
from src.db import connect, init_db
from tests.conftest import seed_kr_companies


def test_update_company_profile_does_not_overwrite_sector(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    seed_kr_companies(conn, ["000001"])  # sector='기타'로 시드됨(conftest 기본값)

    monkeypatch.setattr(
        dart_mod, "fetch_company_profile",
        lambda api_key, corp: {"corp_cls": "Y", "induty_code": "26110"},
    )
    dart_mod._update_company_profile(conn, "fake-key", "000001", "corp123", 0)

    row = conn.execute(
        "SELECT market, sector FROM company WHERE stock_code='000001'"
    ).fetchone()
    assert row["market"] == "KOSPI"  # market은 여전히 DART(corp_cls)에서 갱신됨
    assert row["sector"] == "기타"  # sector는 DART가 더 이상 건드리지 않음(KRX 전용)


def test_upsert_company_preserves_existing_sector_when_new_value_blank(tmp_path):
    """ingest_dart_full()의 company 적재가 빈 market/sector로 기존 값을 지우지 않는지 확인.

    실사용 재현 버그: get_universe("full")이 반환하는 universe는 market/sector를 모르면
    ""로 채우는데(_dart_listed_universe), 기존 INSERT OR REPLACE는 이 빈 값으로 KRX 백필
    sector를 통째로 덮어써 지워버렸다(매일 새벽 2시 backfill_full.py가 이 경로를 탄다).
    """
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    seed_kr_companies(conn, ["005930"])  # market='KOSPI', sector='기타'로 시드
    conn.execute("UPDATE company SET sector='전기·전자' WHERE stock_code='005930'")
    conn.commit()

    # universe 쪽이 market/sector를 모르는 상황("") 그대로 재적재를 시도
    dart_mod._upsert_company(conn, "005930", "삼성전자", "", "")
    conn.commit()

    row = conn.execute(
        "SELECT market, sector FROM company WHERE stock_code='005930'"
    ).fetchone()
    assert row["market"] == "KOSPI"  # 빈 값이 기존 시장 정보를 지우면 안 됨
    assert row["sector"] == "전기·전자"  # 빈 값이 기존 KRX 섹터를 지우면 안 됨


def test_upsert_company_applies_new_value_when_non_blank(tmp_path):
    """새 market/sector 값이 실제로 있으면(빈 문자열이 아니면) 정상적으로 갱신된다."""
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    seed_kr_companies(conn, ["005930"])  # market='KOSPI', sector='기타'로 시드

    dart_mod._upsert_company(conn, "005930", "삼성전자", "KOSPI", "전기·전자")
    conn.commit()

    row = conn.execute(
        "SELECT name, market, sector FROM company WHERE stock_code='005930'"
    ).fetchone()
    assert row["name"] == "삼성전자"
    assert row["sector"] == "전기·전자"


def test_upsert_company_inserts_new_code_with_blank_values_unchanged():
    """처음 보는 종목코드는 지금처럼 빈 market/sector로도 그대로 삽입된다(회귀 없음)."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE company(stock_code TEXT PRIMARY KEY, name TEXT, market TEXT, sector TEXT)"
    )

    dart_mod._upsert_company(conn, "999999", "신규상장사", "", "")
    conn.commit()

    row = conn.execute(
        "SELECT name, market, sector FROM company WHERE stock_code='999999'"
    ).fetchone()
    assert row["name"] == "신규상장사"
    assert row["market"] == ""
    assert row["sector"] == ""
