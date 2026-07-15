"""상장주식수 이상치 가드 — 절대기준(100억주) + 직전 분기 대비 배수 급증 감지.

서린바이오(038070) 사고 재현: 2025Q3 공시에서 상장주식수가 직전 분기 대비
정확히 1000배로 뛰었으나, 절대기준(1e10)만으로는 91억주가 통과해버렸다.
"""
from __future__ import annotations

from src.db import connect, init_db
from src.ingest.dart import _shares_jump_anomalous, _ingest_shares


def test_shares_jump_anomalous_true_for_1000x_jump():
    assert _shares_jump_anomalous(prev=9_100_676, new=9_100_676_000) is True


def test_shares_jump_anomalous_false_for_normal_quarterly_growth():
    # 서린바이오 실제 정상 분기 증가 패턴(약 20만주 증가)
    assert _shares_jump_anomalous(prev=9_100_676, new=9_300_676) is False


def test_shares_jump_anomalous_false_when_no_previous_value():
    # 첫 데이터 포인트는 비교 기준이 없으므로 통과(절대기준만 적용됨)
    assert _shares_jump_anomalous(prev=None, new=9_100_676_000) is False


def test_shares_jump_anomalous_true_for_extreme_drop():
    # 100분의 1 미만으로 급락도 이상치(단위 오류의 반대 방향)
    assert _shares_jump_anomalous(prev=9_100_676, new=90_000) is True


def test_ingest_shares_skips_insert_when_jump_anomalous(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO financials(stock_code, quarter, disclosed_date, account_key, account_name, amount) "
        "VALUES ('038070', '2025Q2', '2025-08-14', 'shares_outstanding', '상장주식수', 9100676)"
    )
    conn.commit()

    monkeypatch.setattr(
        "src.ingest.dart.fetch_shares",
        lambda api_key, corp, year, reprt: 9_100_676_000,
    )

    n = _ingest_shares(conn, "dummy_key", "038070", "corp123", 2025, "Q3", "2025Q3", "2025-11-14", 0)

    assert n == 0
    row = conn.execute(
        "SELECT amount FROM financials WHERE stock_code='038070' AND quarter='2025Q3' AND account_key='shares_outstanding'"
    ).fetchone()
    assert row is None
