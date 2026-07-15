"""src/backtest/data_access.py의 _is_alive() 회귀 테스트 (son-checker 이슈 #23 BUG-1).

_is_alive()의 판정 로직 자체는 이번에 바꾸지 않았다(모르면 살아있다고 보는 폴백은
의도된 정책). 다만 ingest_delisting()이 빈 문자열 대신 NULL을 쓰도록 바뀌었으므로,
NULL과 빈 문자열 둘 다 기존과 동일하게 "모름 → 살아있음"으로 처리되는지, 그리고
실제 delisting_date가 있을 때는 asof와 정확히 비교해 판정하는지 회귀로 고정한다.
이 파일이 생기기 전에는 _is_alive()를 직접 겨냥한 테스트가 없었다(metrics_at 경유뿐).
"""
from __future__ import annotations

import sqlite3

from src.backtest.data_access import _is_alive
from src.db import init_db


def _conn(tmp_path) -> sqlite3.Connection:
    db = tmp_path / "alive.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def test_is_alive_true_before_delisting_date(tmp_path):
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO delisting(stock_code, name, delisting_date) VALUES(?,?,?)",
        ("000030", "우리은행", "2019-02-12"),
    )
    conn.commit()
    assert _is_alive(conn, "000030", "2018-01-01") is True


def test_is_alive_false_on_or_after_delisting_date(tmp_path):
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO delisting(stock_code, name, delisting_date) VALUES(?,?,?)",
        ("000030", "우리은행", "2019-02-12"),
    )
    conn.commit()
    assert _is_alive(conn, "000030", "2020-01-01") is False


def test_is_alive_true_when_no_delisting_row(tmp_path):
    conn = _conn(tmp_path)
    assert _is_alive(conn, "005930", "2026-01-01") is True


def test_is_alive_true_when_delisting_date_is_null(tmp_path):
    """정확한 상폐일을 모르는(NULL) 경우 — ingest_delisting()이 이제 NULL을 쓰므로,
    그런 행이 있어도 기존과 동일하게 '모름 → 살아있음'으로 처리돼야 한다."""
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO delisting(stock_code, name, delisting_date) VALUES(?,?,?)",
        ("999999", "신규상폐추정", None),
    )
    conn.commit()
    assert _is_alive(conn, "999999", "2026-01-01") is True
