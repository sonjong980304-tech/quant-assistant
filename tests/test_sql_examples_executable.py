"""SQL_USER 프롬프트의 모든 few-shot 예시 SQL이 실제로 문법 오류 없이 실행되는지 (TDD).

실제 US 데이터로 검증하다가 발견한 버그: 애플 PER 예시가 us_prices.market_cap을
참조했는데 그 컬럼은 us_company에만 있다(KR prices와 다른 구조 — 실행 시
sqlite3.OperationalError). 프롬프트 문자열은 앞으로도 계속 수정될 것이므로,
예시가 실제로 깨지지 않는지 자동 검증하는 회귀 테스트를 남긴다.
"""
from __future__ import annotations

import re
import sqlite3

from src.db import init_db
from src.legacy.graph import prompts

_EXAMPLE_SQL = re.compile(r"^A: (SELECT.*?;)", re.MULTILINE | re.DOTALL)


def _extract_example_sqls() -> list[str]:
    return _EXAMPLE_SQL.findall(prompts.SQL_USER)


def _seeded_conn(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO company(stock_code, name, market, sector) VALUES "
        "('005930','삼성전자','KOSPI','전기·전자')"
    )
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) VALUES "
        "('005930','2026Q1','2026-05-15','net_income',1000000),"
        "('005930','2026Q1','2026-05-15','revenue',5000000),"
        "('005930','2026Q1','2026-05-15','operating_profit',800000),"
        "('005930','2026Q1','2026-05-15','total_liabilities',2000000),"
        "('005930','2026Q1','2026-05-15','total_equity',3000000),"
        "('005930','2026Q1','2026-05-15','controlling_net_income',900000),"
        "('005930','2026Q1','2026-05-15','controlling_equity',2800000)"
    )
    conn.execute("INSERT INTO prices(stock_code, date, close, market_cap) VALUES ('005930','2026-07-10',70000,300000000)")
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, updated_at) VALUES "
        "('AAPL','Apple Inc.','NASDAQ','Technology',3000000000000,'2026-07-12T00:00:00')"
    )
    conn.execute("INSERT INTO us_prices(stock_code, date, close) VALUES ('AAPL','2026-07-10',210.5)")
    conn.execute(
        "INSERT INTO us_financials(stock_code, as_of_date, period_type, statement_type, item_key, item_value) VALUES "
        "('AAPL','2026-03-31','quarterly','income_stmt','Net Income',20000000000)"
    )
    conn.commit()
    return conn


def test_all_sql_user_examples_execute_without_sql_error(tmp_path):
    conn = _seeded_conn(tmp_path)
    examples = _extract_example_sqls()
    assert len(examples) >= 5  # 프롬프트에 예시가 실제로 있는지도 함께 확인

    failures = []
    for sql in examples:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError as exc:
            failures.append((sql[:60], str(exc)))
    assert not failures, f"실행 실패한 예시 SQL: {failures}"
