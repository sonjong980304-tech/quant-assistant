"""top_market_cap_stocks() 시가총액 상위 N종목 동적 조회 테스트 (US-2).

.omc/specs/brainstorming-factcheck-eval.md 표본 선정 방식(Round 11): 재무제표·주가
표본은 DB에서 동적으로 상위 N종목을 조회한다 — 하드코딩 금지(순위가 바뀌어도 재현
가능해야 함). seeded_db 픽스처(tests/conftest.py)가 이미 company+prices에 종목별로
다른 market_cap(1e12~12e12, stock_code 순)을 시드해두므로 그대로 재사용해 실제
쿼리 결과가 시가총액 내림차순으로 정렬되는지 검증한다.
"""
from __future__ import annotations

from src.db import connect
from src.eval.factcheck.sample import top_market_cap_stocks


def test_top_market_cap_stocks_returns_descending_top_n(seeded_db):
    conn = connect(seeded_db)
    try:
        result = top_market_cap_stocks(conn, 3)
    finally:
        conn.close()

    # seeded_db: market_cap = (순번+1)*1e12 → 000012=12e12, 000011=11e12, 000010=10e12
    assert [r["stock_code"] for r in result] == ["000012", "000011", "000010"]
    assert [r["market_cap"] for r in result] == [12e12, 11e12, 10e12]
    assert [r["name"] for r in result] == ["카타전지", "아자게임", "바사전자"]


def test_top_market_cap_stocks_result_shape(seeded_db):
    conn = connect(seeded_db)
    try:
        result = top_market_cap_stocks(conn, 1)
    finally:
        conn.close()

    assert len(result) == 1
    assert set(result[0].keys()) == {"stock_code", "name", "market_cap"}
