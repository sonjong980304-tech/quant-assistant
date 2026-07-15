"""미국 주가 데이터 에이전트(src/agents/data_price_us.py) 단위/통합 테스트 (TDD, HA-4).

한국판(src/agents/data_price_kr.py, HA-3)과 계약을 맞춘다 — 이후 HA-7(미국주식 도메인
에이전트)이 HA-6(한국주식 도메인 에이전트)과 대칭으로 호출할 수 있어야 하기 때문이다.

검증 대상:
- us_prices/us_company 테이블(이미 yfinance 단일 출처, src/backtest/data_access_us.py)을
  그대로 조회하며 신규 병합 로직을 두지 않는다(정적 가드로 확인) — AC10.
- SQL은 반드시 execute_sql(HA-1 실행기, src/agents/exec_runtime.py)을 경유한다
  (conn.execute() 직접 호출 금지). execute_sql은 파라미터 바인딩이 없어 티커를 SQL
  문자열에 직접 끼워 넣으므로, 형식이 아닌 입력(SQL 인젝션 시도 포함)은 조용히 걸러진다.
- 반환은 metrics_at_us/get_cross_section과 동일한 계약(stock_code 필드를 가진
  list[dict])이라 HA-2 재무데이터와 stock_code 키로 merge 가능하고,
  compute_technical_indicator의 rows 인자로 그대로 넣을 수 있다.
- 기술지표는 compute_technical_indicator(이미 완성된 TA-Lib 프리미티브)를
  us_price_history_batch(신규 history_fn, us_prices 테이블 배치조회)와 함께 재사용한다
  — 새 지표 계산 로직은 만들지 않는다.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.agents.data_price_us import (
    get_latest_price_us,
    get_price_history_us,
    get_price_snapshot_us,
)
from src.db import connect, connect_readonly, init_db


def _seed_prices(tmp_path, rows: list[tuple]) -> str:
    """rows: (stock_code, date, close, open, high, low, volume).

    회사정보(name/exchange/sector/market_cap)는 AAPL/MSFT 한 벌만 고정 시드한다
    (us_company는 us_prices와 별도 테이블 — data_access_us.py와 동일 구조).
    """
    db = tmp_path / "prices_us.db"
    init_db(str(db))
    conn = connect(str(db))
    try:
        for code, name, exchange, sector, market_cap in [
            ("AAPL", "Apple", "NASDAQ", "Technology", 3.0e12),
            ("MSFT", "Microsoft", "NASDAQ", "Technology", 2.5e12),
        ]:
            conn.execute(
                "INSERT INTO us_company(stock_code, name, exchange, sector, market_cap, updated_at) "
                "VALUES (?,?,?,?,?,?)",
                (code, name, exchange, sector, market_cap, "2026-07-01"),
            )
        for code, date, close, open_, high, low, volume in rows:
            conn.execute(
                "INSERT INTO us_prices(stock_code, date, open, high, low, close, volume) "
                "VALUES (?,?,?,?,?,?,?)",
                (code, date, open_, high, low, close, volume),
            )
        conn.commit()
    finally:
        conn.close()
    return str(db)


# ── 정적 가드: conn.execute() 직접 호출 금지 (AC10 신규 병합 로직 없음은 코드리뷰로 확인) ──
def test_module_does_not_call_conn_execute_directly():
    """SQL 실행은 반드시 execute_sql(HA-1) 경유 — conn.execute()를 직접 호출하지 않는다."""
    src = Path(__file__).resolve().parent.parent / "src" / "agents" / "data_price_us.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "execute"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "conn"
        ):
            pytest.fail("conn.execute() 직접 호출 발견 — execute_sql()을 경유해야 한다")


# ── get_latest_price_us: 최신 스냅샷 조회 ────────────────────────────────────
def test_get_latest_price_us_returns_latest_row_for_single_code(tmp_path):
    db = _seed_prices(tmp_path, [
        ("AAPL", "2026-07-10", 195.0, 194.0, 196.0, 193.0, 5.0e7),
        ("AAPL", "2026-07-11", 198.2, 196.5, 199.0, 195.5, 4.2e7),
    ])
    conn = connect_readonly(db)
    try:
        rows = get_latest_price_us(conn, "AAPL")
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0]["stock_code"] == "AAPL"
    assert rows[0]["date"] == "2026-07-11"  # 최신 날짜 채택(과거 07-10 아님)
    assert rows[0]["close"] == 198.2
    assert rows[0]["name"] == "Apple"
    assert rows[0]["exchange"] == "NASDAQ"
    assert rows[0]["sector"] == "Technology"
    assert rows[0]["market_cap"] == 3.0e12


def test_get_latest_price_us_normalizes_lowercase_ticker(tmp_path):
    db = _seed_prices(tmp_path, [("AAPL", "2026-07-11", 198.2, 196.5, 199.0, 195.5, 4.2e7)])
    conn = connect_readonly(db)
    try:
        rows = get_latest_price_us(conn, "aapl")
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0]["stock_code"] == "AAPL"


def test_get_latest_price_us_accepts_list_of_multiple_codes(tmp_path):
    db = _seed_prices(tmp_path, [
        ("AAPL", "2026-07-11", 198.2, 196.5, 199.0, 195.5, 4.2e7),
        ("MSFT", "2026-07-11", 430.0, 425.0, 432.0, 424.0, 2.0e7),
    ])
    conn = connect_readonly(db)
    try:
        rows = get_latest_price_us(conn, ["AAPL", "MSFT"])
    finally:
        conn.close()
    assert {r["stock_code"] for r in rows} == {"AAPL", "MSFT"}


def test_get_latest_price_us_returns_empty_list_for_unknown_code(tmp_path):
    db = _seed_prices(tmp_path, [("AAPL", "2026-07-11", 198.2, 196.5, 199.0, 195.5, 4.2e7)])
    conn = connect_readonly(db)
    try:
        rows = get_latest_price_us(conn, "ZZZZ")
    finally:
        conn.close()
    assert rows == []


def test_get_latest_price_us_filters_malformed_ticker_before_building_sql(tmp_path):
    """execute_sql은 파라미터 바인딩이 없어 티커를 SQL 문자열에 직접 끼워 넣는다 —
    형식에 맞지 않는(SQL 인젝션 시도 포함) 입력은 걸러지고 DB는 손상되지 않아야 한다."""
    db = _seed_prices(tmp_path, [("AAPL", "2026-07-11", 198.2, 196.5, 199.0, 195.5, 4.2e7)])
    conn = connect_readonly(db)
    try:
        rows = get_latest_price_us(conn, "AAPL'; DROP TABLE us_prices; --")
    finally:
        conn.close()
    assert rows == []  # 형식 불일치로 걸러짐(쿼리 자체가 실행되지 않음)

    verify_conn = connect(db)
    try:
        count = verify_conn.execute("SELECT COUNT(*) FROM us_prices").fetchone()[0]
    finally:
        verify_conn.close()
    assert count == 1


# ── execute_sql(HA-1 실행기) 경유 확인 (스파이) ──────────────────────────────
def test_get_latest_price_us_goes_through_execute_sql(tmp_path, monkeypatch):
    """execute_sql이 실제로 호출되는지 monkeypatch 스파이로 검증한다."""
    import src.agents.data_price_us as mod

    db = _seed_prices(tmp_path, [("AAPL", "2026-07-11", 198.2, 196.5, 199.0, 195.5, 4.2e7)])
    calls: list[tuple] = []
    real_execute_sql = mod.execute_sql

    def spy_execute_sql(sql, conn, *args, **kwargs):
        calls.append((sql, conn))
        return real_execute_sql(sql, conn, *args, **kwargs)

    monkeypatch.setattr(mod, "execute_sql", spy_execute_sql)

    conn = connect_readonly(db)
    try:
        rows = mod.get_latest_price_us(conn, "AAPL")
    finally:
        conn.close()

    assert len(calls) == 1
    assert "us_prices" in calls[0][0]
    assert rows[0]["stock_code"] == "AAPL"


# ── get_price_snapshot_us: 스냅샷 + 기술지표 통합 ────────────────────────────
def test_get_price_snapshot_us_without_indicators_matches_get_latest_price_us(tmp_path):
    db = _seed_prices(tmp_path, [("AAPL", "2026-07-11", 198.2, 196.5, 199.0, 195.5, 4.2e7)])
    conn = connect_readonly(db)
    try:
        snapshot = get_price_snapshot_us(conn, "AAPL")
        plain = get_latest_price_us(conn, "AAPL")
    finally:
        conn.close()
    assert snapshot == plain


def test_get_price_snapshot_us_attaches_indicators_via_compute_technical_indicator(tmp_path):
    """compute_technical_indicator를 그대로 호출해 지표가 붙는지 확인한다 — 새 지표 계산
    로직을 만들지 않았는지가 핵심. history_fn으로 us_price_history_batch가 전달되는지도 확인."""
    db = _seed_prices(tmp_path, [("AAPL", "2026-07-11", 198.2, 196.5, 199.0, 195.5, 4.2e7)])
    calls: list[dict] = []

    def fake_indicator_fn(conn, rows, asof, indicators, **kwargs):
        calls.append({"rows": rows, "asof": asof, "indicators": indicators, "kwargs": kwargs})
        out = []
        for r in rows:
            nr = dict(r)
            nr["rsi_14"] = 55.5
            out.append(nr)
        return out

    conn = connect_readonly(db)
    try:
        result = get_price_snapshot_us(
            conn, "AAPL", indicators=[{"name": "rsi"}], indicator_fn=fake_indicator_fn,
        )
    finally:
        conn.close()

    assert len(calls) == 1
    assert calls[0]["indicators"] == [{"name": "rsi"}]
    assert calls[0]["rows"][0]["stock_code"] == "AAPL"  # 원 필드(stock_code) 그대로 전달
    assert "history_fn" in calls[0]["kwargs"]  # US 전용 history_fn이 주입됨
    assert result[0]["rsi_14"] == 55.5
    assert result[0]["stock_code"] == "AAPL"  # 원 필드 보존


def test_get_price_snapshot_us_defaults_asof_to_latest_row_date_when_not_given(tmp_path):
    db = _seed_prices(tmp_path, [
        ("AAPL", "2026-07-10", 195.0, 194.0, 196.0, 193.0, 5.0e7),
        ("AAPL", "2026-07-11", 198.2, 196.5, 199.0, 195.5, 4.2e7),
    ])
    calls: list[dict] = []

    def fake_indicator_fn(conn, rows, asof, indicators, **kwargs):
        calls.append({"asof": asof})
        return rows

    conn = connect_readonly(db)
    try:
        get_price_snapshot_us(
            conn, "AAPL", indicators=[{"name": "rsi"}], indicator_fn=fake_indicator_fn,
        )
    finally:
        conn.close()
    assert calls[0]["asof"] == "2026-07-11"


def test_get_price_snapshot_us_no_rows_skips_indicator_call(tmp_path):
    db = _seed_prices(tmp_path, [("AAPL", "2026-07-11", 198.2, 196.5, 199.0, 195.5, 4.2e7)])
    calls: list[dict] = []

    def fake_indicator_fn(conn, rows, asof, indicators, **kwargs):
        calls.append({})
        return rows

    conn = connect_readonly(db)
    try:
        result = get_price_snapshot_us(
            conn, "ZZZZ", indicators=[{"name": "rsi"}], indicator_fn=fake_indicator_fn,
        )
    finally:
        conn.close()
    assert result == []
    assert calls == []  # 빈 rows면 무거운 지표 계산을 호출하지 않는다


# ── 실제 compute_technical_indicator/us_price_history_batch 연동(모킹 없이) ──
def test_get_price_snapshot_us_default_indicator_fn_is_real_compute_technical_indicator(tmp_path):
    """fake 없이 실제 compute_technical_indicator + us_price_history_batch가 동작함을 확인
    (TA-Lib 미설치 환경이면 스킵)."""
    pytest.importorskip("talib")
    db = _seed_prices(tmp_path, [
        (
            "AAPL",
            f"2026-06-{d:02d}" if d <= 30 else f"2026-07-{d - 30:02d}",
            100.0 + d,
            100.0 + d,
            101.0 + d,
            99.0 + d,
            1.0e6,
        )
        for d in range(1, 40)
    ])
    conn = connect_readonly(db)
    try:
        result = get_price_snapshot_us(conn, "AAPL", indicators=[{"name": "sma", "period": 5}])
    finally:
        conn.close()
    assert result
    assert result[0]["stock_code"] == "AAPL"


# ── get_price_history_us: 단일 티커 종가 시계열(차트용) ──────────────────────
def test_get_price_history_us_returns_series_oldest_to_newest(tmp_path):
    db = _seed_prices(tmp_path, [
        ("AAPL", "2026-07-09", 189.0, 188.0, 190.0, 187.0, 1.0e6),
        ("AAPL", "2026-07-10", 192.0, 189.0, 193.0, 188.0, 1.1e6),
        ("AAPL", "2026-07-11", 195.0, 192.0, 196.0, 191.0, 1.2e6),
    ])
    conn = connect_readonly(db)
    try:
        rows = get_price_history_us(conn, "AAPL")
    finally:
        conn.close()
    assert [r["date"] for r in rows] == ["2026-07-09", "2026-07-10", "2026-07-11"]  # 과거→최신
    assert rows[0]["close"] == 189.0
    assert rows[-1]["close"] == 195.0
    assert rows[0]["stock_code"] == "AAPL"


def test_get_price_history_us_respects_days_limit(tmp_path):
    db = _seed_prices(tmp_path, [
        ("AAPL", "2026-07-07", 185.0, 184.0, 186.0, 183.0, 1.0e6),
        ("AAPL", "2026-07-08", 187.0, 185.0, 188.0, 184.0, 1.0e6),
        ("AAPL", "2026-07-09", 189.0, 188.0, 190.0, 187.0, 1.0e6),
        ("AAPL", "2026-07-10", 192.0, 189.0, 193.0, 188.0, 1.1e6),
        ("AAPL", "2026-07-11", 195.0, 192.0, 196.0, 191.0, 1.2e6),
    ])
    conn = connect_readonly(db)
    try:
        rows = get_price_history_us(conn, "AAPL", days=2)
    finally:
        conn.close()
    assert [r["date"] for r in rows] == ["2026-07-10", "2026-07-11"]  # 최신 2개, 과거→최신
    assert len(rows) == 2


def test_get_price_history_us_filters_malformed_ticker(tmp_path):
    db = _seed_prices(tmp_path, [
        ("AAPL", "2026-07-11", 195.0, 192.0, 196.0, 191.0, 1.2e6),
    ])
    conn = connect_readonly(db)
    try:
        rows = get_price_history_us(conn, "AAPL'; DROP TABLE us_prices; --")
    finally:
        conn.close()
    assert rows == []
    verify_conn = connect(db)
    try:
        count = verify_conn.execute("SELECT COUNT(*) FROM us_prices").fetchone()[0]
    finally:
        verify_conn.close()
    assert count == 1


def test_get_price_history_us_goes_through_execute_sql(tmp_path, monkeypatch):
    import src.agents.data_price_us as mod

    db = _seed_prices(tmp_path, [
        ("AAPL", "2026-07-11", 195.0, 192.0, 196.0, 191.0, 1.2e6),
    ])
    calls: list[tuple] = []
    real_execute_sql = mod.execute_sql

    def spy_execute_sql(sql, conn, *args, **kwargs):
        calls.append((sql, conn))
        return real_execute_sql(sql, conn, *args, **kwargs)

    monkeypatch.setattr(mod, "execute_sql", spy_execute_sql)

    conn = connect_readonly(db)
    try:
        rows = get_price_history_us(conn, "AAPL")
    finally:
        conn.close()

    assert len(calls) == 1
    assert "us_prices" in calls[0][0]
    assert rows[0]["stock_code"] == "AAPL"
