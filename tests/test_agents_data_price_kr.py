"""한국 주가 데이터 에이전트(src/agents/data_price_kr.py) 단위/통합 테스트 (TDD, HA-3).

검증 대상:
- prices 테이블은 이미 병합 완료된 단일 테이블이며(krx.py가 종가/시가총액, naver_prices.py가
  시가/고가/저가/거래량을 각각 채움) — 이 에이전트는 신규 병합 로직을 두지 않고 그대로 조회만
  한다(정적 가드로 확인).
- SQL은 반드시 execute_sql(HA-1 실행기, src/agents/exec_runtime.py)을 경유한다(conn.execute()
  직접 호출 금지). execute_sql은 파라미터 바인딩이 없어 종목코드를 SQL 문자열에 직접 끼워
  넣으므로, 6자리 숫자 형식이 아닌 입력은 주입 방지를 위해 조용히 걸러진다.
- 반환은 get_cross_section과 동일한 계약(stock_code 필드를 가진 list[dict])이라 HA-2
  재무데이터와 stock_code 키로 merge 가능하고, compute_technical_indicator의 rows 인자로
  그대로 넣을 수 있다.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.agents.data_price_kr import (
    get_latest_price_kr,
    get_price_history_kr,
    get_price_snapshot_kr,
)
from src.db import connect, connect_readonly, init_db


def _seed_prices(tmp_path, rows: list[tuple]) -> str:
    """rows: (stock_code, date, close, market_cap, open, high, low, volume)."""
    db = tmp_path / "prices.db"
    init_db(str(db))
    conn = connect(str(db))
    try:
        conn.executemany(
            "INSERT INTO prices(stock_code, date, close, market_cap, open, high, low, volume) "
            "VALUES(?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return str(db)


# ── 정적 가드: 신규 병합 로직이 없음 ─────────────────────────────────────────
# 모듈 docstring(설명 텍스트)에는 "ON CONFLICT"/"conn.execute(" 같은 표현이 background
# 설명으로 등장하므로, docstring을 제외한 실제 코드 부분만 검사한다.
def _code_without_module_docstring(src: Path) -> str:
    text = src.read_text(encoding="utf-8")
    parts = text.split('"""', 2)
    return parts[-1] if len(parts) == 3 else text


def test_module_has_no_new_merge_logic():
    """prices 테이블은 이미 병합 완료 — 이 파일이 krx/naver_prices 병합 로직을
    새로 만들거나 그 인제스트 모듈을 import하면 안 된다."""
    src = Path(__file__).resolve().parent.parent / "src" / "agents" / "data_price_kr.py"
    code = _code_without_module_docstring(src)
    assert "src.ingest.krx" not in code
    assert "src.ingest.naver_prices" not in code
    assert "ON CONFLICT" not in code  # upsert/병합 로직의 표식


def test_module_does_not_call_conn_execute_directly():
    """SQL 실행은 반드시 execute_sql(HA-1) 경유 — conn.execute()를 직접 호출하지 않는다.

    docstring 설명 문구(예: "conn.execute() 직접 호출 금지")에는 이 표현이 등장할 수 있어
    단순 텍스트 검색 대신 AST로 실제 함수호출 노드만 검사한다.
    """
    src = Path(__file__).resolve().parent.parent / "src" / "agents" / "data_price_kr.py"
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


# ── get_latest_price_kr: 최신 스냅샷 조회 ────────────────────────────────────
def test_get_latest_price_kr_returns_latest_row_for_single_code(tmp_path):
    db = _seed_prices(tmp_path, [
        ("005930", "2026-07-10", 70000.0, 4.0e14, 69500.0, 70200.0, 69000.0, 1.0e7),
        ("005930", "2026-07-11", 71000.0, 4.1e14, 70000.0, 71500.0, 69800.0, 1.2e7),
    ])
    conn = connect_readonly(db)
    try:
        rows = get_latest_price_kr(conn, "005930")
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0]["stock_code"] == "005930"
    assert rows[0]["date"] == "2026-07-11"  # 최신 날짜 채택(과거 07-10 아님)
    assert rows[0]["close"] == 71000.0
    assert rows[0]["market_cap"] == 4.1e14
    assert rows[0]["open"] == 70000.0
    assert rows[0]["high"] == 71500.0
    assert rows[0]["low"] == 69800.0
    assert rows[0]["volume"] == 1.2e7


def test_get_latest_price_kr_accepts_list_of_multiple_codes(tmp_path):
    db = _seed_prices(tmp_path, [
        ("005930", "2026-07-11", 71000.0, 4.1e14, 70000.0, 71500.0, 69800.0, 1.2e7),
        ("000660", "2026-07-11", 200000.0, 1.3e14, 199000.0, 201000.0, 198000.0, 5.0e6),
    ])
    conn = connect_readonly(db)
    try:
        rows = get_latest_price_kr(conn, ["005930", "000660"])
    finally:
        conn.close()
    by_code = {r["stock_code"]: r for r in rows}
    assert set(by_code) == {"005930", "000660"}
    assert by_code["000660"]["close"] == 200000.0


def test_get_latest_price_kr_returns_empty_list_for_unknown_code(tmp_path):
    db = _seed_prices(tmp_path, [
        ("005930", "2026-07-11", 71000.0, 4.1e14, 70000.0, 71500.0, 69800.0, 1.2e7),
    ])
    conn = connect_readonly(db)
    try:
        rows = get_latest_price_kr(conn, "999999")
    finally:
        conn.close()
    assert rows == []


def test_get_latest_price_kr_filters_malformed_code_before_building_sql(tmp_path):
    """execute_sql은 파라미터 바인딩이 없어 종목코드를 SQL 문자열에 직접 끼워 넣는다 —
    6자리 숫자가 아닌(SQL 인젝션 시도 포함) 입력은 걸러지고, DB는 손상되지 않아야 한다."""
    db = _seed_prices(tmp_path, [
        ("005930", "2026-07-11", 71000.0, 4.1e14, 70000.0, 71500.0, 69800.0, 1.2e7),
    ])
    conn = connect_readonly(db)
    try:
        rows = get_latest_price_kr(conn, "005930'; DROP TABLE prices; --")
    finally:
        conn.close()
    assert rows == []  # 형식 불일치로 걸러짐(쿼리 자체가 실행되지 않음)

    # DB가 실제로 손상되지 않았는지 재확인
    verify_conn = connect(db)
    try:
        count = verify_conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    finally:
        verify_conn.close()
    assert count == 1


# ── execute_sql(HA-1 실행기) 경유 확인 (스파이) ──────────────────────────────
def test_get_latest_price_kr_goes_through_execute_sql(tmp_path, monkeypatch):
    """execute_sql이 실제로 호출되는지 monkeypatch 스파이로 검증한다."""
    import src.agents.data_price_kr as mod

    db = _seed_prices(tmp_path, [
        ("005930", "2026-07-11", 71000.0, 4.1e14, 70000.0, 71500.0, 69800.0, 1.2e7),
    ])
    calls: list[tuple] = []
    real_execute_sql = mod.execute_sql

    def spy_execute_sql(sql, conn, *args, **kwargs):
        calls.append((sql, conn))
        return real_execute_sql(sql, conn, *args, **kwargs)

    monkeypatch.setattr(mod, "execute_sql", spy_execute_sql)

    conn = connect_readonly(db)
    try:
        rows = get_latest_price_kr(conn, "005930")
    finally:
        conn.close()

    assert len(calls) == 1
    assert "prices" in calls[0][0]
    assert rows[0]["stock_code"] == "005930"


# ── get_price_snapshot_kr: 스냅샷 + 기술지표 통합 ────────────────────────────
def test_get_price_snapshot_kr_without_indicators_matches_get_latest_price_kr(tmp_path):
    db = _seed_prices(tmp_path, [
        ("005930", "2026-07-11", 71000.0, 4.1e14, 70000.0, 71500.0, 69800.0, 1.2e7),
    ])
    conn = connect_readonly(db)
    try:
        snapshot = get_price_snapshot_kr(conn, "005930")
        plain = get_latest_price_kr(conn, "005930")
    finally:
        conn.close()
    assert snapshot == plain


def test_get_price_snapshot_kr_attaches_indicators_via_compute_technical_indicator(tmp_path):
    """compute_technical_indicator(HA 이전 세션에 이미 완성된 TA-Lib 프리미티브)를 그대로
    호출해 지표가 붙는지 확인한다 — 새 지표 계산 로직을 만들지 않았는지가 핵심."""
    db = _seed_prices(tmp_path, [
        ("005930", "2026-07-11", 71000.0, 4.1e14, 70000.0, 71500.0, 69800.0, 1.2e7),
    ])
    calls: list[dict] = []

    def fake_indicator_fn(conn, rows, asof, indicators, **kwargs):
        calls.append({"rows": rows, "asof": asof, "indicators": indicators})
        out = []
        for r in rows:
            nr = dict(r)
            nr["rsi_14"] = 55.5
            out.append(nr)
        return out

    conn = connect_readonly(db)
    try:
        result = get_price_snapshot_kr(
            conn, "005930", indicators=[{"name": "rsi"}], indicator_fn=fake_indicator_fn,
        )
    finally:
        conn.close()

    assert len(calls) == 1
    assert calls[0]["indicators"] == [{"name": "rsi"}]
    assert calls[0]["rows"][0]["stock_code"] == "005930"  # 원 필드(stock_code) 그대로 전달
    assert result[0]["rsi_14"] == 55.5
    assert result[0]["stock_code"] == "005930"  # 원 필드 보존


def test_get_price_snapshot_kr_defaults_asof_to_latest_row_date_when_not_given(tmp_path):
    db = _seed_prices(tmp_path, [
        ("005930", "2026-07-11", 71000.0, 4.1e14, 70000.0, 71500.0, 69800.0, 1.2e7),
    ])
    calls: list[dict] = []

    def fake_indicator_fn(conn, rows, asof, indicators, **kwargs):
        calls.append({"asof": asof})
        return rows

    conn = connect_readonly(db)
    try:
        get_price_snapshot_kr(
            conn, "005930", indicators=[{"name": "rsi"}], indicator_fn=fake_indicator_fn,
        )
    finally:
        conn.close()

    assert calls[0]["asof"] == "2026-07-11"  # 조회된 스냅샷의 최신 date를 기준시점으로 사용


def test_get_price_snapshot_kr_uses_explicit_asof_when_given(tmp_path):
    db = _seed_prices(tmp_path, [
        ("005930", "2026-07-11", 71000.0, 4.1e14, 70000.0, 71500.0, 69800.0, 1.2e7),
    ])
    calls: list[dict] = []

    def fake_indicator_fn(conn, rows, asof, indicators, **kwargs):
        calls.append({"asof": asof})
        return rows

    conn = connect_readonly(db)
    try:
        get_price_snapshot_kr(
            conn, "005930", asof="2026-06-30",
            indicators=[{"name": "rsi"}], indicator_fn=fake_indicator_fn,
        )
    finally:
        conn.close()

    assert calls[0]["asof"] == "2026-06-30"


def test_get_price_snapshot_kr_no_rows_skips_indicator_call(tmp_path):
    """조회 결과가 없으면 compute_technical_indicator를 호출하지 않는다(무거운 연산 스킵)."""
    db = _seed_prices(tmp_path, [])
    calls: list[dict] = []

    def fake_indicator_fn(conn, rows, asof, indicators, **kwargs):
        calls.append({})
        return rows

    conn = connect_readonly(db)
    try:
        result = get_price_snapshot_kr(
            conn, "999999", indicators=[{"name": "rsi"}], indicator_fn=fake_indicator_fn,
        )
    finally:
        conn.close()

    assert result == []
    assert calls == []


def test_get_price_snapshot_kr_default_indicator_fn_is_real_compute_technical_indicator(tmp_path):
    """indicator_fn을 주입하지 않으면 실제 compute_technical_indicator(TA-Lib)를 쓴다
    (스모크 테스트 — TA-Lib 수치 정확성 자체는 검증 대상 아님)."""
    pytest.importorskip("talib")
    db = tmp_path / "smoke.db"
    init_db(str(db))
    conn = connect(str(db))
    try:
        rows = []
        import datetime
        base = datetime.date(2026, 6, 1)
        for i in range(30):
            d = (base + datetime.timedelta(days=i)).isoformat()
            rows.append(("005930", d, 70000.0 + i * 10, 4.0e14, 70000.0, 70500.0, 69500.0, 1.0e7))
        conn.executemany(
            "INSERT INTO prices(stock_code, date, close, market_cap, open, high, low, volume) "
            "VALUES(?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    ro_conn = connect_readonly(str(db))
    try:
        result = get_price_snapshot_kr(
            ro_conn, "005930", asof="2026-06-30", indicators=[{"name": "sma", "period": 5}],
        )
    finally:
        ro_conn.close()

    assert result[0]["stock_code"] == "005930"
    assert "sma_5" in result[0]
    assert result[0]["sma_5"] is not None


# ── get_price_history_kr: 단일 종목 종가 시계열(차트용) ──────────────────────
def test_get_price_history_kr_returns_series_oldest_to_newest(tmp_path):
    db = _seed_prices(tmp_path, [
        ("005930", "2026-07-09", 69000.0, 4.0e14, 68500.0, 69500.0, 68000.0, 1.0e7),
        ("005930", "2026-07-10", 70000.0, 4.0e14, 69500.0, 70200.0, 69000.0, 1.1e7),
        ("005930", "2026-07-11", 71000.0, 4.1e14, 70000.0, 71500.0, 69800.0, 1.2e7),
    ])
    conn = connect_readonly(db)
    try:
        rows = get_price_history_kr(conn, "005930")
    finally:
        conn.close()
    assert [r["date"] for r in rows] == ["2026-07-09", "2026-07-10", "2026-07-11"]  # 과거→최신
    assert rows[0]["close"] == 69000.0
    assert rows[-1]["close"] == 71000.0
    assert rows[0]["stock_code"] == "005930"


def test_get_price_history_kr_respects_days_limit(tmp_path):
    db = _seed_prices(tmp_path, [
        ("005930", "2026-07-07", 67000.0, 4.0e14, 66500.0, 67500.0, 66000.0, 1.0e7),
        ("005930", "2026-07-08", 68000.0, 4.0e14, 67500.0, 68500.0, 67000.0, 1.0e7),
        ("005930", "2026-07-09", 69000.0, 4.0e14, 68500.0, 69500.0, 68000.0, 1.0e7),
        ("005930", "2026-07-10", 70000.0, 4.0e14, 69500.0, 70200.0, 69000.0, 1.1e7),
        ("005930", "2026-07-11", 71000.0, 4.1e14, 70000.0, 71500.0, 69800.0, 1.2e7),
    ])
    conn = connect_readonly(db)
    try:
        rows = get_price_history_kr(conn, "005930", days=3)
    finally:
        conn.close()
    # 최신 3개만, 과거→최신 순서
    assert [r["date"] for r in rows] == ["2026-07-09", "2026-07-10", "2026-07-11"]
    assert len(rows) == 3


def test_get_price_history_kr_filters_malformed_code(tmp_path):
    db = _seed_prices(tmp_path, [
        ("005930", "2026-07-11", 71000.0, 4.1e14, 70000.0, 71500.0, 69800.0, 1.2e7),
    ])
    conn = connect_readonly(db)
    try:
        rows = get_price_history_kr(conn, "005930'; DROP TABLE prices; --")
    finally:
        conn.close()
    assert rows == []
    verify_conn = connect(db)
    try:
        count = verify_conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    finally:
        verify_conn.close()
    assert count == 1


def test_get_price_history_kr_goes_through_execute_sql(tmp_path, monkeypatch):
    import src.agents.data_price_kr as mod

    db = _seed_prices(tmp_path, [
        ("005930", "2026-07-11", 71000.0, 4.1e14, 70000.0, 71500.0, 69800.0, 1.2e7),
    ])
    calls: list[tuple] = []
    real_execute_sql = mod.execute_sql

    def spy_execute_sql(sql, conn, *args, **kwargs):
        calls.append((sql, conn))
        return real_execute_sql(sql, conn, *args, **kwargs)

    monkeypatch.setattr(mod, "execute_sql", spy_execute_sql)

    conn = connect_readonly(db)
    try:
        rows = get_price_history_kr(conn, "005930")
    finally:
        conn.close()

    assert len(calls) == 1
    assert "prices" in calls[0][0]
    assert rows[0]["stock_code"] == "005930"
