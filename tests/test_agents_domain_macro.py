"""매크로 도메인 에이전트(src/agents/domain_macro.py) 단위/통합 테스트 (TDD, HA-8).

이 에이전트는 src/ingest/macro_signal.py/macro_pipeline.py의 판정 로직을 절대 재계산하지
않는다 — 매일 07:40 launchd가 이미 계산해 macro_signal 테이블에 append해 둔 최신 1행을
web/app.py의 api_macro_signal() 핸들러와 동일한 SQL 패턴으로 조회만 한다.

조회는 반드시 HA-1 실행기(src/agents/exec_runtime.execute_sql)를 경유해야 한다(conn.execute()
직접 호출 금지). 여기서는 execute_sql_fn 스파이를 주입해 그 경로를 강제로 확인한다.
"""
from __future__ import annotations

from src.agents.domain_macro import answer_macro_question, get_macro_history
from src.agents.exec_runtime import execute_sql
from src.db import connect, connect_readonly, init_db
from src.ingest.macro_signal import run_signal


def _seed_signals(db_path: str) -> None:
    """macro_signal에 3일치 이력을 append(기존 run_signal 그대로 호출, 재계산 로직 아님)."""
    conn = connect(db_path)
    try:
        run_signal(conn, spread=0.6, cnn=50, vix=14.0, as_of="2026-07-12")   # GREEN
        run_signal(conn, spread=0.2, cnn=30, vix=22.0, as_of="2026-07-13")   # YELLOW
        # 종합신호는 금리차 레짐에서만 결정 — CNN/VIX를 일부러 극단값으로 넣어도
        # overall은 스프레드(-0.1 → 역전)에만 의존해야 한다.
        run_signal(conn, spread=-0.1, cnn=0, vix=80.0, as_of="2026-07-14")   # RED
    finally:
        conn.close()


# ---------- 최신 신호 조회 (통합테스트: 실제 DB + execute_sql 경유) ----------

def test_answer_macro_question_returns_latest_signal(tmp_path):
    db = str(tmp_path / "macro.db")
    init_db(db)
    _seed_signals(db)

    conn = connect_readonly(db)
    try:
        res = answer_macro_question("지금 매크로 신호 어때?", conn)
    finally:
        conn.close()

    assert res["available"] is True
    assert res["as_of"] == "2026-07-14"          # 최신(id 최대) 행
    assert res["overall"] == "RED"                # 스프레드 -0.1(역전) → RED
    assert res["spread"] == {"value": -0.1, "regime": "역전"}
    assert res["cnn"] == {"value": 0, "band": "극단공포"}
    assert res["vix"] == {"value": 80.0, "band": "공포"}
    assert res["created_at"] is not None


def test_answer_macro_question_overall_unaffected_by_cnn_vix_extremes(tmp_path):
    """AC10 회귀 정신 그대로: CNN/VIX가 극단값이어도 overall은 스프레드 레짐만 따른다."""
    db = str(tmp_path / "macro2.db")
    init_db(db)
    conn = connect(db)
    try:
        # 스프레드는 '정상'(0.6) 이지만 CNN/VIX는 공포 극단값 — overall은 GREEN이어야 한다.
        run_signal(conn, spread=0.6, cnn=1, vix=90.0, as_of="2026-07-14")
    finally:
        conn.close()

    conn = connect_readonly(db)
    try:
        res = answer_macro_question("공포지수 심각한데 신호는?", conn)
    finally:
        conn.close()

    assert res["overall"] == "GREEN"
    assert res["cnn"]["band"] == "극단공포"
    assert res["vix"]["band"] == "공포"


# ---------- 이력 없음 (예외 대신 available=False, web _signal_payload(None)과 동일 정신) ----------

def test_answer_macro_question_empty_table_is_graceful(tmp_path):
    db = str(tmp_path / "empty.db")
    init_db(db)

    conn = connect_readonly(db)
    try:
        res = answer_macro_question("매크로 신호 알려줘", conn)
    finally:
        conn.close()

    assert res["available"] is False
    assert res["as_of"] is None
    assert res["overall"] is None
    assert res["spread"] == {"value": None, "regime": None}
    assert res["cnn"] == {"value": None, "band": None}
    assert res["vix"] == {"value": None, "band": None}
    assert res["created_at"] is None


# ---------- HA-1 실행기 경유 강제 (conn.execute() 직접 호출 금지) ----------

def test_answer_macro_question_goes_through_execute_sql(tmp_path):
    db = str(tmp_path / "spy.db")
    init_db(db)
    _seed_signals(db)

    calls: list[str] = []

    def spy_execute_sql(sql, conn, **kwargs):
        calls.append(sql)
        return execute_sql(sql, conn, **kwargs)

    conn = connect_readonly(db)
    try:
        res = answer_macro_question("신호?", conn, execute_sql_fn=spy_execute_sql)
    finally:
        conn.close()

    assert len(calls) == 1
    assert "macro_signal" in calls[0]
    assert "ORDER BY id DESC" in calls[0]
    assert res["available"] is True


# ---------- 설명 필드 — 왜 CNN/VIX가 극단값이어도 신호가 안 바뀌는지 ----------

def test_answer_macro_question_explanation_notes_spread_only_decision(tmp_path):
    db = str(tmp_path / "explain.db")
    init_db(db)
    _seed_signals(db)

    conn = connect_readonly(db)
    try:
        res = answer_macro_question("왜 신호가 안 바뀌었어?", conn)
    finally:
        conn.close()

    assert "explanation" in res
    assert "금리차" in res["explanation"] or "스프레드" in res["explanation"]


def test_answer_macro_question_explanation_present_even_when_unavailable(tmp_path):
    """설명은 이력이 없을 때(available=False)도 항상 채워져 있어야 한다."""
    db = str(tmp_path / "explain_empty.db")
    init_db(db)

    conn = connect_readonly(db)
    try:
        res = answer_macro_question("왜 신호가 안 바뀌었어?", conn)
    finally:
        conn.close()

    assert res["available"] is False
    assert "explanation" in res and res["explanation"]


def test_answer_macro_question_question_is_echoed_back(tmp_path):
    db = str(tmp_path / "echo.db")
    init_db(db)
    _seed_signals(db)

    conn = connect_readonly(db)
    try:
        res = answer_macro_question("금리차 어때?", conn)
    finally:
        conn.close()

    assert res["question"] == "금리차 어때?"


# ---------- get_macro_history: spread 시계열(차트용) ----------

def test_get_macro_history_returns_spread_series_oldest_to_newest(tmp_path):
    db = str(tmp_path / "hist.db")
    init_db(db)
    _seed_signals(db)  # 2026-07-12(0.6) / 07-13(0.2) / 07-14(-0.1)

    conn = connect_readonly(db)
    try:
        rows = get_macro_history(conn)
    finally:
        conn.close()

    assert [r["as_of"] for r in rows] == ["2026-07-12", "2026-07-13", "2026-07-14"]  # 과거→최신
    assert rows[0]["spread"] == 0.6
    assert rows[-1]["spread"] == -0.1


def test_get_macro_history_respects_days_limit(tmp_path):
    db = str(tmp_path / "hist2.db")
    init_db(db)
    _seed_signals(db)

    conn = connect_readonly(db)
    try:
        rows = get_macro_history(conn, days=2)
    finally:
        conn.close()

    assert [r["as_of"] for r in rows] == ["2026-07-13", "2026-07-14"]  # 최신 2개, 과거→최신
    assert len(rows) == 2


def test_get_macro_history_empty_table_returns_empty(tmp_path):
    db = str(tmp_path / "hist_empty.db")
    init_db(db)

    conn = connect_readonly(db)
    try:
        rows = get_macro_history(conn)
    finally:
        conn.close()

    assert rows == []


def test_get_macro_history_goes_through_execute_sql(tmp_path):
    db = str(tmp_path / "hist_spy.db")
    init_db(db)
    _seed_signals(db)

    calls: list[str] = []

    def spy_execute_sql(sql, conn, **kwargs):
        calls.append(sql)
        return execute_sql(sql, conn, **kwargs)

    conn = connect_readonly(db)
    try:
        rows = get_macro_history(conn, execute_sql_fn=spy_execute_sql)
    finally:
        conn.close()

    assert len(calls) == 1
    assert "macro_signal" in calls[0]
    assert len(rows) == 3
