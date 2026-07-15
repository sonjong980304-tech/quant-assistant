"""us_financials.py yfinance 재무제표 매퍼 테스트.

.omc/specs/brainstorming-us-market-data-plane.md AC1/AC5/AC6 검증.
yfinance `Ticker(symbol).income_stmt`/`.balance_sheet`/`.cashflow`가 반환하는
형태(행=항목명, 열=기간말일 Timestamp)를 흉내낸 pandas DataFrame을 입력으로
EAV(long) 형식으로 변환하는 순수 로직만 검증한다.
"""
from __future__ import annotations

import pandas as pd

import src.ingest.us_financials as us_financials
from src.db import connect, init_db
from src.ingest.us_financials import ingest_us_financials, normalize_financial_statement
from tests.conftest import seed_us_companies


def test_normalize_financial_statement_converts_wide_to_eav_rows():
    df = pd.DataFrame(
        {
            pd.Timestamp("2025-09-30"): [391_035_000_000.0],
            pd.Timestamp("2024-09-30"): [383_285_000_000.0],
        },
        index=["Total Revenue"],
    )
    rows = normalize_financial_statement("AAPL", "income_stmt", "annual", df)
    assert {"as_of_date": "2025-09-30", "item_key": "Total Revenue", "item_value": 391_035_000_000.0}.items() <= rows[0].items()
    assert len(rows) == 2


def test_normalize_financial_statement_skips_nan_values():
    df = pd.DataFrame(
        {
            pd.Timestamp("2025-09-30"): [100.0, float("nan")],
        },
        index=["Total Revenue", "Rare Line Item"],
    )
    rows = normalize_financial_statement("AAPL", "income_stmt", "annual", df)
    item_keys = {r["item_key"] for r in rows}
    assert item_keys == {"Total Revenue"}


def test_normalize_financial_statement_passes_through_period_and_statement_type():
    df = pd.DataFrame({pd.Timestamp("2025-12-31"): [1.0]}, index=["Total Assets"])
    rows = normalize_financial_statement("MSFT", "balance_sheet", "quarterly", df)
    assert rows[0]["stock_code"] == "MSFT"
    assert rows[0]["statement_type"] == "balance_sheet"
    assert rows[0]["period_type"] == "quarterly"


def test_normalize_financial_statement_empty_dataframe_returns_empty_list():
    df = pd.DataFrame()
    rows = normalize_financial_statement("AAPL", "cashflow", "annual", df)
    assert rows == []


_SAMPLE_STATEMENTS = {
    ("income_stmt", "annual"): pd.DataFrame(
        {pd.Timestamp("2025-09-30"): [391_035_000_000.0]}, index=["Total Revenue"]
    ),
    ("income_stmt", "quarterly"): pd.DataFrame(
        {pd.Timestamp("2025-06-30"): [90_000_000_000.0]}, index=["Total Revenue"]
    ),
    ("balance_sheet", "annual"): pd.DataFrame(
        {pd.Timestamp("2025-09-30"): [350_000_000_000.0]}, index=["Total Assets"]
    ),
    ("balance_sheet", "quarterly"): pd.DataFrame(),
    ("cashflow", "annual"): pd.DataFrame(
        {pd.Timestamp("2025-09-30"): [100_000_000_000.0]}, index=["Free Cash Flow"]
    ),
    ("cashflow", "quarterly"): pd.DataFrame(),
}


def test_ingest_us_financials_upserts_all_statement_types_for_all_companies(tmp_path, monkeypatch):
    db = str(tmp_path / "usf1.db")
    init_db(db)
    conn = connect(db)
    seed_us_companies(conn, ["AAPL"])
    conn.close()

    monkeypatch.setattr(us_financials, "send_slack_alert", lambda *a, **kw: None)
    result = ingest_us_financials(db_path=db, fetch_statements=lambda code: _SAMPLE_STATEMENTS)

    conn = connect(db)
    rows = conn.execute(
        "SELECT statement_type, period_type, item_key, item_value, source FROM us_financials "
        "ORDER BY statement_type, period_type"
    ).fetchall()
    conn.close()
    combos = {(r["statement_type"], r["period_type"]) for r in rows}
    assert combos == {
        ("income_stmt", "annual"),
        ("income_stmt", "quarterly"),
        ("balance_sheet", "annual"),
        ("cashflow", "annual"),
    }
    assert all(r["source"] == "yfinance" for r in rows)
    assert result["succeeded"] == 1
    assert result["failed"] == []


class _CommitCountingConn:
    """conn.commit() 호출 횟수를 세는 프록시. sqlite3.Connection은 속성 재할당이 안 되므로
    얇은 위임 프록시로 감싼다(price_history_batch 테스트의 _CountingConn과 동일 패턴)."""

    def __init__(self, real):
        self._real = real
        self.commit_calls = 0

    def execute(self, *a, **kw):
        return self._real.execute(*a, **kw)

    def commit(self):
        self.commit_calls += 1
        self._real.commit()

    def close(self):
        self._real.close()


def test_ingest_us_financials_commits_periodically_not_only_at_end(tmp_path, monkeypatch):
    """수천 종목 규모 백필 중 프로세스가 죽어도 그때까지 진행이 날아가지 않도록, 전체 루프가
    끝난 뒤 딱 1번이 아니라 commit_every개 종목마다 주기적으로 commit해야 한다."""
    db = str(tmp_path / "usf_commit.db")
    init_db(db)
    conn = connect(db)
    seed_us_companies(conn, ["AAA", "BBB", "CCC", "DDD", "EEE"])
    conn.close()

    monkeypatch.setattr(us_financials, "send_slack_alert", lambda *a, **kw: None)
    real_connect = us_financials.connect
    counting = {}

    def fake_connect(db_path=None):
        c = _CommitCountingConn(real_connect(db_path))
        counting["conn"] = c
        return c

    monkeypatch.setattr(us_financials, "connect", fake_connect)
    ingest_us_financials(
        db_path=db, fetch_statements=lambda code: _SAMPLE_STATEMENTS, commit_every=2,
    )
    # 종목 5개, commit_every=2 → 중간 커밋 최소 2회(2번째/4번째 종목 후) + 마지막 1회 > 1
    assert counting["conn"].commit_calls > 1


def test_ingest_us_financials_codes_param_restricts_to_given_subset(tmp_path, monkeypatch):
    """codes를 지정하면 us_company 전종목이 아니라 그 목록만 처리한다(중단 후 재시작 시
    이미 수집된 종목을 건너뛰고 이어서 진행하기 위함)."""
    db = str(tmp_path / "usf_codes.db")
    init_db(db)
    conn = connect(db)
    seed_us_companies(conn, ["AAA", "BBB", "CCC"])
    conn.close()

    monkeypatch.setattr(us_financials, "send_slack_alert", lambda *a, **kw: None)
    fetched = []

    def fake_fetch(code):
        fetched.append(code)
        return _SAMPLE_STATEMENTS

    result = ingest_us_financials(db_path=db, fetch_statements=fake_fetch, codes=["BBB"])
    assert fetched == ["BBB"]
    assert result["tickers"] == 1
    assert result["succeeded"] == 1


def test_normalize_financial_statement_quarterly_disclosed_date_is_45_days_after_period_end():
    # yfinance는 실제 공시일을 주지 않으므로 보수적 고정지연으로 근사한다:
    # quarterly는 분기말일 + 45일(SEC 10-Q 제출기한 근사).
    df = pd.DataFrame({pd.Timestamp("2025-06-30"): [100.0]}, index=["Total Revenue"])
    rows = normalize_financial_statement("AAPL", "income_stmt", "quarterly", df)
    assert rows[0]["disclosed_date"] == "2025-08-14"  # 2025-06-30 + 45일


def test_normalize_financial_statement_annual_disclosed_date_is_90_days_after_fiscal_year_end():
    # annual은 회계연도말 + 90일(SEC 10-K 제출기한 근사).
    df = pd.DataFrame({pd.Timestamp("2025-09-30"): [100.0]}, index=["Total Revenue"])
    rows = normalize_financial_statement("AAPL", "income_stmt", "annual", df)
    assert rows[0]["disclosed_date"] == "2025-12-29"  # 2025-09-30 + 90일


def test_ingest_us_financials_persists_disclosed_date(tmp_path, monkeypatch):
    db = str(tmp_path / "usf_disc.db")
    init_db(db)
    conn = connect(db)
    seed_us_companies(conn, ["AAPL"])
    conn.close()

    monkeypatch.setattr(us_financials, "send_slack_alert", lambda *a, **kw: None)
    statements = {
        ("income_stmt", "quarterly"): pd.DataFrame(
            {pd.Timestamp("2025-06-30"): [90_000_000_000.0]}, index=["Total Revenue"]
        ),
    }
    ingest_us_financials(db_path=db, fetch_statements=lambda code: statements)

    conn = connect(db)
    row = conn.execute(
        "SELECT disclosed_date FROM us_financials WHERE stock_code='AAPL' AND period_type='quarterly'"
    ).fetchone()
    conn.close()
    assert row["disclosed_date"] == "2025-08-14"


def test_init_db_migrates_us_financials_disclosed_date_column(tmp_path):
    import sqlite3

    db = str(tmp_path / "old_us.db")
    conn = sqlite3.connect(db)
    # 구 스키마(disclosed_date 컬럼 없음) 재현 — 기존 DB 무중단 이행을 검증
    conn.execute(
        "CREATE TABLE us_financials (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "stock_code TEXT NOT NULL, as_of_date TEXT NOT NULL, period_type TEXT NOT NULL, "
        "statement_type TEXT NOT NULL, item_key TEXT NOT NULL, item_value REAL, "
        "source TEXT, collected_at TEXT, "
        "UNIQUE(stock_code, as_of_date, period_type, statement_type, item_key))"
    )
    conn.commit()
    conn.close()

    init_db(db)

    conn = sqlite3.connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(us_financials)")}
    conn.close()
    assert "disclosed_date" in cols


def test_ingest_us_financials_skips_failing_ticker_and_alerts(tmp_path, monkeypatch):
    db = str(tmp_path / "usf2.db")
    init_db(db)
    conn = connect(db)
    seed_us_companies(conn, ["AAPL", "MSFT"])
    conn.close()

    def failing_fetch(code):
        if code == "AAPL":
            raise ConnectionError("AAPL 요청 실패(mock)")
        return _SAMPLE_STATEMENTS

    alerts: list[str] = []
    monkeypatch.setattr(us_financials, "send_slack_alert", lambda msg, **kw: alerts.append(msg))
    result = ingest_us_financials(db_path=db, fetch_statements=failing_fetch)

    assert result["failed"] == ["AAPL"]
    assert result["succeeded"] == 1
    assert len(alerts) == 1

    conn = connect(db)
    rows = conn.execute("SELECT DISTINCT stock_code FROM us_financials").fetchall()
    conn.close()
    assert [r["stock_code"] for r in rows] == ["MSFT"]
