"""scripts/backfill_us_financial_currency.py 검증 (TDD).

배경: SK텔레콤(SKM, NYSE 상장 ADR)의 us_financials 원본 숫자가 원화(KRW) 단위였다
(yfinance financialCurrency='KRW'). 주가(us_prices)는 항상 달러로 거래되지만, 외국기업
ADR은 실적발표 자체를 본국통화로 하는 경우가 흔해 주가통화와 재무제표통화가 다를 수
있다. 이 스크립트는 us_financials에 이미 재무데이터가 존재하는(=실제 계산에 쓰이는)
종목만 대상으로 yfinance financialCurrency를 수집해 us_company.financial_currency에
캐싱한다(scripts/backfill_us_security_type.py와 동일한 "판단은 API, 실행은 캐싱" 원칙 —
매 스크리닝 요청마다 yfinance를 부르지 않는다).

검증 대상:
- backfill_financial_currency: financial_currency가 NULL이고 us_financials에 데이터가
  있는 종목만 대상으로 하고(재무데이터 없는 종목은 스크리닝에 안 쓰이니 수집 불필요),
  이미 채워진 종목은 재수집하지 않으며(idempotent), USD/비USD/실패 카운트를 리포트로
  반환한다. fetch_currency_fn은 DI(테스트는 실제 yfinance 네트워크 호출 없음).
- 실패(예외/빈 응답)는 해당 종목만 격리하고 나머지는 계속 진행한다(us_financials.py의
  종목별 try/except 격리 관례와 동일).
- yfinance 호출에 인위적 sleep을 추가하지 않는다(src/ingest/us_financials.py/us_prices.py의
  기존 yfinance 호출 관례 — 이 코드베이스에서 스로틀은 Naver/FnGuide HTTP 크롤러
  (ThrottledFetcher)에만 있고 yfinance 배치엔 없다. 새 스로틀을 발명하지 않는다).
"""
from __future__ import annotations

from scripts.backfill_us_financial_currency import backfill_financial_currency
from src.db import connect, init_db


def _seed(tmp_path, companies: list[tuple[str, str, str | None]]) -> str:
    """companies: [(stock_code, name, financial_currency|None), ...].

    각 종목에 us_financials 행을 최소 1개씩 심는다(대상 조건: 재무데이터 존재).
    """
    db = str(tmp_path / "cur.db")
    init_db(db)
    conn = connect(db)
    for code, name, currency in companies:
        conn.execute(
            "INSERT INTO us_company(stock_code, name, exchange, financial_currency) "
            "VALUES (?,?,?,?)",
            (code, name, "NYSE", currency),
        )
        conn.execute(
            "INSERT INTO us_financials(stock_code, as_of_date, period_type, statement_type, "
            "item_key, item_value, source) VALUES (?,?,?,?,?,?,?)",
            (code, "2026-03-31", "quarterly", "income_stmt", "Total Revenue", 100.0, "yfinance"),
        )
    conn.commit()
    conn.close()
    return db


def test_backfill_targets_only_codes_with_financials_and_no_currency_yet(tmp_path):
    db = _seed(tmp_path, [("NVDA", "NVIDIA", None), ("SKM", "SK Telecom", None)])
    # 재무데이터가 없는 종목(NOFIN)은 대상에서 제외돼야 한다.
    conn = connect(db)
    conn.execute(
        "INSERT INTO us_company(stock_code, name, exchange, financial_currency) VALUES (?,?,?,?)",
        ("NOFIN", "No Financials Co", "NYSE", None),
    )
    conn.commit()
    conn.close()

    calls = []

    def fake_fetch(code: str) -> str:
        calls.append(code)
        return {"NVDA": "USD", "SKM": "KRW"}[code]

    report = backfill_financial_currency(db_path=db, fetch_currency_fn=fake_fetch)

    assert set(calls) == {"NVDA", "SKM"}  # NOFIN 호출 안 됨
    assert report["total_targets"] == 2
    assert report["usd"] == 1
    assert report["non_usd"] == 1
    assert report["failed"] == 0
    assert report["currencies"] == {"USD": 1, "KRW": 1}


def test_backfill_writes_currency_to_us_company(tmp_path):
    db = _seed(tmp_path, [("SKM", "SK Telecom", None)])
    backfill_financial_currency(db_path=db, fetch_currency_fn=lambda code: "KRW")
    conn = connect(db)
    row = conn.execute("SELECT financial_currency FROM us_company WHERE stock_code='SKM'").fetchone()
    assert row["financial_currency"] == "KRW"
    conn.close()


def test_backfill_is_idempotent_skips_already_filled(tmp_path):
    db = _seed(tmp_path, [("NVDA", "NVIDIA", "USD"), ("SKM", "SK Telecom", None)])
    calls = []

    def fake_fetch(code: str) -> str:
        calls.append(code)
        return "KRW"

    report = backfill_financial_currency(db_path=db, fetch_currency_fn=fake_fetch)
    assert calls == ["SKM"]  # NVDA는 이미 채워져 있어 재수집 안 함
    assert report["total_targets"] == 1


def test_backfill_isolates_failure_per_code_and_continues(tmp_path):
    db = _seed(tmp_path, [("BAD", "Bad Co", None), ("NVDA", "NVIDIA", None)])

    def flaky_fetch(code: str) -> str:
        if code == "BAD":
            raise RuntimeError("yfinance 요청 실패")
        return "USD"

    report = backfill_financial_currency(db_path=db, fetch_currency_fn=flaky_fetch)
    assert report["failed"] == 1
    assert "BAD" in report["failed_codes"]
    assert report["usd"] == 1  # NVDA는 정상 처리됨(BAD 실패가 전체를 막지 않음)

    conn = connect(db)
    row = conn.execute("SELECT financial_currency FROM us_company WHERE stock_code='BAD'").fetchone()
    assert row["financial_currency"] is None  # 실패 종목은 NULL 유지(다음 실행에 재시도)
    conn.close()


def test_backfill_none_response_counts_as_failure(tmp_path):
    """financialCurrency 자체가 없는(None) 응답은 실패로 집계해 다음 실행에 재시도한다."""
    db = _seed(tmp_path, [("WEIRD", "Weird Co", None)])
    report = backfill_financial_currency(db_path=db, fetch_currency_fn=lambda code: None)
    assert report["failed"] == 1
    assert report["usd"] == 0
    assert report["non_usd"] == 0


def test_backfill_no_targets_returns_zero_report(tmp_path):
    db = _seed(tmp_path, [("NVDA", "NVIDIA", "USD")])  # 이미 다 채워져 있음
    report = backfill_financial_currency(db_path=db, fetch_currency_fn=lambda code: "USD")
    assert report["total_targets"] == 0
    assert report["usd"] == 0
