"""src/ingest/universe.py의 ingest_delisting() 단위 테스트 (TDD).

son-checker 이슈 #23 BUG-1: ingest_delisting()이 INSERT OR REPLACE를 쓰면서
delisting_date를 항상 빈 문자열("")로 채운다. 이 함수가 재실행되면 이미 정확히
채워진(수작업/별도 경로로 확보된) delisting_date를 전부 빈 문자열로 덮어써버려,
_is_alive()가 falsy 판정으로 항상 "살아있음"으로 오판하게 되고 생존편향이 재발한다.

실측 확인: 실제 운영 DB(data/market.db)의 delisting 580행은 전부 실제 날짜가 채워져
있다(빈 문자열 0건) — 이 함수가 재실행되는 순간 그 데이터가 파괴될 위험이 있었다.

pykrx는 이 환경에 설치돼 있지 않으므로(셸 기본 python3), sys.modules에 가짜 pykrx
모듈을 주입해 네트워크 없이 결정론적으로 재현한다.
"""
from __future__ import annotations

import sys
import types
from datetime import date

from src.db import connect, init_db
from src.ingest import universe


def _install_fake_pykrx(monkeypatch, past_tickers: dict, now_tickers: dict, names: dict | None = None):
    names = names or {}

    def get_market_ticker_list(on_date=None, market=None):
        # ingest_delisting은 now_str(=on 그대로, "%Y%m%d")로 현재 시점을 조회하고,
        # 그 외(과거로 years_back년 전 계산된 날짜)는 전부 과거 시점 조회다.
        table = now_tickers if on_date == "20260101" else past_tickers
        return list(table.get(market, []))

    def get_market_ticker_name(code):
        return names.get(code, "")

    fake_stock = types.SimpleNamespace(
        get_market_ticker_list=get_market_ticker_list,
        get_market_ticker_name=get_market_ticker_name,
    )
    fake_pykrx = types.ModuleType("pykrx")
    fake_pykrx.stock = fake_stock
    monkeypatch.setitem(sys.modules, "pykrx", fake_pykrx)
    monkeypatch.setitem(sys.modules, "pykrx.stock", fake_stock)


def _seed_db(tmp_path) -> str:
    db = tmp_path / "delisting.db"
    init_db(str(db))
    return str(db)


def test_ingest_delisting_preserves_existing_known_date(tmp_path, monkeypatch):
    """이미 정확한 delisting_date가 있는 종목은, 그 종목이 여전히 상폐 상태로 감지돼도
    ingest_delisting() 재실행으로 값이 지워지거나 덮어써지면 안 된다."""
    db_path = _seed_db(tmp_path)
    conn = connect(db_path)
    conn.execute(
        "INSERT INTO delisting(stock_code, name, delisting_date) VALUES(?,?,?)",
        ("000030", "우리은행", "2019-02-12"),
    )
    conn.commit()
    conn.close()

    # 과거 시점엔 상장돼 있었고, 현재 시점엔 없음 → 여전히 "상폐 추정" 집합에 포함됨.
    _install_fake_pykrx(
        monkeypatch,
        past_tickers={"KOSPI": ["000030"], "KOSDAQ": []},
        now_tickers={"KOSPI": [], "KOSDAQ": []},
        names={"000030": "우리은행"},
    )

    universe.ingest_delisting(db_path, years_back=10, on=date(2026, 1, 1))

    conn = connect(db_path)
    row = conn.execute(
        "SELECT delisting_date FROM delisting WHERE stock_code=?", ("000030",)
    ).fetchone()
    conn.close()
    assert row["delisting_date"] == "2019-02-12"


def test_ingest_delisting_adds_new_delisted_stock_with_null_date(tmp_path, monkeypatch):
    """delisting 테이블에 없던 신규 상폐종목은 여전히 추가돼야 하고, 정확한 날짜를
    모르므로 빈 문자열이 아니라 NULL로 채워야 한다(falsy 의미를 명확히)."""
    db_path = _seed_db(tmp_path)

    _install_fake_pykrx(
        monkeypatch,
        past_tickers={"KOSPI": ["999999"], "KOSDAQ": []},
        now_tickers={"KOSPI": [], "KOSDAQ": []},
        names={"999999": "신규상폐종목"},
    )

    result = universe.ingest_delisting(db_path, years_back=10, on=date(2026, 1, 1))

    assert result["delisted"] == 1
    conn = connect(db_path)
    row = conn.execute(
        "SELECT delisting_date FROM delisting WHERE stock_code=?", ("999999",)
    ).fetchone()
    conn.close()
    assert row is not None
    assert row["delisting_date"] is None


def test_ingest_delisting_does_not_add_still_listed_stocks(tmp_path, monkeypatch):
    """과거·현재 시점 모두 상장 중인 종목은 delisting에 추가되지 않는다(회귀)."""
    db_path = _seed_db(tmp_path)

    _install_fake_pykrx(
        monkeypatch,
        past_tickers={"KOSPI": ["005930"], "KOSDAQ": []},
        now_tickers={"KOSPI": ["005930"], "KOSDAQ": []},
        names={"005930": "삼성전자"},
    )

    result = universe.ingest_delisting(db_path, years_back=10, on=date(2026, 1, 1))

    assert result["delisted"] == 0
    conn = connect(db_path)
    row = conn.execute(
        "SELECT delisting_date FROM delisting WHERE stock_code=?", ("005930",)
    ).fetchone()
    conn.close()
    assert row is None
