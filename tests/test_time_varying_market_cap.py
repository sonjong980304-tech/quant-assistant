"""시점별 상장주식수 기반 시가총액 계산 (look-ahead 방지 + staleness + 이상치 가드).

financials 에 분기별 shares_outstanding 이 disclosed_date 와 함께 적재돼 있으므로,
그 시점 값을 조회해 market_cap = 종가 × 그 시점 상장주식수 로 계산한다
(_shares_outstanding_at, src/ingest/metrics.py). scripts/backfill_marketcap.py가
이 함수로 prices.market_cap을 일괄 재계산하는 운영 경로다.

US(SEC) 선례 data_access_us_sec._shares_outstanding_at 과 동일 사상의 KR 버전이다.
"""
from __future__ import annotations

from src.db import connect, init_db
from src.ingest.metrics import _shares_outstanding_at


def _conn(tmp_path):
    db = str(tmp_path / "shares.db")
    init_db(db)
    return connect(db)


def _seed_shares(conn, code, rows):
    """rows: [(quarter, disclosed_date, amount)]"""
    for q, disc, amt in rows:
        conn.execute(
            "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, account_name, amount) "
            "VALUES (?,?,?,'shares_outstanding','상장주식수',?)",
            (code, q, disc, amt),
        )
    conn.commit()


# ── _shares_outstanding_at: look-ahead 방지 ──────────────────────────────────

def test_shares_at_returns_latest_disclosed_before_asof(tmp_path):
    """asof 시점까지 공시된(disclosed_date<=asof) 최신 상장주식수. 삼성전자 50:1 분할 경계."""
    conn = _conn(tmp_path)
    _seed_shares(conn, "005930", [
        ("2017Q4", "2018-04-02", 129_098_494),
        ("2018Q1", "2018-05-15", 128_386_494),   # 분할 전
        ("2018Q2", "2018-08-14", 6_419_324_700),  # 50:1 액면분할 후
        ("2018Q3", "2018-11-14", 6_419_324_700),
    ])
    # 분할 공시(2018-08-14) 전 → 분할 전 주식수(2018Q1)
    assert _shares_outstanding_at(conn, "005930", "2018-06-01") == 128_386_494
    # 분할 공시 후 → 분할 후 주식수(2018Q2). 미래 공시가 새어나오지 않음
    assert _shares_outstanding_at(conn, "005930", "2018-09-01") == 6_419_324_700


def test_shares_at_none_before_first_disclosure(tmp_path):
    """asof 가 첫 공시보다 이르면 look-ahead 방지로 None(미래값 안 씀)."""
    conn = _conn(tmp_path)
    _seed_shares(conn, "005930", [("2018Q2", "2018-08-14", 6_419_324_700)])
    assert _shares_outstanding_at(conn, "005930", "2018-06-01") is None


def test_shares_at_missing_returns_none(tmp_path):
    conn = _conn(tmp_path)
    assert _shares_outstanding_at(conn, "999999", "2020-06-01") is None


# ── _shares_outstanding_at: staleness 폴백 ────────────────────────────────────

def test_shares_at_stale_returns_none(tmp_path):
    """마지막 공시가 asof 대비 너무 오래(>400일)면 None(3단계에서 상수 폴백)."""
    conn = _conn(tmp_path)
    _seed_shares(conn, "111111", [("2018Q2", "2018-08-14", 1_000_000)])
    assert _shares_outstanding_at(conn, "111111", "2020-06-01") is None  # ~657일 경과


# ── _shares_outstanding_at: 이상치(×1000 단위오류 원복형 스파이크) 가드 ──────────

def test_shares_at_rejects_transient_1000x_spike(tmp_path):
    """한 분기만 ×1000 튀었다 원복하는 단위오류(003240 실측 패턴)는 이상치로 None."""
    conn = _conn(tmp_path)
    _seed_shares(conn, "003240", [
        ("2021Q1", "2021-05-15", 1_113_400),
        ("2021Q2", "2021-08-14", 1_113_400_000),  # ×1000 오류
        ("2021Q3", "2021-11-14", 1_113_400),
        ("2021Q4", "2022-04-01", 1_113_400),
    ])
    # 스파이크 공시 직후 asof → 이상치로 None
    assert _shares_outstanding_at(conn, "003240", "2021-09-01") is None
    # 정상 분기 → 정상값(가드가 정상값까지 죽이지 않음)
    assert _shares_outstanding_at(conn, "003240", "2021-12-01") == 1_113_400


# ── 마이그레이션(backfill_marketcap): 재계산 + 확정 스파이크 오염 정리 ──────────

def _seed_price(conn, code, d, close, market_cap):
    conn.execute(
        "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,?)",
        (code, d, close, market_cap),
    )
    conn.commit()


def test_backfill_marketcap_recomputes_time_varying(tmp_path):
    """저장된 market_cap 이 시점별 값과 다르면 그 시점 주식수로 재계산(UPDATE)."""
    from scripts.backfill_marketcap import backfill_marketcap

    db_path = str(tmp_path / "m.db")
    init_db(db_path)
    conn = connect(db_path)
    _seed_shares(conn, "005930", [
        ("2024Q1", "2024-05-15", 1_000_000),
        ("2024Q2", "2024-08-14", 2_000_000),
    ])
    # 잘못된(옛 상수) cap 저장 → 시점별 값으로 교정돼야 한다
    _seed_price(conn, "005930", "2024-06-28", 100.0, 999_999_900)  # 틀린 값
    conn.close()

    backfill_marketcap(db_path=db_path)

    conn = connect(db_path)
    row = conn.execute(
        "SELECT market_cap FROM prices WHERE stock_code='005930' AND date='2024-06-28'"
    ).fetchone()
    assert row["market_cap"] == round(100.0 * 1_000_000)  # 그 시점(1M주)로 교정


def test_backfill_marketcap_nulls_confirmed_spike_pollution(tmp_path):
    """구버전 무가드 backfill 이 써둔 ×1000 스파이크 오염(저장 cap==종가×스파이크주식수)은 NULL 로 정리."""
    from scripts.backfill_marketcap import backfill_marketcap

    db_path = str(tmp_path / "m.db")
    init_db(db_path)
    conn = connect(db_path)
    _seed_shares(conn, "003240", [
        ("2021Q3", "2021-11-15", 1_113_400),
        ("2021Q4", "2022-02-25", 1_113_400_000),  # ×1000 단위오류(실측)
        ("2021Q4R", "2022-03-17", 1_113_400),       # 정정 원복
    ])
    # 정상 행(스파이크 전): 이미 올바른 시점별 cap → 유지
    _seed_price(conn, "003240", "2021-12-01", 1_000_000, round(1_000_000 * 1_113_400))
    # 오염 행(스파이크 공시 직후): 저장 cap == 종가×스파이크주식수 → NULL 로 정리돼야
    _seed_price(conn, "003240", "2022-03-01", 1_000_000, round(1_000_000 * 1_113_400_000))
    conn.close()

    backfill_marketcap(db_path=db_path)

    conn = connect(db_path)
    normal = conn.execute(
        "SELECT market_cap FROM prices WHERE stock_code='003240' AND date='2021-12-01'"
    ).fetchone()
    polluted = conn.execute(
        "SELECT market_cap FROM prices WHERE stock_code='003240' AND date='2022-03-01'"
    ).fetchone()
    assert normal["market_cap"] == round(1_000_000 * 1_113_400)  # 정상값 유지
    assert polluted["market_cap"] is None  # ×1000 오염 → NULL 정리


def _seed_daily_shares(conn, code, rows):
    """rows: [(date, shares_outstanding)]"""
    for d, shares in rows:
        conn.execute(
            "INSERT INTO daily_shares(stock_code, date, shares_outstanding, source) VALUES (?,?,?,'pykrx')",
            (code, d, shares),
        )
    conn.commit()


def test_backfill_marketcap_prefers_daily_shares_over_financials(tmp_path):
    """분기 결산일 '사이'에 액면분할이 나면 financials는 분할 전 값에 멈춰 있지만(최대 3개월
    지연), daily_shares는 그 날짜에 실제 발효 중이던 분할 후 값을 즉시 반영한다 — 이 시차가
    바로 daily_shares 테이블을 새로 만든 이유(원래 버그)이므로, 두 소스가 서로 다른 값을 줄 때
    daily_shares가 이긴다는 것을 회귀로 고정한다."""
    from scripts.backfill_marketcap import backfill_marketcap

    db_path = str(tmp_path / "m.db")
    init_db(db_path)
    conn = connect(db_path)
    # financials: 분할 전 분기값(다음 분기 보고서 전까지 정체)
    _seed_shares(conn, "005930", [("2024Q1", "2024-05-15", 1_000_000)])
    # daily_shares: 같은 날짜에 이미 분할이 반영된 실제값(50:1 분할)
    _seed_daily_shares(conn, "005930", [("2024-06-28", 50_000_000)])
    _seed_price(conn, "005930", "2024-06-28", 100.0, None)
    conn.close()

    backfill_marketcap(db_path=db_path)

    conn = connect(db_path)
    row = conn.execute(
        "SELECT market_cap FROM prices WHERE stock_code='005930' AND date='2024-06-28'"
    ).fetchone()
    assert row["market_cap"] == round(100.0 * 50_000_000)  # financials(1M)이 아닌 daily_shares(50M)


def test_backfill_marketcap_falls_back_to_financials_when_no_daily_shares(tmp_path):
    """daily_shares에 그 종목 데이터가 아예 없으면(백필 미대상 등) 기존 financials 경로로
    폴백한다 — daily_shares 도입이 기존 동작을 깨지 않는다는 회귀 확인."""
    from scripts.backfill_marketcap import backfill_marketcap

    db_path = str(tmp_path / "m.db")
    init_db(db_path)
    conn = connect(db_path)
    _seed_shares(conn, "005930", [("2024Q1", "2024-05-15", 1_000_000)])
    # daily_shares 행 없음
    _seed_price(conn, "005930", "2024-06-28", 100.0, 999_999_900)  # 틀린 값
    conn.close()

    backfill_marketcap(db_path=db_path)

    conn = connect(db_path)
    row = conn.execute(
        "SELECT market_cap FROM prices WHERE stock_code='005930' AND date='2024-06-28'"
    ).fetchone()
    assert row["market_cap"] == round(100.0 * 1_000_000)  # financials 폴백값으로 교정


def test_backfill_marketcap_dry_run_makes_no_change(tmp_path):
    """dry_run=True 는 UPDATE 하지 않고 규모만 집계한다."""
    from scripts.backfill_marketcap import backfill_marketcap

    db_path = str(tmp_path / "m.db")
    init_db(db_path)
    conn = connect(db_path)
    _seed_shares(conn, "005930", [("2024Q1", "2024-05-15", 1_000_000)])
    _seed_price(conn, "005930", "2024-06-28", 100.0, 999_999_900)  # 틀린 값
    conn.close()

    report = backfill_marketcap(db_path=db_path, dry_run=True)

    conn = connect(db_path)
    row = conn.execute(
        "SELECT market_cap FROM prices WHERE stock_code='005930' AND date='2024-06-28'"
    ).fetchone()
    assert row["market_cap"] == 999_999_900  # 변경 없음
    assert report["mode"] == "dry-run"
    assert report["market_cap_changed"] == 1  # 바뀔 행 수는 집계
