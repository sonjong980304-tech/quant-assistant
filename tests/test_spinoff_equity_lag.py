"""물적분할(스핀오프) 시차 가드(src/data_quality.is_equity_ratio_anomalous) 검증.

배경
----
물적분할이 나면 주가(=시가총액)는 시장에서 즉시 급락 반영되지만, 자기자본(재무제표)은
다음 분기 DART 공시가 나와야 갱신된다. 그 시차 구간엔 effective_quarter_at 이 여전히
"분할 전(자회사 포함) 큰 자기자본"을 쓰므로 PBR = 시총 ÷ 자기자본 이 비정상적으로 낮게
계산돼 저평가 오탐이 난다. 다음 분기 재무제표가 공시되면 자기자본이 줄어 자동 정상화된다.

기존 블랭킷 가격이상치 가드(detect_price_quality_anomalies, 종목 전체 영구 제외)와 달리
이 가드는:
  · (종목, 특정 asof) 조합에서만,
  · 시총·재무 결합 밸류에이션 배수(PBR/PSR/PER 등)만 결측 처리하고,
  · 다음 분기 공시가 반영되면 별도 해제 로직 없이 자연히 정상 계산으로 돌아온다.

이 파일은 2개 층을 검증한다:
1) is_equity_ratio_anomalous: 순수 판정 함수(급락 감지 / 자기자본 갱신 후 정상화 / 정상종목·
   데이터부족 시 False).
2) 배선: metrics_at 이 걸린 시점의 PBR/PSR/PER 만 None 으로 만들고, 가격전용(return_12m)·
   순수 재무비율(debt_ratio)은 건드리지 않으며, 다음 분기 공시 후엔 PBR 이 정상 계산되는지.
"""
from __future__ import annotations

import sqlite3

from src.backtest.data_access import metrics_at
from src.data_quality import is_equity_ratio_anomalous
from src.db import init_db

# 물적분할 시뮬레이션 종목의 공시 일정(분기 → 공시일).
_DISCLOSED = {
    "2022Q1": "2022-05-15",
    "2022Q2": "2022-08-15",
    "2022Q3": "2022-11-15",
    "2022Q4": "2023-03-31",
    "2023Q1": "2023-05-15",
    "2023Q2": "2023-08-15",
}
# 분할 전 분기는 자회사 포함 큰 자기자본(1000), 분할이 반영된 2023Q2부터 작은 자기자본(400).
_EQUITY = {q: (400.0 if q == "2023Q2" else 1000.0) for q in _DISCLOSED}

# close == market_cap 로 두고(1주 가정) 시총이 분할로 1000→400 으로 "여러 날에 걸쳐" 하락.
# (하루 만에 반토막 나면 블랭킷 가격이상치 가드가 종목 전체를 빼버리므로, 인접일 종가비가
#  항상 0.5 초과가 되게 완만히 계단식으로 내린다.)
_PRICES = [
    ("2022-08-01", 1000.0),  # return_12m(1년 전) 기준점 겸 분할 전
    ("2023-04-25", 1000.0),
    ("2023-05-01", 1000.0),  # 스테일 asof(2023-08-01)의 3개월 전 기준점
    ("2023-05-20", 1000.0),  # 정상화 asof(2023-08-20)의 3개월 전 기준점
    ("2023-07-03", 1000.0),
    ("2023-07-05", 850.0),
    ("2023-07-07", 720.0),
    ("2023-07-11", 610.0),
    ("2023-07-13", 520.0),
    ("2023-07-17", 440.0),
    ("2023-07-19", 400.0),
    ("2023-07-31", 400.0),
    ("2023-08-01", 400.0),  # 스테일 구간(2023Q2 공시 2023-08-15 이전)
    ("2023-08-15", 400.0),
    ("2023-08-20", 400.0),  # 2023Q2 공시 이후(자기자본 400 반영 → 정상화)
    ("2023-08-25", 400.0),
]


def _conn(tmp_path, name="spinoff.db") -> sqlite3.Connection:
    db = tmp_path / name
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def _seed_spinoff_company(conn, code="000010", name="분할테스트"):
    conn.execute(
        "INSERT INTO company(stock_code, name, market, sector) VALUES (?,?,?,?)",
        (code, name, "KOSPI", "전기·전자"),
    )
    for q, disclosed in _DISCLOSED.items():
        eq = _EQUITY[q]
        rows = [
            ("controlling_equity", eq),
            ("total_equity", eq),
            ("net_income", 50.0),
            ("controlling_net_income", 50.0),
            ("revenue", 300.0),
            ("total_liabilities", 500.0),
        ]
        for key, amount in rows:
            conn.execute(
                "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
                "VALUES (?,?,?,?,?)",
                (code, q, disclosed, key, amount),
            )
    for d, v in _PRICES:
        conn.execute(
            "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,?)",
            (code, d, v, v),
        )
    conn.commit()


def _seed_stable_company(conn, code="000020", name="안정전자"):
    """시총·자기자본이 모두 안정적인 정상 종목(오탐 회귀용)."""
    conn.execute(
        "INSERT INTO company(stock_code, name, market, sector) VALUES (?,?,?,?)",
        (code, name, "KOSPI", "전기·전자"),
    )
    for q, disclosed in _DISCLOSED.items():
        for key, amount in (("controlling_equity", 1000.0), ("total_equity", 1000.0)):
            conn.execute(
                "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
                "VALUES (?,?,?,?,?)",
                (code, q, disclosed, key, amount),
            )
    for d, _ in _PRICES:
        conn.execute(
            "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,?)",
            (code, d, 1000.0, 1000.0),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# 1) is_equity_ratio_anomalous: 순수 판정
# ---------------------------------------------------------------------------
def test_flags_spinoff_lag_window(tmp_path):
    """스테일 구간: 시총 400/유효 자기자본 1000(분할 전) → 비율 0.4, 3개월 전 1.0 대비 급락 → True."""
    conn = _conn(tmp_path)
    _seed_spinoff_company(conn)
    assert is_equity_ratio_anomalous(conn, "000010", "2023-08-01") is True
    conn.close()


def test_not_flagged_after_next_quarter_disclosed(tmp_path):
    """2023Q2 공시 후: 자기자본 400 반영 → 비율 1.0 = 3개월 전 1.0, 급변 없음 → False(자동 정상화)."""
    conn = _conn(tmp_path)
    _seed_spinoff_company(conn)
    assert is_equity_ratio_anomalous(conn, "000010", "2023-08-20") is False
    conn.close()


def test_not_flagged_for_stable_stock(tmp_path):
    """시총·자기자본이 안정적인 종목은 어느 시점에서도 급변이 없어 False(오탐 회귀)."""
    conn = _conn(tmp_path)
    _seed_stable_company(conn)
    assert is_equity_ratio_anomalous(conn, "000020", "2023-08-01") is False
    conn.close()


def test_not_flagged_when_reference_missing(tmp_path):
    """3개월 전 기준 데이터(가격/공시)가 없으면 판정 불가 → 안전하게 False(결측 처리 안 함)."""
    conn = _conn(tmp_path)
    _seed_spinoff_company(conn)
    # 최초 상장 직후처럼 3개월 전에 가격·재무가 전무한 시점.
    assert is_equity_ratio_anomalous(conn, "000010", "2021-01-15") is False
    conn.close()


# ---------------------------------------------------------------------------
# 2) 배선: metrics_at 이 걸린 시점의 밸류에이션 배수만 결측 처리
# ---------------------------------------------------------------------------
def _row(rows, code):
    return next((r for r in rows if r["stock_code"] == code), None)


def test_metrics_at_nulls_valuation_multiples_during_lag(tmp_path):
    """스테일 구간: PBR/PSR/PER 은 None, 가격전용(return_12m)·순수 재무비율(debt_ratio)은 정상."""
    conn = _conn(tmp_path)
    _seed_spinoff_company(conn)
    row = _row(metrics_at(conn, "2023-08-01"), "000010")
    assert row is not None  # 유니버스에서 완전히 빠지는 게 아니다
    assert row["pbr"] is None
    assert row["psr"] is None
    assert row["per"] is None
    # 가격만 쓰는 지표(모멘텀)와 순수 재무비율(시총 불사용)은 영향받지 않는다.
    assert row["return_12m"] is not None
    assert row["debt_ratio"] is not None
    conn.close()


def test_metrics_at_pbr_recovers_after_next_quarter(tmp_path):
    """2023Q2 공시 후: 자기자본 400 반영 → PBR/PSR/PER 이 다시 정상 계산된다(자동 정상화)."""
    conn = _conn(tmp_path)
    _seed_spinoff_company(conn)
    row = _row(metrics_at(conn, "2023-08-20"), "000010")
    assert row is not None
    assert row["pbr"] is not None
    assert row["psr"] is not None
    assert row["per"] is not None
    conn.close()
