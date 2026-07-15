"""액면분할/병합 미반영(수정주가 누락)으로 생긴 종가 불연속을 정수배로 보정하는
scripts/fix_split_discontinuities.py 검증.

배경: prices.close에 "정수배(2x/10x/20x…)로 하루 만에 점프"하는 구간이 있는 종목은
과거 분할/병합 이벤트가 수정주가로 소급 반영되지 않은 데이터 버그다. split_date 이전
모든 close에 ratio를 곱하면(과거를 현재 기준으로 소급 조정) 진짜 수정주가가 된다.

이 테스트는 tmp DB에 축소 재현한 패턴으로:
  (a) 정수배 점프 탐지, (b) 비정수배는 별도 그룹 분류(보정 대상 아님),
  (c) 보정 후 점프 소멸(연속 시계열), (d) 정상 종목 회귀(무변경),
  (e) 종목당 이벤트 2개 이상 케이스 누적 적용, (f) market_cap 재계산,
  (g) 드라이런은 DB 미변경.
"""
from __future__ import annotations

from datetime import date, timedelta

from scripts.fix_split_discontinuities import (
    apply_corrections,
    detect_splits,
    main,
    match_round_ratio,
)
from src.db import connect, init_db


def _seed_series(conn, code, segments, start="2016-01-01"):
    """segments=[(n, close), ...]를 연속 거래일로 심는다. 각 세그먼트의 첫 날짜 리스트 반환."""
    d = date.fromisoformat(start)
    boundaries = []
    for n, close in segments:
        boundaries.append(d.isoformat())
        for _ in range(n):
            conn.execute(
                "INSERT INTO prices(stock_code, date, close) VALUES (?,?,?)",
                (code, d.isoformat(), float(close)),
            )
            d += timedelta(days=1)
    conn.commit()
    return boundaries


def _closes(conn, code):
    rows = conn.execute(
        "SELECT close FROM prices WHERE stock_code=? ORDER BY date", (code,)
    ).fetchall()
    return [r["close"] for r in rows]


# ---------------------------------------------------------------------------
# match_round_ratio: 정수배 판정
# ---------------------------------------------------------------------------
def test_match_round_ratio_exact_and_within_tolerance():
    assert match_round_ratio(20.0) == 20
    assert match_round_ratio(10.0) == 10
    # 8% 이내 근사 (42.53 → 40)
    assert match_round_ratio(42.53) == 40
    # 하락 방향(1/ratio) 매칭은 이 임계값(ratio>2 상승만)에선 발생하지 않음


def test_match_round_ratio_non_integer_returns_none():
    assert match_round_ratio(67000.0) is None
    assert match_round_ratio(3.5) is None  # 3과 4 사이, 8% 밖
    assert match_round_ratio(1.5) is None


# ---------------------------------------------------------------------------
# detect_splits: 탐지·분류
# ---------------------------------------------------------------------------
def test_detect_finds_reverse_split(tmp_path):
    db = str(tmp_path / "m.db")
    init_db(db)
    conn = connect(db)
    # 017170(훈영) 축소: 105.0 저유동 구간 → 정확히 20배(2100.0) 점프
    bnds = _seed_series(conn, "S20", [(40, 105.0), (40, 2100.0)])
    det = detect_splits(conn)
    assert "S20" in det["integer_events"]
    evs = det["integer_events"]["S20"]
    assert len(evs) == 1
    assert evs[0].matched_ratio == 20
    assert abs(evs[0].ratio - 20.0) < 1e-9
    assert evs[0].date == bnds[1]  # 점프 행 날짜 = 새 세그먼트 첫 날
    assert "S20" not in det["other_jumps"]
    conn.close()


def test_detect_ignores_normal_stock(tmp_path):
    db = str(tmp_path / "m.db")
    init_db(db)
    conn = connect(db)
    # 완만한 상승(일 변동 < 100%)만 있는 정상 종목
    d = date(2016, 1, 1)
    for i in range(80):
        conn.execute(
            "INSERT INTO prices(stock_code, date, close) VALUES (?,?,?)",
            ("NORMAL", (d + timedelta(days=i)).isoformat(), 10000.0 + i * 10),
        )
    conn.commit()
    det = detect_splits(conn)
    assert "NORMAL" not in det["integer_events"]
    assert "NORMAL" not in det["other_jumps"]
    conn.close()


def test_detect_non_integer_goes_to_other_group(tmp_path):
    db = str(tmp_path / "m.db")
    init_db(db)
    conn = connect(db)
    # 008080(에스와이) 축소: 1.0 placeholder 고정 → 67000.0 (비정수배, 데이터오류 의심)
    _seed_series(conn, "OTHER67", [(30, 1.0), (30, 67000.0)])
    det = detect_splits(conn)
    assert "OTHER67" not in det["integer_events"]
    assert "OTHER67" in det["other_jumps"]
    assert len(det["other_jumps"]["OTHER67"]) >= 1
    conn.close()


def test_detect_multiple_integer_events(tmp_path):
    db = str(tmp_path / "m.db")
    init_db(db)
    conn = connect(db)
    # 103650류: 종목당 정수배 점프 2개 (×10, 그다음 ×5)
    _seed_series(conn, "MULTI", [(30, 10.0), (30, 100.0), (30, 500.0)])
    det = detect_splits(conn)
    assert "MULTI" in det["integer_events"]
    evs = det["integer_events"]["MULTI"]
    assert len(evs) == 2
    matched = sorted(e.matched_ratio for e in evs)
    assert matched == [5, 10]
    conn.close()


# ---------------------------------------------------------------------------
# apply_corrections: 실제 보정 (tmp DB — 실제 data/market.db 아님)
# ---------------------------------------------------------------------------
def test_apply_removes_jump_single(tmp_path):
    db = str(tmp_path / "m.db")
    init_db(db)
    conn = connect(db)
    _seed_series(conn, "S20", [(40, 105.0), (40, 2100.0)])
    det = detect_splits(conn)
    apply_corrections(conn, det["integer_events"])
    # 보정 후 전 구간 종가가 2100.0으로 연속(점프 소멸)
    assert set(_closes(conn, "S20")) == {2100.0}
    # 재탐지 시 점프가 사라져야 함
    assert "S20" not in detect_splits(conn)["integer_events"]
    conn.close()


def test_apply_leaves_normal_stock_unchanged(tmp_path):
    db = str(tmp_path / "m.db")
    init_db(db)
    conn = connect(db)
    _seed_series(conn, "S20", [(40, 105.0), (40, 2100.0)])
    d = date(2016, 1, 1)
    for i in range(80):
        conn.execute(
            "INSERT INTO prices(stock_code, date, close) VALUES (?,?,?)",
            ("NORMAL", (d + timedelta(days=i)).isoformat(), 10000.0 + i * 10),
        )
    conn.commit()
    before = _closes(conn, "NORMAL")
    det = detect_splits(conn)
    apply_corrections(conn, det["integer_events"])
    assert _closes(conn, "NORMAL") == before  # 회귀: 정상 종목 무변경
    conn.close()


def test_apply_multiple_events_cumulative(tmp_path):
    db = str(tmp_path / "m.db")
    init_db(db)
    conn = connect(db)
    # ×10 → ×5. 가장 오래된 구간은 ×50, 중간은 ×5, 최신은 ×1 (누적)
    _seed_series(conn, "MULTI", [(30, 10.0), (30, 100.0), (30, 500.0)])
    det = detect_splits(conn)
    apply_corrections(conn, det["integer_events"])
    # 누적 보정 후 전 구간이 500.0으로 연속
    assert set(_closes(conn, "MULTI")) == {500.0}
    conn.close()


def test_apply_recomputes_market_cap(tmp_path):
    db = str(tmp_path / "m.db")
    init_db(db)
    conn = connect(db)
    _seed_series(conn, "S20", [(40, 105.0), (40, 2100.0)])
    # 상장주식수 1000주가 모든 날짜 이전에 공시됨
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
        "VALUES (?,?,?,?,?)",
        ("S20", "2015Q1", "2015-01-01", "shares_outstanding", 1000.0),
    )
    conn.commit()
    det = detect_splits(conn)
    apply_corrections(conn, det["integer_events"])
    caps = conn.execute(
        "SELECT DISTINCT market_cap FROM prices WHERE stock_code='S20'"
    ).fetchall()
    # 보정 후 close=2100 × 1000주 = 2,100,000 (전 구간 동일)
    assert [c["market_cap"] for c in caps] == [2_100_000.0]
    conn.close()


def test_dry_run_does_not_write(tmp_path):
    db = str(tmp_path / "m.db")
    init_db(db)
    conn = connect(db)
    _seed_series(conn, "S20", [(40, 105.0), (40, 2100.0)])
    conn.close()
    # 드라이런(기본): DB 미변경
    main(["--db", db])
    conn = connect(db)
    assert 105.0 in _closes(conn, "S20")  # 원본 유지
    conn.close()
    # --apply: 실제 보정
    main(["--db", db, "--apply"])
    conn = connect(db)
    assert set(_closes(conn, "S20")) == {2100.0}
    conn.close()
