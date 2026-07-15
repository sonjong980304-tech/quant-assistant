"""가격 데이터 품질 게이트(src/data_quality.py) 검증 — 종목 단위 블랭킷 제외.

배경: prices.close 불연속(액면분할/병합 미반영, 원본 파싱오류 등)을 배율로 "고치는"
접근(scripts/fix_split_discontinuities.py)은 246830 시냅스엠처럼 배율 하나로 안 끝나는
복합 케이스가 있고, 상승 점프만 잡던 기존 임계값(abs(ratio-1)>1.0)은 하락 폭락(예:
246830의 12400→50, ratio=0.004)을 전혀 못 잡는 사각지대가 있었다. "완벽 복원"보다
"신뢰 못 할 종목은 계산에서 아예 뺀다"가 더 안전하다는 방향전환에 따라, 대칭
탐지(ratio>=2.0 또는 ratio<=0.5)로 이상 종목 전체를 찾아 블랭킷 제외하는 함수를 검증한다.

이 파일은 3개 층을 검증한다:
1) detect_price_quality_anomalies: 순수 탐지 함수(캐시 없음, 상승/하락 대칭).
2) get_price_quality_excluded_codes: ingest_meta 캐싱 계층(최초 계산 → 캐시 히트 →
   refresh=True 강제 재계산).
3) 배선: KR 백테스트 유니버스(metrics_at)와 KR 스크리닝(get_cross_section)이 이상
   종목을 후보에서 제외하는지(top_n 선정 이전에), 정상 종목은 영향 없는지(회귀).
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

from src.backtest.data_access import metrics_at
from src.backtest.primitives import get_cross_section
from src.data_quality import (
    detect_price_quality_anomalies,
    get_price_quality_excluded_codes,
)
from src.db import get_meta, init_db


def _conn(tmp_path, name="dq.db") -> sqlite3.Connection:
    db = tmp_path / name
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def _seed_company(conn, code, name="종목", quarter="2025Q1", disclosed="2025-05-15"):
    conn.execute(
        "INSERT INTO company(stock_code, name, market, sector) VALUES (?,?,?,?)",
        (code, name, "KOSPI", "전기·전자"),
    )
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
        "VALUES (?,?,?,?,?)",
        (code, quarter, disclosed, "net_income", 1_000.0),
    )
    conn.execute(
        "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
        "VALUES (?,?,?,?,?)",
        (code, quarter, disclosed, "total_equity", 100_000.0),
    )
    conn.commit()


def _seed_price_series(conn, code, closes, start="2016-01-01"):
    d = date.fromisoformat(start)
    for c in closes:
        conn.execute(
            "INSERT INTO prices(stock_code, date, close) VALUES (?,?,?)",
            (code, d.isoformat(), float(c)),
        )
        d += timedelta(days=1)
    conn.commit()


# ---------------------------------------------------------------------------
# 1) detect_price_quality_anomalies: 순수 탐지(대칭)
# ---------------------------------------------------------------------------
def test_detect_catches_upward_jump(tmp_path):
    conn = _conn(tmp_path)
    # 017170류: 105.0 → 2100.0 (20배 상승 점프)
    _seed_price_series(conn, "UP20", [105.0] * 10 + [2100.0] * 10)
    found = detect_price_quality_anomalies(conn)
    assert "UP20" in found
    conn.close()


def test_detect_catches_downward_crash():
    """246830류: 12400 → 50 (ratio=0.004, 하락 폭락). 기존 상승전용 임계값의 사각지대였다."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path

        conn = _conn(Path(td))
        _seed_price_series(conn, "CRASH", [12400.0] * 10 + [50.0] * 10)
        found = detect_price_quality_anomalies(conn)
        assert "CRASH" in found
        conn.close()


def test_detect_ignores_normal_stock(tmp_path):
    conn = _conn(tmp_path)
    closes = [10000.0 + i * 10 for i in range(60)]  # 완만한 변동, 일 최대비율 <<2배
    _seed_price_series(conn, "NORMAL", closes)
    found = detect_price_quality_anomalies(conn)
    assert "NORMAL" not in found
    conn.close()


def test_detect_boundary_exact_ratio_included():
    """ratio == 2.0 또는 0.5 정확히 경계값도 포함(>=, <=)."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        conn = _conn(Path(td))
        _seed_price_series(conn, "EXACT2X", [100.0] * 5 + [200.0] * 5)  # ratio=2.0 정확히
        found = detect_price_quality_anomalies(conn)
        assert "EXACT2X" in found
        conn.close()


def test_detect_below_threshold_not_flagged(tmp_path):
    """ratio 1.9배(<2.0) 상승은 대칭 임계값 밖 — 플래그되지 않는다."""
    conn = _conn(tmp_path)
    _seed_price_series(conn, "SUB2X", [100.0] * 5 + [190.0] * 5)  # ratio=1.9
    found = detect_price_quality_anomalies(conn)
    assert "SUB2X" not in found
    conn.close()


# ---------------------------------------------------------------------------
# 2) get_price_quality_excluded_codes: ingest_meta 캐싱
# ---------------------------------------------------------------------------
def test_get_excluded_codes_computes_and_caches(tmp_path):
    conn = _conn(tmp_path)
    _seed_price_series(conn, "UP20", [105.0] * 10 + [2100.0] * 10)
    codes = get_price_quality_excluded_codes(conn)
    assert codes == {"UP20"}
    cached = get_meta(conn, "price_quality_excluded_codes")
    assert cached == "UP20"
    conn.close()


def test_get_excluded_codes_empty_result_cached_as_empty_string(tmp_path):
    conn = _conn(tmp_path)
    _seed_price_series(conn, "NORMAL", [10000.0 + i * 10 for i in range(30)])
    codes = get_price_quality_excluded_codes(conn)
    assert codes == set()
    cached = get_meta(conn, "price_quality_excluded_codes")
    assert cached == ""  # 빈 문자열로 캐시(재스캔 방지, None과 구분)
    conn.close()


def test_get_excluded_codes_cache_hit_skips_rescan(tmp_path):
    """캐시가 있으면 재스캔하지 않는다 — DB에 나중에 추가된 이상종목은 refresh 없이는 안 보인다."""
    conn = _conn(tmp_path)
    _seed_price_series(conn, "UP20", [105.0] * 10 + [2100.0] * 10)
    first = get_price_quality_excluded_codes(conn)
    assert first == {"UP20"}

    # 캐시 이후에 새 이상종목이 생겼다고 가정(예: 다음날 신규 적재로 발견된 폭락 케이스).
    _seed_price_series(conn, "CRASH", [12400.0] * 10 + [50.0] * 10)

    stale = get_price_quality_excluded_codes(conn)  # refresh=False(기본) → 캐시 그대로
    assert stale == {"UP20"}  # CRASH는 아직 안 보임(캐시 히트)

    refreshed = get_price_quality_excluded_codes(conn, refresh=True)
    assert refreshed == {"UP20", "CRASH"}
    conn.close()


# ---------------------------------------------------------------------------
# 3) 배선: KR 백테스트 유니버스(metrics_at) + KR 스크리닝(get_cross_section)
# ---------------------------------------------------------------------------
def test_metrics_at_excludes_anomalous_stock_from_backtest_universe(tmp_path):
    conn = _conn(tmp_path)
    _seed_company(conn, "000001", "정상전자")
    _seed_company(conn, "000002", "이상화학")
    _seed_price_series(conn, "000001", [10000.0 + i * 10 for i in range(60)])
    _seed_price_series(conn, "000002", [105.0] * 30 + [2100.0] * 30)  # 이상(20배 점프)

    rows = metrics_at(conn, "2026-06-30")
    codes = {r["stock_code"] for r in rows}
    assert "000001" in codes  # 정상 종목은 영향 없음(회귀)
    assert "000002" not in codes  # 이상 종목은 유니버스에서 제외
    conn.close()


def test_metrics_at_no_false_exclusion_for_normal_stocks_only(tmp_path):
    """정상 종목만 있을 때 제외 로직이 아무것도 건드리지 않는다(순수 회귀)."""
    conn = _conn(tmp_path)
    _seed_company(conn, "000001", "정상전자")
    _seed_price_series(conn, "000001", [10000.0 + i * 10 for i in range(60)])

    rows = metrics_at(conn, "2026-06-30")
    assert {r["stock_code"] for r in rows} == {"000001"}
    conn.close()


def test_get_cross_section_excludes_anomalous_stock_before_top_n():
    """스크리닝 실제 경로(get_cross_section, metrics_at 기본 사용)도 이상종목을 제외한다.

    top_n 선정은 combine/select_stocks에서 일어나므로, 여기(크로스섹션 rows)에서부터
    빠져 있어야 top_n 슬라이싱 이전에 걸러진 것이 보장된다.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        conn = _conn(Path(td))
        _seed_company(conn, "000001", "정상전자")
        _seed_company(conn, "000002", "이상화학")
        _seed_price_series(conn, "000001", [10000.0 + i * 10 for i in range(60)])
        _seed_price_series(conn, "000002", [105.0] * 30 + [2100.0] * 30)

        rows = get_cross_section(conn, "2026-06-30")
        codes = {r["stock_code"] for r in rows}
        assert "000001" in codes
        assert "000002" not in codes
        conn.close()
