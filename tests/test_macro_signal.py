"""매크로 신호 판정(규칙엔진) 테스트 (MAC-3).

.omc/specs/brainstorming-macro-indicator-agent.md AC7/AC8/AC9/AC10/AC11/AC12.
순수 계산 로직(레짐/밴드 분류, 종합신호)은 DB/네트워크 없이 검증하고,
판정 이력 append(macro_signal)만 tmp_path sqlite로 통합 검증한다.

가장 중요한 회귀(AC10): 종합신호(overall)는 오직 금리차 레짐에서만 결정되며
CNN/VIX 값은 신호 계산에 전혀 관여하지 않는다.
"""
from __future__ import annotations

import pytest

from src.db import connect, init_db
from src.ingest.macro_signal import (
    classify_cnn_band,
    classify_spread_regime,
    classify_vix_band,
    compute_signal,
    regime_to_overall,
    run_signal,
)


# --------------------------------------------------------------------------
# AC7 — 금리차 레짐 분류 (>=0.5 정상, 0~0.5 평탄화, <0 역전). 임의 완충구간(스펙 명시).
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "spread,expected",
    [(0.6, "정상"), (0.5, "정상"), (0.3, "평탄화"), (0.0, "평탄화"), (-0.1, "역전")],
)
def test_classify_spread_regime_boundaries(spread, expected):
    assert classify_spread_regime(spread) == expected


# --------------------------------------------------------------------------
# AC8 — CNN 밴드 분류 (CNN 공식 5구간).
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "극단공포"), (24, "극단공포"),
        (25, "공포"), (44, "공포"),
        (45, "중립"), (55, "중립"),
        (56, "탐욕"), (75, "탐욕"),
        (76, "극단탐욕"), (100, "극단탐욕"),
    ],
)
def test_classify_cnn_band_boundaries(value, expected):
    assert classify_cnn_band(value) == expected


# --------------------------------------------------------------------------
# AC9 — VIX 밴드 분류 (<15 안정, 15~20 보통, 20~30 경계, >=30 공포).
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (14.9, "안정"),
        (15.0, "보통"), (19.9, "보통"),
        (20.0, "경계"), (29.9, "경계"),
        (30.0, "공포"),
    ],
)
def test_classify_vix_band_boundaries(value, expected):
    assert classify_vix_band(value) == expected


# --------------------------------------------------------------------------
# 종합신호 매핑 (정상→GREEN, 평탄화→YELLOW, 역전→RED).
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "regime,expected",
    [("정상", "GREEN"), ("평탄화", "YELLOW"), ("역전", "RED")],
)
def test_regime_to_overall(regime, expected):
    assert regime_to_overall(regime) == expected


def test_compute_signal_fills_all_fields():
    sig = compute_signal(spread=0.6, cnn=50, vix=14.0, as_of="2026-07-14")
    assert sig["spread_regime"] == "정상"
    assert sig["overall"] == "GREEN"
    assert sig["cnn_band"] == "중립"
    assert sig["vix_band"] == "안정"
    assert sig["as_of"] == "2026-07-14"
    assert sig["data_missing"] is False
    assert sig["created_at"]  # 생성 시각이 채워진다


# --------------------------------------------------------------------------
# AC10 (핵심 회귀) — overall은 오직 스프레드 레짐에만 의존한다.
# CNN/VIX를 극단값(0,100 / 5,80)으로 바꿔도 결과가 변하지 않아야 한다.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("spread,expected", [(0.6, "GREEN"), (0.2, "YELLOW"), (-0.1, "RED")])
@pytest.mark.parametrize("cnn", [0, 50, 100])
@pytest.mark.parametrize("vix", [5.0, 25.0, 80.0])
def test_overall_depends_only_on_spread_regime(spread, expected, cnn, vix):
    sig = compute_signal(spread=spread, cnn=cnn, vix=vix, as_of="2026-07-14")
    assert sig["overall"] == expected


def test_extreme_cnn_vix_do_not_flip_signal():
    # 같은 스프레드(정상)면 CNN/VIX가 극단공포/극단탐욕이든 overall은 항상 GREEN.
    base = compute_signal(spread=0.6, cnn=50, vix=15.0, as_of="2026-07-14")["overall"]
    extreme_fear = compute_signal(spread=0.6, cnn=0, vix=80.0, as_of="2026-07-14")["overall"]
    extreme_greed = compute_signal(spread=0.6, cnn=100, vix=5.0, as_of="2026-07-14")["overall"]
    assert base == extreme_fear == extreme_greed == "GREEN"


# --------------------------------------------------------------------------
# AC11 — 금리차 수집 실패(spread None) → 직전 신호 유지 + 데이터없음 표시.
# --------------------------------------------------------------------------
def test_spread_missing_keeps_prev_and_marks_data_missing():
    sig = compute_signal(spread=None, cnn=0, vix=80.0, prev_overall="GREEN", as_of="2026-07-14")
    assert sig["spread"] is None
    assert sig["spread_regime"] == "데이터없음"
    assert sig["overall"] == "GREEN"       # 새로 계산하지 않고 직전 신호 유지
    assert sig["prev_overall"] == "GREEN"
    assert sig["data_missing"] is True
    # 참고 지표(CNN/VIX)는 여전히 밴드 분류가 채워진다.
    assert sig["cnn_band"] == "극단공포"
    assert sig["vix_band"] == "공포"


def test_spread_missing_with_no_prev_signal_is_none():
    sig = compute_signal(spread=None, cnn=50, vix=15.0, prev_overall=None, as_of="2026-07-14")
    assert sig["overall"] is None
    assert sig["spread_regime"] == "데이터없음"


# --------------------------------------------------------------------------
# AC12 — 판정은 macro_signal에 날짜별 1행 append(UPDATE 아님). 2회 실행 → 2행.
# --------------------------------------------------------------------------
def test_run_signal_appends_rows_across_runs(tmp_path):
    db = str(tmp_path / "sig.db")
    init_db(db)
    conn = connect(db)
    try:
        run_signal(conn, spread=0.6, cnn=50, vix=14.0, as_of="2026-07-14")
        run_signal(conn, spread=-0.2, cnn=10, vix=40.0, as_of="2026-07-15")
        n = conn.execute("SELECT COUNT(*) FROM macro_signal").fetchone()[0]
        assert n == 2
    finally:
        conn.close()


def test_run_signal_carries_prev_overall_from_last_row(tmp_path):
    db = str(tmp_path / "sig2.db")
    init_db(db)
    conn = connect(db)
    try:
        first = run_signal(conn, spread=0.6, cnn=50, vix=14.0, as_of="2026-07-14")
        second = run_signal(conn, spread=-0.2, cnn=10, vix=40.0, as_of="2026-07-15")
        assert first["overall"] == "GREEN"
        assert first["prev_overall"] is None            # 첫 판정엔 직전값 없음
        assert second["prev_overall"] == "GREEN"        # 직전 판정의 overall을 이어받는다
        assert second["overall"] == "RED"
        # 저장된 최신 행이 두 번째 판정인지 확인
        row = conn.execute(
            "SELECT overall, prev_overall FROM macro_signal ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["overall"] == "RED"
        assert row["prev_overall"] == "GREEN"
    finally:
        conn.close()


def test_run_signal_spread_missing_persists_prev_overall(tmp_path):
    # AC11+AC12: spread 수집 실패로 재실행하면 직전 신호를 유지한 행이 append된다.
    db = str(tmp_path / "sig3.db")
    init_db(db)
    conn = connect(db)
    try:
        run_signal(conn, spread=0.6, cnn=50, vix=14.0, as_of="2026-07-14")   # GREEN
        missing = run_signal(conn, spread=None, cnn=50, vix=14.0, as_of="2026-07-15")
        assert missing["spread_regime"] == "데이터없음"
        assert missing["overall"] == "GREEN"
        n = conn.execute("SELECT COUNT(*) FROM macro_signal").fetchone()[0]
        assert n == 2
    finally:
        conn.close()
