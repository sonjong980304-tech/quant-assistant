"""매크로 신호 판정(규칙엔진) — 순수 규칙기반, LLM 미사용.

.omc/specs/brainstorming-macro-indicator-agent.md 참고.

수집(macro_indicators.py)과 파일을 분리한 이유: 이 모듈은 DB/네트워크 의존 없이
독립적으로 테스트 가능한 순수 계산 로직(레짐/밴드 분류, 종합신호)이 핵심이다.

가장 중요한 규칙(절대 불변)
---------------------------
종합신호(overall: GREEN/YELLOW/RED)는 **오직 금리차 레짐에서만** 결정된다
(정상→GREEN, 평탄화→YELLOW, 역전→RED). CNN/VIX 값은 신호 계산에 전혀 관여하지 않으며
참고 표시(밴드 분류)로만 존재한다. compute_signal에 CNN/VIX를 극단값으로 넣어도
overall은 스프레드 레짐에만 의존한다(test_macro_signal.py AC10 회귀테스트로 고정).

금리차 자체 수집이 실패하면(spread=None) 그날은 새로 계산하지 않고 직전 신호를 유지하며
spread_regime='데이터없음'으로 표시한다(AC11).
"""
from __future__ import annotations

from datetime import date

from ..version import now_iso


# ---------------------------------------------------------------------------
# 금리차 레짐 분류 (AC7)
#   >=0.5%p → 정상 | 0~0.5 → 평탄화 | <0 → 역전
# 주의: 0.5%p 정상 경계는 표준 컨벤션이 아닌 임의 완충구간이다(스펙 §Constraints,
# factcheck로 확인 — 표준은 "0% 미만=역전"만 명시). 판정 단순화를 위한 자체 기준.
# ---------------------------------------------------------------------------
def classify_spread_regime(spread: float) -> str:
    if spread < 0:
        return "역전"
    if spread < 0.5:
        return "평탄화"
    return "정상"


# ---------------------------------------------------------------------------
# CNN Fear&Greed 밴드 분류 (AC8) — CNN 공식 5구간. 참고 표시 전용.
#   0-24 극단공포 | 25-44 공포 | 45-55 중립 | 56-75 탐욕 | 76-100 극단탐욕
# ---------------------------------------------------------------------------
def classify_cnn_band(value: float) -> str:
    if value <= 24:
        return "극단공포"
    if value <= 44:
        return "공포"
    if value <= 55:
        return "중립"
    if value <= 75:
        return "탐욕"
    return "극단탐욕"


# ---------------------------------------------------------------------------
# VIX 밴드 분류 (AC9) — VIX 실무 구간. 참고 표시 전용.
#   <15 안정 | 15-20 보통 | 20-30 경계 | >=30 공포
# ---------------------------------------------------------------------------
def classify_vix_band(value: float) -> str:
    if value < 15:
        return "안정"
    if value < 20:
        return "보통"
    if value < 30:
        return "경계"
    return "공포"


# ---------------------------------------------------------------------------
# 종합신호 매핑 (금리차 레짐 단독 결정 — AC10)
# ---------------------------------------------------------------------------
_REGIME_TO_SIGNAL = {"정상": "GREEN", "평탄화": "YELLOW", "역전": "RED"}
_DATA_MISSING = "데이터없음"


def regime_to_overall(regime: str) -> str:
    """레짐(정상/평탄화/역전)을 종합신호(GREEN/YELLOW/RED)로 매핑."""
    return _REGIME_TO_SIGNAL[regime]


# ---------------------------------------------------------------------------
# 판정 계산 (순수 함수) — macro_signal 한 행에 해당하는 dict 반환
# ---------------------------------------------------------------------------
def compute_signal(
    spread: float | None,
    cnn: float | None,
    vix: float | None,
    prev_overall: str | None = None,
    as_of: str | None = None,
) -> dict:
    """세 지표값으로 판정 dict를 만든다. overall은 스프레드 레짐만으로 결정.

    spread가 None(수집 실패)이면 새로 계산하지 않고 직전 신호(prev_overall)를 유지하며
    spread_regime='데이터없음'으로 표시한다(AC11). CNN/VIX 밴드는 참고용으로 항상 채운다.
    반환 dict의 data_missing은 편의용 플래그이며 테이블 컬럼은 아니다(spread_regime로 표현).
    """
    as_of = as_of or date.today().strftime("%Y-%m-%d")
    cnn_band = classify_cnn_band(cnn) if cnn is not None else None
    vix_band = classify_vix_band(vix) if vix is not None else None

    if spread is None:
        # AC11: 금리차 수집 실패 → 직전 신호 유지 + 데이터없음
        return {
            "as_of": as_of,
            "spread": None,
            "spread_regime": _DATA_MISSING,
            "cnn_value": cnn,
            "cnn_band": cnn_band,
            "vix_value": vix,
            "vix_band": vix_band,
            "overall": prev_overall,
            "prev_overall": prev_overall,
            "created_at": now_iso(),
            "data_missing": True,
        }

    regime = classify_spread_regime(spread)
    return {
        "as_of": as_of,
        "spread": spread,
        "spread_regime": regime,
        "cnn_value": cnn,
        "cnn_band": cnn_band,
        "vix_value": vix,
        "vix_band": vix_band,
        "overall": regime_to_overall(regime),
        "prev_overall": prev_overall,
        "created_at": now_iso(),
        "data_missing": False,
    }


# ---------------------------------------------------------------------------
# 이력 저장 (append) + 직전 신호 조회
# ---------------------------------------------------------------------------
def get_prev_overall(conn) -> str | None:
    """가장 최근 판정의 overall(직전 신호). 이력이 없으면 None."""
    row = conn.execute(
        "SELECT overall FROM macro_signal ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def persist_signal(conn, signal: dict) -> None:
    """판정 dict를 macro_signal에 append(INSERT, UPDATE 아님 — 이력 추적)."""
    conn.execute(
        "INSERT INTO macro_signal(as_of, spread, spread_regime, cnn_value, cnn_band, "
        "vix_value, vix_band, overall, prev_overall, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            signal["as_of"], signal["spread"], signal["spread_regime"],
            signal["cnn_value"], signal["cnn_band"], signal["vix_value"], signal["vix_band"],
            signal["overall"], signal["prev_overall"], signal["created_at"],
        ),
    )
    conn.commit()


def run_signal(
    conn,
    spread: float | None,
    cnn: float | None,
    vix: float | None,
    as_of: str | None = None,
) -> dict:
    """직전 신호를 읽어 판정을 계산하고 macro_signal에 append한 뒤 판정 dict를 반환한다."""
    prev = get_prev_overall(conn)
    signal = compute_signal(spread, cnn, vix, prev_overall=prev, as_of=as_of)
    persist_signal(conn, signal)
    return signal
