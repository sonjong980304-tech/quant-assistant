"""차트 렌더링(src/agents/charting.py) 검증 — 시각 파트(TDD 강제 아님)이지만, 반환된
base64 문자열이 실제로 유효한 PNG인지 자동으로 검증한다(브리핑 "검증 방식").

- base64.b64decode 로 디코딩 가능하고,
- 디코딩된 바이트가 PNG 매직바이트(\\x89PNG\\r\\n\\x1a\\n)로 시작하며,
- IHDR 청크의 폭/높이가 0이 아닌 유효한 PNG인지 확인한다(한글 제목 케이스 최소 1건 포함).
"""
from __future__ import annotations

import base64
import struct

from src.agents.charting import (
    render_bar_chart_base64,
    render_line_chart_base64,
    render_scatter_chart_base64,
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _assert_valid_png(b64: str) -> tuple[int, int]:
    """base64 → PNG 디코딩 + 매직바이트/치수 검증. (width, height) 반환."""
    assert isinstance(b64, str) and b64
    assert not b64.startswith("data:")  # 데이터 URI 접두사 없이 순수 base64
    raw = base64.b64decode(b64)
    assert raw[:8] == _PNG_MAGIC
    # PNG: 8바이트 시그니처 + 4바이트 길이 + "IHDR" + 4바이트 width + 4바이트 height
    width, height = struct.unpack(">II", raw[16:24])
    assert width > 0 and height > 0
    return width, height


def test_render_line_chart_base64_returns_valid_png_with_korean_title():
    """한글 제목 케이스 — save_nav_chart 폰트 버그(DejaVu Sans에 한글 없음)를 피해야 한다."""
    b64 = render_line_chart_base64(
        ["2024-01-01", "2024-02-01", "2024-03-01"],
        {"전략": [1.0, 1.1, 1.2]},
        "백테스트 결과",
        ylabel="NAV",
    )
    _assert_valid_png(b64)


def test_render_line_chart_base64_multi_series():
    """여러 라인(전략 vs 벤치마크) 동시 렌더."""
    b64 = render_line_chart_base64(
        ["a", "b", "c"],
        {"전략": [1.0, 1.1, 1.2], "벤치마크": [1.0, 1.05, 1.08]},
        "전략 vs 벤치마크",
    )
    _assert_valid_png(b64)


def test_render_line_chart_base64_single_series():
    b64 = render_line_chart_base64(
        ["2026-01-01", "2026-01-02"], {"005930 종가": [70000.0, 71000.0]}, "종가 추이"
    )
    _assert_valid_png(b64)


def test_render_line_chart_base64_handles_none_values_without_crash():
    """결측(None)이 섞여도 예외 없이 렌더링된다(벤치마크 일부 결측 등)."""
    b64 = render_line_chart_base64(
        ["a", "b", "c"], {"시리즈": [1.0, None, 1.2]}, "결측 포함"
    )
    _assert_valid_png(b64)


def test_render_line_chart_base64_many_points_limits_xticks():
    """포인트가 많아도(라벨 과밀 방지: 최대 12개) 예외 없이 유효 PNG."""
    dates = [f"2024-{(i % 12) + 1:02d}-01" for i in range(60)]
    b64 = render_line_chart_base64(dates, {"전략": [1.0 + i * 0.01 for i in range(60)]}, "장기 추이")
    _assert_valid_png(b64)


# ── 산점도(scatter) 렌더링 — 라인차트와 동일한 한글폰트/Agg/PNG 검증 수준 ────────────

def test_render_scatter_chart_base64_returns_valid_png_with_korean_labels():
    """한글 축라벨/제목/점라벨 케이스 — 라인차트와 동일 폰트 폴백을 재사용해야 한다."""
    b64 = render_scatter_chart_base64(
        [5.0, 8.0, 3.0], [12.0, 20.0, 9.0], ["가", "나", "다"],
        "이익수익률(%)", "투하자본수익률(%)", "마법공식 산점도",
    )
    _assert_valid_png(b64)


def test_render_scatter_chart_base64_without_labels():
    """labels=None이어도 예외 없이 유효 PNG."""
    b64 = render_scatter_chart_base64([1.0, 2.0], [3.0, 4.0], None, "x", "y", "제목")
    _assert_valid_png(b64)


def test_render_scatter_chart_base64_many_points():
    """점이 많아도(라벨 과밀) 예외 없이 유효 PNG."""
    x = [float(i) for i in range(200)]
    y = [float(i * 2) for i in range(200)]
    labels = [f"종목{i}" for i in range(200)]
    b64 = render_scatter_chart_base64(x, y, labels, "EY", "ROC", "대량 산점도")
    _assert_valid_png(b64)


# ── 막대그래프(bar) 렌더링 — quantile_bucket_means(분위별 평균) 시각화용. 실서버 재현
#    버그: "분위별 평균 GPA를 막대그래프로" 요청에 응답할 렌더 함수 자체가 없었다. ──────

def test_render_bar_chart_base64_returns_valid_png_with_korean_labels():
    """한글 라벨/제목 케이스 — 라인/산점도차트와 동일 폰트 폴백을 재사용해야 한다."""
    b64 = render_bar_chart_base64(
        ["1분위", "2분위", "3분위", "4분위", "5분위"],
        [12.06, 15.82, 15.90, 17.52, 19.31],
        "PBR 분위", "평균 GPA(%)", "PBR 분위수별 평균 GPA",
    )
    _assert_valid_png(b64)


def test_render_bar_chart_base64_handles_negative_values():
    """평균값이 음수(적자 구간 등)여도 예외 없이 유효 PNG."""
    b64 = render_bar_chart_base64(["1분위", "2분위"], [-3.5, 4.2], "분위", "값", "제목")
    _assert_valid_png(b64)


def test_render_bar_chart_base64_many_bars_limits_xticks():
    """막대가 많아도(라벨 과밀 방지) 예외 없이 유효 PNG."""
    labels = [f"그룹{i}" for i in range(30)]
    values = [float(i) for i in range(30)]
    b64 = render_bar_chart_base64(labels, values, "그룹", "값", "대량 막대그래프")
    _assert_valid_png(b64)
