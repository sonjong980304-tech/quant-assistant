"""차트 렌더링(src/agents/charting.py) 검증 — 시각 파트(TDD 강제 아님)이지만, 반환된
base64 문자열이 실제로 유효한 PNG인지 자동으로 검증한다(브리핑 "검증 방식").

- base64.b64decode 로 디코딩 가능하고,
- 디코딩된 바이트가 PNG 매직바이트(\\x89PNG\\r\\n\\x1a\\n)로 시작하며,
- IHDR 청크의 폭/높이가 0이 아닌 유효한 PNG인지 확인한다(한글 제목 케이스 최소 1건 포함).
"""
from __future__ import annotations

import base64
import struct

from src.agents.charting import render_line_chart_base64

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
