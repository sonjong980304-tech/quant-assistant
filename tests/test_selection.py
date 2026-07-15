"""select_stocks의 criteria 필드 검증 테스트.

LLM이 파이프라인 criteria에 존재하지 않는 필드명(예: forward_12m_return)을 지어내면,
기존 코드는 .get()이 조용히 None을 반환해 모든 종목을 걸러내 버렸다(에러 없이 빈 결과).
이제는 존재하지 않는 필드면 즉시 ValueError로 명확히 실패해야 한다.
"""
from __future__ import annotations

import pytest

from src.backtest.selection import select_stocks


def _fake_rows():
    return [
        {"stock_code": "000001", "name": "가", "sector": "화학", "market": "KOSPI",
         "quarter": "2025Q1", "per": 8.0, "roe": 12.0},
        {"stock_code": "000002", "name": "나", "sector": "화학", "market": "KOSPI",
         "quarter": "2025Q1", "per": 15.0, "roe": 8.0},
        {"stock_code": "000003", "name": "다", "sector": "금융", "market": "KOSPI",
         "quarter": "2025Q1", "per": 5.0, "roe": 20.0},
    ]


def test_select_stocks_raises_on_unknown_criteria_field():
    criteria = [{"key": "forward_12m_return", "direction": "low", "weight": 1.0}]
    with pytest.raises(ValueError, match="forward_12m_return"):
        select_stocks(_fake_rows(), criteria, combine="zscore", n=5)


def test_select_stocks_still_selects_with_known_fields():
    criteria = [{"key": "per", "direction": "low", "weight": 1.0}]
    picked = select_stocks(_fake_rows(), criteria, combine="zscore", n=2)
    assert [r["name"] for r in picked] == ["다", "가"]  # per 낮은 순 5<8
