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


def _us_rows_with_dual_class():
    """GOOG(Class C)/GOOGL(Class A)처럼 한 회사가 복수 티커로 상장된 상황 재현.

    operating_profit 내림차순으로 GOOGL > GOOG > NVDA > MSFT > AAPL이 되도록 값 설정.
    """
    return [
        {"stock_code": "GOOGL", "name": "Alphabet Inc. Class A Common Stock",
         "sector": "Tech", "market": "NASDAQ", "quarter": "2026Q1", "operating_profit": 500.0},
        {"stock_code": "GOOG", "name": "Alphabet Inc. Class C Capital Stock",
         "sector": "Tech", "market": "NASDAQ", "quarter": "2026Q1", "operating_profit": 500.0},
        {"stock_code": "NVDA", "name": "NVIDIA Corporation Common Stock",
         "sector": "Tech", "market": "NASDAQ", "quarter": "2026Q1", "operating_profit": 400.0},
        {"stock_code": "MSFT", "name": "Microsoft Corporation Common Stock",
         "sector": "Tech", "market": "NASDAQ", "quarter": "2026Q1", "operating_profit": 300.0},
        {"stock_code": "AAPL", "name": "Apple Inc. Common Stock",
         "sector": "Tech", "market": "NASDAQ", "quarter": "2026Q1", "operating_profit": 200.0},
    ]


def test_select_stocks_includes_sibling_ticker_of_same_company_without_using_up_n():
    """상위 3개 '기업'을 요청하면, 형제 티커(GOOG/GOOGL)는 제거하지 않고 같이 보여주되
    서로 다른 기업 수를 셀 때는 하나로 카운트해야 한다 (dedup 아님, 라벨링만)."""
    criteria = [{"key": "operating_profit", "direction": "high", "weight": 1.0}]
    picked = select_stocks(_us_rows_with_dual_class(), criteria, combine="zscore", n=3)

    codes = [r["stock_code"] for r in picked]
    # GOOGL, GOOG(형제), NVDA, MSFT 4개 행 = 서로 다른 기업 3개(Alphabet/NVIDIA/Microsoft)
    assert codes == ["GOOGL", "GOOG", "NVDA", "MSFT"]
    assert "AAPL" not in codes  # 4번째 기업(Apple)은 n=3 초과라 제외

    by_code = {r["stock_code"]: r for r in picked}
    assert by_code["GOOGL"]["_same_company"] is False
    assert by_code["GOOG"]["_same_company"] is True  # GOOGL과 같은 회사(Alphabet)
    assert by_code["NVDA"]["_same_company"] is False
    assert by_code["MSFT"]["_same_company"] is False


def test_select_stocks_kr_names_unaffected_by_company_grouping():
    """한국 종목명(Class 접미어 없음)은 그룹핑이 사실상 no-op이어야 한다 — 기존 동작 그대로."""
    criteria = [{"key": "per", "direction": "low", "weight": 1.0}]
    picked = select_stocks(_fake_rows(), criteria, combine="zscore", n=2)
    assert [r["_same_company"] for r in picked] == [False, False]
