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


# ── 섹터 중립 z-score (sector_neutral=True) ──────────────────────────────────
# "섹터 중립화": valid 전체를 한 덩어리로 평균/표준편차를 구하지 않고, sector 그룹마다 따로
# 평균/표준편차로 z-score를 구해(섹터 내부 상대순위) 전체에서 다시 비교한다. raw 값이 큰
# 섹터로 결과가 쏠리던 문제(전기전자 몰림)를 없앤다. sector_neutral=False(기본값)면 기존
# global zscore와 완전히 동일해야 한다(회귀 없음).
def _two_sector_momentum_rows():
    """A섹터=[10,20,30], B섹터=[100,200,300]. raw 값만 보면 B가 항상 이기지만, 섹터 내부
    상대순위로 보면 각 섹터 1위(A3/B3)끼리 최상위로 비슷해야 한다."""
    spec = [
        ("A1", "에이1", "A", 10.0), ("A2", "에이2", "A", 20.0), ("A3", "에이3", "A", 30.0),
        ("B1", "비1", "B", 100.0), ("B2", "비2", "B", 200.0), ("B3", "비3", "B", 300.0),
    ]
    return [
        {"stock_code": c, "name": nm, "sector": s, "market": "KOSPI",
         "quarter": "2025Q1", "return_12m": v}
        for c, nm, s, v in spec
    ]


def test_sector_neutral_zscore_does_not_concentrate_in_one_sector():
    criteria = [{"key": "return_12m", "direction": "high", "weight": 1.0}]
    picked = select_stocks(
        _two_sector_momentum_rows(), criteria, combine="zscore", n=2, sector_neutral=True
    )
    # 섹터 내부 상대순위 → 각 섹터의 1위끼리가 최상위 → 한 섹터로 쏠리지 않는다.
    assert {r["sector"] for r in picked} == {"A", "B"}


def test_global_zscore_concentrates_in_high_raw_sector():
    """대조군: sector_neutral 없이는 raw 값이 큰 섹터(B)로 쏠린다(기존 동작)."""
    criteria = [{"key": "return_12m", "direction": "high", "weight": 1.0}]
    picked = select_stocks(_two_sector_momentum_rows(), criteria, combine="zscore", n=2)
    assert {r["sector"] for r in picked} == {"B"}


def test_sector_neutral_false_matches_global_zscore_exactly():
    """sector_neutral=False(기본값)는 기존 global zscore와 종목 순서/점수가 완전히 동일(회귀)."""
    criteria = [{"key": "return_12m", "direction": "high", "weight": 1.0}]
    default = select_stocks(_two_sector_momentum_rows(), criteria, combine="zscore", n=6)
    explicit = select_stocks(
        _two_sector_momentum_rows(), criteria, combine="zscore", n=6, sector_neutral=False
    )
    assert [r["stock_code"] for r in default] == [r["stock_code"] for r in explicit]
    assert [r["_score"] for r in default] == [r["_score"] for r in explicit]


def test_sector_neutral_single_stock_sector_no_zero_division():
    """섹터에 종목 1개뿐이면 std=0 → sd or 1.0로 0나눗셈 방지, 그 종목은 z=0(중립)."""
    rows = _two_sector_momentum_rows()
    rows.append({"stock_code": "C1", "name": "씨1", "sector": "C",
                 "market": "KOSPI", "quarter": "2025Q1", "return_12m": 50.0})
    criteria = [{"key": "return_12m", "direction": "high", "weight": 1.0}]
    picked = select_stocks(rows, criteria, combine="zscore", n=7, sector_neutral=True)
    assert len(picked) == 7  # 예외 없이 전 종목 반환
    c1 = next(r for r in picked if r["stock_code"] == "C1")
    assert c1["_score"] == 0.0  # 자기 섹터 평균과 같음 → z=0


def test_sector_neutral_handles_none_sector_group():
    """sector=None(결측)인 행이 섞여도 별도 "None" 그룹으로 묶어 예외 없이 동작한다."""
    rows = _two_sector_momentum_rows()
    rows.append({"stock_code": "N1", "name": "엔1", "sector": None,
                 "market": "KOSPI", "quarter": "2025Q1", "return_12m": 55.0})
    rows.append({"stock_code": "N2", "name": "엔2", "sector": None,
                 "market": "KOSPI", "quarter": "2025Q1", "return_12m": 45.0})
    criteria = [{"key": "return_12m", "direction": "high", "weight": 1.0}]
    picked = select_stocks(rows, criteria, combine="zscore", n=8, sector_neutral=True)
    assert len(picked) == 8


def test_sector_neutral_ignored_for_rank_sum():
    """combine="rank_sum"이면 sector_neutral=True를 줘도 무시되고 기존과 동일(범위 밖)."""
    criteria = [{"key": "return_12m", "direction": "high", "weight": 1.0}]
    with_flag = select_stocks(
        _two_sector_momentum_rows(), criteria, combine="rank_sum", n=6, sector_neutral=True
    )
    without = select_stocks(_two_sector_momentum_rows(), criteria, combine="rank_sum", n=6)
    assert [r["stock_code"] for r in with_flag] == [r["stock_code"] for r in without]


def test_sector_neutral_ignored_for_and_combine():
    """combine="and"이면 sector_neutral=True를 줘도 에러 없이 기존 동작 유지(범위 밖)."""
    criteria = [{"key": "return_12m", "direction": "high", "weight": 1.0}]
    picked = select_stocks(
        _two_sector_momentum_rows(), criteria, combine="and", n=3, sector_neutral=True
    )
    assert isinstance(picked, list)
