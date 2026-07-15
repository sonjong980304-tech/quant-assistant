"""normalize_account 매핑 테스트.

핵심 개선: DART account_id의 접두사 변형(`ifrs_` 구표기)을 표준(`ifrs-full_`)과
동일하게 취급해 _DART_ID_MAP으로 매핑한다. 기존 매핑(이름매칭, 비지배 차단)은 회귀 없이 유지.
"""
from __future__ import annotations

import pytest

from src.ingest.normalize import normalize_account


# ── 사이클 1: account_id 접두사 정규화 (ifrs_ ≡ ifrs-full_) ──────────────
@pytest.mark.parametrize(
    "account_id, expected",
    [
        ("ifrs_ProfitLossAttributableToOwnersOfParent", "controlling_net_income"),
        ("ifrs_EquityAttributableToOwnersOfParent", "controlling_equity"),
        ("ifrs_Revenue", "revenue"),
        ("ifrs_ProfitLoss", "net_income"),
        ("ifrs_Assets", "total_assets"),
        ("ifrs_Equity", "total_equity"),
    ],
)
def test_ifrs_prefix_variant_maps_like_full(account_id, expected):
    """`ifrs_X`(하이픈 없는 구표기)도 `ifrs-full_X`와 동일 표준키로 매핑된다."""
    # account_nm은 비어 있어도 account_id만으로 매핑돼야 한다(계정명 표기에 안 흔들림).
    assert normalize_account("", account_id) == expected


# ── 회귀: 기존 표준(ifrs-full_) 매핑 유지 ────────────────────────────────
@pytest.mark.parametrize(
    "account_id, expected",
    [
        ("ifrs-full_ProfitLossAttributableToOwnersOfParent", "controlling_net_income"),
        ("ifrs-full_EquityAttributableToOwnersOfParent", "controlling_equity"),
        ("ifrs-full_Revenue", "revenue"),
    ],
)
def test_full_prefix_still_maps(account_id, expected):
    assert normalize_account("아무이름", account_id) == expected


# ── 회귀: 이름 매칭 유지 (account_id 없을 때) ────────────────────────────
@pytest.mark.parametrize(
    "name, expected",
    [
        ("매출액", "revenue"),
        ("영업수익", "revenue"),
        ("판매비와관리비", "sga"),
        ("매출원가", "cost_of_sales"),
        ("당기순이익", "net_income"),
    ],
)
def test_name_matching_still_works(name, expected):
    assert normalize_account(name, None) == expected


# ── 회귀: 비지배지분은 여전히 차단(None) ────────────────────────────────
@pytest.mark.parametrize(
    "name, account_id",
    [
        ("비지배지분", "ifrs-full_ProfitLossAttributableToNoncontrollingInterests"),
        ("비지배지분", "ifrs_ProfitLossAttributableToNoncontrollingInterests"),
        ("비지배지분에귀속되는당기순이익(손실)", "ifrs-full_ProfitLossAttributableToNoncontrollingInterests"),
    ],
)
def test_noncontrolling_still_excluded(name, account_id):
    """'비지배'는 _EXCLUDE로 차단돼 controlling_* 로 오매칭되지 않는다."""
    assert normalize_account(name, account_id) is None


# ── 사이클 2: _DART_ID_MAP 확장 (이름매칭 계정을 표준 element ID로 승격) ──
@pytest.mark.parametrize(
    "account_id, expected",
    [
        ("ifrs-full_CostOfSales", "cost_of_sales"),
        ("ifrs-full_GrossProfit", "gross_profit"),
        ("ifrs-full_CurrentAssets", "current_assets"),
        ("ifrs-full_NoncurrentAssets", "non_current_assets"),
        ("ifrs-full_CurrentLiabilities", "current_liabilities"),
        ("ifrs-full_NoncurrentLiabilities", "non_current_liabilities"),
        ("ifrs-full_CashFlowsFromUsedInOperatingActivities", "operating_cashflow"),
        ("ifrs-full_DividendsPaid", "dividend"),
        # 접두사 변형(ifrs_)도 확장된 코드에 동일 적용
        ("ifrs_CurrentAssets", "current_assets"),
        ("ifrs_CostOfSales", "cost_of_sales"),
    ],
)
def test_expanded_id_map_promotes_name_only_accounts(account_id, expected):
    """표준 element ID만으로(계정명 없이) 매핑된다 — 비표준 계정명에 안 흔들림."""
    assert normalize_account("", account_id) == expected


@pytest.mark.parametrize(
    "account_id",
    [
        "ifrs-full_CashFlowsFromUsedInInvestingActivities",   # 투자활동 — 영업현금흐름 아님
        "ifrs-full_CashFlowsFromUsedInFinancingActivities",   # 재무활동 — 영업현금흐름 아님
    ],
)
def test_investing_financing_cashflow_not_mapped(account_id):
    """영업활동 외 현금흐름은 operating_cashflow로 오매칭되지 않는다(None)."""
    assert normalize_account("", account_id) is None
