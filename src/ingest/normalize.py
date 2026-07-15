"""계정과목명 정규화.

회사/공시마다 계정과목 표기가 비표준이다.
예) '매출액' / '수익(매출액)' / '영업수익' / '매출' → revenue
DART account_nm(또는 더미 원본명)을 표준 account_key로 매핑한다.
"""
from __future__ import annotations

import re

from ..db import ACCOUNT_KEYS  # noqa: F401  (account_key 표준 키 참조)

# 우선순위 순서로 검사 (먼저 매칭되는 키 채택)
# 각 키: 원본명에 포함되면 매칭되는 키워드 패턴들.
# 순서 중요: 더 구체적인 계정을 먼저 둔다(매출원가→매출, 비유동자산→유동자산 충돌 방지).
_RULES: list[tuple[str, list[str]]] = [
    ("shares_outstanding", ["발행주식", "유통주식", "주식수", "상장주식"]),
    ("cost_of_sales", ["매출원가"]),
    ("gross_profit", ["매출총이익"]),
    ("sga", ["판매비와관리비", "판매비및관리비", "판관비"]),
    # 지배주주 귀속 순이익 — 연결 net_income('당기순이익')보다 먼저 매칭돼야 한다(부분문자열 충돌).
    # account_id(ifrs-full_ProfitLossAttributableToOwnersOfParent)가 가장 신뢰도 높다.
    ("controlling_net_income", ["지배기업의소유주에게귀속되는당기순이익", "지배기업소유주지분순이익",
                                "지배기업소유주순이익", "지배주주순이익", "지배기업지분순이익",
                                "지배기업의소유주에게귀속되는순이익"]),
    # net_income: '당기순이익'만. '순이익' 단독은 '법인세차감전 순이익'을 오매칭하므로 제외.
    ("net_income", ["당기순이익", "분기순이익", "반기순이익", "당기순손익"]),
    ("operating_cashflow", ["영업활동현금흐름", "영업활동으로인한현금흐름", "영업활동으로 인한 현금흐름"]),
    ("operating_profit", ["영업이익", "영업손익"]),
    ("interest_expense", ["이자비용", "금융비용"]),
    ("revenue", ["매출액", "영업수익", "수익(매출액)"]),
    # 비유동 먼저(부분문자열 충돌): '비유동자산'에 '유동자산'이 포함됨
    ("non_current_assets", ["비유동자산"]),
    ("current_assets", ["유동자산"]),
    ("non_current_liabilities", ["비유동부채"]),
    ("current_liabilities", ["유동부채"]),
    ("total_assets", ["자산총계", "자산 총계"]),
    ("total_liabilities", ["부채총계", "부채 총계"]),
    # 지배주주지분: 보고서별 표기가 제각각이라 패턴을 넓게 둔다. '비지배'는 _EXCLUDE로 사전 차단.
    # (가장 신뢰도 높은 매칭은 _DART_ID_MAP의 account_id이며, normalize_account가 먼저 시도한다.)
    ("controlling_equity", ["지배기업소유주지분", "지배기업소유지분", "지배기업 소유주지분",
                            "지배기업 소유지분", "지배기업의소유주에게귀속되는자본",
                            "지배기업소유주에게귀속되는자본", "지배기업소유주귀속", "지배주주지분"]),
    ("total_equity", ["자본총계", "자본 총계"]),
    ("depreciation", ["감가상각비"]),
    ("dividend", ["배당금의지급", "배당금지급"]),
]

# 아래 표현이 계정명에 있으면 매칭하지 않는다 (세전이익/계속·중단영업/주당 배제).
# '비지배'는 '비지배지분'이 '지배주주지분' 패턴에 부분문자열로 오매칭되는 것을 차단한다.
_EXCLUDE_KEYWORDS = ["차감전", "차감후", "계속영업", "중단영업", "주당", "자본과부채", "매출채권",
                     "기타영업", "기타수익", "기타의영업", "투자영업", "비지배"]

# DART 표준 account_id (재무제표 표준계정) 매핑 (보조 — 계정명보다 신뢰도 높아 우선 적용)
_DART_ID_MAP = {
    "ifrs-full_Revenue": "revenue",
    "ifrs-full_OperatingIncomeLoss": "operating_profit",
    "dart_OperatingIncomeLoss": "operating_profit",
    "ifrs-full_ProfitLoss": "net_income",
    "ifrs-full_ProfitLossAttributableToOwnersOfParent": "controlling_net_income",
    "ifrs-full_Assets": "total_assets",
    "ifrs-full_Liabilities": "total_liabilities",
    "ifrs-full_Equity": "total_equity",
    # 지배기업 소유주귀속 자본(지배주주지분) — 보고서별 계정명 표기가 흔들려도 표준코드로 안정 매핑
    "ifrs-full_EquityAttributableToOwnersOfParent": "controlling_equity",
    # 이름매칭에만 의존하던 계정을 표준 element ID로 승격(계정명 표기 변형에 견고).
    "ifrs-full_CostOfSales": "cost_of_sales",
    "ifrs-full_GrossProfit": "gross_profit",
    "ifrs-full_CurrentAssets": "current_assets",
    "ifrs-full_NoncurrentAssets": "non_current_assets",
    "ifrs-full_CurrentLiabilities": "current_liabilities",
    "ifrs-full_NoncurrentLiabilities": "non_current_liabilities",
    # 현금흐름표: 영업활동만 매핑(투자/재무 활동은 표준코드가 달라 자동 제외)
    "ifrs-full_CashFlowsFromUsedInOperatingActivities": "operating_cashflow",
    "ifrs-full_DividendsPaid": "dividend",
}


def _canon_account_id(account_id: str | None) -> str | None:
    """account_id 접두사 정규화. DART가 같은 표준요소를 `ifrs_X`(구표기)와
    `ifrs-full_X`(현행) 두 가지로 내보내므로, 구표기를 현행으로 통일해 _DART_ID_MAP 조회를
    한 벌로 유지한다. (예: ifrs_Revenue → ifrs-full_Revenue)"""
    if account_id and account_id.startswith("ifrs_"):
        return "ifrs-full_" + account_id[len("ifrs_"):]
    return account_id


def normalize_account(account_name: str, account_id: str | None = None) -> str | None:
    """원본 계정명(+선택적 account_id) → 표준 account_key. 매칭 실패 시 None."""
    name = re.sub(r"\s+", "", account_name or "")
    # 세전이익/원가/주당 등 혼동 항목은 먼저 배제
    if any(e in name for e in _EXCLUDE_KEYWORDS):
        return None
    aid = _canon_account_id(account_id)
    if aid and aid in _DART_ID_MAP:
        return _DART_ID_MAP[aid]
    for key, patterns in _RULES:
        for p in patterns:
            if re.sub(r"\s+", "", p) in name:
                return key
    return None


# 더미/표시용: 회사별 비표준 표기를 일부러 섞어 정규화가 동작함을 보인다.
VARIANT_NAMES: dict[str, list[str]] = {
    "revenue": ["매출액", "영업수익", "수익(매출액)", "매출"],
    "operating_profit": ["영업이익", "영업이익(손실)"],
    "net_income": ["당기순이익", "당기순이익(손실)", "분기순이익"],
    "total_assets": ["자산총계"],
    "total_liabilities": ["부채총계"],
    "total_equity": ["자본총계"],
    "shares_outstanding": ["발행주식수", "상장주식수"],
}


def variant_for(account_key: str, salt: int) -> str:
    """account_key에 대해 결정적으로 비표준 표기 하나를 선택 (더미용)."""
    opts = VARIANT_NAMES.get(account_key, [account_key])
    return opts[salt % len(opts)]
