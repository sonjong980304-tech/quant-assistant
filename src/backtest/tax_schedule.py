"""매도 증권거래세+농어촌특별세 합산 실효세율의 연도별 스케줄.

1차 출처 검증 근거: references/securities_transaction_tax_rate_history.md.
날짜는 프로젝트 관례대로 'YYYY-MM-DD' 문자열 사전식 비교로 처리한다(data_access.py의
asof 비교와 동일한 관례 — datetime 파싱 불필요). 코스피/코스닥은 세부 구성(거래세 vs
거래세+농특세)만 다를 뿐 합산 실효세율은 항상 같은 해에 일치하므로 시장 구분 없이 단일
스케줄을 쓴다.
"""
from __future__ import annotations

# (시행일, 세율) 내림차순. stt_rate_at은 date 이상인 첫 threshold를 찾는다.
_SCHEDULE = [
    ("2026-01-01", 0.0020),
    ("2025-01-01", 0.0015),
    ("2024-01-01", 0.0018),
    ("2023-01-01", 0.0020),
    ("2021-01-01", 0.0023),
    ("2019-06-03", 0.0025),
]
_PRE_2019_RATE = 0.0030  # 1996-04 ~ 2019-06-02 (이 프로젝트 가격 데이터는 2014년부터라 그 이전은 다루지 않음)


def stt_rate_at(date: str) -> float:
    """asof 날짜(YYYY-MM-DD)에 적용되는 매도 거래세+농특세 합산 실효세율."""
    for threshold, rate in _SCHEDULE:
        if date >= threshold:
            return rate
    return _PRE_2019_RATE
