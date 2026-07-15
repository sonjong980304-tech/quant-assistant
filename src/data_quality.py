"""가격 데이터 품질 게이트 — 신뢰 못 할 종목을 계산에서 통째로 제외한다.

배경
----
prices.close 에 "하루 만에 종가가 2배 이상 뛰거나 절반 이하로 떨어지는" 구간을 가진
종목이 있다. 원인은 대부분 액면분할/병합이 수정주가로 소급 반영되지 않은 데이터
버그이거나(예: 저유동 동전주가 오랫동안 동일 종가로 고정됐다가 정수배로 점프),
드물게는 원본 파싱 오류다(예: 종가 1원이 몇 달 이어지다 67000원으로 점프).

원인이 다양하고 계속 새로운 예외 케이스가 발견되므로(예: 246830 시냅스엠은 상승
정수배 점프 외에 하락 폭락 구간도 섞여 있어 배율 하나로 "복원"이 안 됐고, 상승만
보던 기존 임계값(abs(ratio-1)>1.0, scripts/fix_split_discontinuities.py)은 하락
폭락을 아예 못 잡는 사각지대였다), 배율을 찾아 곱해서 "고치는" 접근 대신 **신뢰
못 할 종목 전체를 계산에서 빼는** 더 안전하고 유지보수하기 쉬운 접근으로 전환한다.

판정
----
종목의 전체 가격 이력 중 어디든 인접 거래일 종가비가 2배 이상(ratio>=2.0) 이거나
1/2 이하(ratio<=0.5)면 그 종목 전체를 제외한다 — 종목 단위 블랭킷 제외다(특정
구간만 빼는 정교한 방식이 아니다. 단순하고 안전한 쪽을 택함). 상승/하락을 대칭으로
본다: 배율로 보면 2배 점프든 1/2배 급락이든 크기가 같은 비정상 변동이다.

캐싱
----
전체 스캔(prices ~840만 행, LAG 윈도우+DISTINCT)은 실측 약 6~7초 걸린다(2026-07
data/market.db 기준). 백테스트 1회 실행이 리밸런싱 시점마다 유니버스(=metrics_at)를
여러 번 구성하는 구조라 요청마다 돌리기엔 비싸므로, ingest_meta에 결과를 CSV로
캐싱한다(기존 usdkrw_rate/inactive_codes 캐싱 관례와 동일 패턴 — src/ingest/dart.py
inactive_codes, src/ingest/exchange_rate.py usdkrw_rate 참고). 최초 호출 시 계산해
저장하고, 이후 호출은 ingest_meta 단건 조회(수 ms)로 끝난다.

캐시가 있으면 재스캔하지 않으므로, 새로 적재된 가격에서 생긴 이상치는 즉시 반영되지
않는다(다음 refresh=True 호출 전까지). 이 트레이드오프는 의도적이다 — "신뢰 못 할
종목은 어차피 과거 이력이 오염된 것"이므로 하루이틀 늦게 걸러져도(최악의 경우) 매
요청마다 6~7초를 태우는 것보다 안전하다.
"""
from __future__ import annotations

import sqlite3

from .db import get_meta, set_meta
from .version import now_iso

# fix_split_discontinuities.py 의 상승전용 임계값(JUMP_THRESHOLD=1.0, 즉 ratio>2.0)을
# 하락 방향(ratio<=0.5)까지 대칭으로 확장했다. 정수배 여부는 따지지 않는다 — 이 모듈은
# "정수배로 보정 가능한가"가 아니라 "이 종목의 과거 이력을 신뢰할 수 있는가"만 판단한다.
JUMP_RATIO_HIGH = 2.0
JUMP_RATIO_LOW = 0.5

_CACHE_KEY = "price_quality_excluded_codes"
_CACHE_KEY_AT = "price_quality_excluded_at"


def detect_price_quality_anomalies(
    conn: sqlite3.Connection,
    ratio_high: float = JUMP_RATIO_HIGH,
    ratio_low: float = JUMP_RATIO_LOW,
) -> set[str]:
    """전체 이력을 스캔해, 인접 거래일 종가비가 [ratio_low, ratio_high] 구간 밖인 종목코드 집합.

    scripts/fix_split_discontinuities._raw_jumps 와 동일한 LAG 윈도우 SQL을 대칭
    임계값으로 재사용한다. 캐시를 거치지 않는 순수 스캔 함수라 실제 대용량 DB에서는
    수 초가 걸릴 수 있다(캐싱은 get_price_quality_excluded_codes가 담당).
    """
    rows = conn.execute(
        """
        WITH ch AS (
          SELECT stock_code, date, close,
                 LAG(close) OVER (PARTITION BY stock_code ORDER BY date) AS prev_close
          FROM prices WHERE close IS NOT NULL AND close > 0
        )
        SELECT DISTINCT stock_code FROM ch
        WHERE prev_close IS NOT NULL AND prev_close > 0
          AND (close / prev_close >= ? OR close / prev_close <= ?)
        """,
        (ratio_high, ratio_low),
    ).fetchall()
    return {r["stock_code"] for r in rows}


def get_price_quality_excluded_codes(
    conn: sqlite3.Connection, refresh: bool = False
) -> set[str]:
    """제외 대상 종목코드 집합(ingest_meta 캐시 우선).

    refresh=False(기본): 캐시가 있으면 그대로 반환(재스캔 없음). 캐시가 없으면(최초
    호출) 자동으로 스캔해 채운다.
    refresh=True: 캐시 유무와 무관하게 강제로 재스캔하고 캐시를 갱신한다(예: 일일
    가격 적재 후 최신화하려는 배치 호출용).
    """
    if not refresh:
        cached = get_meta(conn, _CACHE_KEY)
        if cached is not None:
            return set(cached.split(",")) if cached else set()

    codes = detect_price_quality_anomalies(conn)
    try:
        set_meta(conn, _CACHE_KEY, ",".join(sorted(codes)))
        set_meta(conn, _CACHE_KEY_AT, now_iso())
        conn.commit()
    except sqlite3.OperationalError:
        # 읽기전용 연결(connect_readonly — LLM SQL 실행/에이전트 응답 경로가 방어적으로
        # 쓰는 연결)에서는 캐시에 쓸 수 없다("attempt to write a readonly database").
        # 계산 결과 자체는 정상 반환하고 캐시 저장만 건너뛴다 — 다음에 쓰기가능
        # 연결(배치 갱신 스크립트 등)이 호출될 때 캐시가 채워진다.
        pass
    return codes
