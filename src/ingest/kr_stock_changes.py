"""KRX 기업 주요 변동이력(상호/업종/액면 변경) 수집·정규화·적재.

pykrx `get_stock_major_changes(ticker)` 는 종목별로 날짜(index)와 상호변경전/후·업종변경전/후·
액면변경전/후·대표이사변경전/후 컬럼을 가진 DataFrame 을 돌려준다(실측: 삼성전자 조회 시 1975년
부터의 전체 이력이 시점조회 없이 한 번에 온다). kr_trading_status 는 KRX 가 '오늘 스냅샷'만 줘서
우리가 매일 diff 로 미래 구간을 쌓아야 했지만, 이 API 는 과거 이력을 통째로 반환하므로 한 번
순회로 전체 이력을 적재할 수 있다.

저장 범위(YAGNI): 스크리닝/백테스트/회사명 매칭에 쓰는 상호·업종·액면 변경만 저장하고 대표이사
변경은 저장하지 않는다. 대표이사만 바뀐 날짜 행(우리가 저장하는 3종이 모두 비는 행)은 스킵한다.

핵심 활용: 자연어→SQL 질의에서 회사명 매칭이 현재 사명(company.name)으로 실패할 때 예전 사명
(name_before/name_after)으로도 종목코드를 찾는다(domain_kr.find_stock_code 폴백). 액면변경(분할/
병합)은 저장만 해두고 이번 스코프에서 분할보정 도구엔 연결하지 않는다(범위 밖).

네트워크(pykrx 호출)는 DI(fetch_fn 주입)로 분리해 단위 테스트한다(kr_trading_status
관례와 동일). 라이브 pykrx 호출은 스크립트를 사용자가 직접 실행할 때만 일어난다. 전종목 순회는
기존 per-ticker pykrx 모듈(krx.py)과 동일한 관례를 그대로 재사용한다: 종목별로 재시도/지연 없이
1회 호출하고 실패는 격리(스킵+Slack 알림)한 뒤 다음 종목으로 계속한다(pykrx 가 조회 실패를 빈
DataFrame 으로 흡수하므로 별도 재시도가 불필요 — 추측으로 새 지연·재시도 값을 만들지 않는다).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

from ..db import connect, init_db
from .notify import send_slack_alert
from .robust import log_ingest

# pykrx get_stock_major_changes 출력 컬럼(순서 고정). 대표이사변경전/후는 저장하지 않는다.
_COL_NAME_BEFORE = "상호변경전"
_COL_NAME_AFTER = "상호변경후"
_COL_SECTOR_BEFORE = "업종변경전"
_COL_SECTOR_AFTER = "업종변경후"
_COL_PAR_BEFORE = "액면변경전"
_COL_PAR_AFTER = "액면변경후"


# ---------------------------------------------------------------------------
# 정규화 헬퍼
# ---------------------------------------------------------------------------
def _clean_text(raw: object) -> Optional[str]:
    """상호/업종 값 정규화. pykrx '없음' 표기 "-" 와 빈 문자열/NaN 은 None 으로."""
    if raw is None:
        return None
    if isinstance(raw, float) and raw != raw:  # NaN
        return None
    s = str(raw).strip()
    if s in ("", "-"):
        return None
    return s


def _clean_par(raw: object) -> Optional[int]:
    """액면가 값 정규화. pykrx '변경없음' 센티널 0 은 None 으로, 그 외는 정수로."""
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return None
    return v if v != 0 else None


def _normalize_date(idx: object) -> Optional[str]:
    """pykrx 날짜 인덱스(Timestamp) 또는 문자열을 'YYYY-MM-DD' 로 정규화한다."""
    if idx is None:
        return None
    if hasattr(idx, "strftime"):  # pandas Timestamp / datetime
        return idx.strftime("%Y-%m-%d")
    s = str(idx).strip().split(" ")[0].replace("/", "-")
    if not s:
        return None
    if len(s) == 8 and s.isdigit():  # 'YYYYMMDD'
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s[:10]


def parse_major_changes(df: object, stock_code: str) -> list[dict]:
    """pykrx get_stock_major_changes DataFrame → kr_stock_changes 정규화 행 리스트.

    각 행: {stock_code, changed_at, name_before, name_after, sector_before, sector_after,
    par_before, par_after}. '없음'(텍스트 "-" / 액면 0)은 None 으로 정규화하고, 우리가 저장하는
    상호·업종·액면이 모두 없는 행(= 대표이사만 바뀐 날짜)은 스킵한다(스크리닝/매칭에 무관, YAGNI).
    빈/None DataFrame 은 빈 리스트.
    """
    rows: list[dict] = []
    if df is None or len(df) == 0:
        return rows
    for idx, r in df.iterrows():
        name_before = _clean_text(r.get(_COL_NAME_BEFORE))
        name_after = _clean_text(r.get(_COL_NAME_AFTER))
        sector_before = _clean_text(r.get(_COL_SECTOR_BEFORE))
        sector_after = _clean_text(r.get(_COL_SECTOR_AFTER))
        par_before = _clean_par(r.get(_COL_PAR_BEFORE))
        par_after = _clean_par(r.get(_COL_PAR_AFTER))
        if not any((name_before, name_after, sector_before, sector_after,
                    par_before is not None, par_after is not None)):
            continue  # 저장 대상(상호/업종/액면)이 없는 행 → 대표이사만 변경, 스킵
        rows.append({
            "stock_code": stock_code,
            "changed_at": _normalize_date(idx),
            "name_before": name_before,
            "name_after": name_after,
            "sector_before": sector_before,
            "sector_after": sector_after,
            "par_before": par_before,
            "par_after": par_after,
        })
    return rows


# ---------------------------------------------------------------------------
# 실제 pykrx 호출(라이브 전용, 지연 import — pykrx import 시 자동 로그인 회피)
# ---------------------------------------------------------------------------
def _default_fetch(stock_code: str) -> list[dict]:
    """실제 pykrx get_stock_major_changes 호출 → 정규화 행.

    krx.py(per-ticker pykrx 순회)와 동일하게 재시도/지연 없이 1회 호출한다 — pykrx 자체가
    조회 실패를 빈 DataFrame 으로 흡수(@dataframe_empty_handler)하고, 진짜 네트워크 예외는
    상위 ingest 루프가 종목별로 격리(스킵+Slack)한다. get_stock_major_changes 는 DataFrame 을
    돌려주므로 robust.call_with_retry(결과 truthiness 검사)는 여기 부적합하다(빈 이력 종목을
    매번 재시도하거나 DataFrame 진리값 모호성 예외를 유발)."""
    from pykrx import stock  # 지연 import(네트워크 의존)

    df = stock.get_stock_major_changes(stock_code)
    return parse_major_changes(df, stock_code)


# ---------------------------------------------------------------------------
# 수집: 전종목 순회 → 이력 upsert
# ---------------------------------------------------------------------------
def ingest_stock_changes(
    db_path: str | None = None,
    codes: list[str] | None = None,
    fetch_fn: Optional[Callable[[str], list[dict]]] = None,
    commit_every: int = 20,
    updated_at: str | None = None,
) -> dict:
    """종목별 기업 주요 변동이력을 받아 kr_stock_changes 에 upsert 한다.

    - codes: 순회할 종목코드 리스트(기본=company 테이블 전체). 전종목 백필/부분 갱신 공통 경로.
    - fetch_fn(code) -> 정규화 행 리스트(DI, 기본=실제 pykrx 호출). 테스트는 mock 을 주입해
      네트워크 없이 검증한다(라이브 pykrx 호출은 절대 하지 않음).
    - 종목별 실패는 격리(스킵+Slack 알림)하고 다음 종목으로 계속한다(전종목 백필이 한 종목
      때문에 통째로 죽지 않게 — krx.py/naver_prices.py 관례). commit_every 개마다 주기 커밋.
    - UNIQUE(stock_code, changed_at) + INSERT OR REPLACE 로 재실행 멱등(값 정정도 반영).

    반환: {"tickers": 성공 종목 수, "rows_stored": 저장 행 수, "failed": 실패 종목코드 리스트,
           "total_codes": 순회 대상 종목 수}.
    """
    fetch_fn = fetch_fn or _default_fetch
    updated_at = updated_at or datetime.now(timezone.utc).isoformat()

    init_db(db_path)
    conn = connect(db_path)
    try:
        if codes is None:
            codes = [r["stock_code"] for r in conn.execute("SELECT stock_code FROM company").fetchall()]

        stored = 0
        done = 0
        failed: list[str] = []
        for code in codes:
            try:
                rows = fetch_fn(code)
            except Exception as exc:  # noqa: BLE001 — 종목별 실패 격리(다음 종목으로 계속)
                failed.append(code)
                log_ingest({"source": "kr_stock_changes", "stock_code": code,
                            "status": "fail", "error": str(exc)})
                send_slack_alert(f"[kr_stock_changes] {code} 수집 실패: {exc}")
                continue
            for r in rows or []:
                conn.execute(
                    "INSERT OR REPLACE INTO kr_stock_changes"
                    "(id, stock_code, changed_at, name_before, name_after, sector_before, "
                    "sector_after, par_before, par_after, updated_at) VALUES "
                    "((SELECT id FROM kr_stock_changes WHERE stock_code=? AND changed_at=?), "
                    "?,?,?,?,?,?,?,?,?)",
                    (r["stock_code"], r["changed_at"],
                     r["stock_code"], r["changed_at"], r["name_before"], r["name_after"],
                     r["sector_before"], r["sector_after"], r["par_before"], r["par_after"],
                     updated_at),
                )
                stored += 1
            done += 1
            if done % commit_every == 0:
                conn.commit()
        conn.commit()
        return {
            "tickers": done,
            "rows_stored": stored,
            "failed": failed,
            "total_codes": len(codes),
        }
    finally:
        conn.close()


if __name__ == "__main__":
    print(ingest_stock_changes())
