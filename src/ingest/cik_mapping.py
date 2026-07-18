"""SEC 티커→CIK(회사고유번호) 매핑 — company_tickers.json 파서 + us_company 백필.

.omc/specs/brainstorming-sec-edgar-us-financials-backfill.md AC1/AC2:
SEC 공식 `company_tickers.json`(무료, 인증 불필요)으로 us_company 의 NASDAQ/NYSE/
NYSE Amex 전종목에 CIK 를 매핑한다. 매핑률<95% 면 매핑 실패 종목 목록을 포함한 명시적
경고 리포트를 돌려준다(조용히 스킵 금지 — 스펙 Constraints).

CIK 는 SEC XBRL API 경로(`data.sec.gov/api/xbrl/companyfacts/CIK{10자리}.json`)와
us_financials_sec.cik 저장에 쓰이는 10자리 zero-pad 문자열로 정규화한다.

네트워크는 fetch_tickers_fn 주입(DI)으로 분리해 단위 테스트한다(us_universe.py 의
fetch_exchange, us_financials.py 의 fetch_statements 주입 관례와 동일).
"""
from __future__ import annotations

from typing import Callable, Optional

from ..db import connect, init_db

# SEC 공식 티커→CIK 매핑 파일. 인증/API키 불필요, User-Agent 헤더는 예의상 부여한다.
_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
# 프로젝트 연락처를 담은 User-Agent(SEC 권고). us_financials_sec 의 SEC API 호출과 공유한다.
_USER_AGENT = "dart-text2sql-wiki (sonjong980304@gmail.com)"
# 매핑률 경고 임계값(스펙 Constraints/AC2). 이 값 미만이면 warning=True + 실패목록 보고.
MAPPING_WARN_THRESHOLD = 0.95


def format_cik(cik: int | str) -> str:
    """CIK 를 SEC 표준 10자리 zero-pad 문자열로 정규화한다.

    company_tickers.json 은 cik_str 을 int(예: 320193)로 주지만, XBRL API 경로와
    저장은 10자리(예: '0000320193')를 쓴다. 이미 패딩된 문자열도 그대로 안전하게 통과.
    """
    return str(int(str(cik).strip())).zfill(10)


def parse_company_tickers(raw: dict) -> dict[str, str]:
    """SEC company_tickers.json 원시 JSON 을 {티커(대문자): CIK(10자리)} dict 로 변환.

    원시 형식: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    (인덱스 문자열 키로 감싼 dict). 티커는 대문자로 정규화한다(us_company.stock_code 는
    NASDAQ 스크리너 심볼로 대문자).
    """
    mapping: dict[str, str] = {}
    for entry in raw.values():
        ticker = (entry.get("ticker") or "").strip().upper()
        cik_raw = entry.get("cik_str")
        if not ticker or cik_raw is None:
            continue
        mapping[ticker] = format_cik(cik_raw)
    return mapping


def _fetch_company_tickers() -> dict:
    """SEC company_tickers.json 를 실제로 내려받아 원시 dict 로 반환(지연 import)."""
    import requests

    resp = requests.get(
        _COMPANY_TICKERS_URL,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def backfill_ciks(
    db_path: str | None = None,
    fetch_tickers_fn: Optional[Callable[[], dict]] = None,
) -> dict:
    """us_company 의 cik 가 아직 NULL 인 종목에 SEC 매핑으로 CIK 를 UPDATE 하고 리포트 반환.

    fetch_tickers_fn 미지정 시 실제 SEC(_fetch_company_tickers)를 쓴다. 테스트는 항상 DI 로
    가짜 매핑을 주입한다. 이미 cik 가 채워진 종목은 대상에서 제외(멱등, 기존 backfill 관례).
    fetch_tickers_fn 은 이미 파싱된 {티커:CIK} dict 를 돌려줘도 되고(테스트 관례), SEC 원시
    JSON({"0":{...}} 형태)을 돌려줘도 된다 — 후자는 parse_company_tickers 로 자동 정규화한다.

    반환: {"total", "matched", "unmatched"(리스트), "matched_rate", "threshold", "warning"}.
    매핑률<95% 면 warning=True 이고 unmatched 에 실패 종목이 명시된다(AC2, 조용한 스킵 금지).
    """
    fetch_tickers_fn = fetch_tickers_fn or _fetch_company_tickers
    init_db(db_path)
    conn = connect(db_path)
    try:
        codes = [
            r["stock_code"]
            for r in conn.execute(
                "SELECT stock_code FROM us_company WHERE cik IS NULL ORDER BY stock_code"
            ).fetchall()
        ]
        if not codes:
            return {
                "total": 0, "matched": 0, "unmatched": [],
                "matched_rate": 1.0, "threshold": MAPPING_WARN_THRESHOLD, "warning": False,
            }

        raw = fetch_tickers_fn()
        # 원시 SEC JSON({"0":{"ticker":...}})이면 파싱, 이미 {티커:CIK} 면 그대로 사용.
        if raw and isinstance(next(iter(raw.values())), dict):
            mapping = parse_company_tickers(raw)
        else:
            mapping = {str(k).upper(): str(v) for k, v in raw.items()}

        matched = 0
        unmatched: list[str] = []
        for code in codes:
            cik = mapping.get(code.strip().upper())
            if cik is None:
                unmatched.append(code)
                continue
            conn.execute("UPDATE us_company SET cik = ? WHERE stock_code = ?", (cik, code))
            matched += 1
        conn.commit()

        total = len(codes)
        matched_rate = matched / total if total else 1.0
        return {
            "total": total,
            "matched": matched,
            "unmatched": unmatched,
            "matched_rate": matched_rate,
            "threshold": MAPPING_WARN_THRESHOLD,
            "warning": matched_rate < MAPPING_WARN_THRESHOLD,
        }
    finally:
        conn.close()
