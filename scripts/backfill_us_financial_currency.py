"""us_company.financial_currency 배치 수집 (HA15 후속 — 재무제표 보고통화 확인).

배경: SK텔레콤(SKM, NYSE 상장 ADR)의 "26년 1분기 영업이익 1위" 재현 결과가 이상해서
조사해보니 us_financials 원본 숫자가 원화(KRW) 단위였다(yfinance
`Ticker('SKM').get_info()['financialCurrency'] == 'KRW'`, NVDA 등은 'USD'). 주가
(us_prices)는 항상 달러로 거래되지만, 외국기업 ADR은 실적발표 자체를 본국통화로 하는
경우가 흔해 주가통화와 재무제표통화가 다를 수 있다. 이 스크립트가 없으면
metrics_at_us()가 시가총액(달러)을 순이익(원화, 훨씬 큰 숫자)으로 나누는 식의 통화
불일치 계산오류가 전 종목에 조용히 섞여 있다(PER 0.035처럼 비현실적 값 → 랭킹 오염).

"판단은 API, 실행은 캐싱" 원칙(scripts/backfill_us_security_type.py와 동일): 매
스크리닝 요청마다 yfinance를 부르지 않도록, 이 스크립트가 1회 배치로 financialCurrency를
수집해 us_company.financial_currency 컬럼에 캐싱한다. 스크리닝/백테스트 런타임
(src/backtest/data_access_us.py::metrics_at_us)은 이 캐시만 읽어 'USD'가 아닌
(NULL이 아닌) 종목의 재무 파생 필드를 무효화한다.

대상 범위: us_financials에 이미 재무데이터가 존재하는(=실제 계산에 쓰이는) 종목만
대상으로 한다 — 재무데이터가 없는 종목은 애초에 metrics_at_us에서 계산되지 않으므로
통화 확인도 불필요하다(전체 종목이 아니라 서브셋으로 범위를 좁혀도 된다는 지시 반영).

idempotent: financial_currency가 이미 채워진 종목은 재수집 대상에서 제외한다(다시
돌려도 안전, backfill_us_security_type.py/backfill_marketcap.py 등 기존 관례와 동일).

rate limiting: 이 코드베이스의 기존 yfinance 배치(src/ingest/us_financials.py의
ingest_us_financials, src/ingest/us_prices.py)는 종목당 인위적 sleep을 두지 않는다
(스로틀은 Naver/FnGuide HTTP 크롤러 전용 ThrottledFetcher에만 있음, us_universe.py의
_REQUEST_DELAY_SEC도 investing.com Selenium 스크래핑 전용). 이 스크립트도 그 기존
관례를 그대로 따른다(새 스로틀 로직을 발명하지 않는다) — 종목별 try/except로 격리해
한 건 실패가 전체를 막지 않게 하는 것으로 충분하다.

실행: python3 scripts/backfill_us_financial_currency.py   ← 수천 건 개별 API 호출,
      네트워크 상황에 따라 오래 걸릴 수 있음(백그라운드 실행 권장).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import connect, init_db
from src.ingest.robust import log_ingest


def _fetch_currency_yf(stock_code: str) -> Optional[str]:
    """yfinance financialCurrency 단일 조회. 지연 import(us_financials.py와 동일 관례)."""
    import yfinance as yf

    t = yf.Ticker(stock_code)
    info = t.get_info()
    return info.get("financialCurrency")


def backfill_financial_currency(
    db_path: str | None = None,
    fetch_currency_fn: Callable[[str], Optional[str]] | None = None,
    limit: int | None = None,
) -> dict:
    """us_financials에 데이터가 있고 financial_currency가 아직 NULL인 종목만 대상으로,
    yfinance financialCurrency를 수집해 us_company.financial_currency에 UPDATE한다.

    fetch_currency_fn 미지정 시 실제 yfinance(_fetch_currency_yf)를 쓴다. 테스트는 항상
    DI로 가짜 함수를 주입한다(ingest_us_financials의 fetch_statements와 동일 관례).
    종목마다 즉시 commit한다(ingest_us_financials.py의 동시쓰기 락 회피 이유와 동일).

    반환: {"total_targets", "usd", "non_usd", "failed", "failed_codes", "currencies"}.
    """
    fetch_currency_fn = fetch_currency_fn or _fetch_currency_yf
    init_db(db_path)
    conn = connect(db_path)
    try:
        sql = (
            "SELECT DISTINCT uc.stock_code FROM us_company uc "
            "WHERE uc.financial_currency IS NULL "
            "AND EXISTS (SELECT 1 FROM us_financials uf WHERE uf.stock_code = uc.stock_code) "
            "ORDER BY uc.stock_code"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        codes = [r["stock_code"] for r in conn.execute(sql).fetchall()]

        usd = 0
        non_usd = 0
        failed_codes: list[str] = []
        currencies: dict[str, int] = {}

        for code in codes:
            try:
                currency = fetch_currency_fn(code)
            except Exception as exc:  # noqa: BLE001 — 종목별 실패를 격리해 다음 종목으로 계속
                failed_codes.append(code)
                log_ingest({
                    "source": "us_financial_currency", "stock_code": code,
                    "status": "fail", "error": str(exc),
                })
                continue
            if not currency:
                failed_codes.append(code)
                log_ingest({
                    "source": "us_financial_currency", "stock_code": code,
                    "status": "fail", "error": "financialCurrency 없음(빈 응답)",
                })
                continue

            currencies[currency] = currencies.get(currency, 0) + 1
            if currency == "USD":
                usd += 1
            else:
                non_usd += 1
            conn.execute(
                "UPDATE us_company SET financial_currency = ? WHERE stock_code = ?",
                (currency, code),
            )
            conn.commit()

        return {
            "total_targets": len(codes),
            "usd": usd,
            "non_usd": non_usd,
            "failed": len(failed_codes),
            "failed_codes": failed_codes,
            "currencies": currencies,
        }
    finally:
        conn.close()


def main() -> None:
    print("us_company.financial_currency 배치 수집 시작", flush=True)
    report = backfill_financial_currency()
    print(f"\n대상 종목 {report['total_targets']}개")
    print(f"USD {report['usd']}개 / 비USD {report['non_usd']}개 / 실패 {report['failed']}개")
    if report["currencies"]:
        print("통화별 카운트:")
        for cur, n in sorted(report["currencies"].items(), key=lambda kv: -kv[1]):
            print(f"  {cur}: {n}")
    if report["failed_codes"]:
        print(f"실패 종목(다음 실행에 재시도됨): {report['failed_codes'][:20]}"
              + (" ..." if len(report["failed_codes"]) > 20 else ""))


if __name__ == "__main__":
    main()
