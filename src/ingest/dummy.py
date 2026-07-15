"""더미 데이터 생성 (DART/pykrx 키 없이 전체 파이프라인 테스트용).

결정적(seed 고정) 이므로 재현 가능. PER/부채비율/영업이익률 등이
회사별로 다양하게 분포하도록 설계해 질의가 의미 있게 동작한다.
일부 회사는 적자(순이익<0 → PER None)로 만들어 NULL 처리도 검증한다.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from ..db import connect, init_db, set_meta
from ..version import (
    estimate_available_quarter,
    estimate_disclosed_date,
    now_iso,
    recent_quarters,
    shift_quarter,
)
from .companies import COMPANIES
from .metrics import compute_metrics
from .normalize import variant_for

PRICE_DAYS = 5  # 최근 거래일 수 (스냅샷 다양성)


def generate_dummy(db_path: str | None = None, today: date | None = None, seed: int = 42) -> dict:
    today = today or date.today()
    init_db(db_path)
    conn = connect(db_path)
    try:
        # 기존 시장 데이터 초기화 (wiki/result_cache는 보존)
        for t in ("financials", "prices", "metrics", "company"):
            conn.execute(f"DELETE FROM {t}")

        latest_q = estimate_available_quarter(today)
        quarters = recent_quarters(latest_q, 12)  # 최근 3개년(12분기)

        n_fin = 0
        for idx, (code, name, market, sector) in enumerate(COMPANIES):
            conn.execute(
                "INSERT INTO company(stock_code, name, market, sector) VALUES(?,?,?,?)",
                (code, name, market, sector),
            )
            rng = random.Random(int(code) + seed)

            # 회사 규모: 인덱스가 작을수록 대형 (분기 매출, 원)
            base_rev_q = max(8e11, 70e12 * (0.90 ** idx))
            margin = rng.uniform(0.03, 0.22)
            is_loss = rng.random() < 0.15  # 15% 적자기업
            if is_loss:
                margin = rng.uniform(-0.08, 0.01)
            net_margin_ratio = rng.uniform(0.55, 0.85)  # 순이익/영업이익 (대략)
            equity_mult = rng.uniform(1.8, 4.5)  # 자본/연매출
            debt_ratio_t = rng.uniform(0.3, 2.6)  # 부채/자본
            shares = float(rng.randint(8_000_000, 900_000_000))

            # 분기별 재무 생성
            net_by_q: dict[str, float] = {}
            equity_latest = liab_latest = 0.0
            for qi, q in enumerate(quarters):
                growth = 1.0 + 0.015 * qi + rng.uniform(-0.08, 0.08)
                revenue = base_rev_q * growth
                op = revenue * margin
                net = op * net_margin_ratio if op >= 0 else op * rng.uniform(1.0, 1.4)
                equity = base_rev_q * 4 * equity_mult  # 연매출 환산 * 배수
                liab = equity * debt_ratio_t
                assets = equity + liab
                net_by_q[q] = net
                equity_latest, liab_latest = equity, liab

                rows = {
                    "revenue": revenue,
                    "operating_profit": op,
                    "net_income": net,
                    "total_assets": assets,
                    "total_liabilities": liab,
                    "total_equity": equity,
                    "shares_outstanding": shares,
                }
                for key, amount in rows.items():
                    conn.execute(
                        """INSERT INTO financials
                             (stock_code, quarter, disclosed_date, account_key, account_name, amount)
                           VALUES (?,?,?,?,?,?)""",
                        (
                            code,
                            q,
                            estimate_disclosed_date(q),
                            key,
                            variant_for(key, idx),
                            round(amount),
                        ),
                    )
                    n_fin += 1

            # 시가총액 역산 (PER/PBR이 그럴듯하도록)
            ni_ttm = sum(net_by_q[q] for q in quarters[-4:])
            if ni_ttm > 0:
                per_t = rng.uniform(5, 45)
                if rng.random() < 0.15:
                    per_t = rng.uniform(45, 80)  # 일부 고PER
                market_cap = per_t * ni_ttm
            else:
                pbr_t = rng.uniform(0.4, 2.5)
                market_cap = pbr_t * equity_latest
            target_close = rng.uniform(8000, 300000)
            shares_i = max(1_000_000, round(market_cap / target_close))

            # 최근 PRICE_DAYS 거래일 종가 (날짜별 ±2% 변동)
            for di in range(PRICE_DAYS):
                day = today - timedelta(days=di)
                wobble = 1.0 + rng.uniform(-0.02, 0.02) * (di > 0)
                close = round(market_cap * wobble / shares_i, 1)
                cap = round(close * shares_i)
                conn.execute(
                    """INSERT INTO prices(stock_code, date, close, market_cap)
                       VALUES (?,?,?,?)""",
                    (code, day.isoformat(), close, cap),
                )

        # 적재 메타: 실제 공시된 최신 분기 / 종가 스냅샷 날짜
        set_meta(conn, "latest_disclosed_quarter", latest_q)
        set_meta(conn, "latest_price_date", today.isoformat())
        conn.commit()

        n_metrics = compute_metrics(conn, today)
        return {
            "companies": len(COMPANIES),
            "financials": n_fin,
            "quarters": quarters,
            "latest_quarter": latest_q,
            "price_date": today.isoformat(),
            "metrics": n_metrics,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    info = generate_dummy()
    print("더미 데이터 생성 완료:")
    for k, v in info.items():
        print(f"  {k}: {v}")
