"""공시시즌 재무 증분 갱신 (LLM 미사용, 수집 비용 0).

분기보고서 공시 마감(1Q≈5/15, 반기≈8/14, 3Q≈11/14, 사업보고서≈익년 3/31)
전후로 새 공시를 점검한다. DART에 실제로 공시된 분기만 적재하며, 날짜로 분기를
단순 추정하지 않는다 — fetch_all_accounts 응답이 비어 있지 않은(=공시 존재)
분기만 받는다.

이미 financials에 들어 있는 (종목, 분기)는 스킵하고, 새로 공시된 분기만 수집한다.
사업보고서(Q4)의 FLOW 계정(손익·현금흐름)은 연간 누적이므로
Q4 단독 = 연간 - (Q1+Q2+Q3)로 차감한다(dart.ingest_dart와 동일 규칙).

실행: python3 scripts/update_financials.py
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.db import connect, init_db, set_meta
from src.ingest.dart import (
    FLOW_KEYS,
    REPRT_CODE,
    _ingest_shares,
    _insert_revision,
    _parse_all,
    fetch_all_accounts,
    get_corp_codes,
)
from src.version import (
    estimate_available_quarter,
    estimate_disclosed_date,
    recent_quarters,
)

# universe.py가 있으면 사용, 없으면 COMPANIES 폴백 (다른 에이전트가 작성 중일 수 있음)
try:
    from src.ingest.universe import get_universe

    def _codes():
        return [row[0] for row in get_universe()]
except Exception:
    from src.ingest.companies import COMPANIES

    def _codes():
        return [row[0] for row in COMPANIES]


def _q_sort_key(q: str) -> tuple:
    return int(q[:4]), int(q[5])


def _existing_pairs(conn) -> set:
    """이미 적재된 (종목, 분기) 집합."""
    rows = conn.execute("SELECT DISTINCT stock_code, quarter FROM financials").fetchall()
    return {(r["stock_code"], r["quarter"]) for r in rows}


def _candidate_quarters(today: date, back: int = 5) -> list:
    """공시되었을 법한 최근 분기 후보(공시시즌 경계 흔들림 대비 여유 포함)."""
    latest_q = estimate_available_quarter(today)
    return recent_quarters(latest_q, back)


def update_financials(db_path: str | None = None, today: date | None = None, sleep: float = 0.4) -> dict:
    """새로 공시된 (종목, 분기)만 증분 적재. 이미 있는 쌍은 스킵."""
    if not CONFIG.has_dart_key:
        raise RuntimeError("DART_API_KEY 없음 — .env의 DART_API_KEY 확인")

    today = today or date.today()
    api_key = CONFIG.dart_api_key
    init_db(db_path)
    conn = connect(db_path)
    try:
        corp_map = get_corp_codes(api_key)
        existing = _existing_pairs(conn)
        cand_quarters = _candidate_quarters(today)
        # 후보 분기가 속한 연도(사업보고서 Q4 차감을 위해 같은 연도 1~4Q를 함께 조회)
        years = sorted({int(q[:4]) for q in cand_quarters})
        want_qs = set(cand_quarters)

        new_pairs: list = []
        actual_latest = None
        n_rows = 0

        for code in _codes():
            corp = corp_map.get(code)
            if not corp:
                continue
            # 이 종목에서 받을 게 하나라도 있는지(전부 이미 있으면 API 호출 자체를 생략)
            todo = [q for q in cand_quarters if (code, q) not in existing]
            if not todo:
                continue

            for year in years:
                # 차감을 위해 한 연도의 4개 보고서를 모두 받아둔다(연결 우선, 없으면 별도).
                reports = {}  # qn -> (data, rcept)
                for qn in (1, 2, 3, 4):
                    rows = fetch_all_accounts(api_key, corp, year, REPRT_CODE[qn], "CFS")
                    time.sleep(sleep)
                    data, rcept = _parse_all(rows)
                    if not data:
                        rows = fetch_all_accounts(api_key, corp, year, REPRT_CODE[qn], "OFS")
                        time.sleep(sleep)
                        data, rcept = _parse_all(rows)
                    reports[qn] = (data, rcept)

                for qn in (1, 2, 3, 4):
                    quarter = f"{year}Q{qn}"
                    if quarter not in want_qs or (code, quarter) in existing:
                        continue
                    data, rcept = reports[qn]
                    if not data:
                        # DART에 아직 공시 없음 → 추정으로 만들지 않고 스킵
                        continue
                    disclosed = (
                        f"{rcept[:4]}-{rcept[4:6]}-{rcept[6:8]}"
                        if rcept and len(rcept) == 8
                        else estimate_disclosed_date(quarter)
                    )
                    rcept_no = rcept or disclosed  # 접수번호 미상이면 공시일을 멱등 센티널로
                    for key, (amt, nm) in data.items():
                        if key in FLOW_KEYS and qn == 4:
                            prior = sum(
                                reports[i][0][key][0]
                                for i in (1, 2, 3)
                                if key in reports[i][0]
                            )
                            value = amt - prior
                        else:
                            value = amt
                        conn.execute(
                            """INSERT OR REPLACE INTO financials
                                 (stock_code, quarter, disclosed_date, account_key, account_name, amount)
                               VALUES (?,?,?,?,?,?)""",
                            (code, quarter, disclosed, key, nm, round(value)),
                        )
                        _insert_revision(conn, code, quarter, disclosed, key, nm, round(value), rcept_no)
                        n_rows += 1
                    # 새 분기 재무와 함께 그 분기 상장주식수도 적재(동일 quarter, dart.py와 동일 패턴).
                    # ⚠️ 분기당 stockTotqySttus 호출 1회 추가 → 일일 한도 소모 증가.
                    n_rows += _ingest_shares(
                        conn, api_key, code, corp, year, qn, quarter, disclosed, sleep
                    )
                    new_pairs.append((code, quarter))
                    existing.add((code, quarter))
                    actual_latest = max(actual_latest or quarter, quarter, key=_q_sort_key)
                conn.commit()

        if actual_latest:
            set_meta(conn, "latest_disclosed_quarter", actual_latest)

        print(f"새 공시 적재: {len(new_pairs)}개 (종목,분기), {n_rows}행, 최신분기 {actual_latest}")
        return {
            "new_pairs": len(new_pairs),
            "financials_rows": n_rows,
            "latest_disclosed_quarter": actual_latest,
            "checked_quarters": cand_quarters,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    print(update_financials())
