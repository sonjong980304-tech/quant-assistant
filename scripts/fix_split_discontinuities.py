"""액면분할/병합 미반영(수정주가 누락)으로 생긴 prices.close 불연속을 정수배로 보정.

배경
----
네이버 수정주가는 최근 데이터만 소급 조정하고, 오래된 분할/병합 이벤트의 과거 구간은
조정해주지 않는다. 그 결과 저유동 종목(관리종목/동전주)에서 "하루 만에 정확히 2배·10배·
20배 등으로 점프"하는 구간이 남는다 — 이것이 진짜 액면병합인데 수정주가로 소급 반영되지
않은 데이터 버그다. split_date 이전 모든 close에 ratio를 곱하면(과거를 현재 기준으로 소급
조정) 표준 수정주가가 된다.

분류
----
- 100%(2배) 초과 점프를 가진 종목 중, "최대 점프의 배율이 정수배(±8%)"면 → **정수그룹**
  (보정 대상). 종목당 여러 분할 이벤트가 있을 수 있으므로 그 종목의 모든 정수배 점프를 보정.
- 최대 점프가 정수배가 아니면 → **기타그룹** (감자·특수 코퍼레이트액션 또는 원본 데이터오류
  의심 — 배율 곱셈으로 고칠 문제가 아님. 리포트만 남기고 절대 수정하지 않는다).

안전장치
--------
- 기본은 **드라이런**: 리포트만 출력하고 실제 UPDATE는 하지 않는다.
- `--apply` 를 줘야만 반영하며, 반영 직전 DB를 `<db>.bak-<timestamp>` 로 자동 백업한다.

실행
----
    python3 scripts/fix_split_discontinuities.py            # 드라이런 리포트
    python3 scripts/fix_split_discontinuities.py --apply    # 실제 반영(백업 후)
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backfill_marketcap import _effective_shares, _shares_series
from src.config import CONFIG
from src.db import connect

# 진짜 액면분할/병합에서 나타나는 정수 배율 후보.
ROUND_RATIOS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 100]
RATIO_TOL = 0.08      # 배율 근사 허용오차 (±8%)
JUMP_THRESHOLD = 1.0  # abs(ratio-1) > 1.0 → 100%(2배) 초과 점프만 후보


@dataclass
class SplitEvent:
    """정수배로 판정된 분할/병합 이벤트 1건."""

    stock_code: str
    date: str          # 점프 행 날짜(=새 세그먼트 첫 거래일) = split_date
    close: float       # 점프 후 종가
    prev_close: float  # 점프 전 종가
    ratio: float       # close / prev_close (보정 계수, 원본 배율 그대로 사용)
    matched_ratio: int  # 매칭된 정수 배율(2/10/20…, 분류·리포트 표기용)


@dataclass
class OtherJump:
    """정수배가 아닌 >100% 점프 1건(기타그룹, 보정하지 않음)."""

    stock_code: str
    date: str
    close: float
    prev_close: float
    ratio: float


def match_round_ratio(
    ratio: float, round_ratios: list[int] = ROUND_RATIOS, tol: float = RATIO_TOL
) -> int | None:
    """ratio 가 정수 배율(±tol)에 매칭되면 **가장 가까운** 정수를, 아니면 None 을 반환.

    상승(ratio)·하락(1/ratio) 양방향을 모두 시도한다(하락은 현재 임계값에선 사실상
    발생하지 않으나 스펙대로 유지). 여러 후보가 tol 안이면 상대오차가 가장 작은 것을 고른다
    (예: 9.6 → 9가 아니라 10). 보정 계수는 원본 ratio를 쓰므로 이 값은 라벨·분류용이다.
    """
    if ratio <= 0:
        return None
    best: int | None = None
    best_err = tol
    for r in round_ratios:
        err = min(abs(ratio - r) / r, abs(1.0 / ratio - r) / r)
        if err < best_err:
            best_err, best = err, r
    return best


def _raw_jumps(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """인접 거래일 종가비가 2배 초과(또는 1/2 미만)인 모든 점프 행(후보 superset)."""
    return conn.execute(
        """
        WITH ch AS (
          SELECT stock_code, date, close,
                 LAG(close) OVER (PARTITION BY stock_code ORDER BY date) AS prev_close
          FROM prices WHERE close IS NOT NULL AND close > 0
        )
        SELECT stock_code, date, close, prev_close FROM ch
        WHERE prev_close IS NOT NULL AND prev_close > 0
          AND (close / prev_close > 2.0 OR prev_close / close > 2.0)
        ORDER BY stock_code, date
        """
    ).fetchall()


def detect_splits(
    conn: sqlite3.Connection,
    threshold: float = JUMP_THRESHOLD,
    round_ratios: list[int] = ROUND_RATIOS,
    tol: float = RATIO_TOL,
) -> dict:
    """>threshold(100%) 점프를 종목별로 찾아 정수그룹/기타그룹으로 분류.

    분류 기준은 종목의 **최대 점프(abs(ratio-1) 최댓값)**:
      - 최대 점프가 정수배 → 정수그룹. 그 종목의 **모든** 정수배 점프를 이벤트로 수집.
      - 최대 점프가 비정수배 → 기타그룹. 그 종목의 모든 >threshold 점프를 수집(리포트용).

    반환: {"integer_events": {code: [SplitEvent]}, "other_jumps": {code: [OtherJump]}}
    """
    # 종목별로 (모든 >threshold 점프, 각 점프의 정수매칭 여부)를 모은다.
    per_code: dict[str, list[tuple[sqlite3.Row, float, int | None]]] = {}
    for row in _raw_jumps(conn):
        ratio = row["close"] / row["prev_close"]
        if abs(ratio - 1.0) <= threshold:
            continue  # 상승 2배 초과만 통과(하락은 임계값상 제외됨)
        matched = match_round_ratio(ratio, round_ratios, tol)
        per_code.setdefault(row["stock_code"], []).append((row, ratio, matched))

    integer_events: dict[str, list[SplitEvent]] = {}
    other_jumps: dict[str, list[OtherJump]] = {}
    for code, jumps in per_code.items():
        # 최대 점프(abs(ratio-1) 최댓값)로 종목 분류.
        max_jump = max(jumps, key=lambda j: abs(j[1] - 1.0))
        if max_jump[2] is not None:  # 최대 점프가 정수배 → 정수그룹
            evs = [
                SplitEvent(code, r["date"], r["close"], r["prev_close"], ratio, matched)
                for (r, ratio, matched) in jumps
                if matched is not None
            ]
            integer_events[code] = evs
        else:  # 기타그룹
            other_jumps[code] = [
                OtherJump(code, r["date"], r["close"], r["prev_close"], ratio)
                for (r, ratio, _m) in jumps
            ]
    return {"integer_events": integer_events, "other_jumps": other_jumps}


def _recalc_marketcap_for_codes(conn: sqlite3.Connection, codes: set[str]) -> int:
    """영향받은 종목의 market_cap 을 backfill_marketcap 로직 재사용으로 재계산.

    market_cap = close × (그 시점 공시된 상장주식수). 새 계산식을 만들지 않고
    backfill_marketcap._shares_series / _effective_shares 를 그대로 재사용한다.
    """
    series = _shares_series(conn)
    updated = 0
    for code in codes:
        ser = series.get(code)
        if not ser:
            continue
        rows = conn.execute(
            "SELECT date, close FROM prices WHERE stock_code=? AND close IS NOT NULL",
            (code,),
        ).fetchall()
        for r in rows:
            shares = _effective_shares(ser, r["date"])
            if not shares:
                continue
            conn.execute(
                "UPDATE prices SET market_cap=? WHERE stock_code=? AND date=?",
                (round(r["close"] * shares), code, r["date"]),
            )
            updated += 1
    conn.commit()
    return updated


def apply_corrections(
    conn: sqlite3.Connection, integer_events: dict[str, list[SplitEvent]]
) -> dict:
    """각 (code, split_date, ratio) 이벤트에 대해 date < split_date 의 close 에 ratio 를 곱한다.

    종목당 이벤트가 여러 개면 각 UPDATE 가 자기 split_date 이전 행에만 곱하므로,
    더 오래된 구간일수록 여러 이벤트가 자연히 누적 곱해진다(순서 무관). 보정 후 영향받은
    종목의 market_cap 을 재계산한다.
    """
    affected: set[str] = set()
    for code, events in integer_events.items():
        for ev in events:
            conn.execute(
                "UPDATE prices SET close = close * ? WHERE stock_code=? AND date < ?",
                (ev.ratio, code, ev.date),
            )
        if events:
            affected.add(code)
    conn.commit()
    cap_updated = _recalc_marketcap_for_codes(conn, affected)
    return {
        "affected_codes": len(affected),
        "events": sum(len(v) for v in integer_events.values()),
        "market_cap_updated": cap_updated,
    }


# ---------------------------------------------------------------------------
# 리포트(드라이런)
# ---------------------------------------------------------------------------
def _name(conn: sqlite3.Connection, code: str) -> str:
    row = conn.execute("SELECT name FROM company WHERE stock_code=?", (code,)).fetchone()
    return row["name"] if row and row["name"] else "(이름없음)"


def _cumulative_factor(events: list[SplitEvent], on_date: str) -> float:
    """on_date 종가에 최종 적용될 누적 배율 = split_date > on_date 인 이벤트들의 ratio 곱."""
    factor = 1.0
    for ev in events:
        if ev.date > on_date:
            factor *= ev.ratio
    return factor


def _affected_rows(conn: sqlite3.Connection, code: str, events: list[SplitEvent]) -> int:
    """보정으로 값이 바뀌는 행 수 = date < max(split_date) 인 행 수."""
    last = max(ev.date for ev in events)
    return conn.execute(
        "SELECT COUNT(*) AS n FROM prices WHERE stock_code=? AND date < ?", (code, last)
    ).fetchone()["n"]


def _samples(conn: sqlite3.Connection, code: str, events: list[SplitEvent], k: int = 3) -> list:
    """가장 이른 split 직전 몇 개 행의 (날짜, 전, 후) 샘플."""
    first = min(ev.date for ev in events)
    rows = conn.execute(
        "SELECT date, close FROM prices WHERE stock_code=? AND date < ? ORDER BY date DESC LIMIT ?",
        (code, first, k),
    ).fetchall()
    out = []
    for r in rows:
        after = r["close"] * _cumulative_factor(events, r["date"])
        out.append((r["date"], r["close"], round(after, 2)))
    return list(reversed(out))


def _constant_run(conn: sqlite3.Connection, code: str, jump_date: str, prev_close: float) -> int:
    """jump_date 직전에서 prev_close 와 정확히 같은 종가가 몇 거래일 연속됐는지."""
    rows = conn.execute(
        "SELECT close FROM prices WHERE stock_code=? AND date < ? ORDER BY date DESC LIMIT 120",
        (code, jump_date),
    ).fetchall()
    run = 0
    for r in rows:
        if r["close"] == prev_close:
            run += 1
        else:
            break
    return run


def build_report(conn: sqlite3.Connection, detection: dict) -> str:
    integer_events = detection["integer_events"]
    other_jumps = detection["other_jumps"]
    lines: list[str] = []

    total_events = sum(len(v) for v in integer_events.values())
    total_rows = sum(_affected_rows(conn, c, evs) for c, evs in integer_events.items())
    lines.append("=" * 78)
    lines.append(f"[정수배 그룹] 보정 대상 종목 {len(integer_events)}개 · 이벤트 {total_events}건 · "
                 f"영향행 합계 {total_rows:,}행")
    lines.append("=" * 78)
    for code in sorted(integer_events):
        evs = integer_events[code]
        rows_n = _affected_rows(conn, code, evs)
        lines.append(f"\n● {code} {_name(conn, code)}  (이벤트 {len(evs)}건, 영향행 {rows_n:,})")
        for ev in sorted(evs, key=lambda e: e.date):
            lines.append(
                f"    - {ev.date}: {ev.prev_close:g} → {ev.close:g}  "
                f"(ratio={ev.ratio:.3f} ≈ {ev.matched_ratio}배)"
            )
        for d, before, after in _samples(conn, code, evs):
            lines.append(f"        보정샘플 {d}: {before:g} → {after:g}")

    lines.append("\n" + "=" * 78)
    lines.append(f"[기타 그룹] 비정수배 >100% 점프 종목 {len(other_jumps)}개 (보정 안 함 — 리포트만)")
    lines.append("=" * 78)
    for code in sorted(other_jumps):
        jumps = other_jumps[code]
        lines.append(f"\n○ {code} {_name(conn, code)}")
        for j in sorted(jumps, key=lambda x: x.date):
            run = _constant_run(conn, code, j.date, j.prev_close)
            if run >= 10 or (j.prev_close is not None and j.prev_close <= 10.0):
                verdict = f"데이터오류 의심(점프 전 {run}거래일 연속 {j.prev_close:g} 고정)"
            else:
                verdict = "특수 코퍼레이트액션 의심(감자 등)"
            lines.append(
                f"    - {j.date}: {j.prev_close:g} → {j.close:g}  "
                f"(ratio={j.ratio:.2f})  → {verdict}"
            )
    return "\n".join(lines)


def _backup_db(db_path: str) -> str:
    """반영 직전 DB 스냅샷 백업. WAL 체크포인트 후 메인 파일을 복사한다."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = f"{db_path}.bak-{ts}"
    conn = connect(db_path)
    try:
        conn.execute("PRAGMA wal_checkpoint(FULL)")
    finally:
        conn.close()
    shutil.copy2(db_path, dst)
    return dst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="액면분할/병합 미반영 종가 불연속 정수배 보정")
    ap.add_argument("--apply", action="store_true",
                    help="실제 DB에 UPDATE 반영(기본: 드라이런). 반영 전 자동 백업.")
    ap.add_argument("--db", default=None, help="DB 경로(기본: CONFIG.db_path)")
    args = ap.parse_args(argv)

    db_path = args.db or CONFIG.db_path
    conn = connect(db_path)
    try:
        detection = detect_splits(conn)
        print(build_report(conn, detection))
        if args.apply:
            bak = _backup_db(db_path)
            print(f"\n[백업] {bak}")
            result = apply_corrections(conn, detection["integer_events"])
            print(f"[적용 완료] {result}")
        else:
            print("\n[드라이런] 실제 DB는 변경되지 않았습니다. 반영하려면 --apply 를 붙이세요.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
