"""raw_reports 원문 → financials 재파싱 (DART 재호출 없음).

normalize 규칙이 개선되면, 재수집(DART 재호출) 대신 보관된 원문(raw_reports)을 다시 파싱해
financials를 재생성한다. 라이브 수집과 **동일한** 기록 로직(_write_reports_year)을 써서
Q4 흐름항목 차분 등이 정확히 일치하도록 한다.

- financials만 갱신(INSERT OR REPLACE). raw_reports는 읽기만 하며 절대 수정하지 않는다.
- 상장주식수(shares_outstanding)는 원문에 없으므로 건드리지 않는다(원본 값 유지).

실행: REFEED_DB=data/market_refeed.db python scripts/reparse_from_raw.py
      (REFEED_DB 미지정 시 CONFIG.db_path=data/market.db)
"""
import json
import os
import sys
import zlib
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.db import connect
from src.ingest.dart import REPRT_CODE, _parse_all, _write_reports_year
from src.version import estimate_available_quarter, recent_quarters

# reprt_code(11013 등) → qn(1..4)
_QN_BY_REPRT = {v: k for k, v in REPRT_CODE.items()}

_dbp = os.environ.get("REFEED_DB") or None
conn = connect(_dbp)
db_label = _dbp or CONFIG.db_path
print(f"재파싱 대상 DB: {db_label}", flush=True)

# 라이브 수집과 동일한 wanted/연도 범위 (refeed_full.py와 일치)
today = date(2026, 7, 1)
latest_q = estimate_available_quarter(today)
wanted = set(recent_quarters(latest_q, CONFIG.data_years * 4))
print(f"대상 분기 범위: {min(wanted)} ~ {max(wanted)} ({len(wanted)}개 분기)", flush=True)

before = conn.execute("SELECT COUNT(*) FROM financials").fetchone()[0]
codes = [r[0] for r in conn.execute(
    "SELECT DISTINCT stock_code FROM raw_reports ORDER BY stock_code")]
print(f"원문 보유 종목: {len(codes)}종목 / financials 시작 행수: {before:,}", flush=True)

total_written = 0
for idx, code in enumerate(codes, 1):
    # 이 종목의 원문을 (연도 → {qn: rows}) 로 재구성
    by_year: dict[int, dict[int, list]] = defaultdict(dict)
    for year, reprt, payload in conn.execute(
        "SELECT bsns_year, reprt_code, payload FROM raw_reports WHERE stock_code=?",
        (code,),
    ):
        qn = _QN_BY_REPRT.get(reprt)
        if qn is None:
            continue
        by_year[year][qn] = json.loads(zlib.decompress(payload).decode("utf-8"))

    for year, qn_rows in by_year.items():
        reports = {qn: ({}, None) for qn in (1, 2, 3, 4)}
        for qn, rows in qn_rows.items():
            reports[qn] = _parse_all(rows)  # (data, rcept)
        n, _latest = _write_reports_year(conn, code, year, reports, wanted)
        total_written += n

    if idx % 100 == 0:
        conn.commit()
        print(f"  {idx}/{len(codes)} 종목 재파싱 (누적 {total_written:,}행)", flush=True)

conn.commit()
after = conn.execute("SELECT COUNT(*) FROM financials").fetchone()[0]
print(f"\n재파싱 완료: {len(codes)}종목, financials {before:,} → {after:,}행 "
      f"(순증 {after - before:+,})", flush=True)
print("DONE", flush=True)
