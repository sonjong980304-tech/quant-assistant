"""전체(full) financials 재수집 — normalize 수정(지배주주지분 account_id 매핑) 반영.

특징:
- in-place 갱신: financials를 DELETE하지 않고 INSERT OR REPLACE로 덮어쓴다(데이터 공백 없음).
- 이어받기: 완료 종목을 ingest_meta('refeed_done')에 누적 저장. DART 일일한도 도달 시
  중단하고, 다시 실행하면 남은 종목부터 재개한다.
- 전체 완료 시 compute_metrics로 metrics 재계산.

실행: python scripts/refeed_full.py   (매일/주기적으로 재실행해 이어가기)
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.db import connect, get_meta, init_db, set_meta
from src.ingest.dart import get_corp_codes, _ingest_one_company
from src.ingest.metrics import compute_metrics
from src.ingest.robust import DartQuotaError, log_ingest
from src.version import estimate_available_quarter, recent_quarters

if not CONFIG.has_dart_key:
    raise SystemExit("DART_API_KEY 없음")

import os
# REFEED_DB 환경변수 지정 시 별도 복사본에 재수집(웹 무중단), 없으면 원본.
_dbp = os.environ.get("REFEED_DB") or None
init_db(_dbp)  # raw_reports 등 신규 테이블 보장(없으면 생성) — 없으면 _save_raw가 실패한다
conn = connect(_dbp)
api_key = CONFIG.dart_api_key
today = date(2026, 7, 1)
years = CONFIG.data_years
latest_q = estimate_available_quarter(today)
wanted = set(recent_quarters(latest_q, years * 4))
years_set = sorted({int(q[:4]) for q in wanted})

print("corp_code 매핑 로드...", flush=True)
try:
    corp_map = get_corp_codes(api_key)
except Exception as exc:  # noqa: BLE001 — 한도 초과 시 zip 아닌 에러응답이라 파싱 실패
    print(f"corp_code 로드 실패(DART 일일한도 초과 추정): {exc}", flush=True)
    print("→ 오늘 한도 소진. 자정(00:00) 리셋 후 다시 실행하면 이어받습니다.", flush=True)
    raise SystemExit(0)
if not corp_map:
    print("corp_code 매핑이 비어있음(한도 초과 추정). 리셋 후 재시도.", flush=True)
    raise SystemExit(0)

universe = conn.execute("SELECT stock_code, name FROM company ORDER BY stock_code").fetchall()
done = set(filter(None, (get_meta(conn, "refeed_done") or "").split(",")))
print(f"전체 {len(universe)}종목 / 완료 {len(done)} / 남음 {len(universe) - len(done)}", flush=True)

processed = 0
try:
    for row in universe:
        code, name = row["stock_code"], row["name"]
        if code in done:
            continue
        corp = corp_map.get(code)
        if not corp:
            done.add(code)
            continue
        try:
            rows, _latest = _ingest_one_company(
                conn, api_key, code, corp, years_set, wanted, CONFIG.api_call_delay
            )
            conn.commit()
            done.add(code)
            processed += 1
            if processed % 10 == 0:
                set_meta(conn, "refeed_done", ",".join(sorted(done)))
                print(f"  {processed}종목 갱신 (최근 {code} {name} {rows}행)", flush=True)
        except DartQuotaError:
            set_meta(conn, "refeed_done", ",".join(sorted(done)))
            print(f"⚠ DART 일일한도 도달 — 이번 회차 {processed}종목 갱신 후 중단.", flush=True)
            print("  내일 다시 'python scripts/refeed_full.py' 실행하면 남은 종목부터 이어받습니다.", flush=True)
            raise SystemExit(0)
        except Exception as exc:  # noqa: BLE001 — 종목 실패 격리
            log_ingest({"stock_code": code, "status": "refeed_error", "error": str(exc)})
            done.add(code)
finally:
    set_meta(conn, "refeed_done", ",".join(sorted(done)))

print(f"이번 회차 {processed}종목 갱신 / 누적 완료 {len(done)}/{len(universe)}", flush=True)
if len(done) >= len(universe):
    n = compute_metrics(conn)
    # 완료 표시(refeed_done 유지 → 재실행 시 즉시 완료 감지). 원본 반영은 swap_refeed.py.
    print(f"🎉 전체 재수집 완료! metrics {n}종목 재계산. (교체는 swap_refeed.py)", flush=True)
print("DONE", flush=True)
