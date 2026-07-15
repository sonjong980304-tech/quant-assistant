"""10년 백필: 광범위 재무(전체재무제표) + 주가 시계열 + metrics 재계산.

실행: python3 scripts/backfill_10y.py   (수 분~수십 분 소요)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import connect
from src.ingest.dart import ingest_dart
from src.ingest.krx import ingest_price_history
from src.ingest.metrics import compute_metrics

print("[1/4] financials 초기화", flush=True)
c = connect()
c.execute("DELETE FROM financials")
c.commit()
c.close()

print("[2/4] DART 10년 광범위 재무 수집 시작... (가장 오래 걸림)", flush=True)
r1 = ingest_dart(years=10)
print("  재무:", r1, flush=True)

print("[3/4] pykrx 10년 월별 주가 수집...", flush=True)
r2 = ingest_price_history(years=10)
print("  주가:", r2, flush=True)

print("[4/4] metrics 재계산 + 결과캐시 초기화", flush=True)
c = connect()
n = compute_metrics(c)
c.execute("DELETE FROM result_cache")
c.commit()
c.close()
print("  metrics:", n, flush=True)
print("DONE", flush=True)
