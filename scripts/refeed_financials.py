"""financials 재수집(revenue 오매칭 정정) + 위키/결과캐시 초기화.

normalize 수정(기타영업수익 등 배제) 반영을 위해 financials를 DELETE 후 재적재.
주가(prices)는 그대로 유지. metrics는 더 이상 사용하지 않음.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import connect
from src.ingest.dart import ingest_dart

print("financials/wiki/result_cache 초기화", flush=True)
c = connect()
c.execute("DELETE FROM financials")
c.execute("DELETE FROM wiki")
c.execute("DELETE FROM result_cache")
c.commit()
c.close()

print("DART 10년 광범위 재수집 (revenue 정정)...", flush=True)
r = ingest_dart(years=10)
print("완료:", r, flush=True)
print("DONE", flush=True)
