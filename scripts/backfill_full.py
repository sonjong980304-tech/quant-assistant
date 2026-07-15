"""full 백필: 코스피+코스닥+코넥스 전체(DART corpCode 유니버스) 10년 재무.

체크포인트(backfill_checkpoint)로 중단 시 마지막 성공 종목부터 재개.
DART 일일 한도로 며칠에 나눠질 수 있음 — 끊기면 이 스크립트를 다시 실행하면 이어짐.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.dart import ingest_dart_full

print("full 백필 시작 (체크포인트 resume)", flush=True)
report = ingest_dart_full(years=10, resume=True)
print("DONE", json.dumps(report, ensure_ascii=False), flush=True)
