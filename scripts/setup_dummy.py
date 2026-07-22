"""더미 데이터 생성 스크립트 (API 키 없이 체험용).

DART/KRX API 키가 없어도 프로젝트를 바로 실행해볼 수 있도록 가짜 회사·재무·주가
데이터를 생성한다. 원래 cli.py(레거시 6노드 파이프라인과 함께 삭제됨)의 setup-dummy
명령이었으나, 파이프라인과 무관한 독립 기능이라 별도 스크립트로 남겼다.

경고: generate_dummy()는 financials/prices/metrics/company 테이블을 먼저 DELETE하고
채운다(wiki/result_cache는 보존). 실데이터가 쌓인 운영 DB(data/market.db)에서 실행하면
그 4개 테이블의 실제 데이터가 전부 사라진다 — 새 프로젝트를 처음 체험해볼 때만 쓸 것.

실행: python3 scripts/setup_dummy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.dummy import generate_dummy

if __name__ == "__main__":
    print("더미 데이터 생성 중...")
    info = generate_dummy()
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("완료. 이제 웹(uvicorn web.app:app --reload)에서 질의하세요.")
