"""DART 키/API 소규모 점검.

전체 수집(약 4~5분) 전에, 실제 DART 응답이 파서와 맞는지 1개사로 확인한다.
실행: python3 scripts/check_dart.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.ingest.dart import get_corp_codes, fetch_single_account
from src.ingest.normalize import normalize_account

print("DART 키 인식:", "있음" if CONFIG.has_dart_key else "없음")
if not CONFIG.has_dart_key:
    sys.exit("키 없음 — .env의 DART_API_KEY 확인")

print("corp_code 다운로드 중...")
m = get_corp_codes(CONFIG.dart_api_key)
print("  매핑 수:", len(m), "| 삼성전자(005930) corp:", m.get("005930"))

corp = m.get("005930")
print("삼성전자 2024 사업보고서(11011) 주요계정 호출...")
rows = fetch_single_account(CONFIG.dart_api_key, corp, 2024, "11011")
print("  행 수:", len(rows))
seen = set()
for r in rows:
    k = normalize_account(r.get("account_nm", ""), r.get("account_id"))
    if k and k not in seen:
        seen.add(k)
        print(f'  {k:20s} <- "{r.get("account_nm")}" = {r.get("thstrm_amount")}')

print("\n정규화 매칭된 계정 키:", sorted(seen))
print("→ revenue/operating_profit/net_income/total_assets/total_liabilities/total_equity 중")
print("  몇 개가 잡히는지 확인하세요. 대부분 잡히면 전체 수집 진행 가능합니다.")
