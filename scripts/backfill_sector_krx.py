"""company.sector를 KRX 정보데이터시스템 업종분류(KOSPI+KOSDAQ)로 재매핑.

기존 DART 회사개황(induty_code) 기반 KSIC 세분류(80개+)를 KRX 자체 업종분류
(20~30개 내외 대분류, [12025] 업종분류 현황과 동일 데이터)로 통일한다.

pykrx.stock.get_market_sector_classifications는 KRX_ID/KRX_PW 환경변수 기반
로그인 세션이 필요하다(.env에 설정).

사용법:
  python scripts/backfill_sector_krx.py --dry-run   # 영향범위만 확인
  python scripts/backfill_sector_krx.py             # 실제 UPDATE
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.db import connect


# KRX 현재 상장분류에 없는 종목(상폐·코넥스 등)의 옛 KSIC 라벨 → KRX 29종 폴백 매핑.
# 백테스트 생존편향 방지를 위해 상폐 종목도 업종 유니버스에 남겨야 하므로 삭제 대신 매핑한다.
KSIC_TO_KRX: dict[str, str] = {
    "금융지원서비스": "기타금융", "기타 금융업": "기타금융", "투자기관": "금융",
    "은행·저축기관": "은행", "보험업": "보험",
    "기계·장비 제조": "기계·장비", "도매·상품중개": "유통", "소매업": "유통",
    "자동차·부품 판매": "유통",
    "전자부품": "전기·전자", "반도체": "전기·전자", "영상·음향기기": "전기·전자",
    "전기장비 제조": "전기·전자", "전자부품·컴퓨터·통신장비": "전기·전자",
    "마그네틱·광학매체": "전기·전자", "컴퓨터·주변기기": "전기·전자",
    "화학물질·화학제품": "화학", "고무·플라스틱 제조": "화학",
    "소프트웨어·게임": "IT 서비스", "기타 정보서비스": "IT 서비스",
    "자료처리·호스팅·포털": "IT 서비스", "정보서비스업": "IT 서비스",
    "컴퓨터프로그래밍·시스템": "IT 서비스",
    "종합건설": "건설", "전문직별 공사": "건설",
    "기타 운송장비 제조": "운송장비·부품", "자동차·트레일러 제조": "운송장비·부품",
    "부동산업": "부동산", "임대업(부동산 제외)": "부동산",
    "의료용기기": "의료·정밀기기", "안경·광학기기": "의료·정밀기기", "측정·정밀기기": "의료·정밀기기",
    "의약품 제조": "제약",
    "1차 금속 제조": "금속", "금속가공제품 제조": "금속",
    "식료품 제조": "음식료·담배", "음료 제조": "음식료·담배",
    "의복·의류 제조": "섬유·의류", "섬유제품 제조": "섬유·의류",
    "기타 제품 제조": "기타제조",
    "무선통신": "통신",
    "비금속광물제품 제조": "비금속",
    "영화·방송영상 제작": "오락·문화", "창작·예술·여가": "오락·문화",
    "펄프·종이 제조": "종이·목재",
    "농업": "농업 임업 및 어업", "임업": "농업 임업 및 어업", "어업": "농업 임업 및 어업",
    "도로화물운송": "운송·창고", "해운": "운송·창고", "해상운송": "운송·창고",
    "인쇄·기록매체 복제": "출판·매체복제", "출판": "출판·매체복제",
    "전기·가스·증기 공급": "전기·가스",
    "연구개발(자연·공학)": "일반서비스", "사업지원서비스": "일반서비스", "광고업": "일반서비스",
    "사회복지서비스": "일반서비스", "음식점·주점업": "일반서비스", "전문서비스업": "일반서비스",
}


def fetch_krx_sectors(date: str) -> dict[str, str]:
    """KOSPI+KOSDAQ 전체 종목의 (종목코드 -> 업종명) 매핑."""
    from pykrx import stock

    mapping: dict[str, str] = {}
    for market in ("KOSPI", "KOSDAQ"):
        df = stock.get_market_sector_classifications(date, market)
        for code, row in df.iterrows():
            mapping[code] = row["업종명"]
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--date", default=None, help="YYYYMMDD (기본: prices 최신일)")
    args = parser.parse_args()

    import os
    if not (os.getenv("KRX_ID") and os.getenv("KRX_PW")):
        print("KRX_ID/KRX_PW 환경변수 없음 — 중단", flush=True)
        return

    conn = connect()
    try:
        if args.date:
            trdd = args.date
        else:
            row = conn.execute("SELECT MAX(date) AS d FROM prices").fetchone()
            trdd = (row["d"] or "").replace("-", "")
        if not trdd:
            print("기준일자를 구할 수 없음(prices 비어있음) — --date로 직접 지정하세요.", flush=True)
            return

        print(f"KRX 업종분류 조회 기준일: {trdd}", flush=True)
        krx_map = fetch_krx_sectors(trdd)
        print(f"KRX 업종분류 수집 완료: {len(krx_map)}종목, 업종 {len(set(krx_map.values()))}개", flush=True)

        rows = conn.execute("SELECT stock_code, sector FROM company").fetchall()
        changed, unmatched, same, fallback = 0, 0, 0, 0
        changes_sample = []
        for r in rows:
            code, old_sector = r["stock_code"], r["sector"]
            new_sector = krx_map.get(code)
            if new_sector is None:
                # KRX 현재 상장분류에 없음(상폐·코넥스) → 옛 KSIC 라벨을 KRX 29종으로 폴백 매핑
                new_sector = KSIC_TO_KRX.get(old_sector)
                if new_sector is None:
                    unmatched += 1
                    continue
                fallback += 1
            if new_sector == old_sector:
                same += 1
                continue
            changed += 1
            if len(changes_sample) < 10:
                changes_sample.append((code, old_sector, new_sector))
            if not args.dry_run:
                conn.execute("UPDATE company SET sector = ? WHERE stock_code = ?", (new_sector, code))

        print(f"\n전체 회사 {len(rows)} / KRX 매칭 {len(rows) - unmatched - fallback} / KSIC 폴백 {fallback} / 미매핑(빈값 등) {unmatched}")
        print(f"변경 대상 {changed} / 기존값과 동일 {same}")
        print("\n변경 샘플:")
        for code, old, new in changes_sample:
            print(f"  {code}: '{old}' -> '{new}'")

        if args.dry_run:
            print("\n[dry-run] 실제 UPDATE는 수행하지 않았습니다.")
        else:
            conn.commit()
            print(f"\n{changed}건 UPDATE 완료 및 commit.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
