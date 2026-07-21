"""복사본(market_refeed.db) 재수집 완료 후 → 원본(market.db)에 financials 반영.

웹 무중단 전략: 원본의 wiki/result_cache/prices/company는 그대로 두고,
복사본에서 재수집된 financials만 원본으로 교체한 뒤 metrics를 재계산한다.

실행: python scripts/swap_refeed.py
주의: financials 교체 순간 짧은 쓰기 잠금이 걸린다(busy_timeout 15초). 가급적 한산한 시간에.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import connect, init_db
from src.ingest.metrics import compute_metrics

ROOT = Path(__file__).resolve().parent.parent
ORIG = str(ROOT / "data" / "market.db")
COPY = str(ROOT / "data" / "market_refeed.db")

if not Path(COPY).exists():
    raise SystemExit(f"복사본 없음: {COPY}")

init_db(ORIG)  # 원본 DB에 raw_reports 등 신규 테이블 보장(없으면 생성)
conn = connect(ORIG)
try:
    # 복사본 재수집이 끝났는지 점검(refeed_done 잔여가 없으면 완료로 간주)
    n_copy = connect(COPY).execute("SELECT COUNT(*) FROM financials").fetchone()[0]
    n_orig = conn.execute("SELECT COUNT(*) FROM financials").fetchone()[0]
    print(f"복사본 financials {n_copy:,}행 / 원본 {n_orig:,}행")

    conn.execute("ATTACH DATABASE ? AS refeed", (COPY,))
    conn.execute("BEGIN IMMEDIATE")
    # financials(지배주주지분·지배주주순이익 포함) 교체
    conn.execute("DELETE FROM main.financials")
    conn.execute("INSERT INTO main.financials SELECT * FROM refeed.financials")
    # financials_revision(백테스트 asof 조회가 실제로 읽는 시점별 스냅샷)도 함께 교체.
    # 예전엔 financials만 바꾸고 이걸 빠뜨려서, 일반 조회(compute_metrics)는 고쳐졌는데
    # 스크리닝/백테스트 경로(metrics_at → _fin asof 지정 → financials_revision 우선 조회)는
    # 옛날 버그값을 계속 썼다(효성화학 2025Q4 controlling_net_income이 차분 안 된 연간
    # 누적치로 남아 PER 0.47배 오류로 재현됨).
    conn.execute("DELETE FROM main.financials_revision")
    conn.execute("INSERT INTO main.financials_revision SELECT * FROM refeed.financials_revision")
    # DART 원본 응답(raw_reports)도 원본 DB에 반영 → 향후 재수집 없이 재파싱 가능
    conn.execute("DELETE FROM main.raw_reports")
    conn.execute("INSERT INTO main.raw_reports SELECT * FROM refeed.raw_reports")
    conn.commit()
    conn.execute("DETACH DATABASE refeed")
    print("financials + financials_revision + raw_reports 교체 완료")

    n = compute_metrics(conn)
    print(f"metrics {n}종목 재계산 완료. 교체 끝.")
finally:
    conn.close()
