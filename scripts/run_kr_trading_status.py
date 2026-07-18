"""KRX 관리종목/매매거래정지 '오늘 현재' 스냅샷 누적 수집 실행 스크립트 (launchd 매일 갱신).

과거 이력은 무료로 구할 수 없어(KRX/KIND 모두 조회일 무시, '오늘 현재' 스냅샷만 반환) 매일
1회 '오늘 현재' 스냅샷을 받아 직전 실행 스냅샷과 diff 해서 앞으로의 지정~해제 구간을 우리
쪽에서 누적으로 쌓는다(src/ingest/kr_trading_status.py 참고). 관리종목/거래정지는 하루 단위로
바뀔 수 있어 상장폐지(주간)보다 자주(매일) 돈다.

라이브 KRX 로그인은 이 스크립트를 실행할 때만 일어난다(KRX_ID/KRX_PW 는 .env → os.environ,
src.config 가 import 시 로드; 값은 출력하지 않는다). 기존 run_us_delisting.py 와 동일한
명명·구조 관례. 실행: python3 scripts/run_kr_trading_status.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.kr_trading_status import ingest_trading_status

if __name__ == "__main__":
    result = ingest_trading_status()
    a, h = result["admin"], result["halt"]
    print(
        f"[kr_trading_status] as_of={result['as_of']} "
        f"admin(open={a['open_total']} +{a['opened']}/-{a['released']}) "
        f"halt(open={h['open_total']} +{h['opened']}/-{h['released']})"
    )
