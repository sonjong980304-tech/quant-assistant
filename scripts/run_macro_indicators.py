"""매크로 지표 파이프라인 실행 스크립트 (launchd용).

FRED 장단기금리차(T10Y2Y)/VIX(VIXCLS) + CNN Fear&Greed를 수집해 macro_indicators에
upsert하고, 금리차 레짐 기반 GREEN/YELLOW/RED 신호를 판정해 macro_signal에 append한 뒤,
직전 판정과 다를 때만 Slack 알림을 보낸다(순수 규칙기반, LLM 미사용).

실행: python3 scripts/run_macro_indicators.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.macro_pipeline import run_macro_pipeline

if __name__ == "__main__":
    result = run_macro_pipeline()
    print(result)
