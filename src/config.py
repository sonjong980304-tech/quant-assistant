"""전역 설정 로드. .env 우선, 없으면 합리적 기본값."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv 미설치여도 동작
    pass

ROOT = Path(__file__).resolve().parent.parent


def _abs(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else ROOT / path)


@dataclass
class Config:
    # --- LLM ---
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai").lower())

    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str | None = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL") or None)
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
    openai_model_sql: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL_SQL", os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
    )
    openai_model_judge: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL_JUDGE", "gpt-5.5")
    )

    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M"))

    # --- DART ---
    dart_api_key: str = field(default_factory=lambda: os.getenv("DART_API_KEY", ""))

    # --- 기록(record) ---
    # 과거 캐시 도출(유사도 재사용) 기능은 폐기됨. wiki 테이블은 '질의 기록 로그'로만 쓰이며,
    # 성공한 질의는 항상 기록된다(별도 토글 없음). WIKI_SIM_THRESHOLD/WIKI_ENABLED는 더 이상 쓰지 않는다.

    # --- 주가 수집 ---
    # 과거 주가 백필 주기. "monthly"=각 월 마지막 거래일 종가만, "daily"=전체 일별.
    price_history_freq: str = field(
        default_factory=lambda: os.getenv("PRICE_HISTORY_FREQ", "monthly").strip().lower()
    )

    # --- DB ---
    db_path: str = field(default_factory=lambda: _abs(os.getenv("DB_PATH", "data/market.db")))

    # --- 데이터 범위/수집 (Phase1) ---
    data_years: int = field(default_factory=lambda: int(os.getenv("DATA_YEARS", "10")))
    api_call_delay: float = field(default_factory=lambda: float(os.getenv("API_CALL_DELAY", "0.4")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))

    # --- 알림 ---
    slack_webhook_url: str = field(default_factory=lambda: os.getenv("SLACK_WEBHOOK_URL", ""))

    # --- 백테스트 거래비용 기본값 ---
    fee_rate: float = field(default_factory=lambda: float(os.getenv("FEE_RATE", "0.00015")))  # 수수료 0.015%
    tax_rate: float = field(default_factory=lambda: float(os.getenv("TAX_RATE", "0.0018")))   # 매도 거래세 0.18%
    slippage_rate: float = field(default_factory=lambda: float(os.getenv("SLIPPAGE_RATE", "0.0010")))  # 슬리피지 0.1%

    @property
    def has_dart_key(self) -> bool:
        return bool(self.dart_api_key.strip())

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key.strip())

    def summary(self) -> str:
        return (
            f"provider={self.llm_provider} "
            f"sql_model={self._active_model('sql')} judge_model={self._active_model('judge')} "
            f"dart_key={'있음' if self.has_dart_key else '없음(더미)'} "
            f"db={self.db_path}"
        )

    def _active_model(self, role: str) -> str:
        if self.llm_provider == "ollama":
            return self.ollama_model
        return self.openai_model_sql if role == "sql" else self.openai_model_judge


# 싱글톤
CONFIG = Config()
