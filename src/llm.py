"""LLM provider 추상화 (ollama | openai).

- 키/데몬이 없으면 available=False. 호출하는 노드는 휴리스틱 폴백으로 전환한다.
  (DART 키가 없어도, OpenAI 키가 없어도 전체 파이프라인이 동작하도록.)
- role: 'sql' | 'judge' | 'refine' 에 따라 모델을 선택한다.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from .config import CONFIG


def _maybe_trace(client: Any, role: str) -> Any:
    """LANGCHAIN_TRACING_V2=true(+ langsmith 설치)면 OpenAI 클라이언트를 LangSmith로
    감싸 호출별 프롬프트/응답/토큰/지연시간을 자동 기록한다(.env의 LANGCHAIN_API_KEY/
    LANGCHAIN_PROJECT는 langsmith SDK가 직접 읽는다 — 이 함수는 켜고 끄는 스위치만 본다).
    role(sql/judge/diagnose)을 트레이스 이름에 반영해 LangSmith 대시보드에서 어떤 역할의
    호출인지 구분되게 한다.

    langsmith 미설치거나 트레이싱 비활성이거나 감싸기 자체가 실패하면(예: 테스트의 fake
    client) 원본 클라이언트를 그대로 반환한다 — 이 파일 전체의 '키/데몬 없으면 조용히
    폴백' 원칙과 동일하게, 트레이싱은 부가기능이라 실패해도 질의 자체를 막지 않는다.
    """
    if os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower() != "true":
        return client
    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client, chat_name=f"llm.{role}")
    except Exception:  # noqa: BLE001 — 트레이싱 실패는 무시하고 원본 클라이언트로 계속
        return client


class LLMResult:
    def __init__(self, text: str, ok: bool = True, error: str | None = None):
        self.text = text
        self.ok = ok
        self.error = error


class LLMClient:
    def __init__(
        self,
        cfg=CONFIG,
        model: str | None = None,
        provider: str | None = None,
        offline: bool = False,
    ):
        self.cfg = cfg
        self.model_override = model
        # offline=True면 키/데몬 유무와 무관하게 LLM을 강제로 끈다(휴리스틱 폴백만).
        # 게이트/평가의 결정론을 보장하기 위한 명시적 오프라인 스위치.
        self.offline = offline
        if model:
            # gpt* → openai, 그 외(exaone/qwen 등) → ollama
            self.provider = provider or ("openai" if str(model).lower().startswith("gpt") else "ollama")
        else:
            self.provider = provider or cfg.llm_provider

    # ---- 가용성 ----
    @property
    def available(self) -> bool:
        if self.offline:  # 명시적 오프라인: 항상 미가용 → 휴리스틱 폴백 강제
            return False
        if self.provider == "openai":
            return self.cfg.has_openai_key
        if self.provider == "ollama":
            return self._ollama_alive()
        return False

    def _ollama_alive(self) -> bool:
        try:
            import requests

            r = requests.get(f"{self.cfg.ollama_host}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def model_for(self, role: str) -> str:
        if self.model_override:
            return self.model_override
        if self.provider == "ollama":
            return self.cfg.ollama_model
        # 진단(diagnose)은 증거를 보고 원인을 추론·분류하는 작업이라, 평가(judge)와 같은
        # 강한 모델(gpt-5.5)을 쓴다. 실패한 질의에서만 호출되므로 빈도가 낮아 비용 영향이 작다.
        if role in ("judge", "diagnose"):
            return self.cfg.openai_model_judge
        if role == "sql":
            return self.cfg.openai_model_sql
        return self.cfg.openai_model

    # ---- 생성 ----
    def complete(
        self,
        prompt: str,
        system: str | None = None,
        role: str = "sql",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResult:
        if not self.available:
            return LLMResult("", ok=False, error="LLM unavailable (키/데몬 없음)")
        try:
            if self.provider == "openai":
                return self._openai(prompt, system, role, temperature, max_tokens)
            if self.provider == "ollama":
                return self._ollama(prompt, system, role, temperature)
            return LLMResult("", ok=False, error=f"unknown provider {self.provider}")
        except Exception as e:  # 호출 실패도 폴백 가능하게
            return LLMResult("", ok=False, error=f"{type(e).__name__}: {e}")

    def _openai(self, prompt, system, role, temperature, max_tokens) -> LLMResult:
        from openai import OpenAI

        client = OpenAI(api_key=self.cfg.openai_api_key, base_url=self.cfg.openai_base_url)
        client = _maybe_trace(client, role)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        model = self.model_for(role)
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        # 일부 신규 모델은 max_tokens 대신 max_completion_tokens를 요구한다(실측 확인:
        # temperature는 그대로 지원됨 — 재시도에서 temperature까지 빼면 결정론이 깨진다).
        try:
            resp = client.chat.completions.create(
                temperature=temperature, max_tokens=max_tokens, **kwargs
            )
        except Exception:
            resp = client.chat.completions.create(
                temperature=temperature, max_completion_tokens=max_tokens, **kwargs
            )
        return LLMResult(resp.choices[0].message.content or "")

    def _ollama(self, prompt, system, role, temperature) -> LLMResult:
        import requests

        payload = {
            "model": self.model_for(role),
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {"temperature": temperature},
        }
        r = requests.post(f"{self.cfg.ollama_host}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        return LLMResult(r.json().get("response", ""))


# ---------------------------------------------------------------------------
# 응답 파싱 헬퍼
# ---------------------------------------------------------------------------
_CODE_FENCE = re.compile(r"```(?:sql|json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """LLM 응답에서 SQL만 추출 (코드펜스/접두사 제거)."""
    text = (text or "").strip()
    m = _CODE_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    # 'SQL:' 같은 접두사 제거
    text = re.sub(r"^\s*(sql|쿼리)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    # 마지막 세미콜론 이후 설명 잘라내기
    if ";" in text:
        text = text[: text.index(";") + 1]
    return text.strip()


def extract_json(text: str) -> dict:
    text = (text or "").strip()
    m = _CODE_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    # 가장 바깥 중괄호 추출
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        text = text[s : e + 1]
    try:
        return json.loads(text)
    except Exception:
        return {}


# 싱글톤
LLM = LLMClient()
