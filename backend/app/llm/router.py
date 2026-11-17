"""
LLM router — Groq (gpt-oss) first, heuristic mock as the last-resort fallback.

Gemini is only used if explicitly enabled (USE_GEMINI=true); by default it is OFF because
its free tier is too slow for a real-time per-step action loop. The agent runs on Groq's
fast gpt-oss model, reading the screen from the accessibility tree + OCR (no LLM vision).
"""
from __future__ import annotations

import logging

from ..config import Settings
from .base import LLMError, LLMProvider, extract_json
from .gemini_provider import GeminiProvider
from .groq_provider import GroqProvider
from .mock_provider import MockProvider

log = logging.getLogger("agent.llm")


class LLMRouter(LLMProvider):
    name = "router"

    def __init__(self, settings: Settings):
        self.chain: list[LLMProvider] = []
        if settings.has_groq:
            self.chain.append(GroqProvider(settings.groq_api_key, settings.groq_model,
                                           settings.groq_vision_model))
        if settings.has_gemini and settings.use_gemini:
            self.chain.append(GeminiProvider(settings.gemini_api_key, settings.gemini_model))
        self.chain.append(MockProvider())  # never fully down
        log.info("LLM chain: %s", " -> ".join(p.name for p in self.chain))

    @property
    def active(self) -> str:
        return self.chain[0].name if self.chain else "none"

    async def complete(self, system: str, user: str, **kw) -> str:
        last: Exception | None = None
        for provider in self.chain:
            try:
                return await provider.complete(system, user, **kw)
            except LLMError as e:
                last = e
                log.warning("provider %s failed, falling back: %s", provider.name, e)
        raise LLMError(f"All providers failed: {last}")

    async def complete_json(self, system: str, user: str, **kw) -> dict:
        return extract_json(await self.complete(system, user, json_mode=True, **kw))

    # Back-compat: the action loop may call this; it's the same fast Groq-first chain.
    async def complete_json_fast(self, system: str, user: str, **kw) -> dict:
        return await self.complete_json(system, user, **kw)
