"""
LLM router: Groq (primary) -> Gemini Flash (backup) -> heuristic mock.

Any single request that fails on the primary is retried on the next provider, so a
Groq rate-limit or outage degrades gracefully instead of killing a task.
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
        # Gemini first when available: stronger UI grounding + vision than the free Groq
        # models. Groq is the fallback, then the heuristic mock.
        if settings.has_gemini:
            self.chain.append(GeminiProvider(settings.gemini_api_key, settings.gemini_model))
        if settings.has_groq:
            self.chain.append(GroqProvider(settings.groq_api_key, settings.groq_model,
                                           settings.groq_vision_model))
        # Mock is always last so the system is never fully down.
        self.chain.append(MockProvider())
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
        raw = await self.complete(system, user, json_mode=True, **kw)
        return extract_json(raw)
