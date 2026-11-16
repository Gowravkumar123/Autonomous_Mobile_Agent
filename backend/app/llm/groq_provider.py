"""Groq provider — OpenAI-compatible chat completions endpoint, called via httpx."""
from __future__ import annotations

import httpx

from .base import LLMError, LLMProvider

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def _payload(self, system: str, user: str, temperature: float,
                 max_tokens: int, json_mode: bool) -> dict:
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def complete(self, system: str, user: str, *, temperature: float = 0.2,
                       max_tokens: int = 1024, json_mode: bool = False) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=45) as client:
                r = await client.post(
                    GROQ_URL, headers=headers,
                    json=self._payload(system, user, temperature, max_tokens, json_mode))
                # Some models (e.g. gpt-oss) can 400 on strict json_object mode with a
                # `json_validate_failed`. Retry once WITHOUT response_format — our own
                # extract_json() parses the result — instead of failing the whole request.
                if r.status_code == 400 and json_mode and "json_validate_failed" in r.text:
                    r = await client.post(
                        GROQ_URL, headers=headers,
                        json=self._payload(system, user, temperature, max_tokens, json_mode=False))
                r.raise_for_status()
                data = r.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            raise LLMError(f"Groq HTTP {e.response.status_code}: {e.response.text[:200]}") from e
        except (httpx.HTTPError, KeyError, IndexError) as e:
            raise LLMError(f"Groq request failed: {e}") from e
