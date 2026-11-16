"""Groq provider — OpenAI-compatible chat completions endpoint, called via httpx.

Supports a multimodal path: when an image is supplied, it switches to the configured
vision model and sends the screenshot alongside the text (OpenAI image_url format).
"""
from __future__ import annotations

import httpx

from .base import LLMError, LLMProvider

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, api_key: str, model: str, vision_model: str = ""):
        self.api_key = api_key
        self.model = model
        self.vision_model = vision_model or model

    def _payload(self, system: str, user: str, temperature: float, max_tokens: int,
                 json_mode: bool, image_b64: str | None) -> dict:
        if image_b64:
            user_content = [
                {"type": "text", "text": user},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]
            model = self.vision_model
        else:
            user_content = user
            model = self.model
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def complete(self, system: str, user: str, *, temperature: float = 0.2,
                       max_tokens: int = 1024, json_mode: bool = False,
                       image_b64: str | None = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    GROQ_URL, headers=headers,
                    json=self._payload(system, user, temperature, max_tokens, json_mode, image_b64))
                # Some models 400 on strict json_object with a `json_validate_failed`.
                # Retry once without response_format — extract_json() parses the result.
                if r.status_code == 400 and json_mode and "json_validate_failed" in r.text:
                    r = await client.post(
                        GROQ_URL, headers=headers,
                        json=self._payload(system, user, temperature, max_tokens, False, image_b64))
                r.raise_for_status()
                data = r.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            raise LLMError(f"Groq HTTP {e.response.status_code}: {e.response.text[:200]}") from e
        except (httpx.HTTPError, KeyError, IndexError) as e:
            raise LLMError(f"Groq request failed: {e}") from e
