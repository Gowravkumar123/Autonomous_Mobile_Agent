"""Google Gemini provider (backup) — REST generateContent, called via httpx."""
from __future__ import annotations

import httpx

from .base import LLMError, LLMProvider

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def complete(self, system: str, user: str, *, temperature: float = 0.2,
                       max_tokens: int = 1024, json_mode: bool = False,
                       image_b64: str | None = None) -> str:
        gen_config: dict = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if json_mode:
            gen_config["responseMimeType"] = "application/json"
        # Disable 2.5's default "thinking" for our fast structured action calls — it
        # otherwise consumes the output budget and can return an empty body.
        if "2.5" in self.model:
            gen_config["thinkingConfig"] = {"thinkingBudget": 0}
        parts: list = [{"text": user}]
        if image_b64:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": gen_config,
        }
        url = GEMINI_URL.format(model=self.model)
        headers = {"x-goog-api-key": self.api_key}
        try:
            async with httpx.AsyncClient(timeout=45) as client:
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except httpx.HTTPStatusError as e:
            raise LLMError(f"Gemini HTTP {e.response.status_code}: {e.response.text[:200]}") from e
        except (httpx.HTTPError, KeyError, IndexError) as e:
            raise LLMError(f"Gemini request failed: {e}") from e
