"""LLM provider interface. Keep it tiny so providers stay swappable."""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any


class LLMError(RuntimeError):
    pass


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def complete(self, system: str, user: str, *, temperature: float = 0.2,
                       max_tokens: int = 1024, json_mode: bool = False,
                       image_b64: str | None = None) -> str:
        """Return the model's raw text response. `image_b64` is an optional JPEG
        screenshot (base64, no data: prefix) for vision-capable models."""
        raise NotImplementedError

    async def complete_json(self, system: str, user: str, **kw: Any) -> dict:
        raw = await self.complete(system, user, json_mode=True, **kw)
        return extract_json(raw)


def extract_json(text: str) -> dict:
    """Best-effort JSON extraction. Models sometimes wrap JSON in prose or fences."""
    text = text.strip()
    # Strip ```json fences.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced {...} block.
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise LLMError(f"Could not parse JSON from model output: {text[:300]!r}")
