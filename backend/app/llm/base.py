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
    tagged = _from_tags(text)
    if tagged:
        return tagged
    raise LLMError(f"Could not parse JSON from model output: {text[:300]!r}")


# Some models (seen: Groq's qwen vision model) ignore JSON mode and answer in XML-ish tags:
#   <thought>…</thought><action>tap</action><ref>4</ref><expectation>…</expectation>
# Recover the same fields instead of wasting the call. The policy engine still gates the
# resulting action exactly as it would a JSON one.
_TAG_FIELDS = ("type", "action", "ref", "text", "x", "y", "direction", "seconds",
               "thought", "expectation")


def _from_tags(text: str) -> dict | None:
    found: dict[str, Any] = {}
    for f in _TAG_FIELDS:
        m = re.search(rf"<{f}>\s*(.*?)\s*</{f}>", text, re.DOTALL | re.IGNORECASE)
        if m:
            found[f] = m.group(1).strip()
    if "type" not in found and "action" in found:
        found["type"] = found.pop("action")
    if "type" not in found:
        return None
    found["type"] = found["type"].strip().lower()
    for f in ("ref",):
        if f in found:
            try:
                found[f] = int(found[f])
            except ValueError:
                found[f] = None
    for f in ("x", "y", "seconds"):
        if f in found:
            try:
                found[f] = float(found[f])
            except ValueError:
                found.pop(f)
    for f in ("text", "direction"):
        if found.get(f, "").lower() in ("", "null", "none"):
            found.pop(f, None)
    return found
