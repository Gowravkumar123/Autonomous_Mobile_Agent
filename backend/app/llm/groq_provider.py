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

    def __init__(self, api_key: str, model: str, vision_model: str = "",
                 reasoning_effort: str = ""):
        self.api_key = api_key
        self.model = model
        self.vision_model = vision_model or model
        # gpt-oss models are reasoning models: their hidden reasoning tokens count against
        # max_tokens. "low" effort = fewer hidden tokens = cheaper, and far less risk of the
        # budget being spent before any visible answer. Only sent to gpt-oss models.
        self.reasoning_effort = reasoning_effort

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
        if self.reasoning_effort and "gpt-oss" in model:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    async def complete(self, system: str, user: str, *, temperature: float = 0.2,
                       max_tokens: int = 1024, json_mode: bool = False,
                       image_b64: str | None = None) -> str:
        from .. import observability as obs
        model = self.vision_model if image_b64 else self.model
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with obs.span("llm.chat", **{
                "gen_ai.system": "groq", "gen_ai.operation.name": "chat",
                "gen_ai.request.model": model, "gen_ai.request.max_tokens": max_tokens,
                "gen_ai.request.temperature": temperature, "llm.vision": bool(image_b64),
                "llm.reasoning_effort": self.reasoning_effort or "default"}) as sp:
            obs.content(sp, "input.value", f"{system}\n\n{user}")
            totals = {"in": 0, "out": 0, "reason": 0, "requests": 0, "queue_s": 0.0}

            def account(resp_json: dict) -> None:
                # Every billable request is counted, including the retries below.
                u = resp_json.get("usage") or {}
                i, o = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
                rz = int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
                totals["in"] += i
                totals["out"] += o
                totals["reason"] += rz
                totals["requests"] += 1
                totals["queue_s"] += float(u.get("queue_time") or 0)
                obs.record_llm(model, i, o, rz)

            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    r = await client.post(
                        GROQ_URL, headers=headers,
                        json=self._payload(system, user, temperature, max_tokens, json_mode,
                                           image_b64))
                    # Some models 400 on strict json_object with a `json_validate_failed`.
                    # Retry once without response_format — extract_json() parses the result.
                    if r.status_code == 400 and json_mode and "json_validate_failed" in r.text:
                        obs.set_attrs(sp, **{"llm.retry.json_mode": True})
                        r = await client.post(
                            GROQ_URL, headers=headers,
                            json=self._payload(system, user, temperature, max_tokens, False,
                                               image_b64))
                    r.raise_for_status()
                    data = r.json()
                    account(data)
                    choice = data["choices"][0]
                    content = (choice.get("message") or {}).get("content") or ""
                    finish = choice.get("finish_reason") or ""
                    # Reasoning ate the whole budget before any answer: retry ONCE with more
                    # room (pay for extra tokens only when actually needed).
                    if not content.strip() and finish == "length":
                        obs.set_attrs(sp, **{"llm.retry.empty_length": True})
                        bigger = min(max_tokens * 2, 4096)
                        r = await client.post(
                            GROQ_URL, headers=headers,
                            json=self._payload(system, user, temperature, bigger, json_mode,
                                               image_b64))
                        r.raise_for_status()
                        data2 = r.json()
                        account(data2)
                        choice = data2["choices"][0]
                        content = (choice.get("message") or {}).get("content") or ""
                        finish = choice.get("finish_reason") or ""
                obs.set_attrs(sp, **{
                    "gen_ai.usage.input_tokens": totals["in"],
                    "gen_ai.usage.output_tokens": totals["out"],
                    "gen_ai.usage.reasoning_tokens": totals["reason"],
                    "gen_ai.response.finish_reasons": finish,
                    "llm.requests": totals["requests"],
                    "groq.queue_time_ms": round(totals["queue_s"] * 1000, 1)})
                obs.content(sp, "output.value", content)
                return content
            except httpx.HTTPStatusError as e:
                obs.mark_error(sp, f"HTTP {e.response.status_code}")
                raise LLMError(f"Groq HTTP {e.response.status_code}: {e.response.text[:200]}") from e
            except (httpx.HTTPError, KeyError, IndexError) as e:
                obs.mark_error(sp, type(e).__name__)
                raise LLMError(f"Groq request failed: {e}") from e
