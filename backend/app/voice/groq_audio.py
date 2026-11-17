"""
Groq voice: text-to-speech (PlayAI) and speech-to-text (Whisper), via the same
GROQ_API_KEY. Used by the /voice/tts and /voice/stt endpoints for the phone assistant.

Note: Groq's PlayAI TTS requires accepting its terms once in the Groq console; until then
it returns 400 and the phone falls back to the device's built-in TTS.
"""
from __future__ import annotations

import logging

import httpx

from ..config import get_settings

log = logging.getLogger("agent.voice")

TTS_URL = "https://api.groq.com/openai/v1/audio/speech"
STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


async def tts(text: str) -> bytes | None:
    """Synthesize speech (WAV bytes) from text. None if unavailable/failed."""
    s = get_settings()
    if not s.has_groq or not text.strip():
        return None
    payload = {"model": s.groq_tts_model, "voice": s.groq_tts_voice,
               "input": text[:1200], "response_format": "wav"}
    headers = {"Authorization": f"Bearer {s.groq_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(TTS_URL, json=payload, headers=headers)
            r.raise_for_status()
            return r.content
    except httpx.HTTPError as e:
        log.warning("Groq TTS failed: %s", e)
        return None


async def stt(audio: bytes, filename: str = "audio.wav") -> str:
    """Transcribe audio bytes to text. Empty string if unavailable/failed."""
    s = get_settings()
    if not s.has_groq or not audio:
        return ""
    headers = {"Authorization": f"Bearer {s.groq_api_key}"}
    files = {"file": (filename, audio, "application/octet-stream")}
    data = {"model": s.groq_stt_model, "response_format": "json"}
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(STT_URL, headers=headers, files=files, data=data)
            r.raise_for_status()
            return r.json().get("text", "")
    except httpx.HTTPError as e:
        log.warning("Groq STT failed: %s", e)
        return ""
