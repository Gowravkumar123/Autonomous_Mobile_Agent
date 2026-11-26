"""
Groq voice: text-to-speech (Orpheus, `canopylabs/orpheus-v1-english`) and speech-to-text
(Whisper), via the same GROQ_API_KEY. Used by /voice/tts, /voice/stt and the voice gateway.

Note: `playai-tts` was decommissioned by Groq (it returned 400 model_decommissioned). If TTS
is unavailable the voice gateway falls back to client-side TTS (`say` events).
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


async def stt_detailed(audio: bytes, filename: str = "audio.wav",
                       language: str = "") -> tuple[str, float]:
    """Transcribe with a confidence in [0,1] (from Whisper's segment no_speech_prob and
    avg_logprob), so the voice layer can ask the user to repeat instead of acting on a
    mishearing. ("", 0.0) if unavailable/failed."""
    s = get_settings()
    if not s.has_groq or not audio:
        return "", 0.0
    headers = {"Authorization": f"Bearer {s.groq_api_key}"}
    files = {"file": (filename, audio, "application/octet-stream")}
    data = {"model": s.groq_stt_model, "response_format": "verbose_json", "temperature": "0"}
    if language:
        data["language"] = language
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(STT_URL, headers=headers, files=files, data=data)
            r.raise_for_status()
            body = r.json()
    except httpx.HTTPError as e:
        log.warning("Groq STT failed: %s", e)
        return "", 0.0
    import math
    segs = body.get("segments") or []
    if not segs:
        text = (body.get("text") or "").strip()
        return text, (0.6 if text else 0.0)
    confs = [(1 - float(sg.get("no_speech_prob", 0))) *
             math.exp(min(0.0, float(sg.get("avg_logprob", 0)))) for sg in segs]
    return (body.get("text") or "").strip(), sum(confs) / len(confs)


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


_TTS_CACHE: dict[str, bytes] = {}


async def tts_cached(text: str) -> bytes | None:
    """TTS with a small cache: the voice layer repeats short lines ("On it.", "Got it.")
    constantly — no need to pay for them twice."""
    key = text.strip()
    if key in _TTS_CACHE:
        return _TTS_CACHE[key]
    audio = await tts(key)
    if audio and len(key) <= 80:
        if len(_TTS_CACHE) >= 64:
            _TTS_CACHE.pop(next(iter(_TTS_CACHE)))
        _TTS_CACHE[key] = audio
    return audio
