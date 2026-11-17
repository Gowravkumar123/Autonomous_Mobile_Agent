"""Central configuration, loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    # Groq's multimodal model, used for action decisions on sparse screens when a
    # screenshot is available (same GROQ_API_KEY, fast). Override if Groq rotates it.
    groq_vision_model: str = "qwen/qwen3.8-27b"
    # Groq voice models for the assistant (same GROQ_API_KEY).
    groq_tts_model: str = "playai-tts"
    groq_tts_voice: str = "Fritz-PlayAI"
    groq_stt_model: str = "whisper-large-v3"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    # Gemini is OFF by default — its free tier is too slow for the per-step loop. Set
    # USE_GEMINI=true to add it as a fallback after Groq.
    use_gemini: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Safety
    # Any monetary amount >= this threshold forces a user-approval pause.
    # 0 means "pause before ANY payment" (the safe default).
    spend_approval_threshold: float = 0.0

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key.strip())

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
