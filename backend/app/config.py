"""Central configuration, loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    # Multimodal Groq model used for action decisions when a screenshot is available.
    # Uses the same GROQ_API_KEY. Override if Groq retires this model.
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    gemini_api_key: str = ""
    # 2.5 Flash: strong UI grounding + vision, free-tier friendly. 2.5 Pro is better but
    # has much tighter free limits (can rate-limit mid-task). Override via GEMINI_MODEL.
    gemini_model: str = "gemini-2.5-flash"

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
