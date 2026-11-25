"""Central configuration, loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    groq_api_key: str = ""
    # Capable model for the per-step action decisions (quality matters most here).
    groq_model: str = "openai/gpt-oss-120b"
    # Model for the simpler structured calls (intent + planning). Defaults to the capable
    # model: a weaker model (e.g. gpt-oss-20b) mis-classified intent and broke the whole
    # run, and these are only ~2 calls/task so the saving is tiny. Opt into a cheaper one
    # by setting GROQ_FAST_MODEL only if your tasks are simple.
    groq_fast_model: str = "openai/gpt-oss-120b"
    # Groq's multimodal model, used for action decisions on sparse screens when a
    # screenshot is available (same GROQ_API_KEY, fast). Override if Groq rotates it.
    groq_vision_model: str = "qwen/qwen3.8-27b"

    # --- Cost controls (tune the quality/spend trade-off; all overridable via env) --- #
    # Action JSON is tiny, so a tight output cap cuts output-token cost ~40% with no
    # quality loss. Vision (an image = many input tokens) is gated to only fire when it
    # actually helps: a sparse tree, or when the text path is stuck on a not-too-rich
    # screen (on dense trees the text path is both better AND cheaper).
    action_max_tokens: int = 700
    intent_max_tokens: int = 500
    plan_max_tokens: int = 600
    vision_sparse_max: int = 8      # use vision when a screen has <= this many elements
    vision_stuck_max: int = 28      # ...or when stuck, but only up to this element count
    max_elements_in_prompt: int = 55
    history_in_prompt: int = 6
    # Groq voice models for the assistant (same GROQ_API_KEY).
    groq_tts_model: str = "playai-tts"
    groq_tts_voice: str = "Fritz-PlayAI"
    groq_stt_model: str = "whisper-large-v3"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    # Gemini is OFF by default — its free tier is too slow for the per-step loop. Set
    # USE_GEMINI=true to add it as a fallback after Groq.
    use_gemini: bool = False

    # Persistence — Postgres (e.g. Neon). Empty = in-memory only (tasks lost on restart).
    database_url: str = ""

    # Authentication. HMAC key for the signed admin/device tokens. EMPTY = auth DISABLED
    # (every endpoint open — only acceptable on a private network). Generate one with
    # `python -m app.auth.cli new-secret`.
    auth_secret: str = ""
    auth_token_days: int = 365

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
