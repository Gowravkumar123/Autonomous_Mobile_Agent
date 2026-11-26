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
    # NOTE gpt-oss is a reasoning model: hidden reasoning counts toward these caps. Too
    # tight and the answer comes back EMPTY (seen live: intent at 500). The provider
    # retries once with a bigger cap on an empty "length" stop, but keep sane floors.
    action_max_tokens: int = 900
    intent_max_tokens: int = 900
    plan_max_tokens: int = 900
    # Intent + planning are simple structured calls: "low" reasoning is cheaper and
    # leaves more of the cap for the answer. Action decisions keep the model default.
    fast_reasoning_effort: str = "low"
    vision_sparse_max: int = 8      # use vision when a screen has <= this many elements
    vision_stuck_max: int = 28      # ...or when stuck, but only up to this element count
    max_elements_in_prompt: int = 55
    history_in_prompt: int = 6
    # Groq voice models for the assistant (same GROQ_API_KEY).
    groq_tts_model: str = "canopylabs/orpheus-v1-english"   # playai-tts was decommissioned
    groq_tts_voice: str = "troy"        # orpheus: autumn diana hannah austin daniel troy
    groq_stt_model: str = "whisper-large-v3"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    # Gemini is OFF by default — its free tier is too slow for the per-step loop. Set
    # USE_GEMINI=true to add it as a fallback after Groq.
    use_gemini: bool = False

    # Persistence — Postgres (e.g. Neon). Empty = in-memory only (tasks lost on restart).
    database_url: str = ""
    # Observability (OpenTelemetry). A built-in per-task timeline is always on. Set Langfuse
    # keys to send traces to Langfuse, or OTEL_ENDPOINT for any OTLP/HTTP backend (Jaeger,
    # Grafana Tempo, ...). Prompts/screens are NOT traced unless TRACE_CONTENT=true.
    otel_service_name: str = "onepiece-agent"
    otel_endpoint: str = ""             # e.g. http://localhost:4318/v1/traces
    otel_headers: str = ""              # "key=value,key2=value2"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    trace_content: bool = False
    llm_prices_json: str = ""           # {"model": [usd_per_1M_in, usd_per_1M_out]} overrides

    # Shared short-lived state (rate limits, pairing codes/lock, browser-session cap). Only
    # needed when running several workers; empty = in-process. e.g. rediss://...upstash.io
    redis_url: str = ""
    # Multi-worker routing over Redis (device sockets, task control, events). On whenever
    # REDIS_URL is set; CLUSTER_ENABLED=false keeps each worker standalone.
    cluster_enabled: bool = True

    # ---- Voice gateway (WS /v1/voice/{session}, WebRTC) ----
    voice_end_silence_ms: int = 700      # end-of-turn silence (VAD endpointing)
    voice_min_confidence: float = 0.35   # below this, ask the user to repeat
    voice_progress_s: float = 20.0       # spoken progress updates while a task works
    voice_partial_s: float = 0.0         # interim transcripts every N s (extra STT cost)
    voice_language: str = ""             # e.g. "en"; "" = auto-detect
    voice_ice_servers: str = "stun:stun.l.google.com:19302"   # WebRTC (comma-separated)

    # Authentication. HMAC key for the signed admin/device tokens. EMPTY = auth DISABLED
    # (every endpoint open — only acceptable on a private network). Generate one with
    # `python -m app.auth.cli new-secret`.
    auth_secret: str = ""
    auth_token_days: int = 365

    # Browser executor (Playwright). Needs `pip install playwright` and
    # `playwright install --with-deps chromium` on the server; absent = mobile only.
    browser_enabled: bool = True
    browser_headless: bool = True
    # Comma-separated allowlist of domains the browser may navigate to (subdomains
    # included). Empty = any domain.
    browser_allowed_domains: str = ""
    browser_search_url: str = "https://duckduckgo.com/html/?q={q}"
    browser_max_sessions: int = 2      # concurrent browser tasks (resource limit)

    # Tools (built-in + MCP servers). The MCP config file is gitignored (it may reference
    # secrets as ${ENV_VAR}); see mcp_servers.example.json.
    mcp_config_path: str = "mcp_servers.json"
    tools_builtin: bool = True         # web.search / web.fetch research tools
    tool_timeout_s: float = 30.0
    tool_rate_per_minute: int = 30     # per tool
    tool_max_calls_per_task: int = 25
    # Show the tool catalog to phone/browser tasks too (costs prompt tokens every step).
    tools_in_ui_tasks: bool = False

    # Long-term memory (pgvector on Postgres; in-memory without a DB).
    memory_enabled: bool = True
    memory_ttl_days: int = 365          # retention: memories expire after this
    memory_learn: bool = True           # propose (PENDING) preferences after a completed task
    memory_include_all_below: int = 20  # at most this many approved -> include all, no retrieval
    # Embeddings: open-source model run locally (fastembed/ONNX) by default; "hashing" is the
    # dependency-free fallback; "api" = any OpenAI-compatible /embeddings endpoint.
    embedding_backend: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: str = "models"  # persistent model cache (gitignored)
    embedding_api_url: str = ""
    embedding_api_key: str = ""

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
