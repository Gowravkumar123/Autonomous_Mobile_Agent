"""
Long-term memory (blueprint section 8: `memory_items` — "user-approved preferences and
useful long-term context", pgvector for semantic retrieval, "clear consent and retention
controls").

Consent model:
  - Only APPROVED memories are ever retrieved into a prompt.
  - "Remember that ..." from the user, or POST /v1/memory, is approved by definition.
  - Preferences the agent LEARNS from a completed task are stored PENDING until the user
    approves them (GET /v1/memory?status=pending, POST /v1/memory/{id}/approve).
  - Secrets are refused outright (OTP/PIN/CVV/password/card or Aadhaar numbers/PAN).
  - Every memory expires (MEMORY_TTL_DAYS) and can be hard-deleted (DELETE /v1/memory/{id}).
  - Memories are scoped to their owner (admin, or one paired device).
  - In prompts, memory text is labelled as DATA, never instructions.

Retrieval: at personal scale (<= MEMORY_INCLUDE_ALL_BELOW approved items for an owner) ALL
are included — no retrieval mistakes, few tokens. Beyond that, top-k by cosine similarity
(pgvector `<=>` on Postgres, numpy-free Python fallback in memory).

Embeddings: Groq has no embedding model, so the default is an OPEN-SOURCE model run locally
via FastEmbed/ONNX (BAAI/bge-small-en-v1.5 — free, private, CPU, real semantic matching:
"cinema" finds "movie theatres", "mom" finds "mother"). If it can't load, a dependency-free
hashing embedder (lexical only) takes over. EMBEDDING_BACKEND=api uses any OpenAI-compatible
endpoint instead. Rows embedded by another embedder are re-embedded on startup.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, field

log = logging.getLogger("agent.memory")

APPROVED, PENDING = "approved", "pending"
MAX_CHARS = 300


class MemoryError_(ValueError):
    """Rejected memory (secret, empty, too long)."""


# --------------------------------------------------------------------------- #
# secret guard — never store credentials or identity numbers
# --------------------------------------------------------------------------- #
_SECRET_WORDS = re.compile(
    r"\b(otp|one[- ]time password|pin|mpin|upi pin|cvv|cvc|password|passcode|pass code|"
    r"secret|api[ _-]?key|private key|seed phrase|recovery phrase|security code)\b", re.I)
_LONG_NUMBER = re.compile(r"(?:\d[ -]?){12,19}")              # card / Aadhaar / account
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")                    # Indian PAN


def reject_reason(content: str) -> str:
    c = (content or "").strip()
    if not c:
        return "empty memory"
    if len(c) > MAX_CHARS:
        return f"too long (max {MAX_CHARS} characters)"
    if _SECRET_WORDS.search(c):
        return "looks like a secret (OTP/PIN/password/key) — never stored"
    if _LONG_NUMBER.search(c):
        return "contains a long number (card/Aadhaar/account) — never stored"
    if _PAN.search(c):
        return "contains a PAN number — never stored"
    return ""


# --------------------------------------------------------------------------- #
# embedders
# --------------------------------------------------------------------------- #
_STOP = set("a an the is are was were be been am i me my mine you your we our to of in on at "
            "for and or with that this it its as by from please remember about".split())
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def _stem(w: str) -> str:
    for suf in ("ing", "ies", "es", "ed", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


class HashingEmbedder:
    """Free, deterministic, dependency-free FALLBACK. Lexical similarity only (shared
    words / word pieces) — used when the open-source model can't be loaded."""

    # Thresholds are per embedder: score scales differ wildly between models.
    retrieval_floor = 0.08       # top-k mode: ignore weaker matches
    forget_min = 0.35            # 'forget X' must match at least this well ...
    forget_margin = 0.05         # ... and beat the runner-up by this much
    dup_threshold = 0.93         # near-identical -> reuse instead of storing twice

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hash{dim}-v1"

    async def load(self) -> None:
        return None

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        return await self.embed(texts)

    def _features(self, text: str):
        words = [_stem(w) for w in (t.lower() for t in _TOKEN.findall(text)) if w not in _STOP]
        for w in words:
            yield "w:" + w, 1.0
            padded = f"#{w}#"
            for i in range(len(padded) - 2):
                yield "c:" + padded[i:i + 3], 0.25
        for a, b in zip(words, words[1:]):
            yield f"b:{a}_{b}", 0.6

    def _one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for feat, weight in self._features(text):
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += weight if (h >> 63) & 1 else -weight
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


class FastEmbedEmbedder:
    """OPEN-SOURCE model run locally on CPU via ONNX (FastEmbed) — free, private (memories
    never leave the server), no PyTorch/GPU. Default BAAI/bge-small-en-v1.5 (384-dim,
    ~130 MB, downloaded once). Multilingual option:
    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.

    bge scores are compressed (unrelated text ~0.5-0.6, related ~0.6-0.75, different
    memories up to ~0.7), so thresholds are calibrated for it and 'forget' needs a margin.
    """

    retrieval_floor = 0.55
    forget_min = 0.62
    forget_margin = 0.06
    dup_threshold = 0.95

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None,
                 threads: int | None = None):
        import os
        import threading
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        self.model_name = model
        self.cache_dir = cache_dir or None
        self.threads = threads
        self.name = f"fastembed:{model}"
        self._model = None
        self._lock = threading.Lock()

    def _load_sync(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(self.model_name, cache_dir=self.cache_dir,
                                            threads=self.threads)
        return self._model

    async def load(self) -> None:
        import asyncio
        await asyncio.to_thread(self._load_sync)

    def _run(self, texts: list[str], query: bool) -> list[list[float]]:
        m = self._load_sync()
        vecs = m.query_embed(texts) if query else m.embed(texts)
        out = []
        for v in vecs:
            v = [float(x) for x in v]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        return await asyncio.to_thread(self._run, texts, False)

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        import asyncio                       # bge uses a query instruction for retrieval
        return await asyncio.to_thread(self._run, texts, True)


async def make_embedder(backend: str = "fastembed", model: str = "", cache_dir: str = "",
                        api_url: str = "", api_key: str = ""):
    """Build the configured embedder; fall back to hashing if it can't be loaded."""
    backend = (backend or "fastembed").lower()
    if backend == "hashing":
        return HashingEmbedder()
    try:
        if backend == "api":
            emb = ApiEmbedder(api_url, api_key, model)
            await emb.embed(["warm-up"])
        else:
            emb = FastEmbedEmbedder(model or "BAAI/bge-small-en-v1.5", cache_dir or None)
            await emb.load()
        log.info("memory embedder: %s", emb.name)
        return emb
    except Exception as e:
        log.warning("memory embedder %s unavailable (%s) — falling back to hashing", backend, e)
        return HashingEmbedder()


class ApiEmbedder:
    """Any OpenAI-compatible /embeddings endpoint (optional; costs per call)."""

    retrieval_floor = 0.3
    forget_min = 0.5
    forget_margin = 0.06
    dup_threshold = 0.95

    def __init__(self, url: str, key: str, model: str):
        self.url = url.rstrip("/")
        self.key = key
        self.model = model
        self.name = f"api:{model}"

    async def load(self) -> None:
        return None

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        return await self.embed(texts)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{self.url}/embeddings",
                             headers={"Authorization": f"Bearer {self.key}"},
                             json={"model": self.model, "input": texts})
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
        out = []
        for d in data:
            v = d["embedding"]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


# --------------------------------------------------------------------------- #
@dataclass
class MemoryItem:
    owner: str
    content: str
    kind: str = "preference"
    source: str = "user"
    status: str = APPROVED
    id: str = field(default_factory=lambda: "mem_" + uuid.uuid4().hex[:14])
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    last_used_at: float | None = None
    use_count: int = 0
    score: float | None = None

    def view(self) -> dict:
        return asdict(self)


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS memory_items (
        id            text PRIMARY KEY,
        owner         text NOT NULL,
        content       text NOT NULL,
        kind          text,
        source        text,
        status        text NOT NULL,
        embedder      text NOT NULL,
        embedding     vector,
        created_at    timestamptz NOT NULL,
        expires_at    timestamptz NOT NULL,
        last_used_at  timestamptz,
        use_count     integer NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS memory_owner_idx ON memory_items (owner, status)",
]
_COLS = "id, owner, content, kind, source, status, created_at, expires_at, last_used_at, use_count"


def _row_to_item(r) -> MemoryItem:
    ts = lambda x: x.timestamp() if hasattr(x, "timestamp") else x   # noqa: E731
    return MemoryItem(id=r[0], owner=r[1], content=r[2], kind=r[3] or "", source=r[4] or "",
                      status=r[5], created_at=ts(r[6]), expires_at=ts(r[7]),
                      last_used_at=ts(r[8]) if r[8] else None, use_count=r[9] or 0)


class MemoryManager:
    def __init__(self, db=None, embedder=None, *, ttl_days: int = 365,
                 include_all_below: int = 20, top_k: int = 6):
        self.db = db
        self.embedder = embedder or HashingEmbedder()
        self.ttl_s = ttl_days * 86400
        self.include_all_below = include_all_below
        self.top_k = top_k
        self._mem: dict[str, tuple[MemoryItem, list[float]]] = {}   # in-memory fallback

    @property
    def persistent(self) -> bool:
        return bool(self.db and getattr(self.db, "ok", False))

    # ------------------------------------------------------------------ #
    async def init(self) -> None:
        if not self.persistent:
            return
        try:
            await self.db.execute("CREATE EXTENSION IF NOT EXISTS vector")
            for stmt in SCHEMA:
                await self.db.execute(stmt)
            await self.purge_expired()
            await self._reembed_foreign()
        except Exception as e:
            log.warning("memory: postgres unavailable, using in-memory store: %s", e)
            self.db = None

    async def _reembed_foreign(self) -> None:
        rows = await self.db.query("SELECT id, content FROM memory_items WHERE embedder <> %s",
                                   (self.embedder.name,))
        if not rows:
            return
        vecs = await self.embedder.embed([c for _, c in rows])
        for (mid, _), v in zip(rows, vecs):
            await self.db.execute("UPDATE memory_items SET embedding = %s::vector, embedder = %s "
                                  "WHERE id = %s", (_vec_literal(v), self.embedder.name, mid))
        log.info("memory: re-embedded %d item(s) with %s", len(rows), self.embedder.name)

    async def purge_expired(self) -> int:
        now = time.time()
        if self.persistent:
            rows = await self.db.query("DELETE FROM memory_items WHERE expires_at <= now() "
                                       "RETURNING id")
            return len(rows)
        dead = [k for k, (m, _) in self._mem.items() if m.expires_at <= now]
        for k in dead:
            self._mem.pop(k, None)
        return len(dead)

    # ------------------------------------------------------------------ #
    async def add(self, owner: str, content: str, *, kind: str = "preference",
                  source: str = "user", approved: bool = True) -> MemoryItem:
        content = re.sub(r"\s+", " ", (content or "")).strip()
        why = reject_reason(content)
        if why:
            raise MemoryError_(why)
        [vec] = await self.embedder.embed([content])
        # De-duplicate: a near-identical memory for this owner is reused, not repeated.
        for existing in await self._scored(owner, vec, statuses=(APPROVED, PENDING), limit=3):
            if (existing.score or 0) >= self.embedder.dup_threshold:
                if approved and existing.status == PENDING:
                    await self.approve(existing.id, owner)
                    existing.status = APPROVED
                return existing
        item = MemoryItem(owner=owner, content=content, kind=kind, source=source,
                          status=APPROVED if approved else PENDING,
                          expires_at=time.time() + self.ttl_s)
        if self.persistent:
            await self.db.execute(
                "INSERT INTO memory_items (id, owner, content, kind, source, status, embedder, "
                "embedding, created_at, expires_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector,"
                "to_timestamp(%s::float8),to_timestamp(%s::float8))",
                (item.id, owner, content, kind, source, item.status, self.embedder.name,
                 _vec_literal(vec), item.created_at, item.expires_at))
        else:
            self._mem[item.id] = (item, vec)
        return item

    async def get(self, mem_id: str) -> MemoryItem | None:
        if self.persistent:
            rows = await self.db.query(f"SELECT {_COLS} FROM memory_items WHERE id = %s",
                                       (mem_id,))
            return _row_to_item(rows[0]) if rows else None
        hit = self._mem.get(mem_id)
        return hit[0] if hit else None

    async def list(self, owner: str | None = None, status: str = APPROVED) -> list[MemoryItem]:
        await self.purge_expired()
        statuses = (APPROVED, PENDING) if status == "all" else (status,)
        if self.persistent:
            sql = f"SELECT {_COLS} FROM memory_items WHERE status = ANY(%s)"
            params: list = [list(statuses)]
            if owner is not None:
                sql += " AND owner = %s"
                params.append(owner)
            rows = await self.db.query(sql + " ORDER BY created_at DESC", tuple(params))
            return [_row_to_item(r) for r in rows]
        items = [m for m, _ in self._mem.values()
                 if m.status in statuses and (owner is None or m.owner == owner)]
        return sorted(items, key=lambda m: m.created_at, reverse=True)

    async def approve(self, mem_id: str, owner: str | None = None) -> bool:
        item = await self.get(mem_id)
        if item is None or (owner is not None and item.owner != owner):
            return False
        if self.persistent:
            await self.db.execute("UPDATE memory_items SET status = %s WHERE id = %s",
                                  (APPROVED, mem_id))
        else:
            item.status = APPROVED
        return True

    async def delete(self, mem_id: str, owner: str | None = None) -> bool:
        item = await self.get(mem_id)
        if item is None or (owner is not None and item.owner != owner):
            return False
        if self.persistent:
            await self.db.execute("DELETE FROM memory_items WHERE id = %s", (mem_id,))
        else:
            self._mem.pop(mem_id, None)
        return True

    # ------------------------------------------------------------------ #
    async def _scored(self, owner: str, vec: list[float], *, statuses=(APPROVED,),
                      limit: int = 6) -> list[MemoryItem]:
        if self.persistent:
            rows = await self.db.query(
                f"SELECT {_COLS}, 1 - (embedding <=> %s::vector) AS sim FROM memory_items "
                "WHERE owner = %s AND status = ANY(%s) AND embedder = %s AND expires_at > now() "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (_vec_literal(vec), owner, list(statuses), self.embedder.name,
                 _vec_literal(vec), limit))
            out = []
            for r in rows:
                item = _row_to_item(r[:10])
                item.score = float(r[10]) if r[10] is not None else 0.0
                out.append(item)
            return out
        now = time.time()
        scored = []
        for m, v in self._mem.values():
            if m.owner == owner and m.status in statuses and m.expires_at > now:
                m.score = _cos(vec, v)
                scored.append(m)
        return sorted(scored, key=lambda m: m.score or 0, reverse=True)[:limit]

    async def relevant(self, owner: str, query: str, k: int | None = None) -> list[MemoryItem]:
        """Approved memories useful for this request (all of them at small scale)."""
        approved = await self.list(owner, APPROVED)
        if not approved:
            return []
        if len(approved) <= self.include_all_below:
            chosen = approved
        else:
            [vec] = await self.embedder.embed_query([query])
            chosen = [m for m in await self._scored(owner, vec, limit=k or self.top_k)
                      if (m.score or 0) >= self.embedder.retrieval_floor]
        await self._mark_used([m.id for m in chosen])
        return chosen

    async def forget_matching(self, owner: str, description: str
                              ) -> tuple[list[MemoryItem], list[MemoryItem]]:
        """'Forget ...' -> hard-delete ONLY a clear winner. Returns (deleted, candidates).

        Deleting is destructive and similarity scores of different memories can be close
        (bge: up to ~0.7 between unrelated facts), so the best match must clear an
        absolute floor AND beat the runner-up by a margin. Otherwise nothing is deleted
        and the candidates are returned so the user can pick (DELETE /v1/memory/{id}).
        """
        [vec] = await self.embedder.embed_query([description])
        cands = await self._scored(owner, vec, statuses=(APPROVED, PENDING), limit=3)
        if not cands or (cands[0].score or 0) < self.embedder.forget_min:
            return [], cands
        runner_up = (cands[1].score or 0) if len(cands) > 1 else 0.0
        if (cands[0].score or 0) - runner_up < self.embedder.forget_margin:
            return [], cands
        await self.delete(cands[0].id, owner)
        return [cands[0]], cands

    async def _mark_used(self, ids: list[str]) -> None:
        if not ids:
            return
        now = time.time()
        if self.persistent:
            await self.db.execute("UPDATE memory_items SET last_used_at = now(), "
                                  "use_count = use_count + 1 WHERE id = ANY(%s)", (ids,))
        else:
            for i in ids:
                if i in self._mem:
                    self._mem[i][0].last_used_at = now
                    self._mem[i][0].use_count += 1
