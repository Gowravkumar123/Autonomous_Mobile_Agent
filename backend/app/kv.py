"""
Short-lived shared state (blueprint: "Redis — short-lived state, locks, queue
coordination, rate limiting").

With one backend process, in-process state is already correct, so the default needs no
Redis. Once you run several workers (or hosts), the counters that protect security and
cost MUST be shared — otherwise each worker gets its own rate limit / brute-force budget.
Set REDIS_URL (e.g. a free Upstash instance) and the same code uses Redis.

Used for: tool rate limits (cost), the device-pairing brute-force lock (security), and the
concurrent browser-session cap (resources).
"""
from __future__ import annotations

import logging
import time
from typing import Protocol

log = logging.getLogger("agent.kv")


class KV(Protocol):
    name: str

    async def incr(self, key: str, ttl_s: int) -> int: ...
    async def get_int(self, key: str) -> int: ...
    async def decr(self, key: str) -> int: ...
    async def setex(self, key: str, value: str, ttl_s: int) -> None: ...
    async def getdel(self, key: str) -> str | None: ...      # atomic read-and-delete
    async def close(self) -> None: ...


class MemoryKV:
    name = "memory"

    def __init__(self) -> None:
        self._d: dict[str, tuple[int, float]] = {}
        self._s: dict[str, tuple[str, float]] = {}

    def _live(self, key: str) -> int:
        v = self._d.get(key)
        if v is None or v[1] <= time.monotonic():
            self._d.pop(key, None)
            return 0
        return v[0]

    async def incr(self, key: str, ttl_s: int) -> int:
        cur = self._live(key)
        exp = self._d[key][1] if key in self._d else time.monotonic() + ttl_s
        self._d[key] = (cur + 1, exp)
        return cur + 1

    async def get_int(self, key: str) -> int:
        return self._live(key)

    async def decr(self, key: str) -> int:
        cur = self._live(key)
        if key in self._d:
            self._d[key] = (max(0, cur - 1), self._d[key][1])
        return max(0, cur - 1)

    async def setex(self, key: str, value: str, ttl_s: int) -> None:
        self._s[key] = (value, time.monotonic() + ttl_s)

    async def getdel(self, key: str) -> str | None:
        v = self._s.pop(key, None)
        return v[0] if v and v[1] > time.monotonic() else None

    async def close(self) -> None:
        pass


class RedisKV:
    name = "redis"

    def __init__(self, client) -> None:
        self.r = client

    @classmethod
    def from_url(cls, url: str) -> "RedisKV":
        import redis.asyncio as redis
        return cls(redis.from_url(url, decode_responses=True))

    async def incr(self, key: str, ttl_s: int) -> int:
        async with self.r.pipeline(transaction=True) as p:      # atomic INCR + EXPIRE NX
            p.incr(key)
            p.expire(key, ttl_s, nx=True)
            n, _ = await p.execute()
        return int(n)

    async def get_int(self, key: str) -> int:
        return int(await self.r.get(key) or 0)

    async def decr(self, key: str) -> int:
        n = int(await self.r.decr(key))
        if n < 0:
            await self.r.set(key, 0, keepttl=True)
            n = 0
        return n

    async def setex(self, key: str, value: str, ttl_s: int) -> None:
        await self.r.set(key, value, ex=ttl_s)

    async def getdel(self, key: str) -> str | None:
        return await self.r.getdel(key)       # atomic: a code can be redeemed only once

    async def close(self) -> None:
        try:
            await self.r.aclose()
        except Exception:
            pass


async def make_kv(url: str = "") -> KV:
    if not url.strip():
        return MemoryKV()
    try:
        kv = RedisKV.from_url(url.strip())
        await kv.r.ping()
        log.info("shared state: redis")
        return kv
    except Exception as e:
        log.warning("REDIS_URL set but unreachable (%s) — using in-process state", e)
        return MemoryKV()


async def over_limit(kv: KV, name: str, limit: int, window_s: int = 60) -> bool:
    """Fixed-window rate limit shared across workers: True if this call exceeds `limit`."""
    bucket = int(time.time() // window_s)
    n = await kv.incr(f"rl:{name}:{bucket}", window_s * 2)
    return n > limit
