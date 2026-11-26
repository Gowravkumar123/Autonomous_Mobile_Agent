"""
Shared short-lived state: the KV contract on in-process memory AND Redis (fakeredis), and
the multi-worker guarantees that motivate Redis — simulated as two workers on one Redis.

Run:  python -m tests.test_kv
"""
from __future__ import annotations

import asyncio
import os

os.environ["DATABASE_URL"] = ""

import fakeredis                                         # noqa: E402

from app.auth.manager import AuthError, AuthManager      # noqa: E402
from app.kv import MemoryKV, RedisKV, over_limit          # noqa: E402
from app.tools.registry import ToolRegistry               # noqa: E402

PASS = 0


def check(cond: bool, label: str) -> None:
    global PASS
    assert cond, f"FAIL: {label}"
    PASS += 1
    print(f"PASS {label}")


async def contract(kv, tag: str):
    check(await kv.incr("c", 60) == 1 and await kv.incr("c", 60) == 2, f"[{tag}] incr")
    check(await kv.get_int("c") == 2 and await kv.decr("c") == 1, f"[{tag}] get/decr")
    check(await kv.decr("nope") == 0, f"[{tag}] decr never goes negative")
    await kv.setex("s", "v", 60)
    check(await kv.getdel("s") == "v" and await kv.getdel("s") is None,
          f"[{tag}] getdel is read-once")
    await kv.setex("short", "v", 1)
    await asyncio.sleep(1.2)
    check(await kv.getdel("short") is None, f"[{tag}] values expire")


async def main():
    await contract(MemoryKV(), "memory")
    shared = fakeredis.aioredis.FakeRedis(decode_responses=True)
    await contract(RedisKV(shared), "redis")

    # Two "workers" = two separate KV handles on ONE Redis.
    w1, w2 = RedisKV(shared), RedisKV(shared)

    hits = [await over_limit(w, "tool:x", 3) for w in (w1, w2, w1, w2)]
    check(hits == [False, False, False, True],
          "rate limit is shared across workers (4th call over a limit of 3 is refused)")

    a1, a2 = AuthManager("k" * 40), AuthManager("k" * 40)
    a1.kv, a2.kv = w1, w2
    code, _ = await a1.start_pairing()
    tok = await a2.complete_pairing(code, "phone-A")
    check(bool(tok), "pairing code issued on worker 1 is redeemed on worker 2")
    try:
        await a1.complete_pairing(code, "phone-B")
        check(False, "code is single-use across workers")
    except AuthError:
        check(True, "code is single-use across workers (atomic GETDEL)")
    for i in range(20):
        try:
            await (a1 if i % 2 else a2).complete_pairing(f"{i:06d}", "x")
        except AuthError:
            pass
    try:
        await a1.complete_pairing("123456", "x")
        check(False, "brute-force lock counts failures from ALL workers")
    except AuthError as e:
        check("too many" in str(e), "brute-force lock counts failures from ALL workers")

    r1, r2 = ToolRegistry(builtins=False, rate_per_minute=2), ToolRegistry(builtins=False,
                                                                            rate_per_minute=2)
    r1.kv, r2.kv = w1, w2
    got = [await r._rate_limited("demo.tool") for r in (r1, r2, r1)]
    check(got == [False, False, True], "tool rate limit shared between two registries")

    print(f"\nKV PASS  ({PASS} checks)")


if __name__ == "__main__":
    asyncio.run(main())
