"""
Fixes from the live BookMyShow run on the phone (2026-11-26):
- scroll on a custom-drawn screen (no scrollable node) failed in a loop -> swipe fallback
- admin tokens minted on another machine were intermittently "not yet valid" -> leeway

Run:  python -m tests.test_act
"""
from __future__ import annotations

import asyncio
import os
import time

os.environ.setdefault("DATABASE_URL", "")
os.environ.setdefault("GROQ_API_KEY", "")

import jwt                                                # noqa: E402

from app.agent.controller import AgentController          # noqa: E402
from app.agent.memory import MemoryManager                # noqa: E402
from app.agent.state import Task                          # noqa: E402
from app.auth.manager import ALG, ISSUER, AuthManager     # noqa: E402
from app.config import Settings                           # noqa: E402
from app.llm.router import LLMRouter                      # noqa: E402
from app.mobile.schema import ActionType, AgentAction     # noqa: E402
from app.policy.engine import PolicyEngine                # noqa: E402
from app.store import TaskStore                           # noqa: E402
from app.ws import DeviceRegistry                         # noqa: E402

PASS = 0


def check(cond: bool, label: str) -> None:
    global PASS
    assert cond, f"FAIL: {label}"
    PASS += 1
    print(f"PASS {label}")


class SeatMapPhone:
    """Like the BMS seat map: no scrollable node, but gestures work."""
    kind = "phone"
    device_id = "p1"

    def __init__(self):
        self.sent: list[dict] = []

    async def request(self, type_, timeout=60.0, **data):
        a = data.get("action") or {}
        self.sent.append(a)
        if a.get("type") == "scroll":
            return {"ok": False, "error": "failed"}
        return {"ok": True}

    async def push(self, *a, **k): ...
    def on_message(self, m): ...
    def fail_all(self, e): ...


async def scroll_fallback() -> None:
    c = AgentController(LLMRouter(Settings(groq_api_key="", gemini_api_key="")),
                        PolicyEngine(0), DeviceRegistry(), TaskStore(), MemoryManager())
    phone, task = SeatMapPhone(), Task(prompt="x")
    await c._act(phone, AgentAction(type=ActionType.SCROLL, direction="down"), task)
    check([a.get("type") for a in phone.sent] == ["scroll", "swipe"] and
          phone.sent[1]["direction"] == "up",
          "failed scroll down -> swipe with the finger moving UP (content scrolls down)")
    check(not any("REFUSED" in h for h in task.history),
          "the model isn't told it failed when the swipe worked")
    phone.sent.clear()
    await c._act(phone, AgentAction(type=ActionType.SCROLL, direction="up"), task)
    check(phone.sent[1]["direction"] == "down", "scroll up -> swipe down")


def jwt_leeway() -> None:
    secret = "s" * 48
    auth = AuthManager(secret)
    now = int(time.time())
    future = jwt.encode({"iss": ISSUER, "sub": "admin", "scope": "admin", "jti": "j1",
                         "iat": now + 5, "nbf": now + 5, "exp": now + 3600}, secret, algorithm=ALG)
    check(auth.verify(future).is_admin, "token minted on a clock 5 s ahead is accepted")
    far = jwt.encode({"iss": ISSUER, "sub": "admin", "scope": "admin", "jti": "j2",
                      "iat": now + 600, "nbf": now + 600, "exp": now + 3600}, secret, algorithm=ALG)
    try:
        auth.verify(far)
        ok = False
    except Exception:
        ok = True
    check(ok, "...but one 10 minutes in the future is still rejected")


async def main() -> None:
    await scroll_fallback()
    jwt_leeway()
    print(f"\nACT PASS  ({PASS} checks)")


if __name__ == "__main__":
    asyncio.run(main())
