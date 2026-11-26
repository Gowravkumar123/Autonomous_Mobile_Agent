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


class BmsListPhone:
    """BookMyShow's Now Showing list (the live movies check, 2026-11-26)."""
    kind = "phone"
    device_id = "p2"

    async def request(self, type_, timeout=60.0, **data):
        if type_ == "observe":
            els = [{"ref": i, "role": "TextView", "text": t} for i, t in enumerate(
                ["Now Showing", "Gunmaaster G9", "Bhogi", "ENE Repeat", "Ramayana"])]
            return {"screen": {"app_package": "com.bt.bms", "activity": "list", "elements": els}}
        return {"ok": True}

    async def push(self, *a, **k): ...
    def on_message(self, m): ...
    def fail_all(self, e): ...


async def answer_in_summary() -> None:
    from app.agent.state import TaskState
    devices = DeviceRegistry()
    c = AgentController(LLMRouter(Settings(groq_api_key="", gemini_api_key="")),
                        PolicyEngine(0), devices, TaskStore(), MemoryManager())
    devices.add(BmsListPhone())                         # type: ignore[arg-type]
    answer = "Now showing in Hyderabad: Gunmaaster G9, Bhogi, ENE Repeat, Ramayana."

    async def decide(task, skill, screen, feedback=""):
        return AgentAction(type=ActionType.DONE, text=answer, thought="list visible")
    c._decide = decide
    task = Task(prompt="which movies are showing in Hyderabad today?", device_id="p2",
                executor="mobile", goal="List movies showing in Hyderabad today",
                skill="generic", plan=["open BookMyShow", "read the list"])
    c.store.create(task)
    await asyncio.wait_for(c._execute_phase(task), 30)
    check(answer in task.summary and task.summary.startswith(("[OK] ", "[UNVERIFIED] ")),
          f"the agent's answer is the task result, honestly labelled ({task.state.value})")
    check(task.state in (TaskState.COMPLETED, TaskState.WAITING_FOR_USER),
          "read-only answer ends the task (no further steps)")

    # Second live run: the model put the list in `thought` and left `text` empty.
    async def decide_thought(task, skill, screen, feedback=""):
        return AgentAction(type=ActionType.DONE, text=None,
                           thought="The Now Showing screen lists: Gunmaaster G9, Bhogi.")
    c._decide = decide_thought
    t2 = Task(prompt="which movies?", device_id="p2", executor="mobile", goal="List movies",
              skill="generic", plan=["read"])
    c.store.create(t2)
    await asyncio.wait_for(c._execute_phase(t2), 30)
    check("Gunmaaster G9, Bhogi" in t2.summary,
          "answer left in `thought` (text empty) still becomes the result")


async def main() -> None:
    await scroll_fallback()
    await answer_in_summary()
    jwt_leeway()
    print(f"\nACT PASS  ({PASS} checks)")


if __name__ == "__main__":
    asyncio.run(main())
