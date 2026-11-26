"""
LIVE memory run: real Groq model + the open-source embedding model (in-memory store).
1) "remember ..." -> stored;  2) a later request uses it to fill a fact;
3) learning proposes a PENDING preference;  4) "forget ..." deletes it.

Run:  python -m tests.live_memory
"""
from __future__ import annotations

import asyncio
import os

os.environ["DATABASE_URL"] = ""

from app.agent.controller import AgentController       # noqa: E402
from app.agent.memory import FastEmbedEmbedder, MemoryManager   # noqa: E402
from app.agent.state import Task, TaskState            # noqa: E402
from app.config import get_settings                    # noqa: E402
from app.llm.router import LLMRouter                   # noqa: E402
from app.policy.engine import PolicyEngine             # noqa: E402
from app.store import TaskStore                        # noqa: E402
from app.ws import DeviceRegistry                      # noqa: E402

ME = "phone-A"


async def main():
    s = get_settings()
    assert s.has_groq, "GROQ_API_KEY missing in backend/.env"
    emb = FastEmbedEmbedder(cache_dir=os.path.join(os.path.dirname(__file__), "..", "models"))
    await emb.load()
    mem = MemoryManager(embedder=emb)
    c = AgentController(LLMRouter(s), PolicyEngine(0), DeviceRegistry(), TaskStore(), mem)

    t1 = Task(prompt="Remember that my mom is saved as Amma in my WhatsApp", owner=ME)
    await c.run_task(t1)
    print("1) remember   ->", t1.state.value, "|", t1.summary)

    t2 = Task(prompt="Send good morning to mom on WhatsApp", owner=ME)
    await c._plan_phase(t2)                                   # plan only: no phone needed
    print("2) recall     -> memory_context:", t2.memory_context)
    print("               intent facts   :", t2.facts, "| clarifications:", t2.clarifications)

    t3 = Task(prompt="Book 2 tickets for Scene, aisle seats please, Telugu", owner=ME,
              executor="mobile", facts={"item": "Scene", "passengers": 2,
                                        "seat_preference": "aisle", "language": "Telugu"},
              summary="[OK] Booked 2 aisle seats for Scene (Telugu) at PVR",
              state=TaskState.COMPLETED)
    await c._learn(t3)
    print("3) learn      -> pending:", [m.content for m in await mem.list(ME, "pending")])

    t4 = Task(prompt="Forget how my mom is saved in WhatsApp", owner=ME)
    await c.run_task(t4)
    print("4) forget     ->", t4.state.value, "|", t4.summary)
    print("   approved now:", [m.content for m in await mem.list(ME)])


if __name__ == "__main__":
    asyncio.run(main())
