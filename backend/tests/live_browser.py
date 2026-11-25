"""
LIVE browser run with the real Groq model (uses GROQ_API_KEY from backend/.env and
spends a few dozen requests). Not part of the automated suite.

The agent gets a natural-language goal and the local CineTest site, and must plan,
pick the browser executor, navigate, and stop at the payment boundary on its own.
Approvals are granted by a watcher (standing in for the user).

Run:  python -m tests.live_browser
"""
from __future__ import annotations

import asyncio
import os

os.environ["DATABASE_URL"] = ""            # keep live test rows out of Neon

from app.agent.controller import AgentController       # noqa: E402
from app.agent.memory import MemoryManager             # noqa: E402
from app.agent.state import TERMINAL, Task, TaskState  # noqa: E402
from app.config import get_settings                    # noqa: E402
from app.llm.router import LLMRouter                   # noqa: E402
from app.policy.engine import PolicyEngine             # noqa: E402
from app.store import TaskStore                        # noqa: E402
from app.ws import DeviceRegistry                      # noqa: E402
from tests.site_server import serve                    # noqa: E402

GOAL = ("On the CineTest website, book 2 seats for the movie Scene at the 7:00 PM show. "
        "Stop before payment.")


async def main():
    s = get_settings()
    assert s.has_groq, "GROQ_API_KEY missing in backend/.env"
    srv, base = serve()
    store, devices = TaskStore(), DeviceRegistry()
    c = AgentController(LLMRouter(s), PolicyEngine(spend_threshold=0), devices, store,
                        MemoryManager())
    task = Task(prompt=GOAL, owner="admin", start_url=base + "/index.html")
    store.create(task)

    async def stand_in_user():
        # Runs until cancelled after run_task returns (answers whatever is pending).
        while True:
            if task.pending_approval_id:
                rec = c.get_approval(task.pending_approval_id)
                print(f"  [user] approving: {rec.reason} (risk {rec.risk_level})")
                c.decide_approval(task.pending_approval_id, True, approver="admin")
            fut = c._answers.get(task.id)
            if fut is not None and not fut.done():
                asked = [e.detail for e in task.events if e.kind == "state_change"][-1:]
                q = c._pending_prompts.get(task.id, ("", {}))[1].get("questions")
                print(f"  [user] agent asked: {q or asked} -> answering 'any, you choose'")
                c.resolve_answer(task.id, {"clarification_answer": "any, you choose"})
            await asyncio.sleep(0.3)
    helper = asyncio.create_task(stand_in_user())
    try:
        await asyncio.wait_for(c.run_task(task), 300)
    finally:
        helper.cancel()
        srv.shutdown()

    print("\n--- LIVE RESULT ---")
    print("executor :", task.executor, "| skill:", task.skill)
    print("plan     :", task.plan)
    print("history  :")
    for h in task.history:
        print("   •", h)
    print("state    :", task.state.value)
    print("summary  :", task.summary)
    print("approvals:", [(r.reason, r.risk_level, r.decision) for r in c.approvals_for(task.id)])
    print("evidence :", task.evidence)


if __name__ == "__main__":
    asyncio.run(main())
