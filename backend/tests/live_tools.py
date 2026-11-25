"""
LIVE tools run with the real Groq model (uses GROQ_API_KEY from backend/.env).
No executor is forced: intent must judge that tools suffice -> API executor; the model
must pick the right MCP tools, and the write tool must go through approval.

Run:  python -m tests.live_tools
"""
from __future__ import annotations

import asyncio
import os

os.environ["DATABASE_URL"] = ""

from app.agent.controller import AgentController       # noqa: E402
from app.agent.memory import MemoryManager             # noqa: E402
from app.agent.state import Task                       # noqa: E402
from app.config import get_settings                    # noqa: E402
from app.llm.router import LLMRouter                   # noqa: E402
from app.policy.engine import PolicyEngine             # noqa: E402
from app.store import TaskStore                        # noqa: E402
from app.tools.registry import ToolRegistry            # noqa: E402
from app.ws import DeviceRegistry                      # noqa: E402

GOAL = ("What's the current weather in Hyderabad, and what is 17 + 25? Then save a note "
        "titled 'Trip' containing the weather.")


async def main():
    s = get_settings()
    assert s.has_groq, "GROQ_API_KEY missing in backend/.env"
    reg = ToolRegistry({"servers": {"demo": {
        "command": "python", "args": ["-m", "tests.mcp_test_server"],
        "trust_annotations": True, "tools": {"delete_all_notes": "deny"}}}})
    await reg.start()
    store, devices = TaskStore(), DeviceRegistry()
    c = AgentController(LLMRouter(s), PolicyEngine(0), devices, store, MemoryManager())
    c.tools = reg
    task = Task(prompt=GOAL, owner="admin")
    store.create(task)

    async def stand_in_user():
        while True:
            if task.pending_approval_id:
                rec = c.get_approval(task.pending_approval_id)
                print(f"  [user] approving: {rec.reason[:110]}")
                c.decide_approval(task.pending_approval_id, True, approver="admin")
            fut = c._answers.get(task.id)
            if fut is not None and not fut.done():
                q = c._pending_prompts.get(task.id, ("", {}))[1].get("questions")
                print(f"  [user] agent asked {q} -> 'you choose'")
                c.resolve_answer(task.id, {"clarification_answer": "you choose"})
            await asyncio.sleep(0.3)
    helper = asyncio.create_task(stand_in_user())
    try:
        await asyncio.wait_for(c.run_task(task), 240)
    finally:
        helper.cancel()
        await reg.stop()

    print("\n--- LIVE RESULT ---")
    print("executor :", task.executor, "| hint:", task.executor_hint, "| skill:", task.skill)
    print("history  :")
    for h in task.history:
        print("   •", h)
    print("state    :", task.state.value)
    print("answer   :", task.summary)
    print("approvals:", [(r.reason[:70], r.decision) for r in c.approvals_for(task.id)])


if __name__ == "__main__":
    asyncio.run(main())
