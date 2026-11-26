"""
LIVE trace with the real Groq model: runs the tools scenario and prints the built-in
timeline — where the time went (model vs tools vs human approval) and tokens / est. cost.

Run:  python -m tests.live_trace
"""
from __future__ import annotations

import asyncio
import os

os.environ["DATABASE_URL"] = ""

from app import observability as obs                  # noqa: E402
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
    reg = ToolRegistry({"servers": {"demo": {"command": "python",
                                             "args": ["-m", "tests.mcp_test_server"],
                                             "trust_annotations": True}}})
    await reg.start()
    c = AgentController(LLMRouter(s), PolicyEngine(0), DeviceRegistry(), TaskStore(),
                        MemoryManager())
    c.tools = reg
    _ = c.graph                                           # warm (as the app does at startup)
    task = Task(prompt=GOAL, owner="admin")
    c.store.create(task)

    async def user():
        while True:
            if task.pending_approval_id:
                await asyncio.sleep(1.5)                  # a human takes a moment
                c.decide_approval(task.pending_approval_id, True, approver="admin")
            fut = c._answers.get(task.id)
            if fut is not None and not fut.done():
                q = c._pending_prompts.get(task.id, ("", {}))[1].get("questions")
                print("  [user asked]", q)
                await asyncio.sleep(1.0)
                c.resolve_answer(task.id, {"clarification_answer": "Yes, save it"})
            await asyncio.sleep(0.2)
    h = asyncio.create_task(user())
    try:
        await asyncio.wait_for(c.run_task(task), 240)
    finally:
        h.cancel()
        await reg.stop()

    tl = obs.STORE.timeline(task.id)
    root = next(s for s in tl if s["name"] == "task")
    by: dict[str, float] = {}
    for sp in tl:
        if sp["name"] in ("llm.chat", "tool.call", "approval.wait", "memory.recall"):
            by[sp["name"]] = by.get(sp["name"], 0) + sp["duration_ms"]
    print(f"\nstate {task.state.value} | {len(tl)} spans | total {root['duration_ms']/1000:.1f}s")
    for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"  {k:15} {v/1000:6.2f}s  ({100*v/root['duration_ms']:4.1f}%)")
    print("\nLLM calls:")
    for sp in tl:
        if sp["name"] == "llm.chat":
            a = sp["attributes"]
            print(f"  {sp['start_ms']/1000:6.2f}s  {sp['duration_ms']:6.0f}ms  in={a.get('gen_ai.usage.input_tokens')}"
                  f" out={a.get('gen_ai.usage.output_tokens')} (reasoning {a.get('gen_ai.usage.reasoning_tokens')})"
                  f" queue={a.get('groq.queue_time_ms')}ms effort={a.get('llm.reasoning_effort')}")
    u = task.usage
    print(f"\nusage: {u['llm_calls']} calls, {u['input_tokens']} in / {u['output_tokens']} out "
          f"({u['reasoning_tokens']} reasoning) -> est ${u['est_cost_usd']:.5f} at paid-tier prices")


if __name__ == "__main__":
    asyncio.run(main())
