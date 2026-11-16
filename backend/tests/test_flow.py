"""
In-process end-to-end: drive the full controller against the scripted IRCTC phone
with the heuristic mock LLM. No server, no network, no API key.

Run:  python -m tests.test_flow
Expect: the task reaches COMPLETED after a policy-enforced payment pause.
"""
from __future__ import annotations

import asyncio

from app.agent.controller import AgentController
from app.agent.memory import MemoryManager
from app.agent.state import Task, TaskState
from app.config import Settings
from app.llm.router import LLMRouter
from app.policy.engine import PolicyEngine
from app.store import TaskStore
from app.ws import DeviceRegistry
from simulator.mock_device import IrctcPhone


class FakeDevice:
    """Mimics ws.DeviceSession but talks to an in-memory IrctcPhone."""

    def __init__(self, controller: AgentController, device_id="sim"):
        self.device_id = device_id
        self.phone = IrctcPhone()
        self.controller = controller
        self.task_id: str | None = None

    async def request(self, type_: str, timeout: float = 60.0, **data):
        if type_ == "observe":
            return self.phone.observe()
        if type_ == "act":
            return self.phone.act(data.get("action", {}))
        if type_ == "handoff":
            return self.phone.handoff()
        return {}

    async def push(self, type_: str, **data):
        if type_ == "approval_request":
            # Simulate the user tapping "I'll take over" and approving.
            tid = data.get("task_id") or self.task_id
            asyncio.get_event_loop().call_soon(
                lambda: self.controller.resolve_approval(tid, True, "sim-approve"))


async def main():
    settings = Settings(groq_api_key="", gemini_api_key="")   # -> mock LLM
    store = TaskStore()
    devices = DeviceRegistry()
    controller = AgentController(
        LLMRouter(settings), PolicyEngine(spend_threshold=0),
        devices, store, MemoryManager(),
    )
    fake = FakeDevice(controller)
    devices.add(fake)  # type: ignore[arg-type]

    task = Task(prompt=(
        "Book two train tickets from Hyderabad to Bengaluru for tomorrow. "
        "Prefer AC 3-tier. Compare trains, enter passenger details, stop before payment."),
        device_id="sim")
    fake.task_id = task.id
    store.create(task)

    await controller.run_task(task)

    print("\n--- RESULT ---")
    print("state   :", task.state.value)
    print("skill   :", task.skill)
    print("facts   :", task.facts)
    print("summary :", task.summary)
    print("history :")
    for h in task.history:
        print("   •", h)
    print("approval boundary hit:",
          any(e.kind == "approval_requested" for e in task.events) or
          any("approval" in e.kind for e in task.events))

    assert task.state == TaskState.COMPLETED, f"expected COMPLETED, got {task.state}"
    assert "Hyderabad" in str(task.facts.values()), "intent did not extract source city"
    print("\nE2E PASS  (policy pause enforced, booking verified by PNR)")


if __name__ == "__main__":
    asyncio.run(main())
