"""
Persistence E2E against the real Postgres in DATABASE_URL (Neon).

Runs the full simulated IRCTC task with the DB attached, then simulates a backend
RESTART (fresh store, hydrate from Postgres) and checks the task and its audit trail
came back. Cleans up its own rows. Skips if DATABASE_URL is not set.

Run:  python -m tests.test_persistence
"""
from __future__ import annotations

import asyncio

from app.agent.controller import AgentController
from app.agent.memory import MemoryManager
from app.agent.state import Task, TaskState
from app.audit.logger import set_audit_db
from app.config import Settings, get_settings
from app.db.postgres import Database
from app.llm.router import LLMRouter
from app.policy.engine import PolicyEngine
from app.store import TaskStore
from app.ws import DeviceRegistry
from tests.test_flow import FakeDevice

FLUSH_WAIT = 1.0


async def main():
    url = get_settings().database_url.strip()
    if not url:
        print("SKIP: DATABASE_URL not set")
        return

    print("connecting...", flush=True)
    db = Database(url)
    assert await db.init(), "could not reach Postgres"
    print("db ready; running task...", flush=True)
    set_audit_db(db)

    # ---- run a full task with persistence on ----
    store = TaskStore(); store.db = db
    devices = DeviceRegistry()
    controller = AgentController(LLMRouter(Settings(groq_api_key="", gemini_api_key="")),
                                 PolicyEngine(spend_threshold=0), devices, store, MemoryManager())
    fake = FakeDevice(controller); devices.add(fake)  # type: ignore[arg-type]
    # Same prompt as test_flow: the heuristic mock LLM is wording-sensitive, and a missed
    # fact makes it ask a clarification that the fake device never answers (hangs).
    task = Task(prompt=("Book two train tickets from Hyderabad to Bengaluru for tomorrow. "
                        "Prefer AC 3-tier. Compare trains, enter passenger details, stop "
                        "before payment."), device_id="sim")   # must match FakeDevice
    fake.task_id = task.id
    store.create(task)
    await asyncio.wait_for(controller.run_task(task), timeout=60)
    await asyncio.sleep(FLUSH_WAIT)
    await db.flush()                      # push anything still queued
    print("ran task      :", task.id, task.state.value)

    try:
        # ---- simulate a server restart: brand-new store hydrated from Postgres ----
        fresh = TaskStore(); fresh.db = db
        n = await fresh.hydrate()
        restored = fresh.get(task.id)
        print("hydrated      :", n, "task(s); ours restored:", bool(restored))
        assert restored is not None, "task did not survive the restart"
        assert restored.state == TaskState.COMPLETED, f"state {restored.state}"
        assert restored.history == task.history, "history mismatch"
        assert len(restored.events) == len(task.events), "events mismatch"

        rows = await db.query("SELECT count(*) FROM audit_events WHERE task_id=%s", (task.id,))
        audits = int(rows[0][0])
        print("audit_events  :", audits, "rows for this task")
        assert audits > 0, "no audit rows written"
        print("\nPERSISTENCE PASS  (task + events + audit trail survive a restart)")
    finally:
        await db.execute("DELETE FROM audit_events WHERE task_id=%s", (task.id,))
        await db.execute("DELETE FROM tasks WHERE id=%s", (task.id,))
        print("cleaned up test rows")
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
