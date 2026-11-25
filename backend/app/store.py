"""
Task store + event broadcasting.

In-memory for this reference build (swap for PostgreSQL in production — the Task model
is already a Pydantic schema). Also fans task events out to any subscribed UI clients
(e.g. a dashboard or the companion app's status stream).
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .agent.state import Task


class TaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._subscribers: set[asyncio.Queue] = set()
        self.db = None  # optional app.db.postgres.Database — durable copy of every task

    # ---- persistence (Postgres) ----
    async def hydrate(self) -> int:
        """On startup, reload recent tasks from Postgres so history survives restarts.
        Tasks that were mid-flight when the server died are marked FAILED (their
        driver coroutine is gone) rather than left looking alive."""
        if not self.db:
            return 0
        from .agent.state import TERMINAL, TaskState
        n = 0
        for data in await self.db.load_tasks():
            try:
                t = Task(**data)
            except Exception:
                continue
            if t.state not in TERMINAL and t.state != TaskState.WAITING_FOR_USER:
                t.state = TaskState.FAILED
                t.error = t.error or "backend restarted while this task was running"
            self._tasks[t.id] = t
            n += 1
        return n

    def schedule_persist(self, task: Task) -> None:
        """Non-blocking durable write (coalesced + batched by the DB layer)."""
        if self.db:
            self.db.enqueue_task(task.model_dump(mode="json"))

    def create(self, task: Task) -> Task:
        self._tasks[task.id] = task
        self.schedule_persist(task)
        return task

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def all(self) -> list[Task]:
        return sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)

    # ---- event stream (for dashboards / live UI) ----
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def publish(self, task_id: str, kind: str, detail: str = "", **data) -> None:
        event = {"task_id": task_id, "kind": kind, "detail": detail, "data": data}
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def events(self) -> AsyncIterator[dict]:
        q = self.subscribe()
        try:
            while True:
                yield await q.get()
        finally:
            self.unsubscribe(q)
