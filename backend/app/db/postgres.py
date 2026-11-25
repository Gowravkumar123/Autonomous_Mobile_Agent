"""
Postgres persistence (Neon) — durable tasks, task events and an append-only audit trail.

Implements the core of the blueprint's data model (section 8): tasks survive a backend
restart, and every sensitive decision is recorded in `audit_events`. Uses pg8000, a pure-
Python driver, so it needs no native libpq DLL (works on the locked-down dev box AND the
Linux server) and speaks Neon's SSL + SCRAM channel binding.

Design rules:
- The agent must NEVER wait on, or fail because of, the database. Writes are enqueued
  and flushed in the background; every DB error is logged and swallowed. With no
  DATABASE_URL the app runs purely in-memory.
- ONE reused connection on ONE dedicated worker thread (pg8000 is synchronous). A remote
  SSL+SCRAM handshake per write was far too slow; reusing the connection makes each
  write ~1 round-trip, and the single thread never starves the default executor.
- Writes are COALESCED: a task emits dozens of state changes, but only its latest
  snapshot matters, so a flush writes one upsert per task + all pending audit rows in a
  single transaction.
"""
from __future__ import annotations

import asyncio
import json
import logging
import ssl
import urllib.parse as up
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

log = logging.getLogger("agent.db")

FLUSH_DELAY_S = 0.5

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id          text PRIMARY KEY,
        device_id   text,
        prompt      text,
        goal        text,
        skill       text,
        state       text,
        summary     text,
        error       text,
        created_at  timestamptz,
        updated_at  timestamptz,
        data        jsonb NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS tasks_updated_idx ON tasks (updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS tasks_state_idx ON tasks (state)",
    """
    CREATE TABLE IF NOT EXISTS audit_events (
        id       bigserial PRIMARY KEY,
        task_id  text,
        ts       timestamptz NOT NULL DEFAULT now(),
        actor    text,
        action   text,
        detail   text,
        data     jsonb
    )
    """,
    "CREATE INDEX IF NOT EXISTS audit_task_idx ON audit_events (task_id)",
    """
    CREATE TABLE IF NOT EXISTS devices (
        device_id   text PRIMARY KEY,
        jti         text NOT NULL,
        paired_at   timestamptz NOT NULL DEFAULT now(),
        revoked     boolean NOT NULL DEFAULT false
    )
    """,
]

# Multi-row statements: on a high-latency link (dev box -> Neon us-east-1 ~1s/statement)
# one statement per table per flush instead of one per row is the difference between
# seconds and minutes.
_TASK_ROW = "(%s, %s, %s, %s, %s, %s, %s, %s, to_timestamp(%s), to_timestamp(%s), %s::jsonb)"
_UPSERT_HEAD = ("INSERT INTO tasks (id, device_id, prompt, goal, skill, state, summary, "
                "error, created_at, updated_at, data) VALUES ")
_UPSERT_TAIL = """
    ON CONFLICT (id) DO UPDATE SET
        device_id = EXCLUDED.device_id, prompt = EXCLUDED.prompt,
        goal = EXCLUDED.goal, skill = EXCLUDED.skill, state = EXCLUDED.state,
        summary = EXCLUDED.summary, error = EXCLUDED.error,
        updated_at = EXCLUDED.updated_at, data = EXCLUDED.data
"""
_AUDIT_ROW = "(%s, %s, %s, %s, %s::jsonb)"
_AUDIT_HEAD = "INSERT INTO audit_events (task_id, actor, action, detail, data) VALUES "
_CHUNK = 400   # rows per statement (stays far below Postgres' 65535-parameter limit)


def _multi(head: str, row_sql: str, rows: list[tuple], tail: str = ""):
    """Yield (sql, flat_params) for multi-row statements in chunks."""
    for i in range(0, len(rows), _CHUNK):
        chunk = rows[i:i + _CHUNK]
        sql = head + ", ".join([row_sql] * len(chunk)) + tail
        yield sql, tuple(v for row in chunk for v in row)


def _task_params(t: dict) -> tuple:
    return (t.get("id"), t.get("device_id"), t.get("prompt"), t.get("goal"),
            t.get("skill"), t.get("state"), t.get("summary"), t.get("error"),
            float(t.get("created_at") or 0), float(t.get("updated_at") or 0),
            json.dumps(t, default=str))


class Database:
    def __init__(self, url: str):
        u = up.urlparse(url)
        self._kw = dict(
            user=up.unquote(u.username or ""),
            password=up.unquote(u.password or ""),
            host=u.hostname,
            port=u.port or 5432,
            database=(u.path or "/").lstrip("/") or "postgres",
            ssl_context=ssl.create_default_context(),
            timeout=20,
        )
        self.host = u.hostname
        self.ok = False
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pg")
        self._conn = None
        self._dirty: dict[str, dict] = {}      # task_id -> latest snapshot
        self._audit: list[tuple] = []           # pending audit rows
        self._flush_scheduled = False

    # ------------------------------------------------------------------ #
    # worker-thread internals (only ever run on self._worker)
    # ------------------------------------------------------------------ #
    def _run(self, fn: Callable[[Any], Any]) -> Any:
        """Run fn(conn) on the reused connection; reconnect + retry once on failure."""
        for attempt in range(2):
            try:
                if self._conn is None:
                    import pg8000.dbapi as pg
                    self._conn = pg.connect(**self._kw)
                return fn(self._conn)
            except Exception:
                try:
                    if self._conn is not None:
                        self._conn.close()
                except Exception:
                    pass
                self._conn = None
                if attempt == 1:
                    raise

    async def _call(self, fn: Callable[[Any], Any]) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._worker, self._run, fn)

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def init(self) -> bool:
        def work(conn):
            cur = conn.cursor()
            for stmt in SCHEMA:
                cur.execute(stmt)
            conn.commit()
        try:
            await self._call(work)
            self.ok = True
            log.info("postgres ready (%s)", self.host)
        except Exception as e:
            self.ok = False
            log.warning("postgres unavailable — running in-memory only: %s", e)
        return self.ok

    async def close(self) -> None:
        await self.flush()
        def work(_conn):
            if self._conn is not None:
                self._conn.close()
                self._conn = None
        try:
            await self._call(work)
        except Exception:
            pass
        self._worker.shutdown(wait=False)

    # ------------------------------------------------------------------ #
    # writes — enqueue (non-blocking) + coalesced background flush
    # ------------------------------------------------------------------ #
    def enqueue_task(self, task: dict) -> None:
        if not self.ok or not task.get("id"):
            return
        self._dirty[task["id"]] = task
        self._schedule_flush()

    def enqueue_audit(self, task_id: str, actor: str, action: str,
                      detail: str = "", data: dict | None = None) -> None:
        if not self.ok:
            return
        self._audit.append((task_id, actor, action, detail, json.dumps(data or {}, default=str)))
        self._schedule_flush()

    def _schedule_flush(self) -> None:
        if self._flush_scheduled:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._flush_scheduled = True
        loop.call_later(FLUSH_DELAY_S, lambda: loop.create_task(self.flush()))

    async def flush(self) -> None:
        """Write every pending task snapshot + audit row in one transaction."""
        self._flush_scheduled = False
        tasks = list(self._dirty.values())
        self._dirty.clear()
        audits, self._audit = self._audit, []
        if not tasks and not audits:
            return

        def work(conn):
            cur = conn.cursor()
            for sql, params in _multi(_UPSERT_HEAD, _TASK_ROW,
                                      [_task_params(t) for t in tasks], _UPSERT_TAIL):
                cur.execute(sql, params)
            for sql, params in _multi(_AUDIT_HEAD, _AUDIT_ROW, audits):
                cur.execute(sql, params)
            conn.commit()
        try:
            t0 = asyncio.get_running_loop().time()
            await self._call(work)
            log.debug("db flush: %d tasks, %d audit rows in %.2fs", len(tasks), len(audits),
                      asyncio.get_running_loop().time() - t0)
        except Exception as e:
            log.warning("db flush failed (%d tasks, %d audit rows dropped): %s",
                        len(tasks), len(audits), e)

    # Back-compat async helpers
    async def upsert_task(self, task: dict) -> None:
        self.enqueue_task(task)

    async def record_audit(self, task_id: str, actor: str, action: str,
                           detail: str = "", data: dict[str, Any] | None = None) -> None:
        self.enqueue_audit(task_id, actor, action, detail, data)

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #
    async def query(self, sql: str, params: tuple = ()) -> list:
        def work(conn):
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            conn.commit()
            return rows
        return await self._call(work)

    async def execute(self, sql: str, params: tuple = ()) -> None:
        def work(conn):
            conn.cursor().execute(sql, params)
            conn.commit()
        await self._call(work)

    # ---- devices (pairing is rare, so write immediately rather than batching) ----
    async def upsert_device(self, device_id: str, jti: str, revoked: bool = False) -> None:
        await self.execute(
            "INSERT INTO devices (device_id, jti, revoked) VALUES (%s, %s, %s) "
            "ON CONFLICT (device_id) DO UPDATE SET jti = EXCLUDED.jti, "
            "revoked = EXCLUDED.revoked, paired_at = now()",
            (device_id, jti, revoked))

    async def set_device_revoked(self, device_id: str, revoked: bool = True) -> None:
        await self.execute("UPDATE devices SET revoked = %s WHERE device_id = %s",
                           (revoked, device_id))

    async def load_devices(self) -> list[tuple]:
        if not self.ok:
            return []
        return await self.query("SELECT device_id, jti, revoked, paired_at FROM devices")

    async def load_tasks(self, limit: int = 200) -> list[dict]:
        if not self.ok:
            return []
        try:
            rows = await self.query("SELECT data FROM tasks ORDER BY updated_at DESC LIMIT %s",
                                    (limit,))
        except Exception as e:
            log.warning("load tasks failed: %s", e)
            return []
        return [json.loads(d) if isinstance(d, str) else d for (d,) in rows]
