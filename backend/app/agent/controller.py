"""
AgentController — the heart of the brain.

Runs one driver coroutine per task implementing the blueprint loop:
    observe -> decide (LLM) -> policy check -> act | approval pause -> verify
The LLM only proposes actions; the deterministic PolicyEngine decides what may run,
and sensitive steps are handed to the user. State transitions are explicit so a task
can pause and resume across device disconnection.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time

from ..audit.logger import audit
from ..llm.base import LLMError
from ..llm.prompts import ACTION_SYSTEM, build_action_user
from ..llm.router import LLMRouter
from ..mobile.schema import (ActionType, AgentAction, ApprovalRequest,
                             ScreenObservation)
from ..policy.engine import PolicyEngine
from ..skills import registry
from ..store import TaskStore
from ..ws import DeviceRegistry, DeviceSession
from .approvals import (APPROVED, EXPIRED, INVALIDATED, PENDING, REJECTED, ApprovalRecord,
                        action_binding)
from .intent import analyze_intent, make_plan
from .memory import MemoryManager
from .state import TERMINAL, Task, TaskState
from .verifier import is_success

log = logging.getLogger("agent.controller")

MAX_STEPS = 40


class AgentController:
    def __init__(self, llm: LLMRouter, policy: PolicyEngine, devices: DeviceRegistry,
                 store: TaskStore, memory: MemoryManager):
        self.llm = llm
        self.policy = policy
        self.devices = devices
        self.store = store
        self.memory = memory
        self._approvals: dict[str, asyncio.Future] = {}
        self._answers: dict[str, asyncio.Future] = {}
        # Last prompt (ask_user / approval_request) pushed per waiting task, so we can
        # re-deliver it when the device reconnects after a flap. (type, data) keyed by task.
        self._pending_prompts: dict[str, tuple[str, dict]] = {}
        # Transaction-bound approval records (approval_id -> record), and pause gates.
        self._approval_records: dict[str, ApprovalRecord] = {}
        self._resume: dict[str, asyncio.Event] = {}
        self.approval_ttl_s = 900.0
        self.tools = None   # app.tools.ToolRegistry (built-in + MCP), set by the app
        from ..kv import MemoryKV
        self.kv = MemoryKV()   # shared short-lived state (Redis when configured)
        from ..cluster import LocalCluster
        self.cluster = LocalCluster()   # multi-worker routing (RedisCluster when configured)
        self._graph = None  # compiled lazily (imports langgraph on first run)

    async def on_device_connected(self, session: DeviceSession) -> None:
        """A device (re)connected: re-push any prompt a waiting task is blocked on.

        Pushes are fire-and-forget; if the socket was mid-flap when we asked, the phone
        never saw the question and the task would hang. Re-delivering on reconnect makes
        that self-heal (OEM battery kills + mobile-network jitter make flaps routine).
        """
        for task_id, (ptype, data) in list(self._pending_prompts.items()):
            task = self.store.get(task_id)
            if not task or task.state is not TaskState.WAITING_FOR_USER:
                continue
            # Deliver to the executing device, or to the owner's phone (browser tasks).
            if task.device_id and session.device_id not in (task.device_id, task.owner):
                continue
            try:
                await session.push(ptype, **data)
                self.store.publish(task_id, "note", "re-sent pending prompt after reconnect")
            except Exception as e:
                log.warning("re-deliver to %s failed: %s", session.device_id, e)

    @property
    def graph(self):
        """The compiled LangGraph ReAct state machine (built once, reused)."""
        if self._graph is None:
            from .graph import build_react_graph
            self._graph = build_react_graph(self)
        return self._graph

    # ----------------------------------------------------------------- #
    # Public: resolve pauses (called by REST endpoints or inbound ws).
    # ----------------------------------------------------------------- #
    def resolve_approval(self, task_id: str, approved: bool, note: str = "",
                         approver: str = "device") -> bool:
        """Decide the task's CURRENT pending approval (phone UI / legacy endpoint)."""
        task = self.store.get(task_id)
        rec = self._approval_records.get(task.pending_approval_id) if task else None
        if rec is not None:
            return self.decide_approval(rec.approval_id, approved, note, approver)[0]
        fut = self._approvals.get(task_id)            # e.g. cancel() racing record creation
        if fut and not fut.done():
            fut.set_result((approved, note))
            return True
        return False

    def decide_approval(self, approval_id: str, approved: bool, note: str = "",
                        approver: str = "") -> tuple[bool, str]:
        """Decide one exact approval by id (blueprint: POST /v1/approvals/{id}/decision)."""
        rec = self._approval_records.get(approval_id)
        if rec is None:
            return False, "unknown approval"
        if rec.decision != PENDING:
            return False, f"already {rec.decision}"
        if rec.expired:
            rec.decide(EXPIRED, "system", "expired before a decision")
            self._persist_approval(rec)
            return False, "approval expired"
        rec.decide(APPROVED if approved else REJECTED, approver, note)
        self._persist_approval(rec)
        fut = self._approvals.get(rec.task_id)
        if fut and not fut.done():
            fut.set_result((approved, note))
        return True, rec.decision

    def get_approval(self, approval_id: str) -> ApprovalRecord | None:
        return self._approval_records.get(approval_id)

    def approvals_for(self, task_id: str) -> list[ApprovalRecord]:
        return [r for r in self._approval_records.values() if r.task_id == task_id]

    def _persist_approval(self, rec: ApprovalRecord) -> None:
        if self.store.db:
            try:
                asyncio.get_running_loop().create_task(self.store.db.upsert_approval(rec.view()))
            except RuntimeError:
                pass

    def resolve_answer(self, task_id: str, facts: dict) -> bool:
        fut = self._answers.get(task_id)
        if fut and not fut.done():
            fut.set_result(facts)
            return True
        return False

    def pause(self, task_id: str) -> bool:
        """Pause at the next ReAct cycle boundary (the current action is allowed to finish)."""
        task = self.store.get(task_id)
        if not task or task.state in TERMINAL:
            return False
        task.pause_requested = True
        self._resume.setdefault(task_id, asyncio.Event()).clear()
        self.store.publish(task_id, "note", "pause requested")
        return True

    def resume(self, task_id: str) -> bool:
        task = self.store.get(task_id)
        if not task or task.state in TERMINAL:
            return False
        task.pause_requested = False
        self._resume.setdefault(task_id, asyncio.Event()).set()
        return True

    async def wait_if_paused(self, task: Task) -> None:
        """Called by the graph at each cycle boundary; blocks while the task is paused."""
        if not task.pause_requested:
            return
        prev = task.state
        task.set_state(TaskState.PAUSED, "paused by user")
        self._emit(task)
        ev = self._resume.setdefault(task.id, asyncio.Event())
        while task.pause_requested and not task.cancel_requested:
            try:
                await asyncio.wait_for(ev.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
        if not task.cancel_requested:
            task.set_state(TaskState.RUNNING if prev == TaskState.PAUSED else prev,
                           "resumed")
            self._emit(task)

    def cancel(self, task_id: str) -> None:
        task = self.store.get(task_id)
        if task:
            task.cancel_requested = True
            # Unblock any pending pauses so the loop can observe the cancel.
            self.resolve_approval(task_id, False, "cancelled", approver="system")
            self.resolve_answer(task_id, {})
            if task_id in self._resume:
                self._resume[task_id].set()

    # ----------------------------------------------------------------- #
    # Main driver
    # ----------------------------------------------------------------- #
    async def run_task(self, task: Task) -> None:
        """Root span for the whole task + per-task token/cost accounting."""
        from .. import observability as obs
        task.usage = obs.begin_task_usage()          # filled live by every LLM call
        with obs.span("task", **{"task.id": task.id, "task.owner": task.owner or "",
                                 "task.prompt_chars": len(task.prompt or "")}) as sp:
            self.cluster.claim_task(task.id)          # peers route control calls here
            try:
                await self._run_task_inner(task)
            finally:
                self.cluster.release_task(task.id)
                u = task.usage or {}
                obs.set_attrs(sp, **{
                    "task.state": task.state.value, "task.executor": task.executor,
                    "task.skill": task.skill, "task.steps": len(task.history),
                    "task.llm_calls": u.get("llm_calls", 0),
                    "task.input_tokens": u.get("input_tokens", 0),
                    "task.output_tokens": u.get("output_tokens", 0),
                    "task.reasoning_tokens": u.get("reasoning_tokens", 0),
                    "task.est_cost_usd": u.get("est_cost_usd", 0.0)})
                if task.state == TaskState.FAILED:
                    obs.mark_error(sp, task.error or "task failed")
                self._emit(task)                         # persist the final usage

    async def _run_task_inner(self, task: Task) -> None:
        session = None
        try:
            await self._plan_phase(task)
            if task.state in TERMINAL:
                return
            session = await self._select_executor(task)
            if task.state in TERMINAL:
                return
            await self._execute_phase(task)
            if task.state == TaskState.COMPLETED:
                asyncio.create_task(self._learn(task))    # off the critical path
        except Exception as e:  # defensive: never leave a task hung
            log.exception("task %s crashed", task.id)
            import traceback
            task.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}"
            task.set_state(TaskState.FAILED, f"internal error: {type(e).__name__}: {e}")
            self._emit(task)
        finally:
            self._approvals.pop(task.id, None)
            self._answers.pop(task.id, None)
            if session is not None:
                if getattr(session, "kind", "") == "browser":
                    await self._close_browser(task, session)
                else:
                    self.devices.remove(session.device_id, session)
            if self.tools is not None:
                self.tools.forget_task(task.id)

    # ----------------------------------------------------------------- #
    # select_executor (blueprint): mobile (paired phone) or browser (Playwright)
    # ----------------------------------------------------------------- #
    # A URL or a bare domain (not an e-mail address), or an explicit "on the website".
    _WEB_RE = re.compile(
        r"https?://|(?<![@\w.])(?:www\.)?[a-z0-9-]+\.(?:com|in|org|net|io|co|gov|edu|app|dev)"
        r"(?:/|\b)(?!@)|\b(?:website|web ?site|on the web|in the browser|in a browser)\b",
        re.IGNORECASE)

    # "on my laptop / in Notepad / in Excel ..." -> the desktop companion, if one is online.
    _DESKTOP_RE = re.compile(
        r"\b(?:on|in|from) (?:my |the )?(?:desktop|computer|laptop|pc|windows)\b|"
        r"\b(?:notepad|excel|word document|calculator|file explorer|outlook desktop)\b",
        re.IGNORECASE)

    def _sessions_of(self, kind: str) -> list:
        return [d for d in self.devices.all() if getattr(d, "kind", "") == kind]

    def _phone_connected(self) -> bool:
        return bool(self._sessions_of("phone"))

    def _choose_executor(self, task: Task) -> str:
        if task.executor in ("mobile", "browser", "api", "desktop"):
            return task.executor
        if task.device_id.startswith("desktop-"):
            return "desktop"
        if self._DESKTOP_RE.search(task.prompt) and (
                self._sessions_of("desktop") or task.owner.startswith("desktop-")):
            return "desktop"
        if task.start_url or self._WEB_RE.search(task.prompt):
            return "browser"
        # Intent judged the goal doable with the registered tools alone (no UI).
        if task.executor_hint == "api" and self.tools is not None and self.tools.names():
            return "api"
        if task.device_id and not task.device_id.startswith(("browser-", "api-")):
            return "mobile"
        return "mobile" if self._phone_connected() else "browser"

    async def _select_executor(self, task: Task):
        """Pick the runtime and, for the browser, start an isolated session for this task."""
        from ..config import get_settings
        s = get_settings()
        task.executor = self._choose_executor(task)
        if task.executor == "api":
            if self.tools is None or not self.tools.names():
                task.set_state(TaskState.FAILED, "API executor chosen but no tools are available")
                return None
            from ..executors.api import ToolSession
            session = ToolSession(f"api-{task.id}",
                                  notify=lambda t, d: self._notify_owner(task, t, d))
            task.device_id = session.device_id
            self.devices.add(session)
            audit(task.id, "executor", "api", tools=self.tools.names())
            task.log("note", f"executor: api ({len(self.tools.names())} tools)")
            self._emit(task)
            return session
        if task.executor == "desktop":
            # Bind to the desktop companion: the named one, the owner's own, or the only
            # one online. Never guess between several.
            if not task.device_id.startswith("desktop-"):
                desks = self._sessions_of("desktop")
                if task.owner.startswith("desktop-"):
                    task.device_id = task.owner
                elif len(desks) == 1:
                    task.device_id = desks[0].device_id
                else:
                    task.set_state(TaskState.FAILED,
                                   "no desktop companion is connected" if not desks else
                                   "several desktops are connected — pick one with device_id")
                    return None
            audit(task.id, "executor", "desktop", device=task.device_id)
            task.log("note", f"executor: desktop ({task.device_id})")
            self._emit(task)
            return None
        if task.executor != "browser":
            audit(task.id, "executor", "mobile", device=task.device_id)
            task.log("note", f"executor: mobile ({task.device_id or 'any phone'})")
            return None

        if not s.browser_enabled:
            task.set_state(TaskState.FAILED, "browser executor is disabled (BROWSER_ENABLED)")
            return None
        # Concurrent-browser cap, counted in the shared KV so it holds across workers (the
        # TTL self-heals the count if a worker dies mid-task).
        if await self.kv.incr("browser:active", 3600) > s.browser_max_sessions:
            await self.kv.decr("browser:active")
            task.set_state(TaskState.FAILED, "too many concurrent browser tasks — try again soon")
            return None
        try:
            from ..executors.browser import BrowserSession
        except ImportError:
            await self.kv.decr("browser:active")
            task.set_state(TaskState.FAILED, "browser executor unavailable (pip install playwright)")
            return None

        allowed = tuple(d for d in s.browser_allowed_domains.split(",") if d.strip())
        session = BrowserSession(
            f"browser-{task.id}", allowed_domains=allowed, search_url=s.browser_search_url,
            headless=s.browser_headless,
            notify=lambda t, d: self._notify_owner(task, t, d))
        try:
            await session.start(task.start_url or "about:blank")
        except Exception as e:
            await session.close()
            await self.kv.decr("browser:active")
            task.set_state(TaskState.FAILED, f"could not start the browser: {e} "
                                             "(run `playwright install --with-deps chromium`)")
            return None
        task.device_id = session.device_id
        self.devices.add(session)                         # same registry as phones
        audit(task.id, "executor", "browser", start_url=task.start_url,
              allowed_domains=list(allowed))
        task.log("note", "executor: browser (isolated Chromium context)")
        self._emit(task)
        return session

    # ----------------------------------------------------------------- #
    # Tool calls (blueprint: unified tool execution layer behind the policy gate)
    # ----------------------------------------------------------------- #
    async def run_tool(self, task: Task, action: AgentAction, screen: ScreenObservation,
                       state: dict | None = None) -> dict:
        """Every tool call goes through here: lookup -> policy (deny/approve/allow) ->
        schema validation -> transaction-bound approval -> dispatch -> normalized result
        fed back to the model as the next observation. Never trusts the model's args."""
        from ..policy.engine import PolicyDecision
        from ..tools.base import APPROVE, DENY, ToolError
        name, args = (action.tool or "").strip(), dict(action.args or {})
        spec = self.tools.get(name) if self.tools is not None else None

        def record(ok: bool, text: str, *, cached: bool = False, error: str = "",
                   elapsed: float = 0.0) -> dict:
            task.tool_results.append({"tool": name, "args": args, "ok": ok,
                                      "text": text[:1500], "cached": cached})
            del task.tool_results[:-12]                     # keep the last 12
            short = json.dumps(args, default=str)
            short = short if len(short) <= 80 else short[:77] + "..."
            label = f"call {name}({short}) -> {'ok' if ok else 'error'}" + \
                    (" (cached)" if cached else "")
            task.history.append(label)
            self.store.publish(task.id, "tool", label, ok=ok)
            audit(task.id, "tool_call", name, args=args, ok=ok, error=error[:300],
                  cached=cached, elapsed_s=round(elapsed, 3))
            if self.store.db is not None:
                self.store.db.enqueue_tool_run(task.id, task.step_id, name,
                                               self.tools.args_hash(name, args) if self.tools else "",
                                               ok, (text if ok else error)[:2000], elapsed)
            return {"status": "run", "pending_validate": False, "feedback": ""}

        if spec is None:
            return record(False, f"unknown tool '{name}' — use one from AVAILABLE TOOLS",
                          error="unknown tool")
        if spec.policy == DENY:
            return record(False, f"{name} is DENIED by policy — do this another way or stop",
                          error="denied by policy")
        try:
            self.tools.validate(spec, args)
        except ToolError as e:
            return record(False, str(e), error=str(e))

        if spec.policy == APPROVE and not (task.pre_authorized and spec.risk <= 3):
            why = ("financial tool" if spec.risk >= 4 else
                   "tool that changes data" if not spec.read_only else "tool from an untrusted server")
            decision = PolicyDecision(False, action, True, spec.risk,
                                      f"{why}: {name}({json.dumps(args, default=str)[:160]})")
            device = (self.devices.get(task.device_id) if task.device_id else None)
            outcome, _ = await self._approval_pause(task, device, decision, action, screen)
            if task.cancel_requested or outcome != "approved":
                task.set_state(TaskState.CANCELLED,
                               "approval expired — tool not called" if outcome == "expired"
                               else "user declined the tool call")
                return {"status": "cancelled"}
            task.approvals_granted += 1
            task.set_state(TaskState.RUNNING)

        from .. import observability as obs
        with obs.span("tool.call", **{"tool.name": name, "tool.source": spec.source,
                                      "tool.policy": spec.policy, "tool.risk": spec.risk}) as sp:
            res = await self.tools.call(spec, args, task.id)
            obs.set_attrs(sp, **{"tool.ok": res.ok, "tool.cached": res.cached})
            if not res.ok:
                obs.mark_error(sp, res.error or "tool error")
        return record(res.ok, res.for_model(), cached=res.cached, error=res.error,
                      elapsed=res.elapsed_s)

    async def _notify_owner(self, task: Task, type_: str, data: dict) -> None:
        """Browser tasks have no screen of their own: surface their questions/approvals on
        the owner's phone if it's connected (else the user answers via the API)."""
        if task.owner and task.owner != "admin":
            phone = self.devices.get(task.owner)
            if phone is not None and getattr(phone, "kind", "") in ("phone", "desktop"):
                await phone.push(type_, **data)

    async def _close_browser(self, task: Task, session) -> None:
        try:
            png = await session.evidence()
            if png:
                from pathlib import Path
                folder = Path(__file__).resolve().parents[2] / "evidence"
                folder.mkdir(exist_ok=True)
                path = folder / f"{task.id}.png"
                path.write_bytes(png)
                task.evidence.append(path.name)
                task.log("evidence", "final screenshot saved", file=path.name)
                self._emit(task)
        except Exception as e:
            log.warning("evidence capture failed: %s", e)
        finally:
            self.devices.remove(session.device_id, session)
            await session.close()
            await self.kv.decr("browser:active")        # release the concurrency slot

    # ----------------------------------------------------------------- #
    # Long-term memory
    # ----------------------------------------------------------------- #
    @staticmethod
    def _owner(task: Task) -> str:
        return task.owner or task.device_id or "admin"

    async def _recall(self, task: Task) -> None:
        """Approved memories relevant to this request -> task.memory_context (data only)."""
        if self.memory is None or not hasattr(self.memory, "relevant"):
            return
        from .. import observability as obs
        with obs.span("memory.recall") as sp:
            try:
                items = await self.memory.relevant(self._owner(task), task.prompt)
                task.memory_context = [m.content for m in items]
                obs.set_attrs(sp, **{"memory.items": len(items)})
                if items:
                    task.log("note", f"memory: {len(items)} approved item(s) in context")
            except Exception as e:                              # never block a task on memory
                obs.mark_error(sp, str(e))
                log.warning("memory recall failed: %s", e)

    async def _apply_memory_requests(self, task: Task) -> list[str]:
        """Handle explicit 'remember ...' / 'forget ...' from this request."""
        from .memory import MemoryError_
        notes: list[str] = []
        if self.memory is None or not hasattr(self.memory, "add"):
            return notes
        owner = self._owner(task)
        for fact in task.memory_remember:
            try:
                item = await self.memory.add(owner, fact, kind="preference",
                                             source=f"task:{task.id}", approved=True)
                notes.append(f"Remembered: {item.content}")
                audit(task.id, "memory_added", item.id, content=item.content)
            except MemoryError_ as e:
                notes.append(f"Not stored ({e}): {fact[:80]}")
        for desc in task.memory_forget:
            deleted, cands = await self.memory.forget_matching(owner, desc)
            if deleted:
                notes.append(f"Forgot: {deleted[0].content}")
                audit(task.id, "memory_deleted", deleted[0].id, content=deleted[0].content)
            elif cands:
                notes.append("Not sure which memory to forget — candidates: " +
                             "; ".join(f"[{m.id}] {m.content}" for m in cands) +
                             " (delete one with DELETE /v1/memory/{id})")
            else:
                notes.append(f"Nothing in memory matches '{desc}'")
        for n in notes:
            task.log("note", n)
        return notes

    async def _learn(self, task: Task) -> None:
        """After a COMPLETED task, propose durable preferences — stored PENDING until the
        user approves them (consent: nothing learned is used before approval)."""
        from ..config import get_settings
        from ..llm.prompts import LEARN_SYSTEM, build_learn_user
        if self.memory is None or not hasattr(self.memory, "add") or \
                not get_settings().memory_learn or task.executor == "" or task.memory_only:
            return
        try:
            data = await self.llm.complete_json_fast(
                LEARN_SYSTEM, build_learn_user(task.prompt, task.facts, task.summary,
                                               task.memory_context),
                temperature=0.1, max_tokens=600)
            proposals = [str(x).strip() for x in (data.get("memories") or [])
                         if str(x).strip()][:3]
        except Exception as e:
            log.debug("memory learn skipped: %s", e)
            return
        added = []
        for p in proposals:
            try:
                item = await self.memory.add(self._owner(task), p, kind="learned",
                                             source=f"task:{task.id}", approved=False)
                if item.status == "pending":
                    added.append(item)
            except Exception:
                continue
        if added:
            task.log("memory_proposed", f"{len(added)} preference(s) await your approval",
                     items=[{"id": m.id, "content": m.content} for m in added])
            self.store.publish(task.id, "memory_proposed", f"{len(added)} to approve",
                               items=[m.view() for m in added])
            self._emit(task)

    async def _plan_phase(self, task: Task) -> None:
        from .. import observability as obs
        with obs.span("plan") as sp:
            await self._plan_phase_inner(task)
            obs.set_attrs(sp, **{"task.skill": task.skill,
                                 "task.executor_hint": task.executor_hint or "ui",
                                 "task.plan_steps": len(task.plan),
                                 "task.clarifications": len(task.clarifications),
                                 "task.memory_only": task.memory_only})

    async def _plan_phase_inner(self, task: Task) -> None:
        task.set_state(TaskState.PLANNING)
        self._emit(task)
        await self._recall(task)
        await analyze_intent(task, self.llm,
                             self.tools.names() if self.tools is not None else None,
                             task.memory_context)
        notes = await self._apply_memory_requests(task)
        if task.memory_only:
            # "Remember that ..." / "forget ..." alone: done — no app, browser or tools.
            task.summary = " | ".join(notes) or "Nothing to remember or forget."
            task.set_state(TaskState.COMPLETED, "memory updated")
            self._emit(task)
            return
        if notes:
            await self._recall(task)          # include what was just remembered
        await make_plan(task, self.llm)
        audit(task.id, "plan", task.goal, skill=task.skill, plan=task.plan)
        self._emit(task)

        if task.clarifications:
            facts = await self._ask_user(task, task.clarifications, task.clarification_choices)
            if task.cancel_requested:
                task.set_state(TaskState.CANCELLED, "cancelled during clarification")
                return
            task.facts.update(facts or {})
            task.clarifications = []

    async def _execute_phase(self, task: Task) -> None:
        """Run the task through the LangGraph ReAct state machine.

        All per-step logic (observe -> validate/reflect -> reason -> act) lives in
        app/agent/graph.py; this just seeds the shared AgentState and drives it. The
        nodes mutate `task` in place (state, history, events), so the outcome is read
        straight off `task` once the graph reaches END.
        """
        from .graph import MAX_STEPS as GRAPH_MAX_STEPS

        skill = registry.get(task.skill)
        task.set_state(TaskState.RUNNING)
        self._emit(task)

        initial_state = {
            "task": task, "skill": skill, "screen": None, "action": None,
            "feedback": "", "react_attempts": 0, "blind": 0, "disconnects": 0,
            "step": 0, "last_sig": "", "pending_validate": False, "opens": 0, "progress_mark": -1, "stall": 0,
            "status": "run",
        }
        # recursion_limit counts node executions (~4 per ReAct cycle); the in-node
        # step cap (GRAPH_MAX_STEPS) is the real limiter — keep headroom above it.
        await self.graph.ainvoke(initial_state,
                                 config={"recursion_limit": GRAPH_MAX_STEPS * 6 + 20})
        self._emit(task)

    # ----------------------------------------------------------------- #
    # Steps
    # ----------------------------------------------------------------- #
    async def _await_device(self, task: Task, timeout: float = 30.0) -> DeviceSession | None:
        """Return a connected device, waiting briefly if it is mid-reconnect."""
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while True:
            # A task bound to a device runs ONLY on that device (never fall back to some
            # other connected phone); unbound tasks may use any connected PHONE.
            dev = (self.devices.get(task.device_id) if task.device_id
                   else next(iter(self._sessions_of("phone")), None))
            if dev is not None:
                return dev
            if task.cancel_requested or loop.time() >= deadline:
                return None
            await asyncio.sleep(1.0)

    async def _observe(self, device: DeviceSession) -> ScreenObservation:
        from .. import observability as obs
        # Device latency is where this deployment hurts most (phone <-> cloud link), so
        # every observe is timed per executor kind.
        with obs.span("device.observe", **{"device.kind": getattr(device, "kind", "phone")}) as sp:
            payload = await device.request("observe")
            screen = ScreenObservation(**(payload.get("screen", payload)))
            obs.set_attrs(sp, **{"device.elements": len(screen.elements),
                                 "device.screenshot": bool(screen.screenshot_b64),
                                 "device.app": screen.app_package})
            return screen

    async def _decide(self, task: Task, skill, screen: ScreenObservation,
                      feedback: str = "") -> AgentAction:
        plan_summary = "; ".join(task.plan) + f"  | BOUNDARY: {task.approval_boundary}"
        # When the accessibility tree is sparse/empty but we have a screenshot, OCR the
        # image so the agent can READ names/prices (and tap them) via tap_xy.
        ocr: list[dict] = []
        if screen.screenshot_b64 and len(screen.elements) <= 10:
            from ..vision.ocr import available, ocr_items
            ocr = await asyncio.get_event_loop().run_in_executor(
                None, ocr_items, screen.screenshot_b64)
            screen.ocr_items = ocr          # the policy hit-tests tap_xy against these
            task.log("note", f"OCR: {'on' if available() else 'OFF'}, {len(ocr)} items, "
                             f"shot={'yes' if screen.screenshot_b64 else 'no'}, "
                             f"els={len(screen.elements)}")
        elif len(screen.elements) <= 6 and task.executor != "api":
            task.log("note", f"no screenshot (vision/OCR unavailable), els={len(screen.elements)}")
        from ..config import get_settings
        s = get_settings()
        # Tool catalog costs prompt tokens every step: always for the API executor, and for
        # UI tasks only when enabled (TOOLS_IN_UI_TASKS) — keeps phone/browser prompts lean.
        catalog = ""
        if self.tools is not None and (task.executor == "api" or s.tools_in_ui_tasks):
            catalog = self.tools.catalog()
        user = build_action_user(task.goal, task.facts, plan_summary, task.history, screen,
                                 max_elements=s.max_elements_in_prompt, ocr_items=ocr,
                                 feedback=feedback, executor=task.executor,
                                 tools_catalog=catalog, tool_results=task.tool_results[-5:],
                                 memories=task.memory_context)
        last_err: Exception | None = None
        # Populated screens: fast gpt-oss text call from the tree (+ grouping). Vision
        # (qwen + a base64 image = many input tokens) is COST-GATED: only fire it on a
        # sparse tree, or when the text path is stuck on a not-too-rich screen — on dense
        # trees the text path is both better and cheaper, so paying for an image is waste.
        n = len(screen.elements)
        use_vision = bool(screen.screenshot_b64) and (
            n <= s.vision_sparse_max or (bool(feedback) and n <= s.vision_stuck_max))
        attempts = [screen.screenshot_b64, None] if use_vision else [None, None]
        for image in attempts:
            try:
                data = await self.llm.complete_json(
                    ACTION_SYSTEM, user, temperature=0.1,
                    max_tokens=s.action_max_tokens, image_b64=image)
                return AgentAction(**data)
            except (LLMError, ValueError, TypeError) as e:
                last_err = e
                log.warning("decide (%s) failed: %s", "vision" if image else "text", e)
        return AgentAction(type=ActionType.ASK_USER,
                           text="I couldn't interpret the screen. What should I do next?",
                           thought=str(last_err))

    async def _act(self, device: DeviceSession, action: AgentAction, task: Task) -> None:
        from .. import observability as obs
        payload = action.model_dump(mode="json")
        err = ""
        with obs.span("device.act", **{"device.kind": getattr(device, "kind", "phone"),
                                       "device.action": action.type.value}) as sp:
            for attempt in range(2):
                try:
                    res = await device.request("act", action=payload)
                    if (isinstance(res, dict) and res.get("ok") is False
                            and action.type == ActionType.SCROLL and not res.get("halt")
                            and getattr(device, "kind", "phone") == "phone"):
                        # Phone scroll needs a scrollable node; custom-drawn screens (e.g.
                        # BookMyShow's seat map) have none, so it failed in a loop live.
                        # A swipe gesture works there: finger UP scrolls content DOWN.
                        swipe = {"type": "swipe",
                                 "direction": "down" if action.direction == "up" else "up"}
                        res2 = await device.request("act", action=swipe)
                        if isinstance(res2, dict) and res2.get("ok") is not False:
                            task.log("note", "scroll had no scrollable view; used a swipe")
                            res = res2
                    if isinstance(res, dict) and res.get("ok") is False:
                        obs.set_attrs(sp, **{"device.act_error": str(res.get("error", ""))[:200]})
                        if res.get("halt"):
                            # The device's local kill switch: stop the task, no retries.
                            task.cancel_requested = True
                            task.log("note", f"stopped on the device: {res.get('error', '')}")
                            audit(task.id, "kill_switch", str(res.get("error", ""))[:200])
                        else:
                            # Tell the model why (e.g. allowlist / workspace refusal).
                            err = str(res.get("error") or "failed")[:120]
                            task.log("note", f"action refused/failed: {err}")
                    break
                except (RuntimeError, OSError, asyncio.TimeoutError):
                    obs.set_attrs(sp, **{"device.retried": True})
                    if attempt == 1:
                        raise
                    fresh = await self._await_device(task)   # reconnected session, if any
                    if fresh is None:
                        raise
                    device = fresh
        label = self._action_label(action)
        if err:
            label += f" -> REFUSED: {err}"
        task.history.append(label)
        self.store.publish(task.id, "action", label, risk=action.risk_level)
        audit(task.id, "action", label, type=action.type.value, ref=action.ref)

    async def _approval_pause(self, task: Task, device: DeviceSession, decision,
                              action: AgentAction, screen: ScreenObservation) -> tuple[str, str]:
        """Create a transaction-bound approval and wait for its decision.

        Returns (decision, action_hash) where decision is approved | rejected | expired |
        cancelled. The hash binds the approval to this exact action + target + amount; the
        caller re-checks it against a fresh screen before executing.
        """
        payload = self._approval_payload(task, decision, action, screen)
        rec = ApprovalRecord(
            task_id=task.id, step_id=task.step_id, action_hash=action_binding(action, screen),
            expires_at=time.time() + self.approval_ttl_s, title=payload.get("title", ""),
            reason=decision.reason, amount=payload.get("amount"),
            action_label=self._action_label(action), risk_level=decision.risk_level)
        self._approval_records[rec.approval_id] = rec
        self.cluster.note_approval(rec.approval_id, task.id)
        payload.update(approval_id=rec.approval_id, expires_in=int(self.approval_ttl_s))
        task.pending_action = action
        task.pending_approval_reason = decision.reason
        task.pending_approval_id = rec.approval_id
        # Future + record exist BEFORE we push, so even an instant answer can't race them.
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._approvals[task.id] = fut
        self._persist_approval(rec)

        task.log("approval_requested", decision.reason, risk=decision.risk_level, **payload)
        task.set_state(TaskState.WAITING_FOR_USER, decision.reason)
        self._emit(task)
        self._pending_prompts[task.id] = ("approval_request", payload)
        if device is not None:            # no device: decided via the API / on reconnect
            try:
                await device.push("approval_request", **payload)
            except Exception as e:
                log.warning("approval push failed (will re-deliver on reconnect): %s", e)
        self.store.publish(task.id, "approval", decision.reason, approval=payload)
        audit(task.id, "approval_requested", decision.reason, approval=payload)

        from .. import observability as obs
        # Human wait time is its own span: it dominates wall-clock for sensitive tasks and
        # must not be mistaken for agent latency.
        with obs.span("approval.wait", **{"approval.id": rec.approval_id,
                                          "approval.risk": decision.risk_level}) as sp:
            try:
                approved, note = await asyncio.wait_for(fut, timeout=self.approval_ttl_s)
                outcome = rec.decision if rec.decision != PENDING else (
                    APPROVED if approved else REJECTED)
                if task.cancel_requested:
                    outcome = "cancelled"
            except asyncio.TimeoutError:
                rec.decide(EXPIRED, "system", "no decision before expiry")
                self._persist_approval(rec)
                outcome, note = EXPIRED, "expired"
            finally:
                self._approvals.pop(task.id, None)
                self._pending_prompts.pop(task.id, None)
                task.pending_approval_id = ""
            obs.set_attrs(sp, **{"approval.outcome": outcome})
        audit(task.id, "approval_decided", outcome, note=note, approval_id=rec.approval_id)
        self.store.publish(task.id, "approval_decided", outcome, approval_id=rec.approval_id)
        return outcome, rec.action_hash

    def invalidate_approval(self, task: Task, action_hash: str, why: str) -> None:
        for rec in self.approvals_for(task.id):
            if rec.action_hash == action_hash and rec.decision == APPROVED:
                rec.decide(INVALIDATED, "system", why)
                self._persist_approval(rec)
        audit(task.id, "approval_invalidated", why)
        task.log("note", f"approval invalidated: {why}")
        self.store.publish(task.id, "note", f"approval invalidated — {why}")

    @staticmethod
    def _with_choices(question: str, options: list[str]) -> str:
        """Bake the choices into the question so Zoro reads them aloud verbatim."""
        opts = [o for o in (options or []) if o]
        if not opts:
            return question
        tail = opts[0] if len(opts) == 1 else ", ".join(opts[:-1]) + ", or " + opts[-1]
        return f"{question} Choose one: {tail}."

    async def _ask_user(self, task: Task, questions: list[str],
                        choices: list[list[str]] | None = None) -> dict:
        choices = choices or []
        # Compose each question WITH its options (spoken + displayed), and pass the raw
        # choices through so the phone can render tappable buttons as well.
        composed = [self._with_choices(q, choices[i] if i < len(choices) else [])
                    for i, q in enumerate(questions)]
        task.set_state(TaskState.WAITING_FOR_USER, "needs clarification")
        self.store.publish(task.id, "ask_user", " | ".join(composed),
                           questions=composed, choices=choices)
        audit(task.id, "ask_user", " | ".join(composed))
        # Surface the question(s) to the paired device/companion UI so the user can answer.
        prompt = {"task_id": task.id, "questions": composed, "choices": choices}
        self._pending_prompts[task.id] = ("ask_user", prompt)
        device = (self.devices.get(task.device_id) if task.device_id
                  else self.devices.any())     # never ask a different phone's user
        if device:
            try:
                await device.push("ask_user", **prompt)
            except Exception as e:
                log.warning("ask_user push failed (will re-deliver on reconnect): %s", e)
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._answers[task.id] = fut
        try:
            facts = await fut
        finally:
            self._answers.pop(task.id, None)
            self._pending_prompts.pop(task.id, None)
        return facts or {}

    # ----------------------------------------------------------------- #
    # Helpers
    # ----------------------------------------------------------------- #
    def _approval_payload(self, task, decision, action: AgentAction, screen) -> dict:
        from ..policy.engine import _detect_amount, _screen_text
        amount = _detect_amount(_screen_text(screen))
        title = ("Payment / authentication — you take over"
                 if decision.risk_level >= 4 else "Approval needed before this step")
        return ApprovalRequest(
            task_id=task.id, step_id=task.step_id, title=title,
            summary=f"{task.goal}", amount=amount, reason=decision.reason,
            risk_level=decision.risk_level,
        ).model_dump(mode="json")

    def _action_label(self, action: AgentAction) -> str:
        if action.type == ActionType.TYPE:
            return f"type '{action.text}' into #{action.ref}"
        if action.type == ActionType.TAP:
            return f"tap #{action.ref}"
        if action.type == ActionType.TAP_XY:
            return f"tap ({action.x:.2f},{action.y:.2f})" if action.x is not None else "tap_xy"
        if action.type == ActionType.SCROLL:
            return f"scroll {action.direction or 'down'}"
        if action.type == ActionType.OPEN_APP:
            return f"open app {action.text}"
        return action.type.value

    def _summarize(self, task: Task, success: bool) -> str:
        head = "[OK] Completed and verified: " if success else "[PAUSED] Reached the boundary (unverified): "
        facts = ", ".join(f"{k}={v}" for k, v in task.facts.items())
        return f"{head}{task.goal}" + (f" ({facts})" if facts else "")

    def _emit(self, task: Task) -> None:
        self.store.publish(task.id, "state", task.state.value, summary=task.summary,
                           error=task.error)
        self.store.schedule_persist(task)   # durable checkpoint in Postgres (if configured)
