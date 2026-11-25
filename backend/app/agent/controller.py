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
import logging

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
            if task.device_id and task.device_id != session.device_id:
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
    def resolve_approval(self, task_id: str, approved: bool, note: str = "") -> bool:
        fut = self._approvals.get(task_id)
        if fut and not fut.done():
            fut.set_result((approved, note))
            return True
        return False

    def resolve_answer(self, task_id: str, facts: dict) -> bool:
        fut = self._answers.get(task_id)
        if fut and not fut.done():
            fut.set_result(facts)
            return True
        return False

    def cancel(self, task_id: str) -> None:
        task = self.store.get(task_id)
        if task:
            task.cancel_requested = True
            # Unblock any pending pauses so the loop can observe the cancel.
            self.resolve_approval(task_id, False, "cancelled")
            self.resolve_answer(task_id, {})

    # ----------------------------------------------------------------- #
    # Main driver
    # ----------------------------------------------------------------- #
    async def run_task(self, task: Task) -> None:
        try:
            await self._plan_phase(task)
            if task.state in TERMINAL:
                return
            await self._execute_phase(task)
        except Exception as e:  # defensive: never leave a task hung
            log.exception("task %s crashed", task.id)
            import traceback
            task.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}"
            task.set_state(TaskState.FAILED, f"internal error: {type(e).__name__}: {e}")
            self._emit(task)
        finally:
            self._approvals.pop(task.id, None)
            self._answers.pop(task.id, None)

    async def _plan_phase(self, task: Task) -> None:
        task.set_state(TaskState.PLANNING)
        self._emit(task)
        await analyze_intent(task, self.llm)
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
            "step": 0, "last_sig": "", "pending_validate": False, "opens": 0,
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
            # other connected phone); unbound tasks may use any device.
            dev = (self.devices.get(task.device_id) if task.device_id
                   else self.devices.any())
            if dev is not None:
                return dev
            if task.cancel_requested or loop.time() >= deadline:
                return None
            await asyncio.sleep(1.0)

    async def _observe(self, device: DeviceSession) -> ScreenObservation:
        payload = await device.request("observe")
        return ScreenObservation(**(payload.get("screen", payload)))

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
            task.log("note", f"OCR: {'on' if available() else 'OFF'}, {len(ocr)} items, "
                             f"shot={'yes' if screen.screenshot_b64 else 'no'}, "
                             f"els={len(screen.elements)}")
        elif len(screen.elements) <= 6:
            task.log("note", f"no screenshot (vision/OCR unavailable), els={len(screen.elements)}")
        from ..config import get_settings
        s = get_settings()
        user = build_action_user(task.goal, task.facts, plan_summary, task.history, screen,
                                 max_elements=s.max_elements_in_prompt, ocr_items=ocr,
                                 feedback=feedback)
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
        payload = action.model_dump(mode="json")
        for attempt in range(2):
            try:
                await device.request("act", action=payload)
                break
            except (RuntimeError, OSError, asyncio.TimeoutError):
                if attempt == 1:
                    raise
                fresh = await self._await_device(task)   # reconnected session, if any
                if fresh is None:
                    raise
                device = fresh
        label = self._action_label(action)
        task.history.append(label)
        self.store.publish(task.id, "action", label, risk=action.risk_level)
        audit(task.id, "action", label, type=action.type.value, ref=action.ref)

    async def _approval_pause(self, task: Task, device: DeviceSession, decision,
                              action: AgentAction, screen: ScreenObservation) -> bool:
        payload = self._approval_payload(task, decision, action, screen)
        task.pending_action = action
        task.pending_approval_reason = decision.reason
        task.log("approval_requested", decision.reason, risk=decision.risk_level, **payload)
        task.set_state(TaskState.WAITING_FOR_USER, decision.reason)
        self._pending_prompts[task.id] = ("approval_request", payload)
        try:
            await device.push("approval_request", **payload)
        except Exception as e:
            log.warning("approval push failed (will re-deliver on reconnect): %s", e)
        self.store.publish(task.id, "approval", decision.reason, approval=payload)
        audit(task.id, "approval_requested", decision.reason, approval=payload)

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._approvals[task.id] = fut
        try:
            approved, note = await fut
        finally:
            self._approvals.pop(task.id, None)
            self._pending_prompts.pop(task.id, None)
        audit(task.id, "approval_decided", f"approved={approved}", note=note)
        self.store.publish(task.id, "approval_decided", f"approved={approved}")
        return approved

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
