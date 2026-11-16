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
            task.error = str(e)
            task.set_state(TaskState.FAILED, f"internal error: {e}")
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
            facts = await self._ask_user(task, task.clarifications)
            if task.cancel_requested:
                task.set_state(TaskState.CANCELLED, "cancelled during clarification")
                return
            task.facts.update(facts or {})
            task.clarifications = []

    async def _execute_phase(self, task: Task) -> None:
        device = self.devices.get(task.device_id) or self.devices.any()
        if device is None:
            task.set_state(TaskState.FAILED, "no paired device connected")
            self._emit(task)
            return

        skill = registry.get(task.skill)
        task.set_state(TaskState.RUNNING)
        self._emit(task)

        for step in range(MAX_STEPS):
            if task.cancel_requested:
                task.set_state(TaskState.CANCELLED, "cancelled by user")
                break

            # ---- observe ----
            try:
                screen = await self._observe(device)
            except (LLMError, asyncio.TimeoutError, RuntimeError) as e:
                task.set_state(TaskState.FAILED, f"lost the device: {e}")
                break
            task.step_id = step
            self.store.publish(task.id, "observation", screen.activity,
                               app=screen.app_package, n_elements=len(screen.elements))

            # ---- verify success first (a confirmation may already be on screen) ----
            if is_success(screen, skill, task):
                task.summary = self._summarize(task, success=True)
                task.set_state(TaskState.COMPLETED, "success marker verified")
                audit(task.id, "verified", task.summary)
                break

            # ---- decide ----
            task.set_state(TaskState.RUNNING)
            action = await self._decide(task, skill, screen)
            task.log("action", f"{action.type.value}", thought=action.thought)

            if action.type == ActionType.DONE:
                # Model claims done but no marker — be honest about uncertainty.
                verified = is_success(screen, skill, task)
                task.summary = self._summarize(task, success=verified)
                task.set_state(TaskState.COMPLETED if verified else TaskState.WAITING_FOR_USER,
                               "model reported done" + ("" if verified else " (unverified — please confirm)"))
                break
            if action.type == ActionType.FAIL:
                task.error = action.text or action.thought
                task.set_state(TaskState.FAILED, task.error or "agent could not proceed")
                break
            if action.type == ActionType.ASK_USER:
                facts = await self._ask_user(task, [action.text or "I need more information."])
                if task.cancel_requested:
                    task.set_state(TaskState.CANCELLED, "cancelled")
                    break
                task.facts.update(facts or {})
                continue

            # ---- policy gate (deterministic) ----
            decision = self.policy.evaluate(action, screen)
            audit(task.id, "policy", decision.reason, risk=decision.risk_level,
                  action=action.type.value, allow=decision.allow)

            if decision.needs_approval:
                approved = await self._approval_pause(task, device, decision, action, screen)
                if task.cancel_requested or not approved:
                    task.set_state(TaskState.CANCELLED, "user declined the sensitive step")
                    break
                task.approvals_granted += 1
                task.set_state(TaskState.RUNNING)
                if decision.risk_level >= 4:
                    # User-only action (payment/auth). The user performs it; we re-observe.
                    await device.request("handoff", approval=self._approval_payload(task, decision, action, screen))
                    continue
                # Level 3 approved: the agent may now perform the original action.
                await self._act(device, action, task)
                continue

            # ---- execute (auto-allowed) ----
            await self._act(device, decision.action, task)
            task.set_state(TaskState.VERIFYING)

        else:
            task.set_state(TaskState.FAILED, f"gave up after {MAX_STEPS} steps")

        self._emit(task)

    # ----------------------------------------------------------------- #
    # Steps
    # ----------------------------------------------------------------- #
    async def _observe(self, device: DeviceSession) -> ScreenObservation:
        payload = await device.request("observe")
        return ScreenObservation(**(payload.get("screen", payload)))

    async def _decide(self, task: Task, skill, screen: ScreenObservation) -> AgentAction:
        plan_summary = "; ".join(task.plan) + f"  | BOUNDARY: {task.approval_boundary}"
        user = build_action_user(task.goal, task.facts, plan_summary, task.history, screen)
        try:
            data = await self.llm.complete_json(ACTION_SYSTEM, user, temperature=0.1, max_tokens=500)
            return AgentAction(**data)
        except (LLMError, ValueError, TypeError) as e:
            log.warning("decide failed: %s", e)
            return AgentAction(type=ActionType.ASK_USER,
                               text="I couldn't interpret the screen. What should I do next?",
                               thought=str(e))

    async def _act(self, device: DeviceSession, action: AgentAction, task: Task) -> None:
        await device.request("act", action=action.model_dump(mode="json"))
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
        await device.push("approval_request", **payload)
        self.store.publish(task.id, "approval", decision.reason, approval=payload)
        audit(task.id, "approval_requested", decision.reason, approval=payload)

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._approvals[task.id] = fut
        approved, note = await fut
        self._approvals.pop(task.id, None)
        audit(task.id, "approval_decided", f"approved={approved}", note=note)
        self.store.publish(task.id, "approval_decided", f"approved={approved}")
        return approved

    async def _ask_user(self, task: Task, questions: list[str]) -> dict:
        task.set_state(TaskState.WAITING_FOR_USER, "needs clarification")
        self.store.publish(task.id, "ask_user", " | ".join(questions), questions=questions)
        audit(task.id, "ask_user", " | ".join(questions))
        # Surface the question(s) to the paired device/companion UI so the user can answer.
        device = self.devices.get(task.device_id) or self.devices.any()
        if device:
            await device.push("ask_user", task_id=task.id, questions=questions)
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._answers[task.id] = fut
        facts = await fut
        self._answers.pop(task.id, None)
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
