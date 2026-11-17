"""
The agent brain as a LangGraph ReAct state machine.

This is the real agentic core: a compiled `StateGraph` whose shared `AgentState`
flows through four nodes that implement a ReAct (Reason + Act) loop with explicit
state management and a bounded self-correction (reflection) sub-loop.

    START -> observe -> validate -> reason -> act -> observe ...  -> END

  observe   (Observation)  pull the current screen from the phone; survive device
                           reconnects and protected/secure screens.
  validate  (Reflection)   the VALIDATOR — did the previous action actually change
                           the screen the way it was expected to? If a navigating
                           action produced NO visible change, raise a reflection and
                           re-loop (up to MAX_REACT_ATTEMPTS) with feedback so the
                           model tries a DIFFERENT approach instead of repeating.
                           Also the success/loop guards live here.
  reason    (Thought+Act)  the LLM reasons (`thought`) and picks exactly ONE tool
                           (the next AgentAction), optionally steered by reflection
                           feedback from the validator.
  act       (Action)       run the chosen tool through the deterministic policy gate
                           (approval / payment pauses, pre-authorization), then back
                           to observe for the next ReAct cycle.

State transitions are explicit so a task can pause (approval / clarification / secure
screen) and resume across device disconnects. The LLM only proposes; the PolicyEngine
decides what may actually run.
"""
from __future__ import annotations

# NOTE: must precede the langgraph import — installs a pure-Python xxhash locally
# when the native DLL is blocked (dev box). No-op on the server.
from . import _compat  # noqa: F401

import asyncio
import hashlib
import logging
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from ..audit.logger import audit
from ..mobile.schema import ActionType, AgentAction, ScreenObservation
from ..skills import registry
from .state import Task, TaskState
from .verifier import is_success

log = logging.getLogger("agent.graph")

MAX_STEPS = 40                 # hard cap on ReAct cycles per task
MAX_REACT_ATTEMPTS = 4         # reflection re-loop cap (ReAct self-correction)

# Tools whose whole point is to move the UI forward: if one of these fires and the
# screen does not change at all, it did not work -> the validator reflects. (TYPE and
# SCROLL are excluded on purpose: typed text may not surface in the tree, and a scroll
# at a list's end legitimately changes nothing.)
VALIDATE_TYPES = {ActionType.TAP, ActionType.TAP_XY, ActionType.ENTER,
                  ActionType.OPEN_APP, ActionType.BACK}


class AgentState(TypedDict, total=False):
    """Shared state threaded through every node — the agent's working memory."""
    task: Task
    skill: Any
    screen: Optional[ScreenObservation]   # latest observation
    action: Optional[AgentAction]         # latest decided tool (Thought + Action)
    feedback: str                         # reflection injected into the next reason step
    react_attempts: int                   # consecutive no-effect reflections (<= MAX)
    blind: int                            # consecutive blank/secure screens
    disconnects: int                      # consecutive device drops
    step: int                             # ReAct cycle counter
    last_sig: str                         # screen signature BEFORE the last action ran
    pending_validate: bool                # an action ran and awaits validation
    status: str                           # run | done | failed | waiting | cancelled


def _screen_sig(screen: ScreenObservation) -> str:
    """A stable fingerprint of what is on screen, for change detection."""
    if not screen.elements:
        return f"empty|{screen.app_package}|{screen.activity}|shot={bool(screen.screenshot_b64)}"
    items = sorted((e.role, e.text, e.resource_id, e.editable) for e in screen.elements)
    blob = repr((screen.app_package, screen.activity, items)).encode()
    return hashlib.blake2b(blob, digest_size=12).hexdigest()


def build_react_graph(controller):
    """Compile the ReAct StateGraph, binding its nodes to a live AgentController."""
    c = controller

    # ------------------------------------------------------------------ #
    # Node: OBSERVE  (Observation)
    # ------------------------------------------------------------------ #
    async def observe(state: AgentState) -> dict:
        task: Task = state["task"]
        step = state.get("step", 0)
        if task.cancel_requested:
            task.set_state(TaskState.CANCELLED, "cancelled by user")
            return {"status": "cancelled"}
        if step >= MAX_STEPS:
            task.set_state(TaskState.FAILED, f"gave up after {MAX_STEPS} steps")
            return {"status": "failed"}

        disconnects = state.get("disconnects", 0)
        blind = state.get("blind", 0)

        # Acquisition loop: re-fetch the device and screen, surviving a transient
        # disconnect (OEM battery kill) and a brief protected/blank screen.
        while True:
            if task.cancel_requested:
                task.set_state(TaskState.CANCELLED, "cancelled by user")
                return {"status": "cancelled"}
            device = await c._await_device(task)
            if device is None:
                task.set_state(TaskState.FAILED, "device disconnected and did not return")
                return {"status": "failed"}
            try:
                screen = await c._observe(device)
            except (asyncio.TimeoutError, RuntimeError, OSError) as e:
                disconnects += 1
                if disconnects > 6:
                    task.set_state(TaskState.FAILED, f"device kept disconnecting: {e}")
                    return {"status": "failed"}
                task.log("note", f"disconnect {disconnects}; waiting to resume ({e})")
                c.store.publish(task.id, "note", f"reconnecting ({disconnects})")
                await asyncio.sleep(2.0)
                continue

            task.step_id = step
            c.store.publish(task.id, "observation", screen.activity,
                            app=screen.app_package, n_elements=len(screen.elements))

            # Blind screen: no tree AND no screenshot -> usually a FLAG_SECURE
            # payment/checkout screen. Don't tap blindly; wait, then hand over.
            if not screen.elements and not screen.screenshot_b64:
                blind += 1
                if blind >= 3:
                    task.summary = ("A protected screen (likely payment/checkout) is blocking "
                                    "the agent from seeing it. Please take over on the phone.")
                    task.log("note", "blind screen (secure) — handing to user")
                    try:
                        await device.push("ask_user", task_id=task.id,
                                          questions=["This screen looks protected and I can't see it — "
                                                     "please take over on your phone."])
                    except Exception:
                        pass
                    task.set_state(TaskState.WAITING_FOR_USER,
                                   "screen is protected (secure) — please take over")
                    return {"status": "waiting"}
                await asyncio.sleep(1.0)
                continue

            return {"screen": screen, "status": "run", "step": step + 1,
                    "blind": 0, "disconnects": 0}

    # ------------------------------------------------------------------ #
    # Node: VALIDATE  (Reflection / verifier + guards)
    # ------------------------------------------------------------------ #
    async def validate(state: AgentState) -> dict:
        task: Task = state["task"]
        skill = state["skill"]
        screen: ScreenObservation = state["screen"]
        updates: dict = {"feedback": "", "pending_validate": False}

        react_attempts = state.get("react_attempts", 0)
        action: Optional[AgentAction] = state.get("action")

        # --- the VALIDATOR: did the previous action achieve a visible effect? ---
        if state.get("pending_validate") and action is not None:
            changed = _screen_sig(screen) != state.get("last_sig", "")
            must_change = action.type in VALIDATE_TYPES
            if must_change and not changed:
                if react_attempts < MAX_REACT_ATTEMPTS:
                    react_attempts += 1
                    label = c._action_label(action)
                    updates["feedback"] = (
                        f"REFLECTION (attempt {react_attempts}/{MAX_REACT_ATTEMPTS}): your previous "
                        f"action [{label}] produced NO visible change on screen — it did not work. "
                        f"Expected: {action.expectation or 'the screen to advance'}. Choose a DIFFERENT "
                        f"approach now — a different element/ref, a tap_xy on the control you can see, "
                        f"or another route to the goal. Do NOT repeat the same action.")
                    task.log("note", f"validate: no effect, reflecting ({react_attempts}/{MAX_REACT_ATTEMPTS})")
                    c.store.publish(task.id, "reflect",
                                    f"no effect — retry {react_attempts}/{MAX_REACT_ATTEMPTS}")
                else:
                    react_attempts = 0  # give up reflecting; loop-guard / step-limit take over
            else:
                react_attempts = 0      # it worked (or wasn't expected to change the screen)
        updates["react_attempts"] = react_attempts

        # --- success marker already on screen? ---
        if is_success(screen, skill, task):
            task.summary = c._summarize(task, success=True)
            task.set_state(TaskState.COMPLETED, "success marker verified")
            audit(task.id, "verified", task.summary)
            updates["status"] = "done"
            return updates

        # --- loop guard: same action repeated with no progress ---
        if len(task.history) >= 4 and len(set(task.history[-4:])) == 1:
            task.summary = c._summarize(task, success=False)
            task.log("note", f"loop detected on '{task.history[-1]}' — stopping")
            device = await c._await_device(task)
            if device:
                try:
                    await device.push("ask_user", task_id=task.id,
                                      questions=["I repeated the same step a few times without "
                                                 "progress — the task may already be done. Please check."])
                except Exception:
                    pass
            task.set_state(TaskState.WAITING_FOR_USER,
                           f"repeated '{task.history[-1]}' without progress — please check")
            updates["status"] = "waiting"
            return updates

        updates["status"] = "run"
        return updates

    # ------------------------------------------------------------------ #
    # Node: REASON  (Thought + Action selection)
    # ------------------------------------------------------------------ #
    async def reason(state: AgentState) -> dict:
        task: Task = state["task"]
        skill = state["skill"]
        screen = state["screen"]
        if task.cancel_requested:
            task.set_state(TaskState.CANCELLED, "cancelled by user")
            return {"status": "cancelled"}
        task.set_state(TaskState.RUNNING)
        action = await c._decide(task, skill, screen, feedback=state.get("feedback", ""))
        task.log("action", f"{action.type.value}", thought=action.thought)
        return {"action": action}

    # ------------------------------------------------------------------ #
    # Node: ACT  (tool execution through the policy gate)
    # ------------------------------------------------------------------ #
    async def act(state: AgentState) -> dict:
        task: Task = state["task"]
        skill = state["skill"]
        screen: ScreenObservation = state["screen"]
        action: AgentAction = state["action"]

        # ---- terminal / control tools ----
        if action.type == ActionType.DONE:
            verified = is_success(screen, skill, task)
            task.summary = c._summarize(task, success=verified)
            task.set_state(TaskState.COMPLETED if verified else TaskState.WAITING_FOR_USER,
                           "model reported done" + ("" if verified else " (unverified — please confirm)"))
            return {"status": "done" if verified else "waiting"}
        if action.type == ActionType.FAIL:
            task.error = action.text or action.thought
            task.set_state(TaskState.FAILED, task.error or "agent could not proceed")
            return {"status": "failed"}
        if action.type == ActionType.ASK_USER:
            facts = await c._ask_user(task, [action.text or "I need more information."])
            if task.cancel_requested:
                task.set_state(TaskState.CANCELLED, "cancelled")
                return {"status": "cancelled"}
            task.facts.update(facts or {})
            return {"status": "run", "pending_validate": False, "feedback": ""}

        device = await c._await_device(task)
        if device is None:
            task.set_state(TaskState.FAILED, "device disconnected")
            return {"status": "failed"}

        # ---- avoid redundant launches; resolve name->package on the device ----
        if action.type == ActionType.OPEN_APP:
            cur = registry.get(task.skill)
            if screen.app_package and screen.app_package in cur.app_packages:
                c.store.publish(task.id, "note", f"already in {screen.app_package}, skip open")
                task.history.append(f"(already in {screen.app_package}; skipped open_app)")
                return {"status": "run", "pending_validate": False}
            await c._act(device, action, task)
            await asyncio.sleep(4.0)   # cold starts need time before the next observe
            return {"status": "run", "last_sig": _screen_sig(screen), "pending_validate": True}

        # ---- deterministic policy gate ----
        decision = c.policy.evaluate(action, screen)
        audit(task.id, "policy", decision.reason, risk=decision.risk_level,
              action=action.type.value, allow=decision.allow)

        if decision.needs_approval:
            # Pre-authorized ("do it"/"send it"): proceed on an external commitment
            # (Level 3) without pausing. Level 4 (payment/OTP/biometric) ALWAYS pauses.
            if task.pre_authorized and decision.risk_level <= 3:
                task.log("note", f"pre-authorized — proceeding with {action.type.value} "
                                 f"(level {decision.risk_level})")
                c.store.publish(task.id, "note", "pre-authorized; not pausing")
                await c._act(device, action, task)
                return {"status": "run", "last_sig": _screen_sig(screen), "pending_validate": True}

            approved = await c._approval_pause(task, device, decision, action, screen)
            if task.cancel_requested or not approved:
                task.set_state(TaskState.CANCELLED, "user declined the sensitive step")
                return {"status": "cancelled"}
            task.approvals_granted += 1
            task.set_state(TaskState.RUNNING)
            device = await c._await_device(task)   # may have reconnected during the wait
            if device is None:
                task.set_state(TaskState.FAILED, "device disconnected during approval")
                return {"status": "failed"}
            if decision.risk_level >= 4:
                # User-only action (payment/auth): the user performs it; we re-observe.
                await device.request("handoff",
                                     approval=c._approval_payload(task, decision, action, screen))
                return {"status": "run", "pending_validate": False}
            await c._act(device, action, task)  # Level 3 approved
            return {"status": "run", "last_sig": _screen_sig(screen), "pending_validate": True}

        # ---- auto-allowed (read-only / reversible preparation) ----
        await c._act(device, decision.action, task)
        task.set_state(TaskState.VERIFYING)
        return {"status": "run", "last_sig": _screen_sig(screen), "pending_validate": True}

    # ------------------------------------------------------------------ #
    # Wiring: observe -> validate -> reason -> act -> observe (while "run")
    # ------------------------------------------------------------------ #
    def _route(nxt: str):
        def route(state: AgentState) -> str:
            return nxt if state.get("status", "run") == "run" else END
        return route

    g = StateGraph(AgentState)
    g.add_node("observe", observe)
    g.add_node("validate", validate)
    g.add_node("reason", reason)
    g.add_node("act", act)

    g.add_edge(START, "observe")
    g.add_conditional_edges("observe", _route("validate"), {"validate": "validate", END: END})
    g.add_conditional_edges("validate", _route("reason"), {"reason": "reason", END: END})
    g.add_conditional_edges("reason", _route("act"), {"act": "act", END: END})
    g.add_conditional_edges("act", _route("observe"), {"observe": "observe", END: END})

    return g.compile()
