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
import re
from typing import Any, Optional, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
    _HAS_LANGGRAPH = True
except Exception as _lg_err:  # not installed / unavailable on this Python
    _HAS_LANGGRAPH = False
    END, START, StateGraph = "__end__", "__start__", None
    logging.getLogger("agent.graph").warning(
        "langgraph unavailable (%s) — using the built-in ReAct driver "
        "(same nodes, no dependency)", _lg_err)

from ..audit.logger import audit
from ..mobile.schema import ActionType, AgentAction, ScreenObservation
from .approvals import action_binding
from ..skills import registry
from .state import Task, TaskState
from .verifier import is_success

log = logging.getLogger("agent.graph")

MAX_STEPS = 40                 # hard cap on ReAct cycles per task
MAX_REACT_ATTEMPTS = 4         # reflection re-loop cap (ReAct self-correction)
MAX_QUESTIONS = 3              # clarifying questions per task (breaks ask-loops)
MAX_SKIPPED_CONFIRMS = 2       # after this, a go/no-go question goes to the user for real
STALL_LIMIT = 5                # cycles with no executed action/tool/state change -> stop
                               # (found by tracing: a silent ask-loop burned 42 LLM calls)

_GO = re.compile(r"^(yes|yeah|ok|okay|sure|save|confirm|proceed|go ahead|continue|do it|send|"
                 r"submit|create|book|add|apply)\b", re.I)
_NO_GO = re.compile(r"^(no|nope|cancel|don'?t|do not|skip|abort|stop|not now|never mind)\b", re.I)


def _is_go_no_go(action: AgentAction) -> bool:
    """A two-option 'proceed or not?' question (e.g. 'Save note' / 'Cancel')."""
    opts = [o.strip() for o in (action.options or []) if o and o.strip()]
    if len(opts) != 2:
        return False
    a, b = opts
    return bool((_GO.match(a) and _NO_GO.match(b)) or (_NO_GO.match(a) and _GO.match(b)))

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
    opens: int                            # deterministic open_app attempts (bootstrap)
    progress_mark: int                    # len(history)+tool results at the last cycle
    stall: int                            # consecutive cycles without any progress
    status: str                           # run | done | failed | waiting | cancelled


# Strong "advance the booking" CTAs. If one of these is visible and the model tries to
# scroll past it, we tap it instead (the free model repeatedly scrolls past 'Book tickets').
# No bare "book" — it false-matched "Facebook". Matched on WORD BOUNDARIES below.
_CTA_PHRASES = (
    "book tickets", "book ticket", "book now", "get tickets", "proceed to pay",
    "proceed", "continue", "select seats", "select seat", "confirm booking",
    "place order", "pay now", "pay later", "add to cart", "checkout",
)
_CTA_RE = re.compile(r"\b(" + "|".join(re.escape(p) for p in _CTA_PHRASES) + r")\b")


def _find_cta(screen: ScreenObservation):
    """Return a visible, clickable element that clearly advances the flow, else None.

    Matches whole words only (so 'Facebook' never matches 'book'), ignores long
    non-button labels, and prefers the longest matching phrase.
    """
    best = None
    best_len = 0
    for e in screen.elements:
        label = f"{e.text} {e.desc}".strip().lower()
        if not label or len(label) > 24 or not (e.clickable or e.enabled):
            continue
        m = _CTA_RE.search(label)
        if m and len(m.group(1)) > best_len:
            best, best_len = e, len(m.group(1))
    return best


def _target_app_name(skill) -> str:
    """Friendly app name the device resolves by launcher label (e.g. 'BookMyShow').

    Falls back to the raw package if we have no friendly name for it.
    """
    from ..llm.prompts import KNOWN_APPS
    pkg = skill.primary_app()
    friendly = KNOWN_APPS.get(pkg, "")
    return friendly.split(" (")[0] if friendly else pkg


def _screen_sig(screen: ScreenObservation) -> str:
    """A stable fingerprint of what is on screen, for change detection."""
    if not screen.elements:
        return f"empty|{screen.app_package}|{screen.activity}|shot={bool(screen.screenshot_b64)}"
    # `checked` is included so toggling a checkbox/seat counts as a visible change.
    items = sorted((e.role, e.text, e.resource_id, e.editable, str(e.checked))
                   for e in screen.elements)
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
        # Cycle boundary: honour a pause here (the previous action has completed).
        await c.wait_if_paused(task)
        if task.cancel_requested:
            task.set_state(TaskState.CANCELLED, "cancelled by user")
            return {"status": "cancelled"}
        if step >= MAX_STEPS:
            task.set_state(TaskState.FAILED, f"gave up after {MAX_STEPS} steps")
            return {"status": "failed"}

        disconnects = state.get("disconnects", 0)
        blind = state.get("blind", 0)
        offline_waits = 0   # consecutive 30s waits with the phone fully offline

        # Acquisition loop: re-fetch the device and screen, surviving a transient
        # disconnect (OEM battery kill), a long network dropout, and a brief
        # protected/blank screen.
        while True:
            if task.cancel_requested:
                task.set_state(TaskState.CANCELLED, "cancelled by user")
                return {"status": "cancelled"}
            device = await c._await_device(task)
            if device is None:
                # The phone<->cloud link can drop for well over 30s. Keep waiting (up to
                # ~2 minutes total) before declaring the task dead.
                offline_waits += 1
                if offline_waits >= 4:
                    task.set_state(TaskState.FAILED, "device offline for ~2 minutes")
                    return {"status": "failed"}
                task.log("note", f"device offline — still waiting ({offline_waits}/4)")
                c.store.publish(task.id, "note", "phone offline — waiting for it to reconnect")
                continue
            offline_waits = 0
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
            # (The API/MCP executor has no screen at all — that's expected, not blind.)
            if task.executor != "api" and not screen.elements and not screen.screenshot_b64:
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

        # --- scroll-streak nudge: the model tends to scroll past the control it should
        #     tap (e.g. a 'Book tickets' CTA). After 2 scrolls in a row, don't give up —
        #     reflect it toward tapping a visible actionable control before the hard loop
        #     guard below would hand off. ---
        if (not updates.get("feedback")
                and len(task.history) >= 2
                and all(h.startswith("scroll") for h in task.history[-2:])):
            updates["feedback"] = (
                "You have scrolled twice in a row. STOP scrolling. Look at the elements NOW "
                "and TAP the most relevant actionable control for the goal — e.g. a 'Book "
                "tickets' / 'Book' button, the movie, a showtime, or a seat. Only scroll "
                "again if nothing relevant is visible.")
            c.store.publish(task.id, "reflect", "scroll streak — nudging a tap")

        # --- stall guard (cost protection): no executed action, tool call or answered
        #     question for STALL_LIMIT cycles => the model is spinning; every extra cycle
        #     is a paid LLM call. Stop and hand over instead. (Tracing showed a silent
        #     ask-loop making 42 calls with a constant prompt before the step cap hit.) ---
        mark = len(task.history) + len(task.tool_results) + len(task.facts)
        stall = state.get("stall", 0) + 1 if mark == state.get("progress_mark", -1) else 0
        updates["progress_mark"], updates["stall"] = mark, stall
        if stall >= STALL_LIMIT:
            task.summary = c._summarize(task, success=False)
            task.log("note", f"no progress for {stall} cycles — stopping to save cost")
            task.set_state(TaskState.WAITING_FOR_USER,
                           f"no progress for {stall} steps — stopped (please check)")
            updates["status"] = "waiting"
            return updates

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

        # Deterministic bootstrap: if the skill targets specific apps and we are NOT in
        # one yet (e.g. sitting on the home launcher, whose tree is sparse and whose
        # screenshot the OEM may block), just OPEN the app instead of asking the LLM to
        # reason about a near-blank screen. This is the #1 cause of the agent "asking
        # instead of acting" at the start. Capped so a non-launching app can't loop.
        opens = state.get("opens", 0)
        if (task.executor in ("", "mobile") and skill.app_packages
                and screen.app_package not in skill.app_packages and opens < 2):
            name = _target_app_name(skill)
            task.log("action", "open_app", thought=f"Not in the target app; opening {name}.")
            return {"action": AgentAction(type=ActionType.OPEN_APP, text=name,
                                          thought=f"Open {name} to begin.",
                                          expectation=f"{name} is open"),
                    "opens": opens + 1}

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

        # ---- deterministic CTA override: never scroll past a clear 'advance' button ----
        # The free model repeatedly scrolls past a visible 'Book tickets'/'Proceed' control;
        # if it tries to scroll while such a button is on screen, tap the button instead.
        if action.type == ActionType.SCROLL:
            cta = _find_cta(screen)
            if cta is not None:
                task.log("note", f"override scroll → tap CTA #{cta.ref} {cta.text!r}")
                c.store.publish(task.id, "note", f"tapping '{cta.text}' instead of scrolling")
                action = AgentAction(type=ActionType.TAP, ref=cta.ref,
                                     thought=f"Tapping '{cta.text}' instead of scrolling past it.",
                                     expectation="advance to the next booking step")
                state["action"] = action

        # ---- tool call (built-in or MCP) — no device involved ----
        if action.type == ActionType.CALL_TOOL:
            return await c.run_tool(task, action, screen, state)

        # ---- terminal / control tools ----
        if action.type == ActionType.DONE:
            if task.executor == "api":
                # Verified only by a real runtime result: at least one successful tool call.
                verified = any(r.get("ok") for r in task.tool_results)
                answer = (action.text or "").strip()
                task.summary = (("[OK] " if verified else "[UNVERIFIED] ") +
                                (answer or task.goal))
                task.set_state(TaskState.COMPLETED if verified else TaskState.WAITING_FOR_USER,
                               "answered from tool results" if verified
                               else "model reported done without a successful tool result")
                return {"status": "done" if verified else "waiting"}
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
            # Redundant go/no-go ("save it? Save / Cancel"): the user already asked for this
            # and the PLATFORM's approval gate still guards anything risky, so skipping the
            # model's own confirmation can never bypass a real approval. The model ignores
            # "don't confirm" instructions, so this is enforced in code.
            skipped = sum(1 for e in task.events
                          if e.kind == "note" and e.detail.startswith("skipped a redundant"))
            if _is_go_no_go(action) and skipped < MAX_SKIPPED_CONFIRMS:
                task.log("note", "skipped a redundant confirmation question (platform approval "
                                 "still applies) — proceeding")
                return {"status": "run", "pending_validate": False,
                        "feedback": "Do NOT ask for confirmation — the user already requested "
                                    "this, and the platform asks for approval itself where "
                                    "needed. Perform the action now with call_tool / the "
                                    "matching action."}
            # Model insists (or a genuine question): ask the user for real — a human "Save"
            # resolves what feedback couldn't. Falls through to the question cap below.
            asked = sum(1 for e in task.events if e.kind == "state_change"
                        and e.detail == "needs clarification")
            if asked >= MAX_QUESTIONS:
                task.log("note", f"question cap ({MAX_QUESTIONS}) reached — not asking again")
                return {"status": "run", "pending_validate": False,
                        "feedback": "You have asked the user enough. Do not ask again: make a "
                                    "sensible choice yourself and proceed, or `fail` if truly "
                                    "blocked."}
            facts = await c._ask_user(task, [action.text or "I need more information."],
                                      [list(action.options or [])])
            if task.cancel_requested:
                task.set_state(TaskState.CANCELLED, "cancelled")
                return {"status": "cancelled"}
            task.facts.update(facts or {})
            return {"status": "run", "pending_validate": False, "feedback": ""}

        device = await c._await_device(task)
        if device is None:
            task.set_state(TaskState.FAILED, "device disconnected")
            return {"status": "failed"}

        # Everything below touches the device. On this flaky mobile->cloud link the socket
        # can drop mid-action; treat that as a soft retry (re-observe once it reconnects)
        # instead of crashing the whole task.
        try:
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

                outcome, bound_hash = await c._approval_pause(task, device, decision,
                                                              action, screen)
                if task.cancel_requested or outcome == "cancelled":
                    task.set_state(TaskState.CANCELLED, "cancelled during approval")
                    return {"status": "cancelled"}
                if outcome != "approved":
                    task.set_state(TaskState.CANCELLED,
                                   "approval expired — the sensitive step was not performed"
                                   if outcome == "expired" else "user declined the sensitive step")
                    return {"status": "cancelled"}
                task.approvals_granted += 1
                task.set_state(TaskState.RUNNING)
                device = await c._await_device(task)   # may have reconnected during the wait
                if device is None:
                    task.set_state(TaskState.FAILED, "device disconnected during approval")
                    return {"status": "failed"}
                if decision.risk_level >= 4:
                    # User-only action (payment/auth): the user performs it; we re-observe.
                    resp = await device.request(
                        "handoff", timeout=330.0,   # the user is paying/signing in (device waits 5 min)
                        approval=c._approval_payload(task, decision, action, screen))
                    if isinstance(resp, dict) and resp.get("stop"):
                        # Executor can't let the user act in place (headless browser):
                        # stop here and tell them where to finish.
                        task.summary = resp.get("message") or "Please finish this step yourself."
                        task.set_state(TaskState.WAITING_FOR_USER, task.summary)
                        return {"status": "waiting"}
                    return {"status": "run", "pending_validate": False}

                # Level 3: re-validate the approval against a FRESH screen immediately before
                # executing. If the target button / amount / app changed while the user was
                # deciding, the approval no longer describes what we'd do -> invalidate and
                # let the loop re-decide (which re-requests approval for the new state).
                fresh = await c._observe(device)
                if action_binding(action, fresh) != bound_hash:
                    c.invalidate_approval(task, bound_hash,
                                          "screen changed after approval (target/amount differ)")
                    task.approvals_granted -= 1
                    return {"status": "run", "pending_validate": False,
                            "feedback": "The screen changed after the user approved; re-check "
                                        "it and request approval again if still needed."}
                await c._act(device, action, task)  # Level 3 approved and still valid
                return {"status": "run", "last_sig": _screen_sig(fresh), "pending_validate": True}

            # ---- auto-allowed (read-only / reversible preparation) ----
            await c._act(device, decision.action, task)
            task.set_state(TaskState.VERIFYING)
            return {"status": "run", "last_sig": _screen_sig(screen), "pending_validate": True}

        except (RuntimeError, OSError, asyncio.TimeoutError) as e:
            task.log("note", f"action interrupted by a disconnect — will re-observe ({e})")
            c.store.publish(task.id, "note", "action interrupted; reconnecting")
            await asyncio.sleep(2.0)
            return {"status": "run", "pending_validate": False, "feedback": ""}

    # ------------------------------------------------------------------ #
    # Wiring: observe -> validate -> reason -> act -> observe (while "run")
    # ------------------------------------------------------------------ #
    def _route(nxt: str):
        def route(state: AgentState) -> str:
            return nxt if state.get("status", "run") == "run" else END
        return route

    def _traced(name: str, fn):
        """One span per graph node execution (react.step + resulting status/action)."""
        from .. import observability as obs

        async def node(state: AgentState) -> dict:
            with obs.span(f"node.{name}", **{"react.step": state.get("step", 0)}) as sp:
                out = await fn(state) or {}
                act = out.get("action")
                obs.set_attrs(sp, **{
                    "react.status": out.get("status", state.get("status", "run")),
                    "react.action": getattr(getattr(act, "type", None), "value", None),
                    "react.reflection": bool(out.get("feedback"))})
                return out
        return node

    nodes = {"observe": _traced("observe", observe), "validate": _traced("validate", validate),
             "reason": _traced("reason", reason), "act": _traced("act", act)}
    edges = {"observe": "validate", "validate": "reason", "reason": "act", "act": "observe"}

    if not _HAS_LANGGRAPH:
        # Built-in driver: runs the exact same nodes + routing, no dependency.
        return _ReActDriver(nodes, edges, start="observe")

    g = StateGraph(AgentState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "observe")
    for src, dst in edges.items():
        g.add_conditional_edges(src, _route(dst), {dst: dst, END: END})
    return g.compile()


class _ReActDriver:
    """Sequential runner for the ReAct graph when langgraph isn't installed.

    Executes the identical nodes and status-gated routing as the compiled StateGraph:
    run a node, merge its state delta, stop if status left "run", else follow the edge.
    """

    def __init__(self, nodes: dict, edges: dict, start: str = "observe"):
        self._nodes = nodes
        self._edges = edges
        self._start = start

    async def ainvoke(self, state: dict, config: dict | None = None) -> dict:
        limit = (config or {}).get("recursion_limit", 10_000)
        node = self._start
        for _ in range(limit):
            state.update(await self._nodes[node](state) or {})
            if state.get("status", "run") != "run":
                break
            node = self._edges[node]
        return state
