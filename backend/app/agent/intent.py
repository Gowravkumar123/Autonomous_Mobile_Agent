"""Intent analyzer + planner. Turn a prompt into a structured, inspectable plan."""
from __future__ import annotations

import logging
import re

from ..llm.base import LLMError
from ..llm.prompts import (INTENT_SYSTEM, PLAN_SYSTEM, build_intent_user,
                           build_plan_user)
from ..llm.router import LLMRouter
from ..skills import registry
from .state import Task

log = logging.getLogger("agent.intent")

# Canonical fact key -> accepted synonyms the model might emit instead.
_FACT_SYNONYMS: dict[str, list[str]] = {
    "from": ["origin", "source", "from_station", "boarding", "departure", "start", "src"],
    "to": ["destination", "dest", "to_station", "arrival", "end", "drop"],
    "date": ["travel_date", "journey_date", "journey", "depart_date", "when", "day"],
    "passengers": ["ticket_count", "tickets", "num_passengers", "count", "quantity",
                   "persons", "pax", "no_of_passengers"],
    "class": ["coach", "seat_class", "travel_class", "berth_class"],
    "item": ["dish", "food", "meal", "cuisine"],
    "to_contact": ["contact", "recipient", "name", "person", "receiver"],
    "message": ["text", "body", "content", "msg"],
}


def normalize_facts(facts: dict) -> dict:
    """Map whatever synonym the model used onto our canonical fact keys."""
    out = dict(facts)
    for canonical, synonyms in _FACT_SYNONYMS.items():
        if canonical in out and out[canonical] not in (None, ""):
            continue
        for syn in synonyms:
            if out.get(syn) not in (None, ""):
                out[canonical] = out[syn]
                break
    return out


async def analyze_intent(task: Task, llm: LLMRouter,
                         tool_names: list[str] | None = None,
                         memories: list[str] | None = None) -> None:
    """Fill goal / skill / app_hint / facts / clarifications / executor_hint on the task."""
    from ..config import get_settings
    s = get_settings()
    try:
        data = await llm.complete_json_fast(INTENT_SYSTEM,
                                            build_intent_user(task.prompt, tool_names, memories),
                                            temperature=0.1, max_tokens=s.intent_max_tokens)
    except LLMError as e:
        log.warning("intent analysis failed: %s", e)
        data = {"goal": task.prompt, "skill": "generic", "facts": {}, "clarifications": []}

    task.goal = data.get("goal") or task.prompt
    task.skill = (data.get("skill") or "generic").strip().lower()
    task.facts = normalize_facts(data.get("facts") or {})
    hint = str(data.get("executor_hint") or "").strip().lower()
    task.executor_hint = "api" if (hint == "api" and tool_names) else ""

    def _strs(v) -> list[str]:
        if isinstance(v, str):
            v = [v]
        return [str(x).strip() for x in (v or []) if str(x).strip()][:5]
    task.memory_remember = _strs(data.get("remember"))
    task.memory_forget = _strs(data.get("forget"))
    task.memory_only = bool(data.get("memory_only")) and bool(
        task.memory_remember or task.memory_forget)
    _enforce_memory_command(task)

    # Clarifications may be plain strings or {question, choices} objects. Normalize to
    # parallel lists: question text + its concrete choices (so the phone can offer them).
    questions: list[str] = []
    choices: list[list[str]] = []
    for c in (data.get("clarifications") or []):
        if isinstance(c, dict):
            q = str(c.get("question") or "").strip()
            ch = [str(x).strip() for x in (c.get("choices") or []) if str(x).strip()]
        else:
            q, ch = str(c).strip(), []
        if q:
            questions.append(q)
            choices.append(ch)
    task.clarifications = questions
    task.clarification_choices = choices

    # Did the user pre-authorize commitments? (payments still always pause.)
    _PREAUTH = ("do it", "just do it", "go ahead", "send it", "just send", "submit it",
                "without asking", "don't ask", "dont ask", "no confirmation",
                "no need to confirm", "just book", "just order", "complete it",
                "finish it", "automatically", "without confirmation")
    low = task.prompt.lower()
    task.pre_authorized = any(p in low for p in _PREAUTH)

    skill = registry.get(task.skill)
    task.app_hint = data.get("app_hint") or skill.primary_app()

    # Guard: if a required fact is missing, surface it as a clarification.
    for req in skill.required_facts:
        if not task.facts.get(req) and req not in " ".join(task.clarifications).lower():
            task.clarifications.append(f"What is the {req.replace('_', ' ')}?")
            task.clarification_choices.append([])

    task.log("note", "intent analyzed", skill=task.skill, app=task.app_hint,
             facts=task.facts, clarifications=task.clarifications)


async def make_plan(task: Task, llm: LLMRouter) -> None:
    from ..config import get_settings
    s = get_settings()
    skill = registry.get(task.skill)
    try:
        data = await llm.complete_json_fast(
            PLAN_SYSTEM, build_plan_user(task.goal, task.facts, skill.playbook,
                                         task.memory_context),
            temperature=0.2, max_tokens=s.plan_max_tokens)
        task.plan = data.get("steps") or []
        task.approval_boundary = data.get("approval_boundary") or skill.approval_boundary
    except LLMError as e:
        log.warning("planning failed, using skill defaults: %s", e)
        task.plan = [line.strip() for line in skill.playbook.splitlines() if line.strip()]
        task.approval_boundary = skill.approval_boundary

    task.log("note", "plan created", steps=task.plan, approval_boundary=task.approval_boundary)


# "Remember ...", "forget ...", "note that ...", "keep in mind ..." at the START of a
# request with no second instruction is a pure memory command. The model marks this
# inconsistently (live: "Forget how my mom is saved" sometimes fell through to execution,
# and with no phone online select_executor launched a BROWSER to "forget"). Enforced here.
_MEMORY_CMD = re.compile(
    r"^\s*(?:please\s+|ok(?:ay)?\s+|hey\s+\w+[,\s]+|zoro[,\s]+)*"
    r"(remember|forget|don'?t forget|note that|keep in mind|make a note that)\b[\s,:-]*(.*)$",
    re.IGNORECASE | re.DOTALL)
_SECOND_CLAUSE = re.compile(r"\b(and then|then|and also|after that|and (?:book|send|order|"
                            r"open|call|buy|pay|play|search|message))\b", re.IGNORECASE)


def _enforce_memory_command(task: Task) -> None:
    m = _MEMORY_CMD.match(task.prompt or "")
    if not m or _SECOND_CLAUSE.search(task.prompt):
        return
    verb, rest = m.group(1).lower(), m.group(2).strip().rstrip(".")
    if rest.lower().startswith("that "):
        rest = rest[5:].strip()
    if not rest:
        return
    if verb == "forget":
        if not task.memory_forget:
            task.memory_forget = [rest]
        task.memory_remember = []
    elif not task.memory_remember:
        task.memory_remember = [f"User noted: {rest}"]
    task.memory_only = True
