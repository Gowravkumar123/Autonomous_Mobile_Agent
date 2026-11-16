"""
Prompt builders. One place to tune how the agent talks to the model.

Every prompt starts with a `MODE:` line. Real models simply read it as context;
the built-in heuristic mock provider uses it to pick a code path. Screen elements
are passed as JSON (structured > prose for reliable tool use).
"""
from __future__ import annotations

import json

from ..mobile.schema import ScreenObservation

# --------------------------------------------------------------------------- #
# Intent analysis
# --------------------------------------------------------------------------- #
INTENT_SYSTEM = (
    "You are the intent analyzer of an autonomous Android agent. Convert the user's "
    "natural-language request into a compact structured goal. Identify the single best "
    "Android app to use, extract concrete facts, and list ONLY the clarifying questions "
    "that are strictly required before you could begin. Reply as JSON with keys: "
    '`goal` (str), `app_hint` (str package or app name or ""), `skill` (one of: '
    "train, food, movie, messaging, generic), `facts` (object of field->value), "
    "`clarifications` (array of strings).\n"
    "Use these canonical snake_case fact keys where they apply: `from`, `to`, `date`, "
    "`passengers`, `class`, `item`, `to_contact`, `message`. Do not invent synonyms "
    "(e.g. use `from` not `origin`, `date` not `travel_date`, `passengers` not `ticket_count`)."
)


def build_intent_user(prompt: str) -> str:
    return f"MODE: intent\nUSER REQUEST:\n{prompt}\n"


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
PLAN_SYSTEM = (
    "You are the task planner of an autonomous Android agent. Given a structured goal "
    "and a skill playbook, produce an ordered, high-level plan. Each step is a short "
    "imperative phrase describing a sub-goal (not a raw tap). Mark the step at which a "
    "sensitive boundary (payment / final submission / authentication) is reached. Reply "
    'as JSON: `steps` (array of strings), `approval_boundary` (string describing the '
    "point to stop at and hand control to the user)."
)


def build_plan_user(goal: str, facts: dict, playbook: str) -> str:
    return (
        "MODE: plan\n"
        f"GOAL: {goal}\n"
        f"FACTS: {json.dumps(facts)}\n"
        f"SKILL PLAYBOOK:\n{playbook}\n"
    )


# --------------------------------------------------------------------------- #
# Action selection (the per-step decision loop)
# --------------------------------------------------------------------------- #
ACTION_SYSTEM = (
    "You are the decision core of an autonomous Android agent. You see the CURRENT SCREEN "
    "(accessibility elements) and must choose EXACTLY ONE next action to progress the goal.\n"
    "If a SCREENSHOT image is attached, use it to understand the layout and locate controls "
    "(e.g. the search bar, buttons, items). Match what you see in the image to an element in "
    "the list and act on its `ref`.\n"
    "Rules:\n"
    "- Target elements by their `ref` number, never by pixel coordinates.\n"
    "- Prefer filling required fields and tapping forward/confirm controls.\n"
    "- Re-read the screen each step; do not assume an action worked.\n"
    "- If the CURRENT APP is already the one you need, do NOT choose `open_app` again — "
    "work inside it. Only use `open_app` from the home screen or a different app.\n"
    "- Tap what you can see: if the contact, restaurant, dish, movie, result or button you "
    "need is already visible in the elements, tap it directly instead of asking the user. "
    "For example, to message a contact whose chat row is visible, tap that row.\n"
    "- To find a specific item (a dish, restaurant, movie, train), use the app's SEARCH: "
    "tap the search field/icon and type the query (e.g. 'biryani'). Do not scroll the feed "
    "repeatedly hoping to find it. Never type an app's own name into a search box.\n"
    "- NEVER type OTPs, PINs, passwords, card numbers or CVVs. If a screen asks for those, "
    "or is a payment/UPI screen, choose action type `approve` and hand control to the user.\n"
    "- If the goal looks complete (confirmation, booking reference, order placed), choose `done`.\n"
    "- Be decisive: when several options match the user's stated preferences, PICK the best "
    "one yourself (e.g. earliest departure, then lowest fare, with seats available) and act — "
    "do NOT ask the user to choose routine, reversible things. Reserve `ask_user` for a genuinely "
    "missing required detail or a consequential, irreversible ambiguity. Never ask the same "
    "question twice — if you already asked, make a reasonable choice and proceed.\n"
    "- If you are stuck or a required detail is truly missing, choose `ask_user` (put the question "
    "in `text`) or `fail`.\n"
    "Reply as JSON with keys: `type` (tap|type|scroll|swipe|back|home|open_app|wait|ask_user|approve|done|fail), "
    "`ref` (int|null), `text` (str|null), `direction` (up|down|left|right|null), `seconds` (num|null), "
    "`thought` (str), `expectation` (str)."
)


def build_action_user(goal: str, facts: dict, plan_summary: str,
                      history: list[str], screen: ScreenObservation,
                      max_elements: int = 70, max_text: int = 70) -> str:
    def clip(s: str) -> str:
        s = s or ""
        return s[:max_text] + "…" if len(s) > max_text else s

    shown = screen.elements[:max_elements]
    elements = [
        {
            "ref": e.ref, "role": e.role, "text": clip(e.text), "desc": clip(e.desc),
            "resource_id": e.resource_id, "editable": e.editable,
            "clickable": e.clickable, "scrollable": e.scrollable,
            "checked": e.checked, "enabled": e.enabled,
        }
        for e in shown
    ]
    more = "" if len(screen.elements) <= max_elements else \
        f"\n(+{len(screen.elements) - max_elements} more elements not shown — scroll if needed)"
    hist = "\n".join(f"  {h}" for h in history[-8:]) or "  (none yet)"
    return (
        "MODE: action\n"
        f"GOAL: {goal}\n"
        f"KNOWN FACTS: {json.dumps(facts)}\n"
        f"PLAN: {plan_summary}\n"
        f"RECENT ACTIONS:\n{hist}\n"
        f"CURRENT APP: {screen.app_package}  SCREEN: {screen.activity}\n"
        f"ELEMENTS (JSON): {json.dumps(elements)}{more}\n"
    )
