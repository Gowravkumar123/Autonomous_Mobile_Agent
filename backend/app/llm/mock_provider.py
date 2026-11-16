"""
Heuristic mock LLM — no network, no key. It makes the whole pipeline runnable for
demos and tests. It is deliberately simple: it reads the MODE marker and the JSON
blocks embedded in the prompt and returns a plausible next action.

A real model (Groq / Gemini) replaces this with genuine reasoning; the contract
(same JSON in, same JSON out) is identical.
"""
from __future__ import annotations

import json
import re

from .base import LLMProvider

# Button-text hints, in priority order, that mean "move forward".
FORWARD_WORDS = [
    "proceed to pay", "pay now", "make payment", "place order", "pay ",
    "continue", "proceed", "next", "search", "book now", "book", "confirm",
    "add passenger", "add", "select", "apply", "submit", "send", "done",
]
# Text that means the task reached a successful end state.
SUCCESS_WORDS = ["confirmed", "booked", "pnr", "order placed", "success",
                 "delivered", "your order", "ticket booked", "message sent", "sent"]
# Field label/id hints -> fact keys we might fill them with.
FIELD_HINTS = {
    "from": ["from", "source", "origin", "boarding"],
    "to": ["to", "destination", "arrival", "dest"],
    "date": ["date", "journey", "when", "day"],
    "search": ["search", "query", "find"],
    "message": ["message", "text", "type a message", "compose"],
    "to_contact": ["contact", "recipient", "name", "to:"],
    "item": ["dish", "food", "item", "restaurant"],
}


class MockProvider(LLMProvider):
    name = "mock"

    async def complete(self, system: str, user: str, *, temperature: float = 0.2,
                       max_tokens: int = 1024, json_mode: bool = False) -> str:
        mode = _mode(user)
        if mode == "intent":
            return json.dumps(self._intent(user))
        if mode == "plan":
            return json.dumps(self._plan(user))
        return json.dumps(self._action(user))

    # ---------------- intent ----------------
    def _intent(self, user: str) -> dict:
        req = _after(user, "USER REQUEST:").lower()
        facts: dict = {}
        skill = "generic"
        app = ""
        if any(w in req for w in ["train", "irctc", "ticket from", "tier", "railway"]):
            skill, app = "train", "cris.org.in.prs.ima"  # IRCTC Rail Connect
            m = re.search(r"from\s+([a-z ]+?)\s+to\s+([a-z ]+?)(?:\s+for|\s+on|$)", req)
            if m:
                facts["from"] = m.group(1).strip().title()
                facts["to"] = m.group(2).strip().title()
            if "tomorrow" in req:
                facts["date"] = "tomorrow"
            if "3" in req and "tier" in req:
                facts["class"] = "AC 3 Tier (3A)"
            words = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
            m2 = re.search(r"(\d+|one|two|three|four|five)\s+(?:train\s+)?ticket", req)
            facts["passengers"] = (words.get(m2.group(1), m2.group(1)) if m2 else "1")
        elif any(w in req for w in ["eat", "food", "biriyani", "biryani", "order", "hungry", "pizza"]):
            skill, app = "food", "in.swiggy.android"
            m = re.search(r"(?:eat|want|order)\s+(?:a\s+|some\s+)?([a-z ]+)", req)
            if m:
                facts["item"] = m.group(1).strip()
        elif any(w in req for w in ["movie", "cinema", "show", "pvr", "bookmyshow"]):
            skill, app = "movie", "com.bt.bms"
        elif any(w in req for w in ["message", "text", "whatsapp", "good morning", "send", "tell"]):
            skill, app = "messaging", "com.whatsapp"
            m = re.search(r"(?:send|text|message|tell)\s+([a-z]+)", req)
            if m:
                facts["to_contact"] = m.group(1).strip().title()
            if "good morning" in req:
                facts["message"] = "Good morning!"
        goal = _after(user, "USER REQUEST:").strip()
        return {"goal": goal, "app_hint": app, "skill": skill,
                "facts": facts, "clarifications": []}

    # ---------------- plan ----------------
    def _plan(self, user: str) -> dict:
        return {
            "steps": [
                "Open the target app",
                "Reach the main search / compose screen",
                "Enter the required details",
                "Review options and make a selection",
                "Prepare the final details up to the approval boundary",
            ],
            "approval_boundary": "Stop before any payment, OTP, or final submission "
                                 "and hand control to the user.",
        }

    # ---------------- action ----------------
    def _action(self, user: str) -> dict:
        facts = _json_block(user, "KNOWN FACTS:") or {}
        elements = _json_block(user, "ELEMENTS (JSON):") or []
        history = _after(user, "RECENT ACTIONS:")

        # 0. Success?  -> done
        joined = " ".join(str(e.get("text", "")) + " " + str(e.get("desc", "")) for e in elements).lower()
        if any(w in joined for w in SUCCESS_WORDS):
            return _act("done", thought="Screen shows a success/confirmation state.")

        # 1. Fill the first empty editable field we can map to a fact.
        for e in elements:
            if not e.get("editable") or (e.get("text") or "").strip():
                continue
            label = f"{e.get('text','')} {e.get('desc','')} {e.get('resource_id','')}".lower()
            val = _match_fact(label, facts)
            if val and f"type '{val}'".lower() not in history.lower():
                return _act("type", ref=e["ref"], text=val,
                            thought=f"Fill field {e['ref']} with {val!r}.",
                            expectation=f"Field shows {val!r}.")

        # 2. Tap a forward / confirm control (payment buttons get caught by policy).
        best = _forward_button(elements)
        if best is not None:
            return _act("tap", ref=best["ref"],
                        thought=f"Tap forward control {best['ref']!r}.",
                        expectation="The app advances to the next screen.")

        # 3. Otherwise scroll to reveal more.
        if any(e.get("scrollable") for e in elements):
            return _act("scroll", direction="down", thought="Reveal more of the screen.")

        return _act("ask_user", text="I don't see a clear next step on this screen. "
                                     "Could you tell me what to tap?",
                    thought="No actionable control found.")


# ------------------------- helpers ------------------------- #
def _mode(user: str) -> str:
    m = re.search(r"MODE:\s*(\w+)", user)
    return m.group(1) if m else "action"


def _after(text: str, marker: str) -> str:
    i = text.find(marker)
    return text[i + len(marker):] if i != -1 else ""


def _json_block(text: str, marker: str):
    frag = _after(text, marker).strip()
    if not frag:
        return None
    # Parse the first balanced JSON value after the marker.
    opener = frag[0]
    if opener not in "[{":
        return None
    closer = "]" if opener == "[" else "}"
    depth = 0
    for i, ch in enumerate(frag):
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(frag[:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _match_fact(label: str, facts: dict):
    # Direct key appears in the label.
    for key, val in facts.items():
        if key.lower() in label:
            return str(val)
    # Synonyms.
    for key, syns in FIELD_HINTS.items():
        if key in facts and any(s in label for s in syns):
            return str(facts[key])
    # Generic search box with a single obvious fact.
    if any(s in label for s in FIELD_HINTS["search"]):
        for k in ("item", "to_contact", "from"):
            if k in facts:
                return str(facts[k])
    return None


def _forward_button(elements: list[dict]):
    candidates = [e for e in elements if e.get("clickable") and e.get("enabled", True)]
    for word in FORWARD_WORDS:
        for e in candidates:
            label = f"{e.get('text','')} {e.get('desc','')}".lower()
            if word in label:
                return e
    return None


def _act(type_: str, ref=None, text=None, direction=None, seconds=None,
         thought="", expectation="") -> dict:
    return {"type": type_, "ref": ref, "text": text, "direction": direction,
            "seconds": seconds, "thought": thought, "expectation": expectation}
