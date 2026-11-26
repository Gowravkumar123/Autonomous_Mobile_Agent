"""
Policy / approval engine — DETERMINISTIC. The LLM proposes; this decides.

The model never grants itself permission. Every proposed action and the screen it
would act on are classified here, and anything at or above the "external commitment"
line is converted into a user-approval pause before the phone ever executes it.

Risk levels (from the blueprint):
  1  Read-only            -> auto
  2  Low-risk preparation -> auto (reversible, in-scope)
  3  External commitment  -> require explicit user approval
  4  Financial / auth     -> user-only; agent pauses and hands over control
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..mobile.schema import ActionType, AgentAction, ScreenObservation

# Phrases on a SCREEN that signal a payment / authentication context. Deliberately
# multi-word: bare words like "upi"/"otp" match payment-app icons and notifications on
# the home screen and caused false pauses. The action-button check below still catches
# "Pay"/"Proceed to pay"/"Place order" buttons, so these only need to cover secret-entry
# and payment-selection screens. ("proceed to pay"/"make payment" used to be here too,
# which escalated EVERY action — even picking a seat — on any page merely showing that
# button; tapping the button itself is still Level 4 via FINANCIAL_ACTION.)
FINANCIAL_SCREEN = [
    "enter upi pin", "upi pin", "card number", "enter cvv", "cvv", "net banking",
    "netbanking", "wallet pin", "one time password", "enter otp", "verify otp",
    "enter pin", "enter mpin", "payment method", "add money", "enter your pin",
]
# Words on a BUTTON the agent is about to tap that commit money.
FINANCIAL_ACTION = [
    "pay", "pay now", "proceed to pay", "make payment", "place order",
    "confirm & pay", "continue to payment", "checkout", "buy", "purchase",
]
# Button text that is an external commitment (level 3).
COMMIT_ACTION = [
    "submit", "confirm booking", "book now", "send", "apply now",
    "place request", "cancel order", "request refund", "confirm",
]
# Field labels we must NEVER type into.
SECRET_FIELDS = ["otp", "pin", "cvv", "password", "card number", "security code"]

APPROVE = ActionType.REQUEST_APPROVAL


@dataclass
class PolicyDecision:
    allow: bool                 # may the phone execute the (possibly rewritten) action?
    action: AgentAction         # the action to actually carry out / surface
    needs_approval: bool
    risk_level: int
    reason: str


class PolicyEngine:
    def __init__(self, spend_threshold: float = 0.0):
        # Amounts >= threshold always require approval. 0 => pause before ANY payment.
        self.spend_threshold = spend_threshold

    def evaluate(self, action: AgentAction, screen: ScreenObservation) -> PolicyDecision:
        screen_text = _screen_text(screen)

        # (a) The model itself asked to hand over / ask / finish. Honor it.
        if action.type in (ActionType.ASK_USER, ActionType.DONE, ActionType.FAIL):
            return PolicyDecision(True, action, action.type == ActionType.ASK_USER,
                                  1, "control/terminal signal")
        if action.type == ActionType.REQUEST_APPROVAL:
            return self._as_approval(action, 4, "model requested approval", screen_text)

        # (b) Never type secrets, even if asked to.
        if action.type == ActionType.TYPE:
            target = _element_label(action.ref, screen)
            if _contains(target, SECRET_FIELDS) or _contains(action.text or "", SECRET_FIELDS):
                return self._as_approval(
                    action, 4, "field requires a secret (OTP/PIN/CVV/password)", screen_text)

        # (c) Financial / authentication SCREEN -> level 4 pause, whatever the action.
        if _contains(screen_text, FINANCIAL_SCREEN):
            return self._as_approval(action, 4, "payment/authentication screen detected", screen_text)

        # (d) Tapping a control that commits money -> level 4. A coordinate tap (tap_xy) is
        # judged by what is UNDER the point (element bounds or OCR text) — found live: on
        # BookMyShow's seat map the "Pay ₹498" button exists only in the screenshot, and a
        # tap_xy on it used to pass as "read-only navigation".
        if action.type == ActionType.TAP_XY:
            label = _label_at(action.x, action.y, screen)
            if label is None and _shows_pay_button(screen):
                return self._as_approval(
                    action, 4, "tap on an unidentified spot while a pay button is on screen",
                    screen_text, _detect_amount(_all_text(screen)))
            label = label or ""
            if _contains_word(label, FINANCIAL_ACTION):
                amount = _detect_amount(label) or _detect_amount(_all_text(screen))
                if self._amount_requires_approval(amount):
                    return self._as_approval(action, 4, f"financial action: {label!r}",
                                             screen_text, amount)
            if _contains_word(label, COMMIT_ACTION):
                return self._as_approval(action, 3, f"external commitment: {label!r}", screen_text)
        if action.type == ActionType.TAP:
            label = _element_label(action.ref, screen)
            if _contains(label, FINANCIAL_ACTION):
                amount = _detect_amount(screen_text)
                if self._amount_requires_approval(amount):
                    return self._as_approval(action, 4, f"financial action: {label!r}",
                                             screen_text, amount)
            if _contains(label, COMMIT_ACTION):
                return self._as_approval(action, 3, f"external commitment: {label!r}", screen_text)

        # (e) Everything else is read-only or reversible preparation -> auto.
        action.risk_level = 2 if action.type in (ActionType.TYPE,) else 1
        action.risk_reason = "reversible preparation" if action.risk_level == 2 else "read-only navigation"
        return PolicyDecision(True, action, False, action.risk_level, action.risk_reason)

    # --------------------------------------------------------------------- #
    def _amount_requires_approval(self, amount: float | None) -> bool:
        if self.spend_threshold <= 0:
            return True  # pause before any payment
        if amount is None:
            return True  # unknown amount -> be safe
        return amount >= self.spend_threshold

    def _as_approval(self, action: AgentAction, level: int, reason: str,
                     screen_text: str, amount: float | None = None) -> PolicyDecision:
        rewritten = action.model_copy(deep=True)
        rewritten.type = APPROVE
        rewritten.risk_level = level
        rewritten.risk_reason = reason
        return PolicyDecision(False, rewritten, True, level, reason)


# --------------------------- text helpers --------------------------- #
def _screen_text(screen: ScreenObservation) -> str:
    return " ".join(
        f"{e.text} {e.desc} {e.resource_id}" for e in screen.elements
    ).lower()


def _element_label(ref: int | None, screen: ScreenObservation) -> str:
    if ref is None:
        return ""
    for e in screen.elements:
        if e.ref == ref:
            return f"{e.text} {e.desc} {e.resource_id}".lower()
    return ""


def _contains(haystack: str, needles: list[str]) -> bool:
    return any(n in haystack for n in needles)


def _contains_word(haystack: str, needles: list[str]) -> bool:
    """Whole-word match (OCR text: 'pay' must not match 'display' / 'paytm')."""
    return any(re.search(rf"(?<![a-z]){re.escape(n)}(?![a-z])", haystack) for n in needles)


def _all_text(screen: ScreenObservation) -> str:
    return (_screen_text(screen) + " " +
            " ".join(str(o.get("text", "")) for o in screen.ocr_items)).lower()


def _shows_pay_button(screen: ScreenObservation) -> bool:
    return any(_contains_word(str(o.get("text", "")).lower(), FINANCIAL_ACTION)
               for o in screen.ocr_items) or _contains_word(_screen_text(screen), FINANCIAL_ACTION)


def _label_at(x: float | None, y: float | None, screen: ScreenObservation) -> str | None:
    """What a tap at normalized (x, y) would hit: the smallest element whose bounds
    contain it, plus the nearest OCR text within a button-sized radius. None if nothing
    identifies the target."""
    if x is None or y is None:
        return None
    parts: list[str] = []
    w = max((e.bounds[2] for e in screen.elements if len(e.bounds) == 4), default=0)
    h = max((e.bounds[3] for e in screen.elements if len(e.bounds) == 4), default=0)
    if w > 0 and h > 0:
        px, py = x * w, y * h
        hits = [e for e in screen.elements if len(e.bounds) == 4
                and e.bounds[0] <= px <= e.bounds[2] and e.bounds[1] <= py <= e.bounds[3]
                and (e.bounds[2] - e.bounds[0]) * (e.bounds[3] - e.bounds[1]) < 0.9 * w * h]
        if hits:
            e = min(hits, key=lambda e: (e.bounds[2] - e.bounds[0]) * (e.bounds[3] - e.bounds[1]))
            label = f"{e.text} {e.desc} {e.resource_id}".strip()
            if label:
                parts.append(label)
    # OCR gives only text CENTRES and buttons are big: include EVERY text within a
    # button-sized radius (nearest first), so a tap on the edge of "Pay ₹498" still reads
    # as a pay tap even if a banner's text centre happens to be closer.
    near = []
    for o in screen.ocr_items:
        try:
            dx, dy = (float(o["x"]) - x) / 0.35, (float(o["y"]) - y) / 0.06
        except (KeyError, TypeError, ValueError):
            continue
        d = (dx * dx + dy * dy) ** 0.5
        if d <= 1.0:
            near.append((d, str(o.get("text", ""))))
    parts += [t for _, t in sorted(near)]
    return " ".join(parts).lower() if parts else None


def _detect_amount(text: str) -> float | None:
    # Matches ₹1,234.50 / Rs 1234 / INR 999 etc.
    m = re.search(r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d{1,2})?)", text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None
