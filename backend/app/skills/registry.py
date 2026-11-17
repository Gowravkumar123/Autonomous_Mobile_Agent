"""All built-in skills, registered by key. Mirrors blueprint section 5."""
from __future__ import annotations

from .base import Skill

_SKILLS: dict[str, Skill] = {}


def register(skill: Skill) -> None:
    _SKILLS[skill.key] = skill


def get(key: str) -> Skill:
    return _SKILLS.get(key, _SKILLS["generic"])


def all_skills() -> list[Skill]:
    return list(_SKILLS.values())


# --------------------------------------------------------------------------- #
register(Skill(
    key="generic",
    title="Generic task",
    app_packages=[],
    playbook=(
        "No specialized playbook. Observe the screen, fill required fields, tap the "
        "obvious forward control, and re-check after each step. Stop and ask the user "
        "if anything is ambiguous or irreversible."
    ),
    approval_boundary="Stop before any payment, submission, or message that leaves the device.",
    # Generic tasks have no reliable text marker; rely on the model's own `done`
    # signal rather than ambient words that cause false positives.
    success_markers=[],
))

register(Skill(
    key="train",
    title="Train booking (IRCTC)",
    app_packages=["cris.org.in.prs.ima"],   # IRCTC Rail Connect
    playbook=(
        "1. Open IRCTC Rail Connect and reach 'Book Ticket'.\n"
        "2. Set the 'From' and 'To' stations and the journey date.\n"
        "3. Run the search; inspect trains, timings, fare and availability.\n"
        "4. Pick a train matching the user's class preference (e.g. AC 3 Tier / 3A) "
        "with available seats.\n"
        "5. Enter passenger name, age, gender and berth preference.\n"
        "6. Review the booking summary (train, date, class, fare, passengers).\n"
        "7. STOP at the payment screen — do not pay."
    ),
    approval_boundary="Pause before payment and any final booking commitment; show the "
                      "train, date, class, passengers and total fare.",
    success_markers=["pnr", "booking confirmed", "ticket booked", "transaction id"],
    required_facts=["from", "to", "date"],
))

register(Skill(
    key="food",
    title="Food ordering (cheapest across apps)",
    app_packages=["in.swiggy.android", "com.application.zomato"],
    playbook=(
        "Goal: get the chosen dish for the lowest effective price.\n"
        "1. For EACH food app the user has installed (e.g. Swiggy, then Zomato):\n"
        "   a. Open the app and search for the dish / restaurant.\n"
        "   b. Note the item price, delivery fee, taxes and ETA.\n"
        "   c. Open the coupons/offers section and record the best applicable coupon "
        "and the resulting discounted total.\n"
        "2. Compare the final out-the-door totals across apps.\n"
        "3. Open the cheapest option, build the cart and apply the best coupon.\n"
        "4. STOP before placing the paid order; report the comparison and the winner."
    ),
    approval_boundary="Pause before placing any paid order; show the app chosen, item, "
                      "coupon applied, and the compared totals so the user can confirm.",
    success_markers=["order placed", "order confirmed", "arriving", "out for delivery"],
    required_facts=["item"],
))

register(Skill(
    key="movie",
    title="Movie booking",
    app_packages=["com.bt.bms"],   # BookMyShow
    playbook=(
        "1. Open the movies app and search the movie (or browse what's near).\n"
        "2. Compare cinemas, showtimes and ticket prices.\n"
        "3. Select a show that fits the user's time/location preference.\n"
        "4. Choose the number of seats and pick seats.\n"
        "5. Review the summary. STOP before payment."
    ),
    approval_boundary="Pause before payment; show cinema, movie, showtime, seats and total.",
    # "ticket" is too generic (it appears on every movie page as "Book tickets"), which
    # falsely completed the task. A real booking only confirms AFTER payment.
    success_markers=["booking confirmed", "booking id", "enjoy the show", "ticket confirmed"],
    min_actions_before_success=5,
))

register(Skill(
    key="messaging",
    title="Send a message",
    app_packages=["com.whatsapp", "com.google.android.apps.messaging"],
    playbook=(
        "1. Open the messaging app.\n"
        "2. Find the intended contact (search by name).\n"
        "3. Open the chat and type the message text in the compose box.\n"
        "4. Sending a message leaves the device — treat 'Send' as an external "
        "commitment and request approval before tapping it, unless the user explicitly "
        "pre-authorized sending."
    ),
    approval_boundary="Confirm the exact recipient and message text before sending.",
    # No text markers: WhatsApp signals "sent" with icons, and words like "delivered"
    # are ambient in the chat list. Success = the message text is on screen AND the
    # send was approved (so it can't false-complete just from opening the app).
    success_markers=[],
    success_requires_fact="message",
    confirm_after_approval=True,
    min_actions_before_success=3,
    required_facts=["to_contact", "message"],
))
