"""Guards against the false-positive completion bug (WhatsApp 'delivered' ambient text)."""
from app.agent.state import Task
from app.agent.verifier import is_success
from app.mobile.schema import ScreenObservation, UiElement
from app.skills import registry


def _screen(*texts):
    return ScreenObservation(elements=[UiElement(ref=i, role="TextView", text=t)
                                       for i, t in enumerate(texts)])


def test_messaging_not_done_just_from_opening_whatsapp():
    """The original bug: 1 tap opens WhatsApp, chat list says 'delivered' -> was 'done'."""
    skill = registry.get("messaging")
    task = Task(skill="messaging", facts={"to_contact": "Shivani", "message": "Tirupati tickets"})
    task.history = ["tap #62"]  # only opened the app
    screen = _screen("Shivani", "ok delivered", "Mom", "see you delivered")
    assert is_success(screen, skill, task) is False


def test_messaging_not_done_before_approval_even_if_text_typed():
    skill = registry.get("messaging")
    task = Task(skill="messaging", facts={"message": "Tirupati tickets"})
    task.history = ["tap", "tap", "type 'Tirupati tickets'"]  # typed, not yet approved/sent
    task.approvals_granted = 0
    screen = _screen("Tirupati tickets", "delivered")
    assert is_success(screen, skill, task) is False


def test_messaging_done_after_approval_and_message_visible():
    skill = registry.get("messaging")
    task = Task(skill="messaging", facts={"message": "Tirupati tickets"})
    task.history = ["tap", "tap", "type 'Tirupati tickets'", "tap send"]
    task.approvals_granted = 1  # user approved the send
    screen = _screen("You: Tirupati tickets", "delivered")
    assert is_success(screen, skill, task) is True


def test_train_still_completes_on_pnr():
    skill = registry.get("train")
    task = Task(skill="train")
    task.history = ["tap"] * 6
    screen = _screen("Booking Confirmed", "PNR 2654318907")
    assert is_success(screen, skill, task) is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("All verifier tests passed.")
