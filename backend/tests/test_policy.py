"""Policy engine is the safety boundary — test it directly, no LLM involved."""
from app.mobile.schema import ActionType, AgentAction, ScreenObservation, UiElement
from app.policy.engine import PolicyEngine


def _screen(*els):
    return ScreenObservation(app_package="x", activity="y", elements=list(els))


def test_payment_screen_forces_approval():
    pe = PolicyEngine(spend_threshold=0)
    screen = _screen(UiElement(ref=0, role="TextView", text="Enter UPI PIN"))
    action = AgentAction(type=ActionType.TAP, ref=0)
    d = pe.evaluate(action, screen)
    assert d.needs_approval and d.risk_level == 4
    assert d.action.type == ActionType.REQUEST_APPROVAL


def test_proceed_to_pay_button_blocked():
    pe = PolicyEngine(spend_threshold=0)
    screen = _screen(UiElement(ref=0, role="Button", text="Proceed to Pay ₹1,250", clickable=True))
    action = AgentAction(type=ActionType.TAP, ref=0)
    d = pe.evaluate(action, screen)
    assert d.needs_approval and d.risk_level == 4


def test_seat_pick_allowed_while_pay_button_visible():
    """A page that SHOWS 'Proceed to pay' must not escalate picking a seat; tapping the
    pay button itself still must."""
    pe = PolicyEngine(spend_threshold=0)
    screen = _screen(
        UiElement(ref=0, role="CheckBox", text="Seat A1", clickable=True),
        UiElement(ref=1, role="Button", text="Proceed to pay ₹500", clickable=True))
    seat = pe.evaluate(AgentAction(type=ActionType.TAP, ref=0), screen)
    assert seat.allow and not seat.needs_approval
    pay = pe.evaluate(AgentAction(type=ActionType.TAP, ref=1), screen)
    assert pay.needs_approval and pay.risk_level == 4


def test_never_type_otp():
    pe = PolicyEngine(spend_threshold=0)
    screen = _screen(UiElement(ref=0, role="EditText", desc="OTP", editable=True))
    action = AgentAction(type=ActionType.TYPE, ref=0, text="123456")
    d = pe.evaluate(action, screen)
    assert d.needs_approval and d.risk_level == 4


def test_send_is_external_commitment():
    pe = PolicyEngine(spend_threshold=0)
    screen = _screen(UiElement(ref=0, role="Button", text="Send", clickable=True))
    action = AgentAction(type=ActionType.TAP, ref=0)
    d = pe.evaluate(action, screen)
    assert d.needs_approval and d.risk_level == 3


def test_normal_navigation_auto_allowed():
    pe = PolicyEngine(spend_threshold=0)
    screen = _screen(UiElement(ref=0, role="Button", text="Search Trains", clickable=True))
    action = AgentAction(type=ActionType.TAP, ref=0)
    d = pe.evaluate(action, screen)
    assert d.allow and not d.needs_approval


def test_typing_a_city_is_reversible_prep():
    pe = PolicyEngine(spend_threshold=0)
    screen = _screen(UiElement(ref=0, role="EditText", desc="From station", editable=True))
    action = AgentAction(type=ActionType.TYPE, ref=0, text="Hyderabad")
    d = pe.evaluate(action, screen)
    assert d.allow and d.risk_level == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All policy tests passed.")
