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


def _bms_seat_map():
    """The live BookMyShow seat screen (2026-11-26): a sparse accessibility tree; the seat
    numbers and the 'Pay ₹498' button exist only in the screenshot (OCR)."""
    els = [UiElement(ref=0, role="FrameLayout", bounds=[0, 0, 1268, 2756]),
           UiElement(ref=1, role="ImageButton", desc="Back", clickable=True,
                     bounds=[40, 180, 150, 300]),
           UiElement(ref=2, role="TextView", text="Scene", bounds=[180, 190, 600, 250]),
           UiElement(ref=3, role="TextView", text="PVR: Preston, Gachibowli Hyderabad",
                     bounds=[180, 260, 900, 310])]
    ocr = [{"text": "11:00 PM", "x": 0.18, "y": 0.187}, {"text": "1", "x": 0.163, "y": 0.272},
           {"text": "2", "x": 0.236, "y": 0.272}, {"text": "3", "x": 0.31, "y": 0.272},
           {"text": "7", "x": 0.605, "y": 0.306}, {"text": "Available", "x": 0.5, "y": 0.83},
           {"text": "Enjoy B1G1 Ticket Free!* with Bandhan Bank", "x": 0.46, "y": 0.874},
           {"text": "Pay ₹498", "x": 0.5, "y": 0.945}]
    return ScreenObservation(app_package="com.bt.bms", activity="seats", elements=els,
                             ocr_items=ocr)


def test_tap_xy_on_ocr_pay_button_is_level4():
    """Live bug: tap_xy(0.50, 0.95) / (0.50, 0.90) on 'Pay ₹498' passed as navigation."""
    pe = PolicyEngine(spend_threshold=0)
    for x, y in ((0.50, 0.95), (0.50, 0.90), (0.62, 0.945)):
        d = pe.evaluate(AgentAction(type=ActionType.TAP_XY, x=x, y=y), _bms_seat_map())
        assert d.needs_approval and d.risk_level == 4 and not d.allow, (x, y, d.reason)
        assert "pay" in d.reason


def test_tap_xy_on_a_seat_is_allowed():
    pe = PolicyEngine(spend_threshold=0)
    d = pe.evaluate(AgentAction(type=ActionType.TAP_XY, x=0.165, y=0.27), _bms_seat_map())
    assert d.allow and not d.needs_approval, d.reason


def test_tap_xy_unidentified_spot_with_pay_on_screen_needs_approval():
    """Fail-safe: if nothing identifies the target and a pay button is showing, ask."""
    pe = PolicyEngine(spend_threshold=0)
    d = pe.evaluate(AgentAction(type=ActionType.TAP_XY, x=0.5, y=0.62), _bms_seat_map())
    assert d.needs_approval and d.risk_level == 4, d.reason


def test_tap_xy_element_hit_test_uses_bounds():
    pe = PolicyEngine(spend_threshold=0)
    screen = _screen(UiElement(ref=0, role="FrameLayout", bounds=[0, 0, 1000, 2000]),
                     UiElement(ref=1, role="Button", text="Book now", clickable=True,
                               bounds=[100, 1800, 900, 1950]),
                     UiElement(ref=2, role="TextView", text="Showtimes",
                               bounds=[100, 200, 900, 300]))
    d = pe.evaluate(AgentAction(type=ActionType.TAP_XY, x=0.5, y=0.93), screen)
    assert d.needs_approval and d.risk_level == 3, d.reason
    d = pe.evaluate(AgentAction(type=ActionType.TAP_XY, x=0.5, y=0.12), screen)
    assert d.allow, d.reason


def test_ocr_word_match_does_not_flag_display_or_paytm():
    pe = PolicyEngine(spend_threshold=0)
    screen = ScreenObservation(app_package="x", activity="y", elements=[],
                               ocr_items=[{"text": "Display settings", "x": 0.5, "y": 0.5},
                                          {"text": "Paytm offers", "x": 0.5, "y": 0.7}])
    d = pe.evaluate(AgentAction(type=ActionType.TAP_XY, x=0.5, y=0.5), screen)
    assert d.allow, d.reason


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All policy tests passed.")
