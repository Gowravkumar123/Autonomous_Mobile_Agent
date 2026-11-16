"""
Mock Android device — a scripted IRCTC phone.

It connects to the backend exactly like the real Kotlin client would (same WebSocket
protocol), serves accessibility-style screens, applies the brain's actions to a small
state machine, and auto-approves the payment handoff so you can watch the full
observe -> plan -> act -> policy-pause -> verify loop run end to end with no hardware.

Run the backend first, then:  python -m simulator.mock_device
"""
from __future__ import annotations

import argparse
import asyncio
import json

import httpx
import websockets

PKG = "cris.org.in.prs.ima"


def el(ref, role, text="", desc="", rid="", clickable=False, editable=False,
       scrollable=False, checked=None, enabled=True):
    return {"ref": ref, "role": role, "text": text, "desc": desc, "resource_id": rid,
            "clickable": clickable, "editable": editable, "scrollable": scrollable,
            "checked": checked, "enabled": enabled, "bounds": [0, 0, 0, 0]}


class IrctcPhone:
    """A tiny scripted model of the IRCTC booking flow."""

    def __init__(self):
        self.screen = "book"
        self.fields = {"from": "", "to": "", "date": ""}

    # ---- render current screen as an accessibility tree ----
    def observe(self) -> dict:
        s = self.screen
        if s == "book":
            els = [
                el(0, "EditText", self.fields["from"], "From station", "cris:id/from_station", editable=True),
                el(1, "EditText", self.fields["to"], "To station", "cris:id/to_station", editable=True),
                el(2, "EditText", self.fields["date"], "Journey date", "cris:id/journey_date", editable=True),
                el(3, "Button", "Search Trains", "", "cris:id/search", clickable=True),
            ]
            activity = "BookTicketActivity"
        elif s == "trains":
            els = [
                el(0, "TextView", "Shatabdi Exp 12007 · 06:00 → 16:30"),
                el(1, "TextView", "AC 3 Tier (3A) · ₹1,250 · AVAILABLE 42", "", "cris:id/avl_3a"),
                el(2, "Button", "Select", "Select AC 3 Tier train 12007", "cris:id/select_3a", clickable=True),
                el(3, "TextView", "Kaveri Exp 16021 · 21:00 → 09:10"),
                el(4, "Button", "Select", "Select AC 3 Tier train 16021", "cris:id/select_3a_2", clickable=True),
            ]
            activity = "TrainListActivity"
        elif s == "passenger":
            els = [
                el(0, "TextView", "Passenger 1: Gowrav, 28, Male (from saved list)"),
                el(1, "CheckBox", "AC 3 Tier (3A)", "Class 3A", "cris:id/class_3a",
                   clickable=True, checked=True),
                el(2, "Button", "Continue", "Continue to review", "cris:id/continue", clickable=True),
            ]
            activity = "PassengerDetailsActivity"
        elif s == "review":
            els = [
                el(0, "TextView", "Shatabdi Exp 12007 · AC 3 Tier · 2 passengers"),
                el(1, "TextView", "Total fare: ₹1,250", "", "cris:id/total"),
                el(2, "Button", "Proceed to Pay ₹1,250", "Proceed to payment",
                   "cris:id/pay", clickable=True),
            ]
            activity = "ReviewActivity"
        else:  # confirmed
            els = [
                el(0, "TextView", "Booking Confirmed!", "", "cris:id/status"),
                el(1, "TextView", "PNR 2654318907 · Shatabdi Exp 12007", "", "cris:id/pnr"),
                el(2, "TextView", "Transaction ID TXN99812 · ₹1,250 paid"),
            ]
            activity = "ConfirmationActivity"
        return {"screen": {"app_package": PKG, "activity": activity, "elements": els,
                           "is_keyboard_open": False}}

    # ---- apply a brain action ----
    def act(self, action: dict) -> dict:
        t = action.get("type")
        ref = action.get("ref")
        if self.screen == "book":
            if t == "type" and ref in (0, 1, 2):
                self.fields[{0: "from", 1: "to", 2: "date"}[ref]] = action.get("text", "")
            elif t == "tap" and ref == 3 and all(self.fields.values()):
                self.screen = "trains"
        elif self.screen == "trains" and t == "tap" and ref in (2, 4):
            self.screen = "passenger"
        elif self.screen == "passenger" and t == "tap" and ref == 2:
            self.screen = "review"
        # On the review screen a tap on "Proceed to Pay" is intercepted by the policy
        # engine and never reaches us as an `act`; the user completes it via `handoff`.
        return {"ok": True, "screen": self.screen}

    def handoff(self) -> dict:
        # The user "paid" — advance to the verified confirmation screen.
        self.screen = "confirmed"
        return {"ok": True}


async def run(base_url: str, prompt: str, device_id: str):
    ws_url = base_url.replace("http", "ws", 1) + f"/ws/device/{device_id}"
    phone = IrctcPhone()
    async with websockets.connect(ws_url) as ws:
        print(f"[sim] connected to {ws_url}")

        # Kick off the task via REST.
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{base_url}/tasks",
                                  json={"prompt": prompt, "device_id": device_id})
            task_id = r.json()["task_id"]
        print(f"[sim] created task {task_id}: {prompt!r}")

        async for raw in ws:
            msg = json.loads(raw)
            mtype = msg.get("type")
            rid = msg.get("rid")

            if mtype == "observe":
                payload = phone.observe()
                print(f"[sim] observe -> screen '{phone.screen}'")
                await ws.send(json.dumps({"rid": rid, "payload": payload}))
            elif mtype == "act":
                action = msg.get("action", {})
                print(f"[sim] act <- {action.get('type')} ref={action.get('ref')} "
                      f"text={action.get('text')!r}")
                payload = phone.act(action)
                await ws.send(json.dumps({"rid": rid, "payload": payload}))
            elif mtype == "handoff":
                print("[sim] >>> PAYMENT HANDOFF: user completes payment (auto-simulated) <<<")
                payload = phone.handoff()
                await ws.send(json.dumps({"rid": rid, "payload": payload}))
            elif mtype == "ask_user":
                questions = msg.get("questions", [])
                print(f"[sim] >>> AGENT ASKS: {questions} <<<")
                print("[sim] auto-answering with saved passenger details")
                await ws.send(json.dumps({
                    "type": "answer", "task_id": msg.get("task_id"),
                    "facts": {"passenger": "Gowrav, 28, Male (saved IRCTC passenger)",
                              "berth": "Lower"}}))
            elif mtype == "approval_request":
                print(f"[sim] >>> APPROVAL REQUESTED: {msg.get('title')} — "
                      f"{msg.get('reason')} (amount={msg.get('amount')}) <<<")
                print("[sim] auto-approving (user tapped 'I'll take over')")
                await ws.send(json.dumps({"type": "approval_response",
                                          "task_id": msg.get("task_id"), "approved": True}))
            else:
                # status / summary pushes
                if mtype == "status":
                    print(f"[sim] status: {msg}")

            # Stop once the task is done.
            if phone.screen == "confirmed":
                # Give the brain a moment to observe the confirmation and verify.
                await asyncio.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--device-id", default="sim-device")
    ap.add_argument("--prompt", default=(
        "Book two train tickets from Hyderabad to Bengaluru for tomorrow. "
        "Prefer AC 3-tier. Compare suitable trains, enter the passenger details "
        "and stop before payment."))
    args = ap.parse_args()
    try:
        asyncio.run(run(args.url, args.prompt, args.device_id))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
