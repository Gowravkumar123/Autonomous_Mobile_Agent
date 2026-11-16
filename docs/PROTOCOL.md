# Device ↔ Brain protocol

One WebSocket per paired device: `ws://<backend>/ws/device/{device_id}`.
The **brain drives**: it sends requests, the device replies. Requests are correlated by
an integer `rid`. All frames are JSON. This is the only coupling between the Kotlin
client and the Python brain — keep it small and stable.

## Brain → device

| `type` | fields | device must |
|--------|--------|-------------|
| `observe` | `rid` | reply with the current screen |
| `act` | `rid`, `action` | perform one action, reply `{ok}` |
| `handoff` | `rid`, `approval` | let the user complete payment/auth, reply `{ok}` when done |
| `approval_request` | `task_id`, `title`, `summary`, `amount`, `reason` | show approval UI (no `rid`; answer later) |
| `status` / `summary` | message fields | update the UI |

## Device → brain

**Response to a request** (observe / act / handoff):
```json
{ "rid": 7, "payload": { "screen": { ... } } }      // observe
{ "rid": 8, "payload": { "ok": true } }             // act / handoff
```

**Unsolicited** (answering a pause):
```json
{ "type": "approval_response", "task_id": "ab12", "approved": true, "note": "" }
{ "type": "answer", "task_id": "ab12", "facts": { "to_contact": "Mom" } }
```

## Screen observation (`observe` payload)

```json
{ "screen": {
  "app_package": "cris.org.in.prs.ima",
  "activity": "ReviewActivity",
  "is_keyboard_open": false,
  "elements": [
    { "ref": 0, "role": "EditText", "text": "", "desc": "From station",
      "resource_id": "cris:id/from_station", "editable": true, "clickable": false,
      "scrollable": false, "checked": null, "enabled": true, "bounds": [0,0,0,0] },
    { "ref": 2, "role": "Button", "text": "Proceed to Pay ₹1,250",
      "clickable": true, "resource_id": "cris:id/pay" }
  ]
}}
```

`ref` is a stable index **for that frame only**; the device keeps a parallel
`ref → AccessibilityNodeInfo` map so actions resolve by `ref`, never by pixels.

## Action (`act.action`)

```json
{ "type": "tap", "ref": 3, "text": null, "direction": null, "seconds": null,
  "thought": "Search for trains", "expectation": "A train list appears" }
```

`type` ∈ `tap | type | scroll | swipe | back | home | open_app | wait`.
(The control signals `ask_user | approve | done | fail` are handled inside the brain and
never sent to the device as an `act`.)

## The approval round-trip

1. The model proposes e.g. `tap "Proceed to Pay"`.
2. The **policy engine** classifies it level 4 and **rewrites it to an approval** — the
   phone never receives that tap.
3. Brain → device `approval_request`; brain → device `handoff` once approved.
4. The **user** performs the payment; the device replies `{ok}` to `handoff`.
5. Brain re-observes and only reports success on a verified marker (PNR, order id…).

## REST control plane (companion UI)

```
POST /tasks                 {prompt, device_id?}      -> {task_id, state}
GET  /tasks                                            -> list
GET  /tasks/{id}                                       -> full task + events
POST /tasks/{id}/approve    {approved, note}
POST /tasks/{id}/answer     {facts}
POST /tasks/{id}/cancel                                 (emergency stop)
GET  /health                                           -> active LLM, devices
GET  /skills                                           -> registered playbooks
WS   /ws/events                                         -> live task event stream
```
