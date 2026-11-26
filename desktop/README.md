# OnePiece desktop companion (Windows)

Lets the agent drive Windows desktop apps. It pairs with the backend like the phone does and
speaks the same device protocol, so the ReAct graph, the policy gate, approvals and handoffs
work on the desktop unchanged.

## Setup

```powershell
cd desktop
pip install -r requirements.txt
# On the server/admin side: POST /v1/devices/pair/start  -> 6-digit code
python -m onepiece_desktop pair --server https://YOUR-SERVER --code 123456
python -m onepiece_desktop run --window "Notepad"      # or omit --window: foreground window
```

Start a task with `executor: "desktop"`, or phrase it as "on my desktop / laptop / PC …".
If several desktops are online, pass `device_id: "desktop-<host>"`.

## How it acts

- Uses **UI Automation patterns** (Invoke, Value, Toggle, Select, Scroll). They work on
  background or minimized windows and never move your mouse or keyboard, so you can keep
  working while it does. Enter is sent to the field's own window, not to whatever has focus.
- `--mouse` also allows real clicks and keystrokes for apps without automation support.
  This takes focus.
- Screenshots come from `PrintWindow` on the target window only. Other windows and the
  rest of the desktop are never captured.

## Safety controls (enforced on this PC, independent of the server)

| Control | Default | Config (`~/.onepiece/desktop.json`) |
|---|---|---|
| Process allowlist: other windows are reported only as blocked, with no content or screenshot | notepad, calc, mspaint, wordpad | `allowed_processes`, or `--allow x.exe` |
| Workspace boundary: any typed text that looks like a file path must be inside it | `~/.onepiece/workspace` | `workspaces` |
| Kill switch: refuses every action and the backend cancels the task. It stays on until reset | mouse in the top-left corner, or `python -m onepiece_desktop stop` | `reset` to clear |
| Password fields: values are never read, and the agent can't type into them | always | – |
| Action log: every action, refusal and handoff, with password text redacted | `~/.onepiece/desktop-actions.jsonl` | – |
| `open_app` only switches to open allowlisted windows or launches configured apps | notepad, calculator | `apps` |

Payment and sign-in steps (Level 4) pause with a handoff. You finish the step yourself, then
press Enter in the console (or create `~/.onepiece/RESUME`). Approvals and questions appear
in the console too.

## Tests

`cd backend && python -m tests.test_desktop`. This drives a minimized WinForms test app
(`tests/cinetest_app.ps1`) and runs a full end-to-end test against a real server over a
WebSocket. It never takes focus.
