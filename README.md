# Autonomous Personal AI Agent

A personal agent that takes a natural-language instruction and completes multi-step
tasks on an Android phone by **operating the installed apps through the accessibility
tree** — booking train/movie tickets, ordering food at the cheapest effective price,
sending a message, and more. It **plans, acts, observes after every step, and hands
control back to you before any payment, OTP, or sensitive action.**

> Reasoning lives in the cloud brain; **authority lives in a deterministic policy
> engine** the model cannot talk its way past. That separation is the whole safety story.

---

## Status — what works today

| Piece | State |
|------|-------|
| Python brain (FastAPI + state machine) | ✅ Runs, tested |
| Groq (primary) → Gemini Flash (backup) → heuristic mock | ✅ Auto-fallback |
| Generic observe→plan→act→verify loop (works on *any* app) | ✅ |
| Skill playbooks: train (IRCTC), food+coupon compare, movie, messaging | ✅ |
| Deterministic policy / payment-pause engine | ✅ Unit-tested |
| Device **simulator** (full IRCTC flow, no phone needed) | ✅ Runs end-to-end |
| Android companion app (accessibility service + WS client) | ✅ Scaffold, builds in Android Studio |
| Real on-device booking against live apps | ⚠️ Needs per-app tuning + a real device |

**The IRCTC example runs today via the simulator** — form fill → train select →
passenger → **policy pause at the ₹ payment screen** → handoff → verified by PNR.

---

## LLM choice (why Groq + Gemini, not a local model)

For reliable multi-step planning over accessibility trees you want a 70B-class model.
Groq's free tier serves exactly that, fast. A 1–4B on-device model is "free" but drops
steps, emits malformed actions, and drains the battery — not good enough to be the
planner. So: **Groq primary, Gemini 2.0 Flash backup** (better vision for the
screenshot fallback), and a built-in heuristic mock so the system runs with no keys at
all. On-device models can come later as a cheap pre-filter, never as the brain.

---

## Architecture

```
 ┌─────────────┐   prompt    ┌──────────────────────── CLOUD BRAIN (Python) ───────────────────────┐
 │  Companion  │────────────▶│  Intent → Plan → [ Observe → Decide(LLM) → POLICY → Act → Verify ]   │
 │  app (UI)   │◀────────────│         LLM router: Groq ▶ Gemini ▶ mock      │                      │
 └─────────────┘  status/    │                                  ▲           ▼                       │
        │         approvals   │                      PolicyEngine (deterministic): risk 1–4,         │
        │                     │                      converts payment/OTP/commit → USER APPROVAL     │
        ▼                     └──────────────────────────────────────────────────────────────────────┘
 ┌─────────────┐   WebSocket (observe / act / handoff)        ▲
 │ Accessibility│◀───────────────────────────────────────────┘
 │  Service     │   serializes the live screen, performs taps/typing/scroll,
 │ (Kotlin)     │   hands control to YOU for payment & authentication
 └─────────────┘
```

Full details in [ARCHITECTURE.md](ARCHITECTURE.md). The device↔brain wire format is in
[docs/PROTOCOL.md](docs/PROTOCOL.md).

---

## Repo layout

```
backend/        Python agent brain (FastAPI + LangGraph-style controller)
  app/
    agent/      intent, planner, controller loop, state machine, memory, verifier
    llm/        provider interface + Groq, Gemini, mock, router
    policy/     deterministic risk/approval engine  ← the safety boundary
    skills/     reusable playbooks (train, food, movie, messaging, generic)
    mobile/     the device↔brain schema (the contract)
    main.py     REST + WebSocket server
  simulator/    scripted IRCTC phone — run the demo with no hardware
  tests/        policy unit tests + in-process end-to-end
android/        Kotlin companion app (AccessibilityService controller + UI)
```

---

## Quickstart (no phone, no API key)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows (PowerShell/CMD)
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# 1) run the brain
python -m uvicorn app.main:app --port 8000

# 2) in a second terminal, run the simulated phone + IRCTC task
cd backend
.venv\Scripts\activate
python -m simulator.mock_device
```

You'll watch the agent fill the form, pick a train, reach the payment screen, **pause
for approval**, hand off, and verify the booking by PNR.

Run the tests:

```bash
python -m tests.test_policy    # safety-engine unit tests
python -m tests.test_flow      # full brain, in-process, no server
```

## Turn on the real models

Copy `backend/.env.example` to `backend/.env` and add either key (or both):

```
GROQ_API_KEY=gsk_...        # https://console.groq.com/keys   (free)
GEMINI_API_KEY=AIza...      # https://aistudio.google.com/apikey (free)
```

Restart the server — `/health` will report `"llm":"groq"` (or `gemini`). With no keys
it stays on `mock`, so nothing breaks.

## Run the real Android app

1. Open `android/` in **Android Studio** (it provisions the Gradle wrapper and a
   launcher icon automatically).
2. Set your backend address in `android/app/build.gradle.kts` (`BACKEND_WS_URL` /
   `BACKEND_HTTP_URL`). `10.0.2.2` already points to your host from the emulator.
3. Run the app on a device/emulator → tap **Enable the controller** → turn the service
   on in Accessibility settings.
4. Type a task, tap **Run task**, and approve when it pauses.

---

## Safety model (enforced outside the LLM)

- **Payment / OTP / PIN / biometric → user-only.** The agent pauses and hands you the phone.
- **Send / submit / cancel / refund → explicit approval** before it happens.
- **The agent never types OTPs, PINs, CVVs or passwords** — the policy engine blocks it.
- **A payment screen is not proof of payment** — success needs a verified marker (PNR, order id…).
- **Emergency stop:** `POST /tasks/{id}/cancel`, or disable the Accessibility service.
- **Audit trail:** every attempt, approval and verification is logged (`backend/audit_logs/`).

See blueprint sections 6 & 10 — all of it is implemented in
[`backend/app/policy/engine.py`](backend/app/policy/engine.py).

---

## Roadmap

1. ✅ Brain + policy + simulator + Android scaffold (this milestone)
2. Real-device IRCTC run; tune the tree serializer against the live app
3. Harden food "cheapest across apps" multi-app comparison
4. Postgres persistence, WebSocket dashboard, richer recovery
5. More skills, regression tests per app version
