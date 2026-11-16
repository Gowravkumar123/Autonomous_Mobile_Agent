# Architecture

Five layers, mapping 1:1 to the blueprint. The guiding boundary: **the AI proposes,
deterministic code disposes.**

## 1. Companion app (Kotlin) — `android/`
Prompts, task progress, approvals, emergency stop, and the **AccessibilityService**
that is the controlled execution device. It only ever:
- serializes the current screen into the accessibility tree (never screenshots-only), and
- performs ONE backend-approved action at a time,
- pausing and handing you the phone for payment/auth.

## 2. Agent orchestrator (Python) — `backend/app/agent/`
A per-task async **controller** implementing an explicit state machine
(`CREATED → PLANNING → RUNNING → WAITING_FOR_USER → VERIFYING → COMPLETED/…`). The core
loop:

```
observe  →  decide (LLM: one action)  →  POLICY CHECK  →  act | approval-pause  →  verify
   ▲                                                                                  │
   └──────────────────────────── re-observe after every meaningful action ───────────┘
```

Each step has an expectation and the loop re-reads the screen instead of assuming the
action worked. `MAX_STEPS` + timeouts bound it. This is the "state machine, not an
unconstrained sequence of LLM actions" the blueprint calls for; swapping in LangGraph
means wrapping these same nodes.

- **Intent analyzer** (`intent.py`): prompt → `{goal, skill, app_hint, facts, clarifications}`.
- **Planner** (`intent.py`): structured goal + skill playbook → ordered plan + approval boundary.
- **Controller** (`controller.py`): the loop, approvals, handoff, verification, audit.
- **Memory** (`memory.py`): user-approved preferences; refuses to store secrets.
- **Verifier** (`verifier.py`): success only on a trusted on-screen marker.

## 3. Mobile automation
Device side (Kotlin `service/`) + the `mobile/schema.py` contract. The brain is device-
agnostic: anything that speaks the WebSocket protocol (the real app, or the `simulator/`)
is a valid device.

## 4. Policy & approval — `backend/app/policy/engine.py`
**Deterministic, LLM-free.** Classifies every proposed action against the current screen:

| Level | Examples | Policy |
|------|----------|--------|
| 1 Read-only | search, browse, read status | auto |
| 2 Low-risk prep | fill a field, pick a filter | auto (reversible) |
| 3 External commitment | submit, send, cancel, refund | **explicit approval** |
| 4 Financial / auth | UPI/card/netbanking, OTP/PIN/biometric | **user-only, agent hands over** |

It rewrites a level 3/4 action into an approval *before* the phone sees it, blocks typing
into secret fields, and treats unknown payment amounts as "must approve."

## 5. Verification, memory & audit
`verifier.py` + `audit/logger.py`. Append-only JSONL per task recording what was
attempted, approved, and verified — with secrets redacted.

## LLM layer — `backend/app/llm/`
A tiny `LLMProvider` interface with three implementations behind a **router** that falls
back on any failure: **Groq → Gemini → mock**. Prompts live in one file (`prompts.py`)
and pass screen elements as JSON (structured beats prose for reliable tool use).

## Skills — `backend/app/skills/`
A skill is a **playbook, not a script**: which app, the usual steps, the fields that
matter, the approval boundary, and success markers. The generic loop already handles any
app; a skill just raises reliability for common tasks. Adding one never touches the core.

## Why this shape
- **Reasoning ≠ authority.** A jailbroken or confused model still cannot pay or submit —
  the policy engine is outside it.
- **Observable + recoverable.** Explicit states + re-observation + bounded retries let a
  task pause across a disconnect and resume safely.
- **App-independent.** New skills and new apps are data/playbooks, not forks of the engine.
