# Backend — the agent brain

FastAPI + an async state-machine controller. Plans tasks, drives the device over a
WebSocket, enforces the policy engine, verifies results.

## Run

```bash
python -m venv .venv && .venv\Scripts\activate       # Windows
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000
```

No API key needed — it falls back to a heuristic mock LLM. Add real models in `.env`
(copy from `.env.example`): `GROQ_API_KEY` (primary) and/or `GEMINI_API_KEY` (backup).

## Demo without a phone

```bash
python -m simulator.mock_device
```
Runs the full IRCTC booking against a scripted device: form fill → train select →
passenger → **policy pause at payment** → handoff → verified by PNR.

## Tests

```bash
python -m tests.test_policy    # the safety boundary, directly
python -m tests.test_flow      # full brain end-to-end, in-process
```

## Layout

```
app/
  main.py        FastAPI: REST control plane + /ws/device + /ws/events
  config.py      settings (.env)
  agent/         intent, planner, controller (the loop), state, memory, verifier
  llm/           base + groq + gemini + mock + router(fallback) + prompts
  policy/        engine.py — deterministic risk 1–4 & approval rewriting
  skills/        base + registry (train, food, movie, messaging, generic)
  mobile/        schema.py — the device↔brain contract
  store.py       in-memory task store + event fan-out  (swap for Postgres)
  ws.py          device session (rid-correlated request/response)
  audit/         append-only JSONL audit log (secrets redacted)
simulator/       scripted IRCTC phone
tests/           policy + end-to-end
```

## Extending

- **New skill:** add a `Skill(...)` in `skills/registry.py` — no core changes.
- **New model:** implement `LLMProvider` and insert it in `llm/router.py`.
- **Persistence:** `Task` is already a Pydantic model; back `store.py` with Postgres.
