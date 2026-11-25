"""
Auth E2E through the real FastAPI app (in-memory, mock LLM, no network).

Run:  python -m tests.test_auth
"""
from __future__ import annotations

import os

# Must be set before the app/config is imported (settings are cached).
os.environ["AUTH_SECRET"] = "test-secret-" + "x" * 40
os.environ["DATABASE_URL"] = ""
os.environ["GROQ_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""

from fastapi.testclient import TestClient            # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from app.main import app                              # noqa: E402

PASS = 0


def check(cond: bool, label: str) -> None:
    global PASS
    assert cond, f"FAIL: {label}"
    PASS += 1
    print(f"PASS {label}")


def bearer(t: str) -> dict:
    return {"Authorization": f"Bearer {t}"}


def main() -> None:
    with TestClient(app) as c:
        auth = app.state.auth
        check(auth.enabled, "auth enabled when AUTH_SECRET is set")

        # ---- public vs protected ----
        h = c.get("/health").json()
        check(h["auth"] == "on" and "devices" not in h, "/health public, hides devices when anonymous")
        check(c.get("/tasks").status_code == 401, "/tasks without token -> 401")
        check(c.get("/tasks", headers=bearer("garbage")).status_code == 401, "bad token -> 401")
        check(c.post("/voice/tts", json={"text": "hi"}).status_code == 401,
              "/voice/tts requires auth (protects the Groq quota)")
        check(c.post("/stop").status_code == 401, "/stop requires auth")

        admin = auth.issue_admin()
        check(c.get("/tasks", headers=bearer(admin)).status_code == 200, "admin token -> /tasks 200")
        check("devices" in c.get("/health", headers=bearer(admin)).json(),
              "/health shows devices to signed-in callers")

        # ---- pairing ----
        check(c.post("/v1/devices/pair/start").status_code == 401, "pair/start needs auth")
        code = c.post("/v1/devices/pair/start", headers=bearer(admin)).json()["code"]
        check(len(code) == 6 and code.isdigit(), "admin gets a 6-digit pairing code")
        wrong = "000000" if code != "000000" else "111111"
        check(c.post("/v1/devices/pair", json={"code": wrong, "device_id": "phone-A"}).status_code == 400,
              "wrong pairing code rejected")
        r = c.post("/v1/devices/pair", json={"code": code, "device_id": "phone-A"})
        check(r.status_code == 200 and r.json()["token"], "correct code pairs the device")
        dev_a = r.json()["token"]
        check(c.post("/v1/devices/pair", json={"code": code, "device_id": "phone-B"}).status_code == 400,
              "pairing code is single-use")

        # ---- device scope ----
        check(c.post("/v1/devices/pair/start", headers=bearer(dev_a)).status_code == 403,
              "device token cannot mint pairing codes (admin only)")
        check(c.get("/v1/devices", headers=bearer(dev_a)).status_code == 403,
              "device token cannot list devices")
        t_admin = c.post("/tasks", headers=bearer(admin),
                         json={"prompt": "open camera", "device_id": "phone-Z"}).json()["task_id"]
        t_dev = c.post("/tasks", headers=bearer(dev_a),
                       json={"prompt": "open camera", "device_id": "phone-Z"}).json()["task_id"]
        mine = {t["task_id"]: t for t in c.get("/tasks", headers=bearer(dev_a)).json()}
        check(t_dev in mine and mine[t_dev]["device_id"] == "phone-A",
              "device-created task is forced onto the device itself")
        check(t_admin not in mine, "device cannot see another device's tasks")
        check(c.get(f"/tasks/{t_admin}", headers=bearer(dev_a)).status_code == 404,
              "device cannot read another device's task")
        check(c.post(f"/tasks/{t_admin}/cancel", headers=bearer(dev_a)).status_code == 404,
              "device cannot cancel another device's task")
        check(c.post("/voice/tts", json={"text": ""}, headers=bearer(dev_a)).status_code in (200, 503),
              "device token may use voice")

        # ---- websockets ----
        try:
            with c.websocket_connect("/ws/device/phone-A"):
                pass
            check(False, "ws without token must be rejected")
        except WebSocketDisconnect:
            check(True, "ws without token rejected")
        try:
            with c.websocket_connect("/ws/device/phone-B", headers=bearer(dev_a)):
                pass
            check(False, "ws for another device must be rejected")
        except WebSocketDisconnect:
            check(True, "device token cannot open another device's socket")
        with c.websocket_connect("/ws/device/phone-A", headers=bearer(dev_a)):
            check("phone-A" in app.state.devices._sessions, "device opens its own socket")
        with c.websocket_connect(f"/ws/device/phone-A?token={dev_a}"):
            check(True, "?token= query fallback works for sockets")

        # ---- re-pair supersedes, revoke kills ----
        code2 = c.post("/v1/devices/pair/start", headers=bearer(admin)).json()["code"]
        dev_a2 = c.post("/v1/devices/pair", json={"code": code2, "device_id": "phone-A"}).json()["token"]
        check(c.get("/tasks", headers=bearer(dev_a)).status_code == 401,
              "re-pairing invalidates the old device token")
        check(c.get("/tasks", headers=bearer(dev_a2)).status_code == 200, "new device token works")
        check(c.post("/v1/devices/phone-A/revoke", headers=bearer(admin)).json()["revoked"],
              "admin revokes the device")
        check(c.get("/tasks", headers=bearer(dev_a2)).status_code == 401, "revoked token rejected")

        # ---- brute-force lock ----
        statuses = [c.post("/v1/devices/pair", json={"code": f"{i:06d}", "device_id": "x"}).status_code
                    for i in range(900000, 900025)]
        check(429 in statuses, "repeated wrong codes lock pairing (429)")

        # tidy: stop the tasks we started
        c.post("/stop", headers=bearer(admin))

    # ---- registry: a stale socket's cleanup must not evict the newer session ----
    from app.ws import DeviceRegistry, DeviceSession
    reg = DeviceRegistry()
    old, new = DeviceSession("p", None), DeviceSession("p", None)   # type: ignore[arg-type]
    reg.add(old)
    reg.add(new)                 # phone reconnected after a flap
    reg.remove("p", old)         # server only now notices the old socket died
    check(reg.get("p") is new, "stale session cleanup keeps the newer live session")
    reg.remove("p", new)
    check(reg.get("p") is None, "current session cleanup removes it")

    print(f"\nAUTH PASS  ({PASS} checks)")


if __name__ == "__main__":
    main()
