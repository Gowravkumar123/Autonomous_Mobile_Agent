# Deployment (on-prem, PM2)

Deploys the Python agent backend on an on-prem server (example IP `10.20.0.13`),
supervised by **PM2** so it restarts on crash and on server reboot.

> The Android companion app and your phone talk to this server at
> `http://10.20.0.13:8000` (REST) and `ws://10.20.0.13:8000` (WebSocket).

---

## 0. Prerequisites on the server

- **Python 3.11+**  (`python3 --version`)
- **Node.js 18+ and PM2**  (`npm install -g pm2`)
- **git**
- Network: TCP port **8000** reachable from the phones/clients (firewall rule below).

---

## 1. Get the code

```bash
sudo mkdir -p /opt/autonomous-agent && sudo chown "$USER" /opt/autonomous-agent
cd /opt/autonomous-agent
git clone https://github.com/Gowravkumar123/Autonomous_Mobile_Agent.git .
cd backend
```

## 2. Create the virtualenv + install deps

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

## 3. Create the `.env` **on the server** (never committed)

```bash
cp .env.example .env
nano .env      # paste GROQ_API_KEY=... (and optionally GEMINI_API_KEY=...)
```

Keep `HOST=0.0.0.0` so the service is reachable on the LAN. The PM2 command already
passes `--host 0.0.0.0 --port 8000`.

## 4. Start under PM2

```bash
# from /opt/autonomous-agent/backend
pm2 start ecosystem.config.js
pm2 status
pm2 logs agent-backend --lines 50
```

`ecosystem.config.js` points PM2 at `.venv/bin/python -m uvicorn app.main:app` with
`cwd` set to `backend/`, so it uses the venv and loads `.env`.

## 5. Persist across reboots

```bash
pm2 save
pm2 startup           # prints a `sudo ... ` command — run that once
```

## 6. Open the firewall (example: ufw)

```bash
sudo ufw allow 8000/tcp
```

## 7. Verify (from another machine on the network)

```bash
curl http://10.20.0.13:8000/health
# -> {"ok":true,"llm":"groq","devices":[]}
```

If `"llm"` is `mock`, the `.env` key wasn't picked up — check step 3 and `pm2 restart`.

---

## Point the Android app at the server

In `android/app/build.gradle.kts` → `defaultConfig`:

```kotlin
buildConfigField("String", "BACKEND_WS_URL",   "\"ws://10.20.0.13:8000\"")
buildConfigField("String", "BACKEND_HTTP_URL", "\"http://10.20.0.13:8000\"")
```

Rebuild the app. The phone must be on a network that can route to `10.20.0.13`.

---

## Day-2 operations

| Task | Command |
|------|---------|
| Logs (live) | `pm2 logs agent-backend` |
| Status / CPU / mem | `pm2 status` / `pm2 monit` |
| Restart | `pm2 restart agent-backend` |
| Stop | `pm2 stop agent-backend` |
| Remove | `pm2 delete agent-backend` |
| **Deploy an update** | `git pull && .venv/bin/pip install -r requirements.txt && pm2 restart agent-backend` |

Audit logs are written to `backend/audit_logs/*.jsonl` (git-ignored).

---

## Before real/production use — hardening (not yet implemented)

The backend currently has **no authentication** and **CORS is open (`*`)**. On a
reachable IP, anyone who can hit `10.20.0.13:8000` could create tasks or connect as a
device. Before trusting it with real accounts:

1. **Add a shared-secret/token** to device pairing and the REST endpoints
   (reject WebSocket/REST calls without it).
2. **Lock down CORS** in `app/main.py` to only the origins you use.
3. **Terminate TLS** — put Nginx/Caddy in front for `https://` / `wss://`, since the
   link carries screen contents and task details. PM2 runs the app on `:8000`; the
   reverse proxy listens on `:443` and forwards to it.
4. Bind the app to `127.0.0.1` and let only the reverse proxy reach it, once TLS is in place.

Ask and I can implement 1–2 (token auth + CORS lockdown) in the app.
