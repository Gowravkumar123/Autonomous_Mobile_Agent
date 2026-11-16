// PM2 process definition for the agent backend (FastAPI/uvicorn).
// Run from the backend/ directory:   pm2 start ecosystem.config.js
//
// PM2 is a Node process manager, but it supervises any executable. We point it at
// the virtualenv's Python and launch uvicorn as a module, so it uses the venv deps
// and loads backend/.env. `interpreter: "none"` tells PM2 not to wrap it in Node.
const path = require("path");
const isWin = process.platform === "win32";
const python = isWin ? ".venv\\Scripts\\python.exe" : ".venv/bin/python";

module.exports = {
  apps: [
    {
      name: "agent-backend",
      cwd: __dirname,                       // the backend/ directory
      script: path.join(__dirname, python),
      args: "-m uvicorn app.main:app --host 0.0.0.0 --port 8000",
      interpreter: "none",                  // execute the Python binary directly
      env: {
        PYTHONUNBUFFERED: "1",
        PYTHONIOENCODING: "utf-8",
      },
      autorestart: true,
      max_restarts: 10,
      min_uptime: "10s",
      restart_delay: 3000,
      max_memory_restart: "600M",
      time: true,                           // timestamp log lines
    },
  ],
};
