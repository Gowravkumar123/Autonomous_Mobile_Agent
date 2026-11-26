"""
LIVE voice run (real Groq Whisper STT, real model, real TTS when available): a spoken
question — synthesized offline by Windows' System.Speech, so it is real speech audio —
is streamed in real time over WS /v1/voice, endpointed by the VAD, transcribed,
answered by the agent through the normal task runtime, and spoken back.

Run (Windows, GROQ_API_KEY in backend/.env):  python -m tests.live_voice
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

os.environ["DATABASE_URL"] = ""
os.environ["REDIS_URL"] = ""
os.environ["EMBEDDING_BACKEND"] = "hashing"
os.environ["MCP_CONFIG_PATH"] = "does-not-exist.json"
os.environ["BROWSER_ENABLED"] = "false"
os.environ["AUTH_SECRET"] = "live-voice-" + "z" * 40

QUESTION = "Search the web and tell me: what is the capital city of Australia?"


def synthesize(text: str, out: Path) -> None:
    ps = ("Add-Type -AssemblyName System.Speech; "
          "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
          "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
          "[System.Speech.AudioFormat.AudioChannel]::Mono); "
          f"$s.SetOutputToWaveFile('{out}', $f); $s.Speak('{text}'); $s.Dispose()")
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], check=True)


async def main():
    import httpx  # noqa: F401
    import uvicorn
    import websockets
    from app.main import app
    from app.voice.audio import frames, wav_to_pcm16k

    wav = Path(tempfile.mkdtemp()) / "q.wav"
    synthesize(QUESTION, wav)
    pcm = wav_to_pcm16k(wav.read_bytes())
    print(f"question audio: {len(pcm) / 32000:.1f}s")

    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        port = so.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning", lifespan="on"))
    srv = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    tok = app.state.auth.issue_admin()
    events, audio = [], {"frames": 0}
    t0 = time.monotonic()
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}/v1/voice/live1",
                                      additional_headers={"Authorization": f"Bearer {tok}"},
                                      max_size=None) as ws:
            async def reader():
                async for m in ws:
                    if isinstance(m, bytes):
                        audio["frames"] += 1
                    else:
                        e = json.loads(m)
                        events.append(e)
                        if e["type"] in ("transcript", "speak", "say", "barge_in"):
                            print(f"  [{time.monotonic() - t0:5.1f}s] {e['type']}: "
                                  f"{e.get('text', '')} {e.get('confidence', '')}")
            r = asyncio.create_task(reader())
            silence = b"\x00" * 640
            for fr in [silence] * 25 + frames(pcm) + [silence] * 60:   # real-time pace
                await ws.send(fr)
                await asyncio.sleep(0.02)
            sess = app.state.voice.sessions["live1"]
            for _ in range(240):
                task = app.state.store.get(sess.task_id) if sess.task_id else None
                if task and task.state.value in ("COMPLETED", "FAILED", "WAITING_FOR_USER") \
                        and any(e["type"] in ("speak", "say") and "Australia" not in
                                e.get("text", "") and e.get("text") != "On it."
                                for e in events[-3:]):
                    break
                await asyncio.sleep(0.5)
            await asyncio.sleep(4)
            r.cancel()
        task = app.state.store.get(sess.task_id) if sess.task_id else None
        print(f"\ntranscript turns: {[(t['transcript'], t['confidence'], t['intent']) for t in sess.turns]}")
        print(f"task: {task.state.value if task else None} | executor {task.executor if task else ''}"
              f" | summary: {task.summary if task else ''}")
        print(f"agent audio frames received: {audio['frames']} "
              f"({'server TTS' if audio['frames'] else 'client TTS fallback (say)'})")
        spoken = " ".join(e.get("text", "") for e in events if e["type"] in ("speak", "say"))
        print("spoken:", spoken)
        ok = task is not None and "canberra" in (spoken + (task.summary or "")).lower()
        print("\nLIVE VOICE", "PASS" if ok else "CHECK OUTPUT")
    finally:
        server.should_exit = True
        await srv


if __name__ == "__main__":
    asyncio.run(main())
