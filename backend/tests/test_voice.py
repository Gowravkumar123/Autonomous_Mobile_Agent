"""
Voice gateway: VAD/audio units, then E2E through the real app over WS /v1/voice —
speech frames in, VAD endpointing, STT, turn manager into the SAME task runtime (mock
LLM + simulated IRCTC phone), narrated replies streamed back as audio, barge-in,
push-to-talk, text fallback, voice-approval safety, ownership. STT/TTS are faked (no
Groq), everything else is real. WebRTC runs only where aiortc/PyAV can load.

Run:  python -m tests.test_voice
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import time

os.environ["AUTH_SECRET"] = "test-secret-" + "v" * 40
os.environ["DATABASE_URL"] = ""
os.environ["REDIS_URL"] = ""
os.environ["EMBEDDING_BACKEND"] = "hashing"
os.environ["GROQ_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["MCP_CONFIG_PATH"] = "does-not-exist.json"
os.environ["MEMORY_LEARN"] = "false"
os.environ["BROWSER_ENABLED"] = "false"
os.environ["VOICE_PROGRESS_S"] = "0"

import numpy as np                                        # noqa: E402

from app.agent.approvals import ApprovalRecord            # noqa: E402
from app.agent.state import Task, TaskState               # noqa: E402
from app.voice import groq_audio                          # noqa: E402
from app.voice.audio import (FRAME_BYTES, RATE, Vad, frames, pcm_to_wav, tone,  # noqa: E402
                             wav_to_pcm16k)
from simulator.mock_device import IrctcPhone              # noqa: E402

PROMPT = ("Book two train tickets from Hyderabad to Bengaluru for tomorrow. "
          "Prefer AC 3-tier. Compare trains, enter passenger details, stop before payment.")
PASS = 0


def check(cond: bool, label: str) -> None:
    global PASS
    assert cond, f"FAIL: {label}"
    PASS += 1
    print(f"PASS {label}")


def silence(ms: int) -> bytes:
    return b"\x00" * (RATE * ms // 1000 * 2)


def noise(ms: int, db: float = -60.0, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    amp = 32767 * 10 ** (db / 20)
    return (rng.standard_normal(RATE * ms // 1000) * amp).clip(-32767, 32767).astype("<i2").tobytes()


def speech(ms: int = 600) -> bytes:
    """Voiced-ish signal: a few harmonics, well above the noise floor."""
    return bytes(np.frombuffer(tone(ms, 180, -18), "<i2").astype(np.int32)
                 .__add__(np.frombuffer(tone(ms, 360, -22), "<i2"))
                 .clip(-32767, 32767).astype("<i2"))


# --------------------------------------------------------------------------- #
# 1. Audio + VAD
# --------------------------------------------------------------------------- #
def unit_audio() -> None:
    v = Vad(end_silence_ms=600)
    evs = [e for fr in frames(noise(1500)) if (e := v.feed(fr))]
    check(not evs, "VAD: background noise alone is not speech")
    stream = noise(500) + speech(700) + noise(900, seed=1)
    evs = [e for fr in frames(stream) if (e := v.feed(fr))]
    kinds = [e.kind for e in evs]
    check(kinds == ["start", "end"], f"VAD: one utterance -> start, end ({kinds})")
    end = evs[-1]
    dur = len(end.pcm) / 2 / RATE * 1000
    check(600 <= end.speech_ms <= 760 and 900 <= dur <= 1700,
          f"VAD: utterance kept with pre-roll + hangover ({end.speech_ms} ms speech, {dur:.0f} ms audio)")
    evs = [e.kind for fr in frames(noise(300) + speech(100) + noise(800, seed=2))
           if (e := v.feed(fr))]
    check(evs == ["start", "discard"], "VAD: a 100 ms click is discarded (min speech)")
    v2 = Vad(end_silence_ms=600)
    loud_bg = noise(2000, db=-35)
    _ = [v2.feed(fr) for fr in frames(loud_bg)]
    check(v2.noise_db > -45, f"VAD: adapts to a noisy room (floor {v2.noise_db:.0f} dBFS)")
    v3 = Vad(end_silence_ms=600)
    kinds = [e.kind for fr in frames(noise(600) + noise(5000, db=-30, seed=5))
             if (e := v3.feed(fr))]
    check(not v3.speaking and kinds.count("start") <= 1,
          f"VAD: a fan switching on can't hold the turn open ({kinds})")
    v2.bias_db = 6.0
    check(not v2.is_speech(v2.noise_db + 12) and v2.is_speech(v2.noise_db + 20),
          "VAD: echo guard raises the bar while the agent talks")

    from app.voice.session import NO, YES
    check(all(YES.match(t) for t in ("Yes, go ahead.", "yes", "Okay.", "sure, do it")) and
          not any(YES.match(t) for t in ("yes but change the date", "yes, wait", "yesterday",
                                         "not sure", "Book two tickets")),
            "consent phrases: plain yes counts; hedged/unrelated speech doesn't")
    check(all(NO.match(t) for t in ("no", "No thanks.", "nope, cancel it")) and
          not NO.match("nothing yet"), "rejection phrases")

    w24 = pcm_to_wav(tone(500, 440, -10, rate=24000), rate=24000)
    pcm = wav_to_pcm16k(w24)
    check(abs(len(pcm) / 2 - 8000) <= 2, "WAV 24 kHz -> PCM16 16 kHz resample")
    check(len(frames(pcm)[0]) == FRAME_BYTES, "20 ms frames of 640 bytes")


# --------------------------------------------------------------------------- #
# 2. E2E over the WebSocket voice gateway
# --------------------------------------------------------------------------- #
SCRIPT: list[tuple[str, float]] = []
TTS_LONG = {"on": False}


async def fake_stt(wav: bytes, filename: str = "", language: str = ""):
    return SCRIPT.pop(0) if SCRIPT else ("", 0.0)


async def fake_tts(text: str):
    ms = 3000 if TTS_LONG["on"] else min(1200, 25 * len(text))
    return pcm_to_wav(tone(ms, 300, -14, rate=24000), rate=24000)


class SimPhone:
    kind = "phone"

    def __init__(self, device_id):
        self.device_id, self.phone = device_id, IrctcPhone()

    async def request(self, type_, timeout=60.0, **data):
        if type_ == "observe":
            return self.phone.observe()
        if type_ == "act":
            return self.phone.act(data.get("action", {}))
        if type_ == "handoff":
            return self.phone.handoff()
        return {}

    async def push(self, type_, **data):
        pass                                   # the approval comes from the app screen

    def on_message(self, msg): ...
    def fail_all(self, exc): ...


class Client:
    """A voice client: sends PCM frames + JSON, collects events and received audio."""

    def __init__(self, ws):
        self.ws = ws
        self.events: list[dict] = []
        self.audio_frames = 0
        self.audio_log: list[float] = []
        self._reader = asyncio.create_task(self._read())

    async def _read(self):
        try:
            async for m in self.ws:
                if isinstance(m, bytes):
                    self.audio_frames += 1
                    self.audio_log.append(time.monotonic())
                else:
                    self.events.append(json.loads(m))
        except Exception:
            pass

    async def talk(self, pcm: bytes, realtime: bool = False):
        for fr in frames(pcm):
            await self.ws.send(fr)
            if realtime:
                await asyncio.sleep(0.02)

    async def say(self, text: str, conf: float = 0.92, ms: int = 600):
        SCRIPT.append((text, conf))
        await self.talk(noise(200) + speech(ms) + noise(900, seed=3))

    def said(self) -> list[str]:
        return [e["text"] for e in self.events if e.get("type") == "speak"]

    async def wait(self, pred, timeout=20.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if pred():
                return True
            await asyncio.sleep(0.05)
        return pred()

    async def close(self):
        await self.ws.close()
        self._reader.cancel()


def free_port() -> int:
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


class _FakeTask:
    def __init__(self, state="RUNNING"):
        self.state = TaskState(state)
        self.goal = self.prompt = "book a train"
        self.history, self.summary, self.error = [], "", ""


async def live_regression() -> None:
    """Replay the exact turns the phone produced on 2026-11-26 (room noise + Zoro's own
    voice): before the fix, noise answered a question and STARTED A TASK on the phone."""
    from app.voice.session import Backend, Pending, VoiceSession
    created, answered = [], []
    task = {"t": None}

    async def lookup(_):
        return task["t"]

    async def ctl(task_id, op, **a):
        if op == "resolve_answer":
            answered.append(a["facts"])
        return True

    async def create(text):
        created.append(text)
        return "t-new"
    q: asyncio.Queue = asyncio.Queue()
    b = Backend(create_task=create, lookup=lookup, ctl=ctl, subscribe=lambda: q,
                unsubscribe=lambda _q: None, stt=fake_stt, tts=fake_tts)
    s = VoiceSession("reg", "phone", b)
    s._open_attention(1000)          # these checks are about the noise/read-back rules

    task["t"] = _FakeTask()
    s.task_id, s.pending = "t1", Pending("ask", "t1")
    live = [("Amen.", 0.40, 400), ("Awesome.", 0.22, 740), ("Valeu, gente.", 0.16, 880),
            ("\u0634\u0643\u0631\u0627", 0.56, 660), ("Thank you.", 0.10, 560),
            ("Thank you.", 0.51, 600)]
    intents = [(await s._route(t, c, "voice", ms))[1] for t, c, ms in live]
    check(not answered and all(i in ("ignored_noise", "unclear") for i in intents),
          f"live noise turns can't answer Zoro's question ({intents})")
    reply, intent = await s._route("Yo creo que no tengo que contestar.", 0.30, "voice", 2440)
    check(intent == "unclear" and "didn't catch" in reply,
          "long unintelligible speech -> one 'didn't catch' (not silence)")

    task["t"], s.task_id, s.pending = None, "", Pending()
    s.cfg.task_confidence = 0.75     # (the first session's threshold)
    reply, intent = await s._route("come il professor Fondi Si", 0.43, "voice", 1400)
    check(not created and intent == "ignored_fragment" and not reply,
          "the live phantom task is no longer started (not even read back: not a request)")
    reply, intent = await s._route("please open the settings app", 0.43, "voice", 1400)
    check(not created and intent == "confirm_task" and "Say yes to start" in reply,
          "a low-confidence real-looking request is READ BACK instead of started")
    reply, intent = await s._route("no", 0.9, "voice", 500)
    check(not created and intent == "confirm_rejected", "...and 'no' drops it")
    await s._route("open whatsapp and message amma", 0.6, "voice", 1600)
    reply, intent = await s._route("yes", 0.9, "voice", 500)
    check(created == ["open whatsapp and message amma"] and intent == "new_task",
          "a read-back request starts only after an explicit yes")
    created.clear()
    s.task_id, task["t"] = "", None
    await s._route("search the web for the weather in Hyderabad", 0.9, "voice", 2200)
    check(created == ["search the web for the weather in Hyderabad"],
          "clearly heard requests start immediately")

    s._recent_spoken.append("the capital of australia is canberra")
    task["t"] = _FakeTask()
    s.task_id, s.pending = "t2", Pending("ask", "t2")
    _, intent = await s._route("The capital of Australia is Canberra.", 0.9, "voice", 2000)
    check(intent == "ignored_echo" and not answered, "Zoro's own voice in the mic is ignored")
    _, intent = await s._route("AC three tier please", 0.9, "voice", 1500, overlapped=False)
    check(intent == "answer" and answered, "a real answer still gets through")
    _, intent = await s._route("hmm maybe", 0.5, "voice", 900, overlapped=True)
    check(intent == "ignored_echo", "low-confidence speech that began over Zoro's voice is ignored")
    await s.close()


# The second live phone session (2026-11-26 15:05): the user asked once, then people
# talked nearby. Before addressing, ~10 of these got "Did you want me to ...?" read-backs.
SESSION2 = [
    ("Bye.", 0.139, 400), ("What's that?", 0.59, 580), ("I don't know.", 0.386, 920),
    ("I'll have another one.", 0.08, 620), ("Thank you.", 0.273, 380),
    ("What is that?", 0.496, 2140), ("Wow, I got a... action candy.", 0.357, 1740),
    ("That's nice.", 0.353, 3800), ("Thank you.", 0.399, 580), ("No.", 0.402, 500),
    ("You can't have that.", 0.352, 680), ("Amen.", 0.356, 1660), ("Thank you.", 0.432, 280),
    ("Take care.", 0.437, 600), ("and i'm not very much", 0.489, 780),
    ("Nothing will happen.", 0.541, 520), ("Thank you.", 0.219, 280),
    ("Thank you.", 0.542, 460), ("Hi, I'm going to turn it off.", 0.541, 1700),
    ("Bye.", 0.126, 380), ("Bye.", 0.31, 380),
    ("I think I found a question. I think I found a question.", 0.23, 2740),
    ("We will sit down and keep sitting for 10 minutes.", 0.434, 1780),
    ("And this kind of...", 0.653, 1160)]


async def addressing_regression() -> None:
    from app.voice.session import Backend, Pending, VoiceSession
    created, answered, ctls = [], [], []
    task = {"t": None}

    async def lookup(_):
        return task["t"]

    async def ctl(task_id, op, **a):
        ctls.append(op)
        if op == "resolve_answer":
            answered.append(a["facts"])
        return True

    async def create(text):
        created.append(text)
        return "t-new"
    q: asyncio.Queue = asyncio.Queue()
    b = Backend(create_task=create, lookup=lookup, ctl=ctl, subscribe=lambda: q,
                unsubscribe=lambda _q: None, stt=fake_stt, tts=fake_tts)
    s = VoiceSession("addr", "phone", b)
    s._attention_until = 0.0                                   # nobody has spoken lately

    reply, intent = await s._route("Hi, Juro. Can you say what is the capital of Australia?",
                                   0.594, "voice", 2600)
    check(intent == "new_task" and created == ["Can you say what is the capital of Australia?"],
          "the live request ('Hi, Juro...') starts at once: name heard, 0.59 >= 0.55")
    task["t"], s.task_id = _FakeTask(), "t-new"                 # it's running now
    s._attention_until = 0.0
    replies = [await s._route(t, c, "voice", ms) for t, c, ms in SESSION2]
    spoken = [r for r, _ in replies if r]
    check(not spoken and len(created) == 1 and not ctls,
          f"all {len(SESSION2)} background turns from the live session stay silent "
          f"({sorted({i for _, i in replies})})")

    task["t"], s.task_id = None, ""                             # task finished
    s._open_attention(8)                                        # Zoro just answered
    for t in ("I don't know.", "Take care.", "That's nice."):
        _, intent = await s._route(t, 0.7, "voice", 900)
        check(intent == "ignored_fragment", f"follow-up window: chatter '{t}' isn't a request")
    reply, intent = await s._route("And what about New Zealand?", 0.8, "voice", 1500)
    check(intent == "new_task" and created[-1] == "And what about New Zealand?",
          "follow-up window: a real follow-up question works without the name")

    task["t"], s.task_id = _FakeTask(), "t-new"
    s._open_attention(8)
    _, intent = await s._route("Book me a cab to the airport", 0.8, "voice", 1500)
    check(intent == "ignored_busy", "while a task runs, un-named requests don't interrupt it")
    reply, intent = await s._route("Zoro, book me a cab", 0.8, "voice", 1500)
    check(intent == "busy" and "still working" in reply, "...named ones get the busy answer")
    _, intent = await s._route("Zoro stop", 0.8, "voice", 800)
    check(intent == "cancel" and "cancel" in ctls, "'Zoro, stop' cancels the running task")

    task["t"], s.task_id = None, ""
    s._attention_until = 0.0
    reply, intent = await s._route("Hey Zoro", 0.8, "voice", 700)
    check(intent == "wake" and s.attention_open(), "'Hey Zoro' alone -> 'Yes?' and listens")
    _, intent = await s._route("search the weather in Hyderabad", 0.8, "voice", 1600)
    check(intent == "new_task", "...then the request needs no name")
    s._attention_until, s.pending = 0.0, Pending()
    _, intent = await s._route("book a cab", 0.9, "voice", 900, addressed=True)
    check(intent == "new_task", "push-to-talk turns are addressed by the button press")
    s._attention_until = 0.0
    s.pending = Pending("ask", "t-x")
    task["t"], s.task_id = _FakeTask(), "t-x"
    _, intent = await s._route("AC three tier", 0.8, "voice", 1200)
    check(intent == "answer", "a pending question keeps Zoro listening (answer needs no name)")
    await s.close()


async def e2e() -> None:
    import httpx
    import uvicorn
    import websockets
    from app.main import app

    groq_audio.stt_detailed = fake_stt
    groq_audio.tts_cached = fake_tts
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning", lifespan="on"))
    srv = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    base, wsb = f"http://127.0.0.1:{port}", f"ws://127.0.0.1:{port}"
    try:
        auth = app.state.auth
        adm = {"Authorization": f"Bearer {auth.issue_admin()}"}
        async with httpx.AsyncClient(base_url=base, timeout=30) as h:
            async def pair(did):
                code = (await h.post("/v1/devices/pair/start", headers=adm)).json()["code"]
                return (await h.post("/v1/devices/pair",
                                     json={"code": code, "device_id": did})).json()["token"]
            tok = await pair("phone-v")
            other = await pair("phone-x")
            hdr = {"Authorization": f"Bearer {tok}"}
            app.state.devices.add(SimPhone("phone-v"))      # its execution target

            try:
                await websockets.connect(f"{wsb}/v1/voice/s1")
                ok = False
            except Exception:
                ok = True
            check(ok, "voice socket without a token is refused")

            ws = await websockets.connect(f"{wsb}/v1/voice/s1?mode=duplex",
                                          additional_headers=hdr, max_size=None)
            c = Client(ws)
            check(await c.wait(lambda: any(e["type"] == "ready" for e in c.events)) and
                  c.events[0]["format"]["rate"] == 16000, "session ready (PCM16 16 kHz, 20 ms)")
            try:
                await websockets.connect(f"{wsb}/v1/voice/s1",
                                         additional_headers={"Authorization": f"Bearer {other}"})
                ok = False
            except Exception:
                ok = True
            check(ok, "another device can't join (hijack) this conversation")

            # ---- spoken goal -> a real task on the same runtime ----
            await c.say(PROMPT)
            check(await c.wait(lambda: any(e.get("type") == "transcript" and e.get("final")
                                           for e in c.events)),
                  "VAD endpointed the utterance -> final transcript")
            check(await c.wait(lambda: "On it." in c.said()), "agent acknowledges by voice")
            check(await c.wait(lambda: c.audio_frames > 10), "speech streamed back as PCM frames")
            sess = app.state.voice.sessions["s1"]
            task = app.state.store.get(sess.task_id)
            check(task is not None and task.owner == "phone-v" and task.device_id == "phone-v",
                  "task created for this device through the normal task path")

            # ---- payment approval: spoken yes is NOT enough ----
            check(await c.wait(lambda: sess.pending.kind == "approval", 60),
                  "task reached an approval; voice narrates it")
            spoke = await c.wait(lambda: any("confirm it in the app" in t for t in c.said()), 10)
            if not spoke:
                print("    risk:", sess.pending.risk, "| said:", c.said())
            check(sess.pending.risk >= 4 and spoke, "Level-4 step: told to confirm in the app")
            await c.say("Yes, go ahead.")
            refused = await c.wait(lambda: any("can't take a spoken yes" in t for t in c.said()))
            if not refused:
                print("    events:", [(e.get("type"), e.get("text") or e.get("state") or "")
                                      for e in c.events if e.get("type") != "task_event"][-14:])
                print("    turns:", [(t["transcript"], t["intent"]) for t in sess.turns])
            check(refused, "spoken 'yes' refused for a payment step")
            recs = app.state.controller.approvals_for(task.id)
            check(recs and recs[-1].decision == "pending", "...and the approval is still pending")
            r = await h.post(f"/v1/approvals/{recs[-1].approval_id}/decision", headers=hdr,
                             json={"approved": True})
            check(r.status_code == 200, "approved on the in-app screen instead")
            check(await c.wait(lambda: task.state == TaskState.COMPLETED, 60),
                  f"task completed ({task.state.value})")
            check(await c.wait(lambda: any(t.startswith("Done") for t in c.said())),
                  "completion summary spoken")
            await asyncio.sleep(1.0)
            check(sum(1 for t in c.said() if t.startswith("Done")) == 1,
                  "...exactly once (final state is emitted twice)")

            # ---- low confidence -> ask to repeat ----
            # Wait until Zoro has finished talking (speech that starts over Zoro's voice
            # is treated as possible echo).
            await c.wait(lambda: not app.state.voice.sessions["s1"].agent_speaking, 10)
            await asyncio.sleep(0.3)
            await c.say("mumble mumble something", conf=0.1, ms=1400)
            check(await c.wait(lambda: any("didn't catch that" in t for t in c.said())),
                  "low-confidence real speech -> asks to repeat (no action)")

            # ---- barge-in ----
            TTS_LONG["on"] = True
            n0 = len(c.events)
            await c.say("what's the status")
            check(await c.wait(lambda: any(e.get("type") == "speak" for e in c.events[n0:])),
                  "status answered by voice")
            await asyncio.sleep(0.6)                  # agent is mid-sentence (3 s of audio)
            t_barge = time.monotonic()
            await c.talk(speech(400) + noise(200), realtime=True)
            check(await c.wait(lambda: any(e.get("type") == "barge_in" for e in c.events[n0:]), 5),
                  "user talking over the agent -> barge_in")
            await asyncio.sleep(0.8)
            late = [t for t in c.audio_log if t > t_barge + 0.5]
            check(len(late) <= 15, f"agent audio stopped promptly ({len(late)} late frames)")
            TTS_LONG["on"] = False
            SCRIPT.clear()
            await c.talk(noise(900, seed=4))          # let the interrupted turn finish

            # ---- reconnect keeps the conversation ----
            await c.close()
            ws = await websockets.connect(f"{wsb}/v1/voice/s1", additional_headers=hdr)
            c = Client(ws)
            check(await c.wait(lambda: any(e.get("type") == "ready" and
                                           e.get("task_id") == task.id for e in c.events)),
                  "reconnect resumes the same session (task remembered)")
            await c.close()

            # ---- text fallback (same session state) ----
            r = await h.post("/v1/voice/s1/turns", headers=hdr, json={"text": "status?"})
            check(r.status_code == 200 and r.json()["reply"].startswith("Done"),
                  "text fallback turn answered from the same task state")
            r = await h.post("/v1/voice/s1/turns", headers={"Authorization": f"Bearer {other}"},
                             json={"text": "status?"})
            check(r.status_code == 404, "...not by another device")

            # ---- push-to-talk ----
            ws = await websockets.connect(f"{wsb}/v1/voice/p1?mode=ptt", additional_headers=hdr)
            c = Client(ws)
            await c.wait(lambda: c.events)
            SCRIPT.append(("stop", 0.9))
            await ws.send(json.dumps({"type": "ptt", "state": "down"}))
            await c.talk(noise(400))                  # quiet speech: VAD would ignore it
            await ws.send(json.dumps({"type": "ptt", "state": "up"}))
            check(await c.wait(lambda: "Nothing is running right now." in c.said()),
                  "push-to-talk: the press/release defines the turn (and 'stop' with nothing "
                  "running doesn't start a task)")
            await c.close()

            # ---- clarification + Level-3 voice approval + stop, on a live task ----
            c_ctl = app.state.controller
            t2 = Task(prompt="send the report", owner="phone-v", device_id="phone-v",
                      state=TaskState.WAITING_FOR_USER, goal="send the report")
            app.state.store.create(t2)
            s1 = app.state.voice.sessions["s1"]
            s1.task_id = t2.id
            fut = asyncio.get_running_loop().create_future()
            c_ctl._answers[t2.id] = fut
            app.state.store.publish(t2.id, "ask_user", "Which format?",
                                    questions=["Which format? Choose one: PDF, or Word."])
            await asyncio.sleep(0.2)
            reply = await s1.submit_text("PDF please")
            check(reply == "Got it." and fut.done() and
                  fut.result() == {"clarification_answer": "PDF please"},
                  "spoken answer resolves the task's clarification question")
            rec = ApprovalRecord(task_id=t2.id, step_id=1, action_hash="h",
                                 expires_at=time.time() + 300, title="Approval needed",
                                 reason="send message", risk_level=3)
            c_ctl._approval_records[rec.approval_id] = rec
            afut = asyncio.get_running_loop().create_future()
            c_ctl._approvals[t2.id] = afut
            t2.pending_approval_id = rec.approval_id
            app.state.store.publish(t2.id, "approval", "send message",
                                    approval={"approval_id": rec.approval_id, "risk_level": 3,
                                              "reason": "send message"})
            await asyncio.sleep(0.2)
            reply = await s1.submit_text("maybe later")
            check("say yes" in reply and rec.decision == "pending",
                  "ambiguous reply to an approval -> asks for yes/no, decides nothing")
            reply = await s1.submit_text("Yes.")
            check(rec.decision == "approved" and rec.approver == "voice:phone-v" and
                  afut.done() and afut.result()[0] is True,
                  "Level-3 step approved by voice, bound to that approval id")
            reply = await s1.submit_text("stop")
            check(t2.cancel_requested and reply == "Okay, stopping.", "'stop' cancels the live task")

            turns = list(app.state.voice.recent_turns)
            intents = [t["intent"] for t in turns]
            check({"new_task", "approval_needs_app", "unclear", "status", "answer",
                   "approve", "cancel"} <= set(intents), f"turns recorded with intents")
            check(all("pcm" not in t and "audio" not in t for t in turns) and
                  all(isinstance(t["transcript"], str) for t in turns),
                  "turn log keeps transcript + timing only (no raw audio)")
            info = (await h.get("/v1/voice/s1", headers=hdr)).json()
            check(info["turns"] and info["owner"] == "phone-v", "GET /v1/voice/{id} shows turns")
    finally:
        server.should_exit = True
        await srv


async def webrtc_check() -> None:
    from app.voice import webrtc
    if not webrtc.available():
        print("SKIP WebRTC (aiortc/PyAV can't load here — e.g. blocked by Windows app "
              "control); run this suite on the server to exercise it")
        return
    from aiortc import RTCPeerConnection
    from aiortc.mediastreams import AudioStreamTrack
    from app.voice.session import Backend, VoiceSession

    got = {"frames": 0}
    q: asyncio.Queue = asyncio.Queue()

    async def lookup(_):
        return None
    backend = Backend(create_task=lambda p: asyncio.sleep(0, "t"), lookup=lookup,
                      ctl=lambda *a, **k: asyncio.sleep(0), subscribe=lambda: q,
                      unsubscribe=lambda _q: None, stt=fake_stt, tts=fake_tts)
    sess = VoiceSession("rtc", "admin", backend)
    orig = sess.feed_audio

    async def counting(pcm):
        got["frames"] += 1
        await orig(pcm)
    sess.feed_audio = counting
    await sess.start()
    server_t = webrtc.WebRtcTransport(sess, [])
    client = RTCPeerConnection()
    ch = client.createDataChannel("control")
    msgs = []
    ch.on("message", lambda m: msgs.append(json.loads(m)))
    client.addTrack(AudioStreamTrack())                  # client mic (silence)
    rx = {"n": 0}

    @client.on("track")
    def on_track(track):
        async def pull():
            while True:
                await track.recv()
                rx["n"] += 1
        asyncio.ensure_future(pull())
    await client.setLocalDescription(await client.createOffer())
    ans = await server_t.answer(client.localDescription.sdp)
    await sess.attach(server_t)
    from aiortc import RTCSessionDescription
    await client.setRemoteDescription(RTCSessionDescription(ans["sdp"], ans["type"]))
    for _ in range(100):
        if got["frames"] > 20 and rx["n"] > 20 and msgs:
            break
        await asyncio.sleep(0.1)
    check(got["frames"] > 20, "WebRTC: client mic audio reaches the session as 20 ms frames")
    check(rx["n"] > 20, "WebRTC: agent audio track flows to the client")
    check(any(m.get("type") == "ready" for m in msgs), "WebRTC: control events on the data channel")
    await client.close()
    await server_t.close()
    await sess.close()


async def main() -> None:
    unit_audio()
    await live_regression()
    await addressing_regression()
    await e2e()
    await webrtc_check()
    print(f"\nVOICE PASS  ({PASS} checks)")


if __name__ == "__main__":
    asyncio.run(main())
