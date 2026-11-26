"""
Voice session (blueprint 4): MIC -> VAD -> STT -> TURN MANAGER -> the SAME task runtime
-> TTS -> SPEAKER, full-duplex with barge-in, or push-to-talk.

Voice is just another channel into the one orchestrator: a turn either answers the
task's pending question, decides a pending approval, controls the task (stop / pause /
resume / status) or starts a new task — through the same functions the REST API uses,
so policy, approvals and audit are identical. It never gets its own tool path.

Safety (blueprint 7 "Voice approval safety"): a spoken "yes" can approve an ordinary
(Level-3) step, bound to that exact approval id, but NEVER a payment/auth (Level-4)
step — those need the authenticated in-app confirmation. A spoken "no" may always reject.

Transports (WebSocket audio, WebRTC) only move 20 ms PCM16 frames and JSON events in and
out; this class is transport-agnostic and testable without any audio stack.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .audio import FRAME_BYTES, FRAME_MS, Vad, frames, pcm_to_wav, wav_to_pcm16k

log = logging.getLogger("agent.voice")

_YES_WORDS = (r"yes|yeah|yep|yup|sure|ok|okay|go ahead|approve[d]?|confirm(ed)?|do it|"
              r"proceed|please do|go for it")
_NO_WORDS = (r"no|nope|nah|don'?t|do not|reject|deny|decline|cancel( it| that)?|stop")
# An affirmative word + at most a short tail ("Yes, go ahead.") — but a hedge ("yes, but
# change the date") is NOT consent.
YES = re.compile(rf"^\W*({_YES_WORDS})\b(?!.*\b(but|wait|not|don'?t|change|instead)\b)"
                 rf"[\w\s,.!'-]{{0,24}}$", re.I)
NO = re.compile(rf"^\W*({_NO_WORDS})\b[\w\s,.!'-]{{0,24}}$", re.I)
STOP = re.compile(r"^\W*(stop|cancel|abort|never ?mind|forget it)\b.{0,20}$", re.I)
PAUSE = re.compile(r"^\W*(pause|hold on|wait( a (sec|second|minute|moment))?)\W*$", re.I)
RESUME = re.compile(r"^\W*(resume|continue|carry on|go on|keep going)\W*$", re.I)
STATUS = re.compile(r"\b(status|what are you doing|where are you|how('?s| is) it going|"
                    r"progress|what'?s happening)\b", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+")
# "Zoro" as Whisper hears it on a phone mic (seen live: "Hi, Juro.").
_NAMES = r"zoro|zorro|zoroo|zoru|juro|jurro|joro|jorro|jiro|jiroh|ziro|soro|sorro|zolo|zaro|jaro|zuro"
NAME_LEAD = re.compile(rf"^\W*(?:(?:hey|hi|hello|ok|okay|yo|oh)\W+)?(?:{_NAMES})\b[\s,.!?:;-]*",
                       re.I)
NAME_TAIL = re.compile(rf"[\s,]+(?:{_NAMES})\W*$", re.I)
# Leading "word," that may be a mishearing of the name (checked fuzzily, see _sounds_like_name).
LEAD_WORD = re.compile(r"^\W*(?:(?:hey|hi|hello|ok|okay|yo|oh)\W+)?([a-z]{3,7})\b[\s,.!?:;-]*", re.I)
_NAME_FORMS = ("zoro", "juro", "jiro", "zorro")
# Real words that sound close to "Zoro" but are not the name.
_NOT_NAME = {"zero", "euro", "euros", "hero", "heroes", "pro", "sorry", "sure", "sir", "door",
             "floor", "four", "your", "yours", "tour", "turn", "zoom", "room", "torro", "giro",
             "jury", "zone", "sorrow", "borrow", "tomorrow", "story", "guru", "who", "hello",
             "hurry", "worry", "car", "jar", "zara", "sara", "laura", "juro"}
# Does an un-named utterance look like a request to an assistant (vs. people chatting)?
REQUEST = re.compile(
    r"\b(book|order|search|find|open|send|call|message|text|play|show|tell|check|buy|get|"
    r"set|remind|compare|look up|read|write|schedule|navigate|reserve|pay|cancel|start|"
    r"what|who|where|when|how|why|which|can you|could you|would you|will you|please|"
    r"help|i want|i need|i'd like)\b", re.I)

# Whisper's stock outputs for noise / silence / breathing (seen live on the phone).
HALLUCINATIONS = {
    "thank you", "thanks", "thank you very much", "thank you so much", "thanks for watching",
    "thank you for watching", "please subscribe", "amen", "bye", "bye bye", "you", "oh", "uh",
    "um", "hmm", "mm", "ah", "okay okay", "so", "the end", "subtitles by the amaraorg community",
}
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}


@dataclass
class VoiceConfig:
    mode: str = "duplex"                 # duplex | ptt
    end_silence_ms: int = 700
    min_confidence: float = 0.35
    task_confidence: float = 0.55        # below this a spoken NEW task is read back first
    answer_confidence: float = 0.6       # below this an answer to Zoro's question is re-asked
    progress_every_s: float = 20.0       # spoken status while a task works (0 = off)
    partial_every_s: float = 0.0         # interim transcripts (costs STT calls; 0 = off)
    barge_in_bias_db: float = 6.0        # echo guard while the agent is talking
    language: str = ""
    # Attention: Zoro reacts to un-named speech only right after it spoke (follow-ups) or
    # when it is waiting for an answer; otherwise the speech must contain its name.
    attention_after_reply_s: float = 8.0
    attention_on_connect_s: float = 12.0


@dataclass
class Backend:
    """The orchestrator surface voice may use — the same calls as the REST API."""
    create_task: Callable[[str], Awaitable[str]]
    lookup: Callable[[str], Awaitable[Any]]
    ctl: Callable[..., Awaitable[Any]]
    subscribe: Callable[[], asyncio.Queue]
    unsubscribe: Callable[[asyncio.Queue], None]
    stt: Callable[[bytes], Awaitable[tuple[str, float]]]
    tts: Callable[[str], Awaitable[bytes | None]]
    record_turn: Callable[[dict], None] = lambda t: None


@dataclass
class Pending:
    kind: str = ""                       # "" | ask | approval
    task_id: str = ""
    approval_id: str = ""
    risk: int = 0
    text: str = ""                       # confirm_task: what we'd start


class VoiceSession:
    def __init__(self, session_id: str, owner: str, backend: Backend,
                 cfg: VoiceConfig | None = None):
        self.id = session_id
        self.owner = owner
        self.b = backend
        self.cfg = cfg or VoiceConfig()
        self.vad = Vad(end_silence_ms=self.cfg.end_silence_ms)
        self.task_id = ""                # the task this conversation is driving
        self.pending = Pending()
        self.transport = None            # .send_json(dict) / .send_audio(bytes)
        self._turns: asyncio.Queue = asyncio.Queue()
        self._say: asyncio.Queue = asyncio.Queue()
        self._gen = 0                    # bumps on barge-in: stale speech stops
        self._streaming = False
        self._client_playing_until = 0.0
        self._ptt = False
        self._last_progress = 0.0
        self._final_spoken = ""
        self._recent_spoken: deque = deque(maxlen=6)   # echo filter
        self._start_overlapped = False
        self._attention_until = 0.0
        self._explicit_until = 0.0       # just tapped Talk / said "Hey Zoro": all speech is for us
        self._last_partial = 0.0
        self._workers: list[asyncio.Task] = []
        self._events_q: asyncio.Queue | None = None
        self.turns: list[dict] = []      # this session's turn log (metadata only)

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def attach(self, transport) -> None:
        """(Re)connect a transport; the conversation (task, pending prompt) carries on."""
        self.transport = transport
        self._open_attention(self.cfg.attention_on_connect_s)
        self._explicit_until = time.monotonic() + self.cfg.attention_on_connect_s
        await self._send({"type": "ready", "session_id": self.id, "mode": self.cfg.mode,
                          "task_id": self.task_id, "pending": self.pending.kind,
                          "format": {"encoding": "pcm16le", "rate": 16000, "channels": 1,
                                     "frame_ms": FRAME_MS}})

    def detach(self, transport) -> None:
        if self.transport is transport:
            self.transport = None
            self._streaming = False

    async def start(self) -> None:
        self._events_q = self.b.subscribe()
        self._workers = [asyncio.create_task(self._turn_loop()),
                         asyncio.create_task(self._speak_loop()),
                         asyncio.create_task(self._event_loop())]

    async def close(self) -> None:
        for w in self._workers:
            w.cancel()
        if self._events_q is not None:
            self.b.unsubscribe(self._events_q)

    async def _send(self, msg: dict) -> None:
        if self.transport is not None:
            try:
                await self.transport.send_json(msg)
            except Exception as e:
                log.debug("voice send failed: %s", e)

    @property
    def agent_speaking(self) -> bool:
        return self._streaming or time.monotonic() < self._client_playing_until

    # ------------------------------------------------------------------ #
    # input: audio frames + control messages from the transport
    # ------------------------------------------------------------------ #
    async def feed_audio(self, pcm: bytes) -> None:
        for fr in frames(pcm) if len(pcm) != FRAME_BYTES else [pcm]:
            await self._feed_frame(fr)

    async def _feed_frame(self, fr: bytes) -> None:
        if self.cfg.mode == "ptt":
            if self._ptt:
                self.vad._buf += fr
                self.vad._speech_frames += 1
            return
        self.vad.bias_db = self.cfg.barge_in_bias_db if self.agent_speaking else 0.0
        ev = self.vad.feed(fr)
        if ev is None:
            if self.vad.speaking and self.cfg.partial_every_s > 0 and \
                    time.monotonic() - self._last_partial >= self.cfg.partial_every_s:
                self._last_partial = time.monotonic()
                asyncio.create_task(self._partial(bytes(self.vad._buf)))
            return
        if ev.kind == "start":
            self._last_partial = time.monotonic()
            self._start_overlapped = self.agent_speaking
            if self.agent_speaking:
                await self.barge_in()
            await self._send({"type": "vad", "state": "speech"})
        elif ev.kind == "end":
            await self._send({"type": "vad", "state": "end", "speech_ms": ev.speech_ms})
            await self._turns.put(("audio", ev.pcm, ev.speech_ms, self._start_overlapped))
        elif ev.kind == "discard":
            await self._send({"type": "vad", "state": "discard"})

    async def handle_json(self, msg: dict) -> None:
        t = msg.get("type")
        if t == "ptt":
            if msg.get("state") == "down":
                await self.barge_in()            # pressing talk interrupts the agent
                self.vad._buf = bytearray()
                self.vad._speech_frames = 0
                self.vad.speaking = True
                self._ptt = True
            else:
                self._ptt = False
                ev = self.vad.force_end()
                if ev is not None and ev.kind == "end":
                    await self._turns.put(("ptt", ev.pcm, ev.speech_ms, False))
        elif t == "text":                        # text fallback, same task state
            await self._turns.put(("text", str(msg.get("text", "")), 0, False))
        elif t == "playback":                    # client-side TTS state (fallback path)
            if msg.get("state") == "start":
                self._client_playing_until = time.monotonic() + float(msg.get("ms", 8000)) / 1000
            else:
                self._client_playing_until = 0.0
        elif t == "interrupt":
            await self.barge_in()
        elif t == "mode" and msg.get("mode") in ("duplex", "ptt"):
            self.cfg.mode = msg["mode"]

    async def barge_in(self) -> None:
        """User started talking over the agent: stop speech now, listen."""
        if not self.agent_speaking and self._say.empty():
            return
        self._gen += 1
        while not self._say.empty():
            self._say.get_nowait()
        self._streaming = False
        self._client_playing_until = 0.0
        await self._send({"type": "barge_in"})

    async def _partial(self, pcm: bytes) -> None:
        text, _ = await self.b.stt(pcm_to_wav(pcm))
        if text:
            await self._send({"type": "transcript", "final": False, "text": text})

    # ------------------------------------------------------------------ #
    # turn manager
    # ------------------------------------------------------------------ #
    async def submit_text(self, text: str) -> str:
        """Text fallback turn (POST /turns): handled now; returns the spoken reply."""
        return await self._handle_turn("text", text, 0.0, 1.0, 0)

    async def _turn_loop(self) -> None:
        while True:
            source, payload, speech_ms, overlapped = await self._turns.get()
            try:
                if source in ("audio", "ptt"):
                    t0 = time.monotonic()
                    text, conf = await self.b.stt(pcm_to_wav(payload))
                    stt_ms = int((time.monotonic() - t0) * 1000)
                    await self._handle_turn("voice", text, stt_ms, conf, speech_ms, overlapped,
                                            addressed=(source == "ptt"))
                else:
                    await self._handle_turn("text", payload, 0, 1.0, 0)
            except Exception:
                log.exception("voice turn failed")
                self.say("Sorry, something went wrong handling that.")

    async def _handle_turn(self, source: str, text: str, stt_ms: float, conf: float,
                           speech_ms: int, overlapped: bool = False,
                           addressed: bool = False) -> str:
        text = (text or "").strip()
        if source == "voice":
            await self._send({"type": "transcript", "final": True, "text": text,
                              "confidence": round(conf, 2)})
        turn = {"turn_id": uuid.uuid4().hex[:16], "session_id": self.id, "owner": self.owner,
                "source": source, "transcript": text[:2000], "confidence": round(conf, 3),
                "speech_ms": speech_ms, "stt_ms": int(stt_ms), "intent": "", "task_id": ""}
        reply, intent = await self._route(text, conf, source, speech_ms, overlapped,
                                          addressed)
        turn["intent"], turn["task_id"] = intent, self.task_id
        self.turns.append(turn)
        self.b.record_turn(turn)
        if reply:
            self.say(reply)
        return reply

    def _noise_gate(self, text: str, conf: float, speech_ms: int,
                    overlapped: bool) -> tuple[str, str] | None:
        """Drop what isn't the user talking to Zoro: room noise, Whisper's stock
        hallucinations, and Zoro's own voice leaking back into the mic. Dropped SILENTLY —
        a "sorry?" reply would itself be picked up and loop."""
        norm = _norm(text)
        if not norm:
            return ("", "ignored_noise")
        if any(norm == r or (len(norm) >= 8 and norm in r) for r in self._recent_spoken):
            return ("", "ignored_echo")
        if norm in HALLUCINATIONS and speech_ms < 1500:
            return ("", "ignored_noise")
        if overlapped and conf < 0.6:
            return ("", "ignored_echo")
        if conf < self.cfg.min_confidence:
            if speech_ms >= 1200:          # real speech we couldn't make out: ask
                return ("Sorry, I didn't catch that. Could you say it again?", "unclear")
            return ("", "ignored_noise")
        if speech_ms < 600 and conf < 0.6:
            return ("", "ignored_noise")
        return None

    def _open_attention(self, seconds: float) -> None:
        self._attention_until = max(self._attention_until, time.monotonic() + seconds)

    def attention_open(self) -> bool:
        return (time.monotonic() < max(self._attention_until, self._explicit_until)
                or self.pending.kind != "")

    async def _route(self, text: str, conf: float, source: str, speech_ms: int = 0,
                     overlapped: bool = False, addressed: bool = False) -> tuple[str, str]:
        if not text:
            return ("Sorry, I didn't catch that. Could you say it again, or type it?", "unclear")
        named = False
        if source == "voice":
            gated = self._noise_gate(text, conf, speech_ms, overlapped)
            if gated is not None:
                # Only ask "say that again?" when Zoro is actually listening to the user.
                if gated[1] == "unclear" and not (addressed or self.attention_open()):
                    return ("", "ignored_unaddressed")
                return gated
            text, named = _strip_name(text)
            if not (addressed or named or self.attention_open()):
                # Not talking to Zoro (people nearby, TV): stay silent.
                return ("", "ignored_unaddressed")
            if named and len(text.split()) < 2 and not (YES.match(text) or NO.match(text) or
                                                         STOP.match(text) or STATUS.search(text)):
                self._open_attention(self.cfg.attention_after_reply_s + 4)
                self._explicit_until = time.monotonic() + self.cfg.attention_after_reply_s + 4
                return ("Yes? What would you like me to do?", "wake")
        explicit = (source != "voice" or addressed or named
                    or time.monotonic() < self._explicit_until)
        task = await self.b.lookup(self.task_id) if self.task_id else None
        state = task.state.value if task is not None else ""
        active = task is not None and state not in TERMINAL

        p = self.pending
        if p.kind == "confirm_task":
            # We read a low-confidence request back; only an explicit yes starts it.
            self.pending = Pending()
            if YES.match(text):
                return await self._start(p.text)
            if NO.match(text) or STOP.match(text):
                return ("Okay, never mind.", "confirm_rejected")
            # Anything else is a fresh request: fall through and route it normally.
        if active and p.kind == "approval":
            if YES.match(text):
                if p.risk >= 4:
                    # Blueprint 7: a spoken yes is never enough for payments/auth.
                    return ("For payments and sign-ins I can't take a spoken yes. Please "
                            "confirm it on the approval screen in the app.", "approval_needs_app")
                ok, res = await self.b.ctl(self.task_id, "decide_approval",
                                           approval_id=p.approval_id, approved=True,
                                           note="spoken approval",
                                           approver=f"voice:{self.owner}")
                self.pending = Pending()
                return (("Okay, going ahead." if ok else f"I couldn't apply that: {res}."),
                        "approve")
            if NO.match(text) or STOP.match(text):
                ok, res = await self.b.ctl(self.task_id, "decide_approval",
                                           approval_id=p.approval_id, approved=False,
                                           note="spoken rejection",
                                           approver=f"voice:{self.owner}")
                self.pending = Pending()
                return ("Okay, I won't do that.", "reject")
            return ("Please say yes to approve or no to cancel"
                    + (" — or confirm in the app." if p.risk >= 4 else "."), "approval_unclear")

        if active and STOP.match(text):
            await self.b.ctl(self.task_id, "cancel")
            self.pending = Pending()
            return ("Okay, stopping.", "cancel")
        if active and p.kind == "ask":
            if source == "voice" and conf < self.cfg.answer_confidence:
                return ("Sorry, I didn't catch your answer. Could you say it again?", "unclear")
            # The question event can reach us a moment before the task starts waiting.
            for _ in range(20):
                if await self.b.ctl(self.task_id, "resolve_answer",
                                    facts={"clarification_answer": text}):
                    break
                await asyncio.sleep(0.1)
            self.pending = Pending()
            return ("Got it.", "answer")
        if active and PAUSE.match(text):
            await self.b.ctl(self.task_id, "pause")
            return ("Pausing.", "pause")
        if active and RESUME.match(text):
            await self.b.ctl(self.task_id, "resume")
            return ("Resuming.", "resume")
        if STATUS.search(text):
            return (self._status_line(task), "status")
        if STOP.match(text) or PAUSE.match(text) or RESUME.match(text):
            return ("Nothing is running right now.", "idle_control")
        if active:
            if not explicit:
                return ("", "ignored_busy")      # chatter while a task runs: stay quiet
            return (f"I'm still working on: {task.goal or task.prompt}. Say stop to cancel "
                    "it first.", "busy")
        if not explicit and (len(text.split()) < 3 or not REQUEST.search(text)):
            return ("", "ignored_fragment")      # a follow-up must look like a request
        if source == "voice" and conf < self.cfg.task_confidence:
            self.pending = Pending("confirm_task", text=text)
            return (f"Did you want me to: {_short(text, 120)}? Say yes to start, or no.",
                    "confirm_task")
        return await self._start(text)

    async def _start(self, text: str) -> tuple[str, str]:
        self.task_id = await self.b.create_task(text)
        self.pending = Pending()
        self._last_progress = time.monotonic()
        return ("On it.", "new_task")

    @staticmethod
    def _status_line(task) -> str:
        if task is None:
            return "Nothing is running right now. What would you like me to do?"
        st = task.state.value
        if st in TERMINAL:
            return _clean_summary(task.summary or task.error or st.lower())
        last = task.history[-1] if task.history else "getting started"
        return f"Still working on {task.goal or task.prompt}. Last step: {last}."

    # ------------------------------------------------------------------ #
    # narrator: task events -> short spoken lines
    # ------------------------------------------------------------------ #
    async def _event_loop(self) -> None:
        while True:
            ev = await self._events_q.get()
            if not self.task_id or ev.get("task_id") != self.task_id:
                continue
            try:
                line = self._narrate(ev)
            except Exception:
                log.exception("narrate failed")
                line = ""
            if line:
                self.say(line)
            await self._send({"type": "task_event", "event": {
                k: ev.get(k) for k in ("task_id", "kind", "detail")}})

    def _narrate(self, ev: dict) -> str:
        kind, detail, data = ev.get("kind"), ev.get("detail", ""), ev.get("data") or {}
        if kind == "ask_user":
            self.pending = Pending("ask", self.task_id)
            qs = data.get("questions") or [detail]
            return " ".join(q for q in qs if q)
        if kind == "approval":
            ap = data.get("approval") or {}
            self.pending = Pending("approval", self.task_id, ap.get("approval_id", ""),
                                   int(ap.get("risk_level") or 0))
            amount = f" for {ap['amount']:g}" if ap.get("amount") else ""
            if self.pending.risk >= 4:
                return (f"I've reached a payment or sign-in step{amount}. Please confirm it "
                        "in the app — for safety I can't accept a spoken yes for that.")
            return (f"I need your OK before this step{amount}: {ap.get('reason') or detail}. "
                    "Say yes to approve, or no to cancel.")
        if kind == "approval_decided":
            self.pending = Pending()
            return ""
        if kind == "state":
            if detail in TERMINAL:
                self.pending = Pending()
                # The final state is emitted more than once (e.g. again to persist usage):
                # announce the outcome once per task.
                if self._final_spoken == ev.get("task_id"):
                    return ""
                self._final_spoken = ev.get("task_id", "")
                if detail == "COMPLETED":
                    return _clean_summary(data.get("summary") or "Done.")
                if detail == "CANCELLED":
                    return "The task was cancelled."
                return "I couldn't finish that. " + _short(data.get("error") or "")
            if detail == "PAUSED":
                return "Paused. Say resume when you're ready."
            summary = data.get("summary") or ""
            if detail == "WAITING_FOR_USER" and summary and self.pending.kind == "" \
                    and self._final_spoken != ev.get("task_id"):
                # The task stopped with a result for the user but it isn't a verified
                # completion (e.g. the model answered after every tool failed, or a step
                # the user must finish elsewhere). Say it, honestly labelled.
                self._final_spoken = ev.get("task_id", "")
                if summary.startswith("[UNVERIFIED]"):
                    return ("I couldn't confirm this with a source, but: "
                            + _clean_summary(summary))
                return _clean_summary(summary)
            return ""
        if kind == "action" and self.cfg.progress_every_s > 0:
            now = time.monotonic()
            if now - self._last_progress >= self.cfg.progress_every_s:
                self._last_progress = now
                return f"Still on it — {_short(detail, 60)}."
        return ""

    # ------------------------------------------------------------------ #
    # speaker: TTS -> paced audio frames (cancellable), or client-side TTS fallback
    # ------------------------------------------------------------------ #
    def say(self, text: str) -> None:
        if text:
            self._say.put_nowait((self._gen, text))

    async def _speak_loop(self) -> None:
        while True:
            gen, text = await self._say.get()
            if gen != self._gen:
                continue
            sentences = _chunks(text)
            nxt = asyncio.create_task(self.b.tts(sentences[0]))
            for i, sentence in enumerate(sentences):
                audio = await nxt
                if i + 1 < len(sentences):           # synthesize ahead while we play
                    nxt = asyncio.create_task(self.b.tts(sentences[i + 1]))
                if gen != self._gen:
                    break
                await self._play(gen, sentence, audio)
            else:
                continue
            if not nxt.done():
                nxt.cancel()

    async def _play(self, gen: int, sentence: str, audio: bytes | None) -> None:
        self._recent_spoken.append(_norm(sentence))
        await self._send({"type": "speak", "text": sentence, "audio": bool(audio)})
        if not audio:
            # Server TTS unavailable: the client speaks it (text fallback) and reports
            # playback so barge-in still works; estimate the duration meanwhile.
            self._client_playing_until = time.monotonic() + 0.4 + len(sentence.split()) / 2.6
            self._attention_until = max(self._attention_until, self._client_playing_until
                                        + self.cfg.attention_after_reply_s)
            await self._send({"type": "say", "text": sentence})
            return
        try:
            pcm = wav_to_pcm16k(audio)
        except Exception as e:
            log.warning("tts decode failed: %s", e)
            await self._send({"type": "say", "text": sentence})
            return
        self._streaming = True
        start = time.monotonic()
        sent_ms = 0
        try:
            for fr in frames(pcm):
                if gen != self._gen or self.transport is None:
                    return
                await self.transport.send_audio(fr)
                sent_ms += FRAME_MS
                # Stay <= ~240 ms ahead of real time so a barge-in stops quickly.
                ahead = sent_ms / 1000 - (time.monotonic() - start)
                if ahead > 0.24:
                    await asyncio.sleep(ahead - 0.2)
            await self._send({"type": "speak_end"})
            self._open_attention(self.cfg.attention_after_reply_s)
        finally:
            if gen == self._gen:
                self._streaming = False


def _chunks(text: str, limit: int = 180) -> list[str]:
    """Sentences for TTS (first audio sooner; long inputs get truncated by some TTS
    models), with over-long sentences split again at commas."""
    out: list[str] = []
    for sent in (x.strip() for x in _SENT.split(text)):
        while len(sent) > limit:
            cut = sent.rfind(", ", 0, limit)
            cut = cut + 1 if cut > 40 else sent.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            out.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if sent:
            out.append(sent)
    return out or [text]


def _sounds_like_name(word: str) -> bool:
    import difflib
    w = word.lower()
    if w in _NOT_NAME:
        return False
    return max(difflib.SequenceMatcher(None, w, f).ratio() for f in _NAME_FORMS) >= 0.6 \
        and w[-1] in "ou"


def _strip_name(text: str) -> tuple[str, bool]:
    """Remove a leading/trailing "Zoro" (any of its mishearings); report if it was there."""
    t = NAME_LEAD.sub("", text, count=1)
    if t == text:
        m = LEAD_WORD.match(text)
        # Only a word set off like a name ("Churro, ...") — not the start of a sentence.
        if m and _sounds_like_name(m.group(1)) and re.match(
                r"^\W*(?:(?:hey|hi|hello|ok|okay|yo|oh)\W+)?[a-z]+\s*[,.!?:;-]", text, re.I):
            t = text[m.end():]
    named = t != text
    t2 = NAME_TAIL.sub("", t)
    named = named or t2 != t
    if not named:
        return text, False
    t2 = t2.strip()
    return (t2[:1].upper() + t2[1:] if t2 else t2), True


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


def _short(s: str, n: int = 140) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def _clean_summary(s: str) -> str:
    s = re.sub(r"^\[(OK|PAUSED|UNVERIFIED)\]\s*", "", s or "")
    s = s.replace("Completed and verified: ", "Done: ")
    return _short(s, 220) or "Done."
