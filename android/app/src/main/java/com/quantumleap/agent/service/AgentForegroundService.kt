package com.quantumleap.agent.service

import ai.picovoice.porcupine.Porcupine
import ai.picovoice.porcupine.PorcupineManager
import ai.picovoice.porcupine.PorcupineManagerCallback
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.IBinder
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import com.quantumleap.agent.BuildConfig
import com.quantumleap.agent.net.Auth.withAuth
import com.quantumleap.agent.net.VoiceConversation
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.Locale

/**
 * Foreground service: keeps the process alive, runs an on-device WAKE WORD (Picovoice
 * Porcupine — low-power, no SpeechRecognizer loop), and handles voice commands + the
 * task's spoken Q&A. Porcupine and the SpeechRecognizer never hold the mic at once:
 * Porcupine is paused for each single-shot recognition and resumed after.
 *
 * Wake word: a custom "Hey Zoro" keyword if `assets/zoro.ppn` is bundled, otherwise the
 * built-in "Jarvis". Needs a free Picovoice AccessKey (set in the app). Falls back to
 * push-to-talk (the notification "Speak" action) if no key / Porcupine can't start.
 */
class AgentForegroundService : Service() {

    private val main = Handler(Looper.getMainLooper())
    private val http = OkHttpClient()
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var recognizer: SpeechRecognizer? = null
    private var player: android.media.MediaPlayer? = null
    private var status = ""

    private var porcupine: PorcupineManager? = null
    @Volatile private var wakeWanted = false

    // Full-duplex conversation with the backend voice gateway (server VAD + barge-in).
    private var conversation: VoiceConversation? = null
    @Volatile private var lastActivity = 0L
    private val idleCheck = object : Runnable {
        override fun run() {
            if (conversation == null) return
            if (System.currentTimeMillis() - lastActivity > CONVERSATION_IDLE_MS) endConversation()
            else main.postDelayed(this, 5000)
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        tts = TextToSpeech(this) { st -> ttsReady = (st == TextToSpeech.SUCCESS) }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForegroundSafely()
        when (intent?.action) {
            ACTION_WAKE_ON -> { wakeWanted = true; startWake(spoken = true) }
            ACTION_WAKE_OFF -> { wakeWanted = false; stopWake() }
            ACTION_LISTEN -> { pauseWake(); speakThen("Listening") { listenCommand(0) } }
            ACTION_STOP -> stopTasks()
            ACTION_CONVERSE -> if (conversation == null) startConversation() else endConversation()
            // While a conversation is live the server's voice session narrates questions and
            // approvals itself, so they aren't spoken twice.
            ACTION_ASK -> if (conversation == null) { pauseWake(); handleQuestion(intent.getStringExtra(EXTRA_PROMPT) ?: "I need more information.") { a ->
                AgentAccessibilityService.instance?.answerClarification(a)
            } }
            ACTION_APPROVE -> if (conversation != null) Unit
            else if (intent.getIntExtra(EXTRA_RISK, 0) >= 4) {
                // Blueprint: a spoken yes is never enough for payment / sign-in.
                speak((intent.getStringExtra(EXTRA_PROMPT) ?: "Payment step") +
                    ". Please confirm it on the approval screen in the app.")
            } else { pauseWake(); handleQuestion(
                (intent.getStringExtra(EXTRA_PROMPT) ?: "Approval needed.") +
                    ". Say approve to continue, or decline.") { a ->
                val l = a.lowercase()
                when {
                    listOf("approve", "yes", "okay", "ok", "proceed", "confirm", "do it").any { it in l } ->
                        AgentAccessibilityService.instance?.answerApproval(true)
                    listOf("decline", "no", "cancel", "stop", "don't").any { it in l } ->
                        AgentAccessibilityService.instance?.answerApproval(false)
                    else -> speak("I didn't catch that; please decide in the app.")
                }
            } }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        conversation?.stop(); conversation = null
        stopWake(); destroyRecognizer(); releasePlayer(); tts?.shutdown(); super.onDestroy()
    }

    // ---------------- full-duplex conversation ---------------- //
    private fun sessionId(): String {
        val prefs = getSharedPreferences("agent", Context.MODE_PRIVATE)
        return prefs.getString("voice_sid", null) ?: ("zoro-" + java.util.UUID.randomUUID()
            .toString().take(12)).also { prefs.edit().putString("voice_sid", it).apply() }
    }

    private fun startConversation() {
        pauseWake()                 // Porcupine and the conversation can't share the mic
        destroyRecognizer()
        lastActivity = System.currentTimeMillis()
        val conv = VoiceConversation(this, sessionId(), object : VoiceConversation.Listener {
            override fun onState(state: String) {
                main.post {
                    Log.i(TAG, "conversation: $state")
                    if (state == "live") status = "Listening - just talk (interrupt any time)"
                    if (state.startsWith("error")) status = "Voice: " + state.removePrefix("error:")
                    if (state == "closed" || state.startsWith("error")) {
                        main.removeCallbacks(idleCheck)
                        conversation = null
                        resumeWake()
                    }
                    updateNotification(listening = porcupine != null)
                }
            }
            override fun onTranscript(text: String, final: Boolean) {
                if (final) main.post { status = "You: $text"; updateNotification(false) }
            }
            override fun onAgentText(text: String) {
                main.post { status = "Zoro: $text"; updateNotification(false) }
            }
            override fun onSay(text: String, done: () -> Unit) {
                main.post { androidSpeakThen(text, done) }
            }
            override fun onActivity() { lastActivity = System.currentTimeMillis() }
        })
        conversation = conv
        conv.start("duplex")
        main.postDelayed(idleCheck, 5000)
        updateNotification(false)
    }

    private fun endConversation() {
        main.removeCallbacks(idleCheck)
        val c = conversation
        conversation = null
        c?.stop()
        status = ""
        updateNotification(false)
        resumeWake()
    }

    // ---------------- wake word (Porcupine) ---------------- //
    private fun accessKey(): String =
        getSharedPreferences("agent", Context.MODE_PRIVATE).getString("pv_key", "") ?: ""

    private fun startWake(spoken: Boolean = false) {
        if (!wakeWanted) return
        val key = accessKey()
        if (key.isBlank()) { if (spoken) speak("Add your Picovoice key first."); return }
        if (porcupine != null) return
        try {
            val builder = PorcupineManager.Builder().setAccessKey(key).setSensitivity(0.6f)
            val custom = customKeywordPath()
            if (custom != null) builder.setKeywordPath(custom)
            else builder.setKeyword(Porcupine.BuiltInKeyword.JARVIS)
            porcupine = builder.build(applicationContext,
                PorcupineManagerCallback { _ -> main.post { onWakeDetected() } })
            porcupine?.start()
            updateNotification(listening = true)
            if (spoken) speak(if (custom != null) "Say hey Zoro." else "Say Jarvis.")
            Log.i(TAG, "wake word on (${if (custom != null) "Hey Zoro" else "Jarvis"})")
        } catch (e: Exception) {
            Log.w(TAG, "Porcupine start failed: ${e.message}")
            porcupine = null
            if (spoken) speak("Wake word could not start; use the Speak button.")
        }
    }

    /** Release Porcupine's mic without clearing the wake preference (so we can resume). */
    private fun pauseWake() {
        try { porcupine?.stop(); porcupine?.delete() } catch (_: Exception) {}
        porcupine = null
    }

    private fun stopWake() { pauseWake(); updateNotification(listening = false) }

    private fun resumeWake() { if (wakeWanted) main.postDelayed({ startWake() }, 400) }

    private fun customKeywordPath(): String? = try {
        val f = File(filesDir, "zoro.ppn")
        if (!f.exists()) assets.open("zoro.ppn").use { inp -> f.outputStream().use { inp.copyTo(it) } }
        f.absolutePath
    } catch (e: Exception) { null }   // no custom keyword bundled -> use built-in

    private fun onWakeDetected() {
        pauseWake()
        // The wake word opens a hands-free conversation (it ends itself when idle).
        startConversation()
    }

    // ---------------- voice flows ---------------- //
    private fun listenCommand(attempt: Int) {
        listenOnce(
            onResult = { runCommand(it) },
            onFail = {
                if (attempt < 1) listenCommand(attempt + 1)
                else { speak("I didn't catch that."); resumeWake() }
            })
    }

    private fun runCommand(command: String) {
        if (command.isNotBlank()) { createTask(command); speak("On it.") }
        resumeWake()
    }

    private fun handleQuestion(question: String, onAnswer: (String) -> Unit) {
        speakThen(question) { listenAnswer(question, onAnswer, 0) }
    }

    private fun listenAnswer(question: String, onAnswer: (String) -> Unit, attempt: Int) {
        listenOnce(
            onResult = { onAnswer(it); resumeWake() },
            onFail = {
                if (attempt < 1) listenAnswer(question, onAnswer, attempt + 1)
                else { speak("I didn't catch that; you can answer in the app."); resumeWake() }
            })
    }

    // ---------------- helpers ---------------- //
    private fun createTask(prompt: String) {
        val body = JSONObject().put("prompt", prompt).toString()
            .toRequestBody("application/json".toMediaType())
        val req = Request.Builder().url("${BuildConfig.BACKEND_HTTP_URL}/tasks").post(body)
            .withAuth(this).build()
        http.newCall(req).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.w(TAG, "task post failed: ${e.message}"); main.post { speak("I couldn't reach the server.") }
            }
            override fun onResponse(call: Call, response: Response) {
                val code = response.code
                response.close()
                if (code == 401) main.post { speak("This phone isn't paired yet. Open the app and enter a pairing code.") }
            }
        })
    }

    private fun stopTasks() {
        val req = Request.Builder().url("${BuildConfig.BACKEND_HTTP_URL}/stop")
            .post("".toRequestBody("application/json".toMediaType())).withAuth(this).build()
        http.newCall(req).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {}
            override fun onResponse(call: Call, response: Response) { response.close(); main.post { speak("Stopped.") } }
        })
    }

    private fun speak(text: String) = speakThen(text) {}

    /** Speak via Groq TTS (nicer voice); fall back to the device TTS on any failure. */
    private fun speakThen(text: String, after: () -> Unit) {
        val body = JSONObject().put("text", text).toString()
            .toRequestBody("application/json".toMediaType())
        val req = Request.Builder().url("${BuildConfig.BACKEND_HTTP_URL}/voice/tts").post(body)
            .withAuth(this).build()
        http.newCall(req).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) { main.post { androidSpeakThen(text, after) } }
            override fun onResponse(call: Call, response: Response) {
                val bytes = if (response.isSuccessful) response.body?.bytes() else null
                response.close()
                main.post {
                    if (bytes == null || bytes.isEmpty()) androidSpeakThen(text, after)
                    else playAudio(bytes, onDone = after, onFail = { androidSpeakThen(text, after) })
                }
            }
        })
    }

    private fun playAudio(bytes: ByteArray, onDone: () -> Unit, onFail: () -> Unit) {
        try {
            val f = File(cacheDir, "tts.wav"); f.writeBytes(bytes)
            releasePlayer()
            player = android.media.MediaPlayer().apply {
                setDataSource(f.absolutePath)
                setOnCompletionListener { main.post(onDone) }
                setOnErrorListener { _, _, _ -> main.post(onFail); true }
                prepare(); start()
            }
        } catch (e: Exception) { onFail() }
    }

    private fun androidSpeakThen(text: String, after: () -> Unit) {
        val t = tts
        if (t == null || !ttsReady) { main.post(after); return }
        t.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(id: String?) {}
            override fun onError(id: String?) { main.post(after) }
            override fun onDone(id: String?) { main.post(after) }
        })
        t.speak(text, TextToSpeech.QUEUE_FLUSH, null, "say")
    }

    private fun releasePlayer() { try { player?.release() } catch (_: Exception) {}; player = null }

    private fun listenOnce(onResult: (String) -> Unit, onFail: () -> Unit) {
        destroyRecognizer()
        main.postDelayed({
            if (!SpeechRecognizer.isRecognitionAvailable(this)) { onFail(); return@postDelayed }
            val r = SpeechRecognizer.createSpeechRecognizer(this)
            recognizer = r
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            }
            r.setRecognitionListener(object : RecognitionListener {
                override fun onResults(results: Bundle?) {
                    val best = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()
                    destroyRecognizer()
                    if (best.isNullOrBlank()) onFail() else onResult(best)
                }
                override fun onError(error: Int) { destroyRecognizer(); onFail() }
                override fun onReadyForSpeech(p: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onEndOfSpeech() {}
                override fun onRmsChanged(v: Float) {}
                override fun onBufferReceived(b: ByteArray?) {}
                override fun onPartialResults(p: Bundle?) {}
                override fun onEvent(t: Int, p: Bundle?) {}
            })
            try { r.startListening(intent) } catch (e: Exception) { onFail() }
        }, 300)
    }

    private fun destroyRecognizer() {
        try { recognizer?.cancel(); recognizer?.destroy() } catch (_: Exception) {}
        recognizer = null
    }

    // ---------------- notification ---------------- //
    private fun updateNotification(listening: Boolean) {
        try {
            getSystemService(NotificationManager::class.java).notify(NOTIF_ID, buildNotification(listening))
        } catch (_: Exception) {}
    }

    private fun buildNotification(listening: Boolean): Notification {
        val channelId = "agent_fg"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            getSystemService(NotificationManager::class.java).createNotificationChannel(
                NotificationChannel(channelId, "Agent", NotificationManager.IMPORTANCE_LOW))
        }
        val listenPI = PendingIntent.getService(
            this, 1, Intent(this, AgentForegroundService::class.java).setAction(ACTION_LISTEN),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val talkPI = PendingIntent.getService(
            this, 3, Intent(this, AgentForegroundService::class.java).setAction(ACTION_CONVERSE),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val stopPI = PendingIntent.getService(
            this, 2, Intent(this, AgentForegroundService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        return Notification.Builder(this, channelId)
            .setContentTitle("OnePiece")
            .setContentText(when {
                status.isNotBlank() -> status
                conversation != null -> "Conversation on - just talk"
                listening -> "Wake word on - say the word, or tap Talk"
                else -> "Tap Talk for a conversation, or Speak for one command"
            })
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setOngoing(true)
            .addAction(Notification.Action.Builder(
                android.R.drawable.ic_btn_speak_now,
                if (conversation != null) "🔇 End talk" else "💬 Talk", talkPI).build())
            .addAction(Notification.Action.Builder(
                android.R.drawable.ic_btn_speak_now, "🎤 Speak", listenPI).build())
            .addAction(Notification.Action.Builder(
                android.R.drawable.ic_delete, "⛔ Stop", stopPI).build())
            .build()
    }

    private fun startForegroundSafely() {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                startForeground(NOTIF_ID, buildNotification(porcupine != null),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
            else startForeground(NOTIF_ID, buildNotification(porcupine != null))
        } catch (e: Exception) { Log.w(TAG, "startForeground failed", e) }
    }

    companion object {
        private const val TAG = "AgentFG"
        private const val NOTIF_ID = 42
        const val ACTION_ASK = "com.quantumleap.agent.ASK"
        const val ACTION_APPROVE = "com.quantumleap.agent.APPROVE"
        const val ACTION_LISTEN = "com.quantumleap.agent.LISTEN"
        const val ACTION_STOP = "com.quantumleap.agent.STOP"
        const val ACTION_WAKE_ON = "com.quantumleap.agent.WAKE_ON"
        const val ACTION_WAKE_OFF = "com.quantumleap.agent.WAKE_OFF"
        const val ACTION_CONVERSE = "com.quantumleap.agent.CONVERSE"
        const val EXTRA_PROMPT = "prompt"
        const val EXTRA_RISK = "risk"
        private const val CONVERSATION_IDLE_MS = 45_000L

        fun keepAlive(ctx: Context) = start(ctx, null, null)
        fun askUser(ctx: Context, q: String) = start(ctx, ACTION_ASK, q)
        fun requestApproval(ctx: Context, s: String, risk: Int = 0) =
            start(ctx, ACTION_APPROVE, s, risk)
        fun converse(ctx: Context) = start(ctx, ACTION_CONVERSE, null)
        fun listen(ctx: Context) = start(ctx, ACTION_LISTEN, null)
        fun wakeOn(ctx: Context) = start(ctx, ACTION_WAKE_ON, null)
        fun wakeOff(ctx: Context) = start(ctx, ACTION_WAKE_OFF, null)

        private fun start(ctx: Context, action: String?, prompt: String?, risk: Int = 0) {
            val i = Intent(ctx, AgentForegroundService::class.java).apply {
                this.action = action
                if (prompt != null) putExtra(EXTRA_PROMPT, prompt)
                putExtra(EXTRA_RISK, risk)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) ctx.startForegroundService(i)
            else ctx.startService(i)
        }
    }
}
