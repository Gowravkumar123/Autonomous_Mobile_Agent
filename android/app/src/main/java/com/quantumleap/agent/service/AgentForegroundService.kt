package com.quantumleap.agent.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import com.quantumleap.agent.BuildConfig
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONObject
import java.io.IOException
import java.util.Locale

/**
 * Foreground service: keeps the process alive, and hosts the voice assistant.
 *
 * Wake word: it continuously listens; when it hears the assistant's name (default
 * "zoro", e.g. "hi zoro"), it greets you, listens for a command, and runs it — all
 * hands-free, in the background. It also speaks the agent's clarifying questions and
 * approvals and takes your spoken answer.
 *
 * NOTE: Android's SpeechRecognizer is not a true always-on hotword engine, so this is a
 * continuous listen-loop (heavier on battery, less precise than a dedicated wake engine).
 */
class AgentForegroundService : Service() {

    private enum class Mode { IDLE, WAKE, BUSY }

    private val main = Handler(Looper.getMainLooper())
    private val http = OkHttpClient()
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    @Volatile private var mode = Mode.IDLE
    private var recognizer: SpeechRecognizer? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        tts = TextToSpeech(this) { st -> ttsReady = (st == TextToSpeech.SUCCESS) }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForegroundSafely()
        when (intent?.action) {
            ACTION_WAKE_ON -> { mode = Mode.WAKE; armWake() }
            ACTION_WAKE_OFF -> { mode = Mode.IDLE; destroyRecognizer() }
            ACTION_ASK -> interrupt {
                speakThenListen(intent.getStringExtra(EXTRA_PROMPT) ?: "I need more information.") { a ->
                    AgentAccessibilityService.instance?.answerClarification(a)
                }
            }
            ACTION_APPROVE -> interrupt {
                val s = intent.getStringExtra(EXTRA_PROMPT) ?: "Approval needed."
                speakThenListen("$s. Say approve to continue, or decline.") { a ->
                    val l = a.lowercase()
                    when {
                        listOf("approve", "yes", "okay", "ok", "proceed", "confirm").any { it in l } ->
                            AgentAccessibilityService.instance?.answerApproval(true)
                        listOf("decline", "no", "cancel", "stop").any { it in l } ->
                            AgentAccessibilityService.instance?.answerApproval(false)
                        else -> speak("I didn't catch that; use the app to decide.")
                    }
                }
            }
        }
        return START_STICKY
    }

    override fun onDestroy() { destroyRecognizer(); tts?.shutdown(); super.onDestroy() }

    // ---------------- wake loop ---------------- //
    private fun assistantName(): String =
        getSharedPreferences("agent", Context.MODE_PRIVATE).getString("name", "zoro") ?: "zoro"

    private fun armWake() {
        if (mode != Mode.WAKE) return
        listen(
            onResult = { text ->
                val name = assistantName().lowercase()
                if (text.lowercase().contains(name)) onWake(text, name)
                else reArm(300)
            },
            onError = { reArm(600) })
    }

    private fun reArm(delay: Long) { if (mode == Mode.WAKE) main.postDelayed({ armWake() }, delay) }

    /** Heard the name. If the utterance already includes a command ("zoro, open camera"),
     *  use it; otherwise greet and listen for the command. */
    private fun onWake(utterance: String, name: String) {
        mode = Mode.BUSY
        val after = utterance.lowercase().substringAfter(name).trim().trimStart(',', '.', ' ')
        val command = after.removePrefix("hi").removePrefix("hey").trim()
        if (command.length > 3) {
            runCommand(command)
        } else {
            speakThenListen("Yes?") { cmd -> runCommand(cmd) }
        }
    }

    private fun runCommand(command: String) {
        if (command.isBlank()) { resumeWake(); return }
        createTask(command)
        speak("On it.")
        resumeWake()
    }

    private fun resumeWake() { mode = Mode.WAKE; reArm(800) }

    private fun interrupt(block: () -> Unit) {
        val wasWake = (mode == Mode.WAKE)
        mode = Mode.BUSY
        destroyRecognizer()
        block()
        // speakThenListen/listen will call back; when done they don't auto-resume, so
        // schedule a resume to wake-mode shortly after.
        if (wasWake) main.postDelayed({ if (mode == Mode.BUSY) resumeWake() }, 12000)
    }

    // ---------------- task ---------------- //
    private fun createTask(prompt: String) {
        val body = JSONObject().put("prompt", prompt).toString()
            .toRequestBody("application/json".toMediaType())
        val req = Request.Builder().url("${BuildConfig.BACKEND_HTTP_URL}/tasks").post(body).build()
        http.newCall(req).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) { Log.w(TAG, "task post failed: ${e.message}") }
            override fun onResponse(call: Call, response: Response) { response.close() }
        })
    }

    // ---------------- speech ---------------- //
    private fun speak(text: String) { if (ttsReady) tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "u") }

    private fun speakThenListen(text: String, onAnswer: (String) -> Unit) {
        val t = tts
        if (t == null || !ttsReady) { listen(onAnswer) { }; return }
        t.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(id: String?) {}
            override fun onError(id: String?) { main.post { listen(onAnswer) { } } }
            override fun onDone(id: String?) { main.post { listen(onAnswer) { } } }
        })
        t.speak(text, TextToSpeech.QUEUE_FLUSH, null, "ask")
    }

    private fun listen(onResult: (String) -> Unit, onError: () -> Unit) {
        main.post {
            if (!SpeechRecognizer.isRecognitionAvailable(this)) { onError(); return@post }
            destroyRecognizer()
            val r = SpeechRecognizer.createSpeechRecognizer(this)
            recognizer = r
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            }
            r.setRecognitionListener(object : RecognitionListener {
                override fun onResults(results: Bundle?) {
                    val best = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()
                    if (best.isNullOrBlank()) onError() else onResult(best)
                }
                override fun onError(error: Int) { onError() }
                override fun onReadyForSpeech(p: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onEndOfSpeech() {}
                override fun onRmsChanged(v: Float) {}
                override fun onBufferReceived(b: ByteArray?) {}
                override fun onPartialResults(p: Bundle?) {}
                override fun onEvent(t: Int, p: Bundle?) {}
            })
            try { r.startListening(intent) } catch (e: Exception) { onError() }
        }
    }

    private fun destroyRecognizer() {
        try { recognizer?.destroy() } catch (_: Exception) {}
        recognizer = null
    }

    private fun startForegroundSafely() {
        val channelId = "agent_fg"
        val nm = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            nm.createNotificationChannel(
                NotificationChannel(channelId, "Agent", NotificationManager.IMPORTANCE_LOW))
        }
        val notif: Notification = Notification.Builder(this, channelId)
            .setContentTitle("OnePiece")
            .setContentText(if (mode == Mode.WAKE) "Listening for \"${assistantName()}\"" else "Connected and ready")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setOngoing(true).build()
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                startForeground(NOTIF_ID, notif, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
            else startForeground(NOTIF_ID, notif)
        } catch (e: Exception) { Log.w(TAG, "startForeground failed", e) }
    }

    companion object {
        private const val TAG = "AgentFG"
        private const val NOTIF_ID = 42
        const val ACTION_ASK = "com.quantumleap.agent.ASK"
        const val ACTION_APPROVE = "com.quantumleap.agent.APPROVE"
        const val ACTION_WAKE_ON = "com.quantumleap.agent.WAKE_ON"
        const val ACTION_WAKE_OFF = "com.quantumleap.agent.WAKE_OFF"
        const val EXTRA_PROMPT = "prompt"

        fun keepAlive(ctx: Context) = start(ctx, null, null)
        fun askUser(ctx: Context, q: String) = start(ctx, ACTION_ASK, q)
        fun requestApproval(ctx: Context, s: String) = start(ctx, ACTION_APPROVE, s)
        fun wakeOn(ctx: Context) = start(ctx, ACTION_WAKE_ON, null)
        fun wakeOff(ctx: Context) = start(ctx, ACTION_WAKE_OFF, null)

        private fun start(ctx: Context, action: String?, prompt: String?) {
            val i = Intent(ctx, AgentForegroundService::class.java).apply {
                this.action = action
                if (prompt != null) putExtra(EXTRA_PROMPT, prompt)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) ctx.startForegroundService(i)
            else ctx.startService(i)
        }
    }
}
