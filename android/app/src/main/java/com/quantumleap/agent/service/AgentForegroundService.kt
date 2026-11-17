package com.quantumleap.agent.service

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
 * Foreground service: keeps the process alive and handles voice by PUSH-TO-TALK.
 *
 * Why not always-on "Hi Zoro": Android's SpeechRecognizer has no real hotword mode — it
 * can only be restarted in a tight loop that never stops, beeps, and drains battery. So
 * instead the persistent notification has a "Speak" action: tap it (or the app mic),
 * say one command, done. Each listen is a single shot — no loop. A true wake word needs
 * a dedicated engine (e.g. Picovoice Porcupine).
 */
class AgentForegroundService : Service() {

    private val main = Handler(Looper.getMainLooper())
    private val http = OkHttpClient()
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var recognizer: SpeechRecognizer? = null
    @Volatile private var busy = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        tts = TextToSpeech(this) { st -> ttsReady = (st == TextToSpeech.SUCCESS) }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForegroundSafely()
        when (intent?.action) {
            ACTION_LISTEN -> startCommandListen()
            ACTION_ASK -> handleQuestion(intent.getStringExtra(EXTRA_PROMPT) ?: "I need more information.") { a ->
                AgentAccessibilityService.instance?.answerClarification(a)
            }
            ACTION_APPROVE -> handleQuestion(
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
            }
        }
        return START_STICKY
    }

    override fun onDestroy() { destroyRecognizer(); tts?.shutdown(); super.onDestroy() }

    // ------------------------------------------------------------------ //
    private fun startCommandListen() {
        if (busy) return
        speakThen("Listening") { listenCommand(0) }
    }

    private fun listenCommand(attempt: Int) {
        listenOnce(
            onResult = { runCommand(it) },
            onFail = { if (attempt < 1) listenCommand(attempt + 1) else speak("I didn't catch that.") })
    }

    private fun runCommand(command: String) {
        if (command.isBlank()) return
        createTask(command)
        speak("On it.")
    }

    private fun handleQuestion(question: String, onAnswer: (String) -> Unit) {
        speakThen(question) { listenAnswer(question, onAnswer, 0) }
    }

    private fun listenAnswer(question: String, onAnswer: (String) -> Unit, attempt: Int) {
        listenOnce(
            onResult = { onAnswer(it) },
            onFail = {
                if (attempt < 1) listenAnswer(question, onAnswer, attempt + 1)
                else speak("I didn't catch that; you can answer in the app.")
            })
    }

    // ------------------------------------------------------------------ //
    private fun createTask(prompt: String) {
        val body = JSONObject().put("prompt", prompt).toString()
            .toRequestBody("application/json".toMediaType())
        val req = Request.Builder().url("${BuildConfig.BACKEND_HTTP_URL}/tasks").post(body).build()
        http.newCall(req).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.w(TAG, "task post failed: ${e.message}"); main.post { speak("I couldn't reach the server.") }
            }
            override fun onResponse(call: Call, response: Response) { response.close() }
        })
    }

    private fun speak(text: String) { if (ttsReady) tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "u") }

    private fun speakThen(text: String, after: () -> Unit) {
        val t = tts
        if (t == null || !ttsReady) { main.post(after); return }
        t.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(id: String?) {}
            override fun onError(id: String?) { main.post(after) }
            override fun onDone(id: String?) { main.post(after) }
        })
        t.speak(text, TextToSpeech.QUEUE_FLUSH, null, "say")
    }

    /** One single-shot recognizer at a time. No looping. */
    private fun listenOnce(onResult: (String) -> Unit, onFail: () -> Unit) {
        destroyRecognizer()
        busy = true
        main.postDelayed({
            if (!SpeechRecognizer.isRecognitionAvailable(this)) { busy = false; onFail(); return@postDelayed }
            val r = SpeechRecognizer.createSpeechRecognizer(this)
            recognizer = r
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            }
            r.setRecognitionListener(object : RecognitionListener {
                override fun onResults(results: Bundle?) {
                    busy = false
                    val best = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()
                    destroyRecognizer()
                    if (best.isNullOrBlank()) onFail() else onResult(best)
                }
                override fun onError(error: Int) { busy = false; destroyRecognizer(); onFail() }
                override fun onReadyForSpeech(p: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onEndOfSpeech() {}
                override fun onRmsChanged(v: Float) {}
                override fun onBufferReceived(b: ByteArray?) {}
                override fun onPartialResults(p: Bundle?) {}
                override fun onEvent(t: Int, p: Bundle?) {}
            })
            try { r.startListening(intent) } catch (e: Exception) { busy = false; onFail() }
        }, 300)
    }

    private fun destroyRecognizer() {
        try { recognizer?.cancel(); recognizer?.destroy() } catch (_: Exception) {}
        recognizer = null
    }

    private fun startForegroundSafely() {
        val channelId = "agent_fg"
        val nm = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            nm.createNotificationChannel(
                NotificationChannel(channelId, "Agent", NotificationManager.IMPORTANCE_LOW))
        }
        val listenPI = PendingIntent.getService(
            this, 1, Intent(this, AgentForegroundService::class.java).setAction(ACTION_LISTEN),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val notif: Notification = Notification.Builder(this, channelId)
            .setContentTitle("OnePiece")
            .setContentText("Tap Speak to give a voice command")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setOngoing(true)
            .addAction(Notification.Action.Builder(
                android.R.drawable.ic_btn_speak_now, "🎤 Speak", listenPI).build())
            .build()
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
        const val ACTION_LISTEN = "com.quantumleap.agent.LISTEN"
        const val EXTRA_PROMPT = "prompt"

        fun keepAlive(ctx: Context) = start(ctx, null, null)
        fun askUser(ctx: Context, q: String) = start(ctx, ACTION_ASK, q)
        fun requestApproval(ctx: Context, s: String) = start(ctx, ACTION_APPROVE, s)
        fun listen(ctx: Context) = start(ctx, ACTION_LISTEN, null)

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
