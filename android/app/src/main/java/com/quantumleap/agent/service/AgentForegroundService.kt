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
import android.os.IBinder
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale

/**
 * Foreground service with two jobs:
 *  1. Keep the app's process alive (persistent notification) so the WebSocket survives
 *     aggressive OEM battery management.
 *  2. Handle the agent's questions/approvals by VOICE in the background — speak the
 *     prompt and listen for the spoken answer — WITHOUT bringing the UI to the front.
 */
class AgentForegroundService : Service() {

    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var busy = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        tts = TextToSpeech(this) { st -> ttsReady = (st == TextToSpeech.SUCCESS) }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForegroundSafely()
        when (intent?.action) {
            ACTION_ASK -> {
                val q = intent.getStringExtra(EXTRA_PROMPT) ?: "I need more information."
                speakThenListen(q) { answer ->
                    AgentAccessibilityService.instance?.answerClarification(answer)
                }
            }
            ACTION_APPROVE -> {
                val summary = intent.getStringExtra(EXTRA_PROMPT) ?: "Approval needed."
                speakThenListen("$summary. Say approve to continue, or decline.") { answer ->
                    val a = answer.lowercase()
                    val yes = listOf("approve", "yes", "okay", "ok", "proceed", "confirm", "sure")
                    val no = listOf("decline", "no", "cancel", "stop", "don't")
                    when {
                        yes.any { it in a } -> AgentAccessibilityService.instance?.answerApproval(true)
                        no.any { it in a } -> AgentAccessibilityService.instance?.answerApproval(false)
                        else -> speak("I didn't catch that; please use the app to decide.")
                    }
                }
            }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        tts?.shutdown()
        super.onDestroy()
    }

    // ------------------------------------------------------------------ //
    private fun startForegroundSafely() {
        val channelId = "agent_fg"
        val nm = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            nm.createNotificationChannel(
                NotificationChannel(channelId, "Agent", NotificationManager.IMPORTANCE_LOW))
        }
        val notif: Notification = Notification.Builder(this, channelId)
            .setContentTitle("Personal AI Agent")
            .setContentText("Connected and ready")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setOngoing(true)
            .build()
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(NOTIF_ID, notif, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
            } else {
                startForeground(NOTIF_ID, notif)
            }
        } catch (e: Exception) {
            Log.w(TAG, "startForeground failed", e)
        }
    }

    private fun speak(text: String) {
        val t = tts ?: return
        if (ttsReady) t.speak(text, TextToSpeech.QUEUE_FLUSH, null, "u")
    }

    private fun speakThenListen(text: String, onAnswer: (String) -> Unit) {
        if (busy) return
        busy = true
        val t = tts
        if (t == null || !ttsReady) { busy = false; listen(onAnswer); return }
        t.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(id: String?) {}
            override fun onError(id: String?) { listen(onAnswer) }
            override fun onDone(id: String?) { listen(onAnswer) }
        })
        t.speak(text, TextToSpeech.QUEUE_FLUSH, null, "ask")
    }

    private fun listen(onAnswer: (String) -> Unit) {
        android.os.Handler(mainLooper).post {
            if (!SpeechRecognizer.isRecognitionAvailable(this)) { busy = false; return@post }
            val recognizer = SpeechRecognizer.createSpeechRecognizer(this)
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            }
            recognizer.setRecognitionListener(object : RecognitionListener {
                override fun onResults(results: Bundle?) {
                    busy = false
                    val best = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()
                    recognizer.destroy()
                    if (!best.isNullOrBlank()) onAnswer(best)
                }
                override fun onError(error: Int) { busy = false; recognizer.destroy() }
                override fun onReadyForSpeech(p: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onEndOfSpeech() {}
                override fun onRmsChanged(v: Float) {}
                override fun onBufferReceived(b: ByteArray?) {}
                override fun onPartialResults(p: Bundle?) {}
                override fun onEvent(t: Int, p: Bundle?) {}
            })
            try { recognizer.startListening(intent) }
            catch (e: Exception) { busy = false; recognizer.destroy() }
        }
    }

    companion object {
        private const val TAG = "AgentFG"
        private const val NOTIF_ID = 42
        const val ACTION_ASK = "com.quantumleap.agent.ASK"
        const val ACTION_APPROVE = "com.quantumleap.agent.APPROVE"
        const val EXTRA_PROMPT = "prompt"

        fun keepAlive(ctx: Context) = start(ctx, null, null)
        fun askUser(ctx: Context, question: String) = start(ctx, ACTION_ASK, question)
        fun requestApproval(ctx: Context, summary: String) = start(ctx, ACTION_APPROVE, summary)

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
