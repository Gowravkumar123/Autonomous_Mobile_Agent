package com.quantumleap.agent.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.view.Gravity
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.quantumleap.agent.BuildConfig
import com.quantumleap.agent.service.AgentAccessibilityService
import com.quantumleap.agent.service.AgentForegroundService
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
 * Companion UI with voice: give commands by speaking, hear the agent's questions read
 * aloud, and answer them by voice — all without touching the target app. Falls back to
 * typing throughout.
 */
class MainActivity : AppCompatActivity() {

    private val http = OkHttpClient()
    private val ui = Handler(Looper.getMainLooper())
    private lateinit var status: TextView
    private lateinit var panel: LinearLayout

    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var handlingQuestionFor: String? = null   // task_id of the question being handled
    private var voiceBusy = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ensureMicPermission()
        tts = TextToSpeech(this) { st -> ttsReady = (st == TextToSpeech.SUCCESS) }

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 64, 48, 48)
        }
        root.addView(TextView(this).apply { text = "OnePiece — Zoro"; textSize = 24f })
        root.addView(TextView(this).apply {
            text = "Backend: ${BuildConfig.BACKEND_HTTP_URL}"; setPadding(0, 8, 0, 24)
        })
        root.addView(Button(this).apply {
            text = "1 · Enable the controller (Accessibility)"
            setOnClickListener { startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)) }
        })

        val prompt = EditText(this).apply {
            hint = "Speak or type a task, e.g. 'order a chicken biryani, cheapest, stop before payment'"
            minLines = 2
        }
        root.addView(TextView(this).apply { text = "2 · Say or type the task"; setPadding(0, 24, 0, 8) })
        root.addView(prompt)

        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        row.addView(Button(this).apply {
            text = "🎤 Speak"
            setOnClickListener {
                if (!hasMic()) { ensureMicPermission(); return@setOnClickListener }
                listen(
                    onResult = { text -> prompt.setText(text); createTask(text) },
                    onError = { toast("Didn't catch that") })
            }
        })
        row.addView(Button(this).apply {
            text = "Run"
            setOnClickListener {
                val text = prompt.text.toString().trim()
                when {
                    text.isEmpty() -> toast("Say or type a task first")
                    AgentAccessibilityService.instance == null -> toast("Enable the controller first")
                    else -> createTask(text)
                }
            }
        })
        root.addView(row)

        // --- hands-free wake word ("Hi <name>") ---
        val prefs = getSharedPreferences("agent", MODE_PRIVATE)
        val nameField = EditText(this).apply {
            hint = "Assistant name (e.g. zoro)"
            setText(prefs.getString("name", "zoro"))
        }
        root.addView(TextView(this).apply {
            text = "3 · Hands-free wake word — say \"Hi <name>\""; setPadding(0, 24, 0, 8)
        })
        root.addView(nameField)
        val wakeRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        wakeRow.addView(Button(this).apply {
            text = "Enable wake word"
            setOnClickListener {
                if (!hasMic()) { ensureMicPermission(); return@setOnClickListener }
                val n = nameField.text.toString().trim().ifEmpty { "zoro" }.lowercase()
                prefs.edit().putString("name", n).apply()
                AgentForegroundService.wakeOn(this@MainActivity)
                speak("Listening. Say hi $n.")
                toast("Listening for 'Hi $n'")
            }
        })
        wakeRow.addView(Button(this).apply {
            text = "Disable"
            setOnClickListener { AgentForegroundService.wakeOff(this@MainActivity); toast("Wake word off") }
        })
        root.addView(wakeRow)

        status = TextView(this).apply { text = "Status: idle"; setPadding(0, 32, 0, 16) }
        root.addView(status)
        panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; visibility = LinearLayout.GONE
        }
        root.addView(panel)

        setContentView(root)
        startLoop()
    }

    override fun onDestroy() {
        tts?.shutdown()
        super.onDestroy()
    }

    // --------------------------------------------------------------- //
    private fun createTask(prompt: String) {
        val body = JSONObject().put("prompt", prompt).toString()
            .toRequestBody("application/json".toMediaType())
        val req = Request.Builder().url("${BuildConfig.BACKEND_HTTP_URL}/tasks").post(body).build()
        http.newCall(req).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                ui.post { toast("Could not reach backend: ${e.message}") }
            }
            override fun onResponse(call: Call, response: Response) {
                response.close(); ui.post { speak("Starting."); toast("Task started") }
            }
        })
    }

    private fun startLoop() {
        ui.post(object : Runnable {
            override fun run() {
                val svc = AgentAccessibilityService.instance
                status.text = "Status: " + (svc?.lastStatus ?: "controller not enabled")
                render(svc)   // voice Q&A is handled by AgentForegroundService in the background
                ui.postDelayed(this, 1000)
            }
        })
    }

    private fun render(svc: AgentAccessibilityService?) {
        panel.removeAllViews()
        if (svc == null) { panel.visibility = LinearLayout.GONE; return }
        val approval = svc.pendingApproval
        val question = svc.pendingQuestion
        if (approval == null && question == null) { panel.visibility = LinearLayout.GONE; return }
        panel.visibility = LinearLayout.VISIBLE

        if (approval != null) {
            panel.addView(TextView(this).apply {
                text = "⚠ ${approval.optString("title")}\n${approval.optString("summary")}" +
                        "\nReason: ${approval.optString("reason")}"
            })
            panel.addView(Button(this).apply {
                text = "Approve / I'll take over"; setOnClickListener { svc.answerApproval(true) }
            })
            panel.addView(Button(this).apply {
                text = "Decline"; setOnClickListener { svc.answerApproval(false) }
            })
            panel.addView(Button(this).apply {
                text = "Resume (after I finished paying)"; setOnClickListener { svc.resumeFromHandoff() }
            })
        }
        if (question != null) {
            val q = question.optJSONArray("questions")?.optString(0) ?: "The agent needs more info."
            panel.addView(TextView(this).apply { text = "❓ $q"; setPadding(0, 16, 0, 8) })
            val answer = EditText(this).apply { hint = "Type an answer (or use the mic above)" }
            panel.addView(answer)
            panel.addView(Button(this).apply {
                text = "Send answer"
                setOnClickListener {
                    val a = answer.text.toString().trim()
                    if (a.isNotEmpty()) { svc.answerClarification(a); ui.postDelayed({ moveTaskToBack(true) }, 400) }
                }
            })
        }
    }

    // ---------------- speech helpers ---------------- //
    private fun speak(text: String) = speakThen(text) {}

    private fun speakThen(text: String, onDone: () -> Unit) {
        val t = tts
        if (t == null || !ttsReady) { onDone(); return }
        val id = "u" + System.currentTimeMillis()
        t.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {}
            override fun onError(utteranceId: String?) { ui.post(onDone) }
            override fun onDone(utteranceId: String?) { ui.post(onDone) }
        })
        t.speak(text, TextToSpeech.QUEUE_FLUSH, null, id)
    }

    private fun listen(onResult: (String) -> Unit, onError: () -> Unit) {
        if (voiceBusy) return
        if (!SpeechRecognizer.isRecognitionAvailable(this)) { onError(); return }
        voiceBusy = true
        val recognizer = SpeechRecognizer.createSpeechRecognizer(this)
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
        }
        recognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) { status.text = "Status: listening…" }
            override fun onResults(results: Bundle?) {
                voiceBusy = false
                val best = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()
                recognizer.destroy()
                if (best.isNullOrBlank()) onError() else onResult(best)
            }
            override fun onError(error: Int) { voiceBusy = false; recognizer.destroy(); onError() }
            override fun onBeginningOfSpeech() {}
            override fun onEndOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
        try { recognizer.startListening(intent) }
        catch (e: Exception) { voiceBusy = false; recognizer.destroy(); onError() }
    }

    // ---------------- permissions ---------------- //
    private fun hasMic() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    private fun ensureMicPermission() {
        if (!hasMic()) ActivityCompat.requestPermissions(
            this, arrayOf(Manifest.permission.RECORD_AUDIO), 1)
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
}
