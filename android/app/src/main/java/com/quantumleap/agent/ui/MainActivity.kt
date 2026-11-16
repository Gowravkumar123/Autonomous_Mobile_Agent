package com.quantumleap.agent.ui

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.Gravity
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.quantumleap.agent.BuildConfig
import com.quantumleap.agent.service.AgentAccessibilityService
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONObject
import java.io.IOException

/**
 * Companion UI: enable the controller, type a natural-language task, watch progress,
 * and answer approvals / resume after a payment handoff. Built programmatically to
 * keep the scaffold to a single file.
 */
class MainActivity : AppCompatActivity() {

    private val http = OkHttpClient()
    private val ui = Handler(Looper.getMainLooper())
    private lateinit var status: TextView
    private lateinit var approvalPanel: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 64, 48, 48)
        }

        root.addView(TextView(this).apply {
            text = "Personal AI Agent"
            textSize = 24f
        })
        root.addView(TextView(this).apply {
            text = "Backend: ${BuildConfig.BACKEND_HTTP_URL}"
            setPadding(0, 8, 0, 24)
        })

        val enableBtn = Button(this).apply {
            text = "1 · Enable the controller (Accessibility)"
            setOnClickListener {
                startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
            }
        }
        root.addView(enableBtn)

        val prompt = EditText(this).apply {
            hint = "e.g. Book two train tickets from Hyderabad to Bengaluru tomorrow, AC 3-tier, stop before payment"
            minLines = 2
        }
        root.addView(TextView(this).apply { text = "2 · Describe the task"; setPadding(0, 24, 0, 8) })
        root.addView(prompt)

        root.addView(Button(this).apply {
            text = "Run task"
            setOnClickListener {
                val text = prompt.text.toString().trim()
                if (text.isEmpty()) {
                    toast("Type a task first")
                } else if (AgentAccessibilityService.instance == null) {
                    toast("Enable the controller first")
                } else {
                    createTask(text)
                }
            }
        })

        status = TextView(this).apply {
            text = "Status: idle"
            setPadding(0, 32, 0, 16)
        }
        root.addView(status)

        approvalPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            visibility = LinearLayout.GONE
        }
        root.addView(approvalPanel)

        setContentView(root)
        startStatusLoop()
    }

    private fun createTask(prompt: String) {
        val body = JSONObject().put("prompt", prompt).toString()
            .toRequestBody("application/json".toMediaType())
        val req = Request.Builder().url("${BuildConfig.BACKEND_HTTP_URL}/tasks").post(body).build()
        http.newCall(req).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) =
                ui.post { toast("Could not reach backend: ${e.message}") }

            override fun onResponse(call: Call, response: Response) =
                ui.post { toast("Task started") }
        })
    }

    private fun startStatusLoop() {
        ui.post(object : Runnable {
            override fun run() {
                val svc = AgentAccessibilityService.instance
                status.text = "Status: " + (svc?.lastStatus ?: "controller not enabled")
                renderApproval(svc)
                ui.postDelayed(this, 1000)
            }
        })
    }

    private fun renderApproval(svc: AgentAccessibilityService?) {
        val pending = svc?.pendingApproval
        approvalPanel.removeAllViews()
        if (svc == null) { approvalPanel.visibility = LinearLayout.GONE; return }

        if (pending != null) {
            approvalPanel.visibility = LinearLayout.VISIBLE
            approvalPanel.addView(TextView(this).apply {
                text = "⚠ ${pending.optString("title")}\n${pending.optString("summary")}" +
                        (pending.optDouble("amount").takeIf { !it.isNaN() }
                            ?.let { "\nAmount: ₹$it" } ?: "") +
                        "\nReason: ${pending.optString("reason")}"
            })
            approvalPanel.addView(Button(this).apply {
                text = "Approve / I'll take over"
                setOnClickListener { svc.answerApproval(true) }
            })
            approvalPanel.addView(Button(this).apply {
                text = "Decline"
                setOnClickListener { svc.answerApproval(false) }
            })
            approvalPanel.addView(Button(this).apply {
                text = "Resume (after I finished paying)"
                setOnClickListener { svc.resumeFromHandoff() }
            })
        } else {
            approvalPanel.visibility = LinearLayout.GONE
        }
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
}
