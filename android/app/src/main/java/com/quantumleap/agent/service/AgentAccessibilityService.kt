package com.quantumleap.agent.service

import android.accessibilityservice.AccessibilityService
import android.graphics.Bitmap
import android.os.Build
import android.util.Base64
import android.util.Log
import android.view.Display
import android.view.accessibility.AccessibilityEvent
import com.quantumleap.agent.BuildConfig
import com.quantumleap.agent.model.AgentAction
import com.quantumleap.agent.net.AgentWebSocketClient
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * The controlled execution device. Enabled explicitly by the user in Accessibility
 * settings. It connects to the paired backend and, on request, reports the screen or
 * performs a single permitted action. All sensitive-action gating happens on the
 * backend; this service pauses and hands control to the user for those.
 */
class AgentAccessibilityService : AccessibilityService(), AgentWebSocketClient.Handler {

    private val serializer = TreeSerializer()
    private lateinit var executor: ActionExecutor
    private var ws: AgentWebSocketClient? = null

    // Set when the brain hands over for payment/auth; the user releases it from the UI.
    @Volatile private var handoffLatch: CountDownLatch? = null
    // The most recent approval the user must answer (surfaced to the UI).
    @Volatile var pendingApproval: JSONObject? = null
        private set
    @Volatile var lastStatus: String = "Idle"
        private set

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        executor = ActionExecutor(this)
        val deviceId = "android-${android.os.Build.MODEL}".replace(" ", "-")
        ws = AgentWebSocketClient(BuildConfig.BACKEND_WS_URL, deviceId, this).also { it.connect() }
        lastStatus = "Connected. Waiting for a task."
        Log.i(TAG, "service connected, device=$deviceId")
    }

    override fun onDestroy() {
        ws?.disconnect()
        instance = null
        super.onDestroy()
    }

    // We don't react to raw events; the brain pulls the screen when it needs it.
    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}
    override fun onInterrupt() {}

    // ------------------------------------------------------------------ //
    // Handler callbacks (called from the ws thread)
    // ------------------------------------------------------------------ //
    override fun onObserve(): JSONObject {
        val root = rootInActiveWindow
        val pkg = root?.packageName?.toString() ?: ""
        val activity = windows.firstOrNull()?.title?.toString() ?: ""
        val screen = serializer.capture(root, pkg, activity)
        captureScreenshotB64()?.let { screen.put("screenshot_b64", it) }
        return screen
    }

    private val screenshotExecutor = Executors.newSingleThreadExecutor()

    /** Take a screenshot of the current screen as a downscaled JPEG (base64), for the
     *  vision model. Returns null if unavailable (API < 30) or on failure. */
    private fun captureScreenshotB64(): String? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return null
        val latch = CountDownLatch(1)
        var out: String? = null
        try {
            takeScreenshot(Display.DEFAULT_DISPLAY, screenshotExecutor,
                object : AccessibilityService.TakeScreenshotCallback {
                    override fun onSuccess(result: AccessibilityService.ScreenshotResult) {
                        try {
                            val hw = Bitmap.wrapHardwareBuffer(result.hardwareBuffer, result.colorSpace)
                            val soft = hw?.copy(Bitmap.Config.ARGB_8888, false)
                            hw?.recycle()
                            if (soft != null) {
                                val scaled = scaleDown(soft, 720)
                                val baos = ByteArrayOutputStream()
                                scaled.compress(Bitmap.CompressFormat.JPEG, 50, baos)
                                out = Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)
                                if (scaled !== soft) scaled.recycle()
                                soft.recycle()
                            }
                        } catch (e: Exception) {
                            Log.w(TAG, "screenshot encode failed", e)
                        } finally {
                            result.hardwareBuffer.close()
                            latch.countDown()
                        }
                    }

                    override fun onFailure(errorCode: Int) {
                        Log.w(TAG, "takeScreenshot failed: $errorCode")
                        latch.countDown()
                    }
                })
            latch.await(2, TimeUnit.SECONDS)
        } catch (e: Exception) {
            Log.w(TAG, "takeScreenshot threw", e)
        }
        return out
    }

    private fun scaleDown(bmp: Bitmap, maxW: Int): Bitmap {
        if (bmp.width <= maxW) return bmp
        val ratio = maxW.toFloat() / bmp.width
        return Bitmap.createScaledBitmap(bmp, maxW, (bmp.height * ratio).toInt(), true)
    }

    override fun onAct(action: JSONObject): JSONObject {
        val ok = executor.execute(AgentAction.from(action), serializer.nodesByRef)
        // Give the UI a moment to settle before the next observation.
        Thread.sleep(500)
        return JSONObject().put("ok", ok)
    }

    override fun onHandoff(approval: JSONObject): JSONObject {
        lastStatus = "Hand over: complete payment/authentication, then tap Resume."
        val latch = CountDownLatch(1)
        handoffLatch = latch
        // The real app surfaces a full-screen approval + "Resume" control here.
        latch.await(5, TimeUnit.MINUTES)
        handoffLatch = null
        lastStatus = "Resumed. Verifying result…"
        return JSONObject().put("ok", true)
    }

    override fun onApprovalRequest(request: JSONObject) {
        pendingApproval = request
        lastStatus = "Approval needed: ${request.optString("reason")}"
        // The UI (MainActivity) observes pendingApproval and shows Approve / Decline.
    }

    override fun onStatus(message: JSONObject) {
        lastStatus = message.optString("message", message.toString())
    }

    // ------------------------------------------------------------------ //
    // Called by the UI
    // ------------------------------------------------------------------ //
    fun answerApproval(approved: Boolean) {
        val req = pendingApproval ?: return
        ws?.sendApprovalResponse(req.optString("task_id"), approved)
        pendingApproval = null
    }

    /** User finished the payment/auth and wants the agent to continue. */
    fun resumeFromHandoff() {
        handoffLatch?.countDown()
    }

    companion object {
        private const val TAG = "AgentA11y"
        @Volatile var instance: AgentAccessibilityService? = null
            private set
    }
}
