package com.quantumleap.agent.net

import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * Thin OkHttp WebSocket wrapper. Receives request frames from the brain and routes
 * them to [Handler]; sends replies back. Auto-reconnects with backoff so a flaky
 * link does not abandon a running task.
 */
class AgentWebSocketClient(
    private val wsUrl: String,
    private val deviceId: String,
    private val handler: Handler,
    /** Returns the device bearer token (null until the phone is paired). */
    private val tokenProvider: () -> String? = { null },
) {
    interface Handler {
        /** Reply with the current screen JSON (wrapped as {"screen": {...}}). */
        fun onObserve(): JSONObject
        /** Execute one action; return {"ok": true/false}. */
        fun onAct(action: JSONObject): JSONObject
        /** User performs a sensitive step (payment/auth). Block until returned; return {"ok": true}. */
        fun onHandoff(approval: JSONObject): JSONObject
        /** Show the approval UI. The user answers later via [sendApprovalResponse]. */
        fun onApprovalRequest(request: JSONObject)
        /** The agent asked a clarifying question; answer later via [sendAnswer]. */
        fun onAskUser(request: JSONObject)
        /** Status / summary pushes for the UI. */
        fun onStatus(message: JSONObject)
    }

    // Longer ping interval tolerates mobile-network jitter to the cloud backend (a 20s
    // pong timeout was dropping the socket on brief hiccups). If a flap does happen, the
    // backend re-delivers any pending prompt on reconnect, so nothing is lost.
    private val client = OkHttpClient.Builder()
        .pingInterval(40, TimeUnit.SECONDS)
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .retryOnConnectionFailure(true)
        .build()

    @Volatile private var socket: WebSocket? = null
    @Volatile private var closed = false
    /** True while the backend rejects us (not paired / revoked): retry slowly, not every 2s. */
    @Volatile var authRejected = false
        private set

    fun connect() {
        closed = false
        open()
    }

    // Each open() gets a new generation; callbacks from older sockets are ignored, so a
    // cancelled/replaced socket can never schedule a duplicate reconnect.
    @Volatile private var generation = 0

    /** Drop the current socket and reconnect now (e.g. right after pairing). */
    fun reconnectNow() {
        authRejected = false
        val old = socket
        if (!closed) open()
        old?.cancel()
    }

    fun disconnect() {
        closed = true
        socket?.close(1000, "bye")
        socket = null
    }

    fun send(json: JSONObject) {
        socket?.send(json.toString())
    }

    fun sendApprovalResponse(taskId: String, approved: Boolean, note: String = "") {
        send(JSONObject()
            .put("type", "approval_response")
            .put("task_id", taskId)
            .put("approved", approved)
            .put("note", note))
    }

    fun sendAnswer(taskId: String, answerText: String) {
        send(JSONObject()
            .put("type", "answer")
            .put("task_id", taskId)
            .put("facts", JSONObject().put("clarification_answer", answerText)))
    }

    @Synchronized
    private fun open() {
        val gen = ++generation
        val builder = Request.Builder().url("$wsUrl/ws/device/$deviceId")
        val token = tokenProvider()
        if (!token.isNullOrBlank()) builder.header("Authorization", "Bearer $token")
        socket = client.newWebSocket(builder.build(), object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                if (gen != generation) return
                authRejected = false
                Log.i(TAG, "connected as $deviceId")
            }

            override fun onMessage(ws: WebSocket, text: String) {
                dispatch(ws, JSONObject(text))
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                if (gen != generation) return
                val code = response?.code
                if (code == 401 || code == 403) {
                    // Not paired, or the token was revoked: no point hammering every 2s.
                    authRejected = true
                    Log.w(TAG, "backend rejected this device ($code) — pair it in the app")
                    scheduleReconnect(gen, 30_000)
                } else {
                    Log.w(TAG, "ws failure: ${t.message}")
                    scheduleReconnect(gen, 2_000)
                }
            }

            override fun onClosing(ws: WebSocket, code: Int, reason: String) {
                ws.close(1000, null)
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                if (gen != generation || closed) return
                if (code == 1008) {                       // policy violation: revoked
                    authRejected = true
                    Log.w(TAG, "backend closed the socket (revoked?) — pair again")
                    scheduleReconnect(gen, 30_000)
                } else {
                    scheduleReconnect(gen, 2_000)
                }
            }
        })
    }

    private fun dispatch(ws: WebSocket, msg: JSONObject) {
        val type = msg.optString("type")
        val rid = if (msg.has("rid")) msg.optInt("rid") else null
        try {
            when (type) {
                "observe" -> reply(ws, rid, JSONObject().put("screen", handler.onObserve()))
                "act" -> reply(ws, rid, handler.onAct(msg.optJSONObject("action") ?: JSONObject()))
                "handoff" -> reply(ws, rid, handler.onHandoff(msg.optJSONObject("approval") ?: JSONObject()))
                "approval_request" -> handler.onApprovalRequest(msg)
                "ask_user" -> handler.onAskUser(msg)
                else -> handler.onStatus(msg)
            }
        } catch (e: Exception) {
            Log.e(TAG, "dispatch error for $type", e)
            if (rid != null) reply(ws, rid, JSONObject().put("ok", false).put("error", e.message))
        }
    }

    private fun reply(ws: WebSocket, rid: Int?, payload: JSONObject) {
        if (rid == null) return
        ws.send(JSONObject().put("rid", rid).put("payload", payload).toString())
    }

    private fun scheduleReconnect(gen: Int, delayMs: Long) {
        if (closed) return
        Thread {
            try { Thread.sleep(delayMs) } catch (_: InterruptedException) {}
            // Only reconnect if nothing newer (e.g. reconnectNow) has replaced this socket.
            if (!closed && gen == generation) open()
        }.start()
    }

    companion object { private const val TAG = "AgentWS" }
}
