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
        /** Status / summary pushes for the UI. */
        fun onStatus(message: JSONObject)
    }

    private val client = OkHttpClient.Builder()
        .pingInterval(20, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    @Volatile private var socket: WebSocket? = null
    @Volatile private var closed = false

    fun connect() {
        closed = false
        open()
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

    private fun open() {
        val request = Request.Builder().url("$wsUrl/ws/device/$deviceId").build()
        socket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                Log.i(TAG, "connected as $deviceId")
            }

            override fun onMessage(ws: WebSocket, text: String) {
                dispatch(ws, JSONObject(text))
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                Log.w(TAG, "ws failure: ${t.message}")
                scheduleReconnect()
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                if (!closed) scheduleReconnect()
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

    private fun scheduleReconnect() {
        if (closed) return
        Thread {
            Thread.sleep(2000)
            if (!closed) open()
        }.start()
    }

    companion object { private const val TAG = "AgentWS" }
}
