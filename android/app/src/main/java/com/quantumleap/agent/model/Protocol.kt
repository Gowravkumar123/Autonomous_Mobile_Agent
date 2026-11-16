package com.quantumleap.agent.model

import org.json.JSONObject

/**
 * The device <-> brain contract, mirrored from backend/app/mobile/schema.py.
 * We keep it hand-rolled over org.json so the client has zero extra codegen.
 *
 * Inbound frame types (brain -> device):
 *   observe            -> reply with the current screen
 *   act {action}       -> perform one action, reply ok
 *   handoff {approval} -> user completes a sensitive step; reply ok when returned
 *   approval_request   -> show the approval UI (no reply on this frame; the user
 *                         answers via an `approval_response` frame)
 *   status / summary   -> UI updates
 */
object Frame {
    const val OBSERVE = "observe"
    const val ACT = "act"
    const val HANDOFF = "handoff"
    const val APPROVAL_REQUEST = "approval_request"
    const val STATUS = "status"
    const val SUMMARY = "summary"
}

/** One decision from the brain. */
data class AgentAction(
    val type: String,
    val ref: Int?,
    val text: String?,
    val direction: String?,
    val seconds: Double?,
    val x: Double?,
    val y: Double?,
    val thought: String,
) {
    companion object {
        fun from(json: JSONObject): AgentAction = AgentAction(
            type = json.optString("type"),
            ref = if (json.isNull("ref")) null else json.optInt("ref"),
            text = if (json.isNull("text")) null else json.optString("text"),
            direction = if (json.isNull("direction")) null else json.optString("direction"),
            seconds = if (json.isNull("seconds")) null else json.optDouble("seconds"),
            x = if (json.isNull("x")) null else json.optDouble("x"),
            y = if (json.isNull("y")) null else json.optDouble("y"),
            thought = json.optString("thought"),
        )
    }
}

/** Helper to build the JSON frames we send back. */
object Reply {
    fun response(rid: Int, payload: JSONObject): JSONObject =
        JSONObject().put("rid", rid).put("payload", payload)

    fun approvalResponse(taskId: String, approved: Boolean, note: String = ""): JSONObject =
        JSONObject()
            .put("type", "approval_response")
            .put("task_id", taskId)
            .put("approved", approved)
            .put("note", note)
}
