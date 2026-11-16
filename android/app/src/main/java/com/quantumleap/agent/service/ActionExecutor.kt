package com.quantumleap.agent.service

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.content.Intent
import android.graphics.Path
import android.graphics.Rect
import android.os.Bundle
import android.view.accessibility.AccessibilityNodeInfo
import com.quantumleap.agent.model.AgentAction

/**
 * Executes ONE permitted action against the current tree. The policy engine has
 * already cleared anything sensitive on the backend; by the time an `act` reaches
 * here it is a read-only or reversible step. Secrets are never typed by the agent.
 */
class ActionExecutor(private val service: AccessibilityService) {

    /** Returns true if the action was dispatched. */
    fun execute(action: AgentAction, nodesByRef: List<AccessibilityNodeInfo>): Boolean {
        return when (action.type) {
            "tap" -> tap(node(action.ref, nodesByRef))
            "type" -> type(node(action.ref, nodesByRef), action.text ?: "")
            "scroll" -> scroll(action.direction ?: "down", nodesByRef)
            "swipe" -> swipe(action.direction ?: "up")
            "back" -> service.performGlobalAction(AccessibilityService.GLOBAL_ACTION_BACK)
            "home" -> service.performGlobalAction(AccessibilityService.GLOBAL_ACTION_HOME)
            "open_app" -> openApp(action.text)
            "wait" -> true
            else -> false
        }
    }

    private fun node(ref: Int?, nodes: List<AccessibilityNodeInfo>): AccessibilityNodeInfo? =
        ref?.let { nodes.getOrNull(it) }

    private fun tap(node: AccessibilityNodeInfo?): Boolean {
        node ?: return false
        // Prefer the semantic click; fall back to a tap gesture at the node centre.
        if (node.isClickable && node.performAction(AccessibilityNodeInfo.ACTION_CLICK)) return true
        var p: AccessibilityNodeInfo? = node
        while (p != null) {
            if (p.isClickable && p.performAction(AccessibilityNodeInfo.ACTION_CLICK)) return true
            p = p.parent
        }
        val r = Rect().also { node.getBoundsInScreen(it) }
        return tapAt(r.exactCenterX(), r.exactCenterY())
    }

    private fun type(node: AccessibilityNodeInfo?, text: String): Boolean {
        node ?: return false
        node.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
        val args = Bundle().apply {
            putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
        }
        return node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
    }

    private fun scroll(direction: String, nodes: List<AccessibilityNodeInfo>): Boolean {
        val scrollable = nodes.firstOrNull { it.isScrollable } ?: return false
        val action = if (direction == "up")
            AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD
        else
            AccessibilityNodeInfo.ACTION_SCROLL_FORWARD
        return scrollable.performAction(action)
    }

    private fun tapAt(x: Float, y: Float): Boolean {
        val path = Path().apply { moveTo(x, y) }
        val stroke = GestureDescription.StrokeDescription(path, 0, 60)
        val gesture = GestureDescription.Builder().addStroke(stroke).build()
        return service.dispatchGesture(gesture, null, null)
    }

    private fun swipe(direction: String): Boolean {
        val dm = service.resources.displayMetrics
        val cx = dm.widthPixels / 2f
        val (y1, y2) = if (direction == "up")
            dm.heightPixels * 0.7f to dm.heightPixels * 0.3f
        else
            dm.heightPixels * 0.3f to dm.heightPixels * 0.7f
        val path = Path().apply { moveTo(cx, y1); lineTo(cx, y2) }
        val stroke = GestureDescription.StrokeDescription(path, 0, 300)
        return service.dispatchGesture(GestureDescription.Builder().addStroke(stroke).build(), null, null)
    }

    /**
     * Launch an app given a package id OR a friendly name ("Swiggy"). The name is
     * resolved dynamically against the launcher's visible app labels, so no hardcoded
     * name->package table is needed — it works for any installed app on any device.
     */
    private fun openApp(nameOrPackage: String?): Boolean {
        val query = nameOrPackage?.trim().orEmpty()
        if (query.isEmpty()) return false
        val pm = service.packageManager

        // 1. Exact package id? launch directly.
        pm.getLaunchIntentForPackage(query)?.let {
            it.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            service.startActivity(it)
            return true
        }

        // 2. Resolve by visible app label.
        val pkg = findPackageByLabel(query) ?: return false
        val intent = pm.getLaunchIntentForPackage(pkg) ?: return false
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        service.startActivity(intent)
        return true
    }

    private fun findPackageByLabel(name: String): String? {
        val pm = service.packageManager
        val q = name.lowercase()
        val launchables = pm.queryIntentActivities(
            Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER), 0)
        var partial: String? = null
        for (ri in launchables) {
            val label = ri.loadLabel(pm).toString().lowercase()
            val pkg = ri.activityInfo.packageName
            if (label == q) return pkg                       // exact label match wins
            if (partial == null && (label.contains(q) || q.contains(label))) partial = pkg
        }
        return partial
    }
}
