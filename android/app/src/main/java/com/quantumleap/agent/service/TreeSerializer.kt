package com.quantumleap.agent.service

import android.graphics.Rect
import android.view.accessibility.AccessibilityNodeInfo
import org.json.JSONArray
import org.json.JSONObject

/**
 * Turns the live accessibility tree into the JSON screen the brain reasons over.
 *
 * Reliability rule from the blueprint: prefer element IDs, text, content
 * descriptions and roles over pixel coordinates. We assign each node a stable
 * `ref` for this frame and keep a parallel list so actions can resolve ref -> node.
 */
class TreeSerializer {

    /** The nodes captured in the last snapshot, indexed by `ref`. */
    val nodesByRef = mutableListOf<AccessibilityNodeInfo>()

    fun capture(root: AccessibilityNodeInfo?, packageName: String, activity: String): JSONObject {
        nodesByRef.clear()
        val elements = JSONArray()
        if (root != null) walk(root, elements)
        return JSONObject()
            .put("app_package", packageName)
            .put("activity", activity)
            .put("is_keyboard_open", false)
            .put("elements", elements)
    }

    private fun walk(node: AccessibilityNodeInfo, out: JSONArray) {
        if (!node.isVisibleToUser) {
            // Still descend — some containers are invisible but hold visible children.
        }
        if (isInteresting(node)) {
            val ref = nodesByRef.size
            nodesByRef.add(node)
            out.put(serialize(node, ref))
        }
        for (i in 0 until node.childCount) {
            val child = node.getChild(i) ?: continue
            walk(child, out)
        }
    }

    /** Skip pure layout noise; keep anything a user could read or act on. */
    private fun isInteresting(n: AccessibilityNodeInfo): Boolean {
        val hasText = !n.text.isNullOrBlank() || !n.contentDescription.isNullOrBlank()
        return hasText || n.isClickable || n.isEditable || n.isCheckable || n.isScrollable
    }

    private fun serialize(n: AccessibilityNodeInfo, ref: Int): JSONObject {
        val bounds = Rect().also { n.getBoundsInScreen(it) }
        return JSONObject()
            .put("ref", ref)
            .put("role", role(n))
            .put("text", n.text?.toString() ?: "")
            .put("desc", n.contentDescription?.toString() ?: "")
            .put("resource_id", n.viewIdResourceName ?: "")
            .put("clickable", n.isClickable)
            .put("editable", n.isEditable)
            .put("scrollable", n.isScrollable)
            .put("checked", if (n.isCheckable) n.isChecked else JSONObject.NULL)
            .put("enabled", n.isEnabled)
            .put("bounds", JSONArray(listOf(bounds.left, bounds.top, bounds.right, bounds.bottom)))
    }

    private fun role(n: AccessibilityNodeInfo): String {
        val cls = n.className?.toString() ?: return "View"
        return cls.substringAfterLast('.')
    }
}
