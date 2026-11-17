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
        if (root != null) walk(root, elements, 0, -1)
        return JSONObject()
            .put("app_package", packageName)
            .put("activity", activity)
            .put("is_keyboard_open", false)
            .put("elements", elements)
    }

    /**
     * Hybrid serialization: a flat list (for stable `ref` targeting) where each element
     * also carries `depth` and `parent` (the ref of its nearest kept ancestor). That
     * preserves the tree's grouping — e.g. a price and a name under the same card share
     * a parent — without losing the flat refs used for actions.
     */
    private fun walk(node: AccessibilityNodeInfo, out: JSONArray, depth: Int, parentRef: Int) {
        var childDepth = depth
        var childParent = parentRef
        if (isInteresting(node)) {
            val ref = nodesByRef.size
            nodesByRef.add(node)
            out.put(serialize(node, ref, depth, parentRef))
            childDepth = depth + 1
            childParent = ref            // this node is the parent of its kept descendants
        }
        for (i in 0 until node.childCount) {
            val child = node.getChild(i) ?: continue
            walk(child, out, childDepth, childParent)
        }
    }

    /** Skip pure layout noise; keep anything a user could read or act on. */
    private fun isInteresting(n: AccessibilityNodeInfo): Boolean {
        val hasText = !n.text.isNullOrBlank() || !n.contentDescription.isNullOrBlank()
        return hasText || n.isClickable || n.isEditable || n.isCheckable || n.isScrollable
    }

    private fun serialize(n: AccessibilityNodeInfo, ref: Int, depth: Int, parent: Int): JSONObject {
        val bounds = Rect().also { n.getBoundsInScreen(it) }
        return JSONObject()
            .put("ref", ref)
            .put("depth", depth)
            .put("parent", parent)
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
