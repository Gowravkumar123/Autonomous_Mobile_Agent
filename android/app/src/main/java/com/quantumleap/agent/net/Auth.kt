package com.quantumleap.agent.net

import android.content.Context
import android.os.Build
import com.quantumleap.agent.BuildConfig
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
 * Device identity + bearer token for the backend.
 *
 * The phone pairs once: an admin gets a 6-digit code from the backend, the user enters it
 * here, and the backend returns a device token bound to this phone's device id. Every
 * HTTP call and the agent WebSocket then carry `Authorization: Bearer <token>`.
 */
object Auth {
    private const val PREFS = "agent_auth"
    private const val KEY_TOKEN = "device_token"
    private val http = OkHttpClient()

    val deviceId: String
        get() = "android-${Build.MODEL}".replace(" ", "-")

    fun token(ctx: Context): String? =
        ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY_TOKEN, null)

    fun isPaired(ctx: Context): Boolean = !token(ctx).isNullOrBlank()

    fun save(ctx: Context, token: String) {
        ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(KEY_TOKEN, token).apply()
    }

    fun clear(ctx: Context) {
        ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().remove(KEY_TOKEN).apply()
    }

    /** Add the bearer header if this phone is paired. */
    fun Request.Builder.withAuth(ctx: Context): Request.Builder {
        val t = token(ctx)
        return if (t.isNullOrBlank()) this else header("Authorization", "Bearer $t")
    }

    /** Exchange a 6-digit pairing code for a device token. Callback runs on a worker thread. */
    fun pair(ctx: Context, code: String, done: (ok: Boolean, message: String) -> Unit) {
        val body = JSONObject().put("code", code.trim()).put("device_id", deviceId).toString()
            .toRequestBody("application/json".toMediaType())
        val req = Request.Builder().url("${BuildConfig.BACKEND_HTTP_URL}/v1/devices/pair")
            .post(body).build()
        http.newCall(req).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                done(false, "Couldn't reach the backend: ${e.message}")
            }

            override fun onResponse(call: Call, response: Response) {
                val text = response.body?.string().orEmpty()
                val code = response.code
                response.close()
                if (code == 200) {
                    val token = JSONObject(text).optString("token")
                    if (token.isNotBlank()) {
                        save(ctx, token)
                        done(true, "Paired as $deviceId")
                    } else {
                        done(false, "Pairing response had no token")
                    }
                } else {
                    val detail = runCatching { JSONObject(text).optString("detail") }.getOrNull()
                    done(false, "Pairing failed ($code): ${detail ?: text}")
                }
            }
        })
    }
}
