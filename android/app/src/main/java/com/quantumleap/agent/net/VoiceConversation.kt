package com.quantumleap.agent.net

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import android.util.Log
import com.quantumleap.agent.BuildConfig
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * Full-duplex voice conversation with the backend voice gateway (WS /v1/voice/{session}).
 *
 * Mic: AudioRecord on VOICE_COMMUNICATION (the platform's echo-cancelled voice path) +
 * AcousticEchoCanceler / NoiseSuppressor / AGC, streamed as 20 ms PCM16 16 kHz frames.
 * Speaker: AudioTrack (voice-communication usage, so AEC uses it as the echo reference)
 * fed with the agent's PCM frames. The SERVER does VAD, turn-taking and barge-in: when it
 * sends `barge_in`, we drop whatever speech is still buffered so the user is heard at once.
 *
 * If server TTS is unavailable the server sends `say` text; [onSay] speaks it locally and
 * we report playback so barge-in keeps working.
 */
class VoiceConversation(
    private val ctx: Context,
    private val sessionId: String,
    private val listener: Listener,
) {
    interface Listener {
        fun onState(state: String)                 // connecting | live | closed | error:<why>
        fun onTranscript(text: String, final: Boolean)
        fun onAgentText(text: String)
        fun onSay(text: String, done: () -> Unit)  // local TTS fallback
        fun onActivity()                           // anything happened (idle timer reset)
    }

    private val http = OkHttpClient.Builder().pingInterval(20, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS).build()
    private var ws: WebSocket? = null
    private var record: AudioRecord? = null
    private var track: AudioTrack? = null
    private var effects = mutableListOf<android.media.audiofx.AudioEffect>()
    @Volatile private var running = false
    private var micThread: Thread? = null
    private var prevAudioMode = AudioManager.MODE_NORMAL

    fun start(mode: String = "duplex") {
        if (running) return
        running = true
        listener.onState("connecting")
        val token = Auth.token(ctx) ?: run { listener.onState("error:not paired"); running = false; return }
        val req = Request.Builder()
            .url("${BuildConfig.BACKEND_WS_URL}/v1/voice/$sessionId?mode=$mode")
            .header("Authorization", "Bearer $token").build()
        ws = http.newWebSocket(req, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                startAudio()
            }
            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                track?.write(bytes.toByteArray(), 0, bytes.size)
            }
            override fun onMessage(webSocket: WebSocket, text: String) = handle(JSONObject(text))
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.w(TAG, "voice socket failed: ${t.message}")
                listener.onState(if (response?.code == 403) "error:not authorized" else "error:${t.message}")
                stop()
            }
            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) = stop()
        })
    }

    fun stop() {
        if (!running) return
        running = false
        try { micThread?.join(300) } catch (_: InterruptedException) {}
        try { record?.stop() } catch (_: Exception) {}
        record?.release(); record = null
        effects.forEach { try { it.release() } catch (_: Exception) {} }; effects.clear()
        try { track?.stop() } catch (_: Exception) {}
        track?.release(); track = null
        try { ws?.close(1000, "bye") } catch (_: Exception) {}
        ws = null
        (ctx.getSystemService(Context.AUDIO_SERVICE) as AudioManager).mode = prevAudioMode
        listener.onState("closed")
    }

    /** Text turn on the same conversation (typed fallback). */
    fun sendText(text: String) {
        ws?.send(JSONObject().put("type", "text").put("text", text).toString())
    }

    fun interrupt() { ws?.send(JSONObject().put("type", "interrupt").toString()) }

    private fun handle(m: JSONObject) {
        listener.onActivity()
        when (m.optString("type")) {
            "ready" -> listener.onState("live")
            "transcript" -> listener.onTranscript(m.optString("text"), m.optBoolean("final"))
            "speak" -> listener.onAgentText(m.optString("text"))
            "barge_in" -> flushSpeech()
            "say" -> {
                ws?.send(JSONObject().put("type", "playback").put("state", "start")
                    .put("ms", 400 + 400 * m.optString("text").split(" ").size).toString())
                listener.onSay(m.optString("text")) {
                    ws?.send(JSONObject().put("type", "playback").put("state", "end").toString())
                }
            }
        }
    }

    private fun flushSpeech() {
        val t = track ?: return
        try { t.pause(); t.flush(); t.play() } catch (_: Exception) {}
    }

    @SuppressLint("MissingPermission")    // RECORD_AUDIO is requested by MainActivity
    private fun startAudio() {
        val am = ctx.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        prevAudioMode = am.mode
        am.mode = AudioManager.MODE_IN_COMMUNICATION        // engages the AEC voice path
        val minIn = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT)
        val rec = AudioRecord(MediaRecorder.AudioSource.VOICE_COMMUNICATION, RATE,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minIn, FRAME_BYTES * 10))
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            listener.onState("error:microphone unavailable"); stop(); return
        }
        record = rec
        val sid = rec.audioSessionId
        if (AcousticEchoCanceler.isAvailable()) AcousticEchoCanceler.create(sid)?.let { it.enabled = true; effects += it }
        if (NoiseSuppressor.isAvailable()) NoiseSuppressor.create(sid)?.let { it.enabled = true; effects += it }
        if (AutomaticGainControl.isAvailable()) AutomaticGainControl.create(sid)?.let { it.enabled = true; effects += it }

        val minOut = AudioTrack.getMinBufferSize(RATE, AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT)
        track = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder().setSampleRate(RATE)
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
            .setBufferSizeInBytes(maxOf(minOut, FRAME_BYTES * 25))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setSessionId(sid)                              // same session: AEC reference
            .build().also { it.play() }

        rec.startRecording()
        micThread = Thread {
            val buf = ByteArray(FRAME_BYTES)
            while (running) {
                var n = 0
                while (n < FRAME_BYTES && running) {
                    val r = rec.read(buf, n, FRAME_BYTES - n)
                    if (r <= 0) break
                    n += r
                }
                if (n == FRAME_BYTES) ws?.send(buf.toByteString(0, FRAME_BYTES))
            }
        }.apply { name = "voice-mic"; start() }
    }

    companion object {
        private const val TAG = "VoiceConv"
        const val RATE = 16000
        const val FRAME_BYTES = 640                         // 20 ms of PCM16 mono @16 kHz
    }
}
