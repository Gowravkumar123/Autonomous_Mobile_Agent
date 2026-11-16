# Android companion app

The controlled execution device: a companion UI + an **AccessibilityService** that
reports the screen and performs backend-approved actions, pausing for you at payment/auth.

## Build

Open this `android/` folder in **Android Studio** (Giraffe+). It will:
- download the Gradle wrapper jar (not committed; see `.gitignore`), and
- generate a launcher icon (none is committed to keep the repo lean).

Or from the CLI, once a wrapper exists: `./gradlew :app:assembleDebug`.
(If you have Gradle installed, run `gradle wrapper` first.)

## Configure the backend address

`app/build.gradle.kts` → `defaultConfig`:
```
buildConfigField("String", "BACKEND_WS_URL",   "\"ws://10.0.2.2:8000\"")
buildConfigField("String", "BACKEND_HTTP_URL", "\"http://10.0.2.2:8000\"")
```
`10.0.2.2` is the host machine from the Android emulator. For a physical phone, use your
PC's LAN IP (same Wi-Fi), and make sure the backend is reachable.

## Run

1. Launch the app → **Enable the controller** → turn the service on in
   *Settings ▸ Accessibility ▸ Personal AI Agent Controller*.
2. Type a task → **Run task**.
3. When it reaches a payment/OTP screen it pauses and shows **Approve / I'll take over**;
   finish paying yourself, then tap **Resume**.

## Key files

| File | Role |
|------|------|
| `service/AgentAccessibilityService.kt` | connects to the brain; observe / act / handoff |
| `service/TreeSerializer.kt` | accessibility tree → JSON screen (ref-addressed) |
| `service/ActionExecutor.kt` | tap / type / scroll / swipe / back / open-app |
| `net/AgentWebSocketClient.kt` | OkHttp WebSocket + auto-reconnect |
| `model/Protocol.kt` | the device↔brain frame types |
| `ui/MainActivity.kt` | enable, prompt, status, approvals |

> This is a working scaffold. Running against a *live* third-party app needs per-app
> tuning of the serializer and action targeting, and apps that block automation,
> show CAPTCHAs, or hide accessibility info must be handed to the user — never bypassed.
