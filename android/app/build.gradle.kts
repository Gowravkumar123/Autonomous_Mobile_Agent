plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.quantumleap.agent"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.quantumleap.agent"
        minSdk = 26          // AccessibilityService gesture dispatch needs API 24+; 26 for safety
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"

        // Points at the on-prem backend. The phone must be on a network that can
        // route to this IP. (Emulator-to-host would be 10.0.2.2 instead.)
        buildConfigField("String", "BACKEND_WS_URL", "\"ws://37.27.53.14:8033\"")
        buildConfigField("String", "BACKEND_HTTP_URL", "\"http://37.27.53.14:8033\"")
    }

    buildFeatures {
        buildConfig = true
        viewBinding = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")

    // Networking (OkHttp carries the WebSocket client)
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // JSON parsing/serialization uses org.json, which ships with the Android SDK.
}
