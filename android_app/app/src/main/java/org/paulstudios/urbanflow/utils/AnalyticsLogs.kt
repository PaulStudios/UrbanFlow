package org.paulstudios.urbanflow.utils

import com.google.firebase.analytics.ktx.analytics
import com.google.firebase.analytics.ktx.logEvent
import com.google.firebase.ktx.Firebase


private val firebaseAnalytics = Firebase.analytics

fun logSelfieAttempt() {
    firebaseAnalytics.logEvent("selfie_attempt") {}
}

fun logSelfieFaceDetectionResult(success: Boolean) {
    firebaseAnalytics.logEvent("selfie_face_detection_result") {
        param("success", success.toString())
    }
}

fun logUserInfoSubmission(success: Boolean) {
    firebaseAnalytics.logEvent("user_info_submission") {
        param("success", success.toString())
    }
}