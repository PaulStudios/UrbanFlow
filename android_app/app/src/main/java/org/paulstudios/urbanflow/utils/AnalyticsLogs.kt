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

fun logDataReceived(success: Boolean, clientId: String, e: String? = null) {
    if (success) {
        firebaseAnalytics.logEvent("data_received_successfully") {
            param("client_id", clientId)
        }
    } else {
        firebaseAnalytics.logEvent("data_receive_error") {
            param("client_id", clientId)
            param("error_message", e ?: "Unknown error")
        }
    }
}

fun logKeyExchange(success: Boolean, clientId: String, e: String? = null) {
    if (success) {
        firebaseAnalytics.logEvent("key_exchange_success") {
            param("client_id", clientId)
        }
    } else {
        firebaseAnalytics.logEvent("key_exchange_error") {
            param("client_id", clientId)
            param("error_message", e ?: "Unknown error")
        }
    }
}

fun logDataSent(success: Boolean, clientId: String, e: String? = null) {
    if (success) {
        firebaseAnalytics.logEvent("data_sent_successfully") {
            param("client_id", clientId)
        }
    } else {
        firebaseAnalytics.logEvent("data_send_error") {
            param("client_id", clientId)
            param("error_message", e ?: "Unknown error")
        }
    }
}