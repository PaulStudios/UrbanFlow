package org.paulstudios.datasurvey.ui.screens.auth.components

import android.util.Patterns

fun isValidEmail(email: String): Boolean {
    return Patterns.EMAIL_ADDRESS.matcher(email).matches()
}

fun calculatePasswordStrength(password: String): Int {
    var strength = 0
    if (password.length >= 8) strength++
    if (password.any { it.isDigit() }) strength++
    if (password.any { it.isUpperCase() }) strength++
    if (password.any { it.isLowerCase() }) strength++
    if (password.any { "!@#\$%^&*()-_=+<>?/{}~".contains(it) }) strength++
    return strength
}

fun getPasswordStrengthLabel(strength: Int): String {
    return when (strength) {
        1 -> "Very Weak"
        2 -> "Weak"
        3 -> "Medium"
        4 -> "Strong"
        5 -> "Very Strong"
        else -> "..."
    }
}