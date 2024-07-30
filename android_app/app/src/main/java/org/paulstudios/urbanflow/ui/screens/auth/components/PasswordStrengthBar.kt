package org.paulstudios.datasurvey.ui.screens.auth.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

@Composable
fun PasswordStrengthBar(strength: Int) {
    val colors = listOf(
        Color(0xFFFF0000), // Red
        Color(0xFFFF4500), // Orange-Red
        Color(0xFFFFA500), // Orange
        Color(0xFFFFFF00), // Yellow
        Color(0xFF7CFC00)  // Lawn Green
    )
    val indicatorWidth = 78.dp
    Row(modifier = Modifier.fillMaxWidth()) {
        for (i in 0 until strength) {
            Box(
                modifier = Modifier
                    .height(8.dp)
                    .width(indicatorWidth)
                    .padding(1.dp)
                    .background(colors[i])
            )
        }
    }
    Spacer(modifier = Modifier.height(4.dp))
    Text(text = getPasswordStrengthLabel(strength))
}