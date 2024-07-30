package org.paulstudios.urbanflow.utils

import android.content.Context
import android.util.Log
import android.widget.TextView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import io.noties.markwon.Markwon
import io.noties.markwon.ext.strikethrough.StrikethroughPlugin
import io.noties.markwon.ext.tables.TablePlugin
import java.io.BufferedReader
import java.io.InputStreamReader

private const val TAG = "MarkdownViewer"

@Composable
fun MarkdownViewerScreen(
    onConsentGiven: () -> Unit,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    val sharedPreferences = context.getSharedPreferences("AppPreferences", Context.MODE_PRIVATE)
    var privacyPolicy by remember { mutableStateOf("") }
    var termsConditions by remember { mutableStateOf("") }
    var showPrivacyPolicy by remember { mutableStateOf(true) }

    LaunchedEffect(Unit) {
        privacyPolicy = loadMarkdownFile(context, "privacy_policy.md")
        termsConditions = loadMarkdownFile(context, "terms_conditions.md")
    }

    if (showPrivacyPolicy) {
        MarkdownContent(
            title = "Privacy Policy",
            content = privacyPolicy,
            onNext = { showPrivacyPolicy = false },
            onDismiss = onDismiss
        )
    } else {
        MarkdownContent(
            title = "Terms and Conditions",
            content = termsConditions,
            onNext = {
                sharedPreferences.edit().putBoolean("consent_given", true).apply()
                onConsentGiven()
            },
            onDismiss = onDismiss
        )
    }
}

@Composable
fun MarkdownContent(
    title: String,
    content: String,
    onNext: () -> Unit,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    val markwon = remember {
        Markwon.builder(context)
            .usePlugin(StrikethroughPlugin.create())
            .usePlugin(TablePlugin.create(context))
            .build()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text(text = title, style = MaterialTheme.typography.headlineSmall, modifier = Modifier.padding(bottom = 16.dp))

        AndroidView(
            factory = { context ->
                TextView(context).apply {
                    markwon.setMarkdown(this, content)
                }
            },
            update = { textView ->
                markwon.setMarkdown(textView, content)
            },
            modifier = Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState())
        )

        Spacer(modifier = Modifier.height(16.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            TextButton(onClick = onDismiss) {
                Text("Decline")
            }
            Button(onClick = onNext) {
                Text(text = "Agree")
            }
        }
    }
}

fun loadMarkdownFile(context: Context, fileName: String): String {
    Log.d(TAG, "loadMarkdownFile: Starting to load file: $fileName")
    return try {
        val inputStream = context.assets.open(fileName)
        val reader = BufferedReader(InputStreamReader(inputStream))
        val content = reader.use { it.readText() }
        Log.d(TAG, "loadMarkdownFile: Successfully loaded file: $fileName, content length: ${content.length}")
        content
    } catch (e: Exception) {
        Log.e(TAG, "loadMarkdownFile: Error loading file: $fileName", e)
        "Error loading content."
    }
}