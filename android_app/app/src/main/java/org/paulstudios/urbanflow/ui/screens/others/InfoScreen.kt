package org.paulstudios.urbanflow.ui.screens.others

import android.content.Context
import android.util.Log
import android.widget.TextView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.navigation.NavHostController
import io.noties.markwon.Markwon
import io.noties.markwon.ext.strikethrough.StrikethroughPlugin
import io.noties.markwon.ext.tables.TablePlugin
import org.paulstudios.urbanflow.data.models.Screen
import org.paulstudios.urbanflow.utils.MarkdownViewerScreen
import org.paulstudios.urbanflow.utils.loadMarkdownFile

private const val TAG = "InfoScreen"

@Composable
fun InfoScreen(navController: NavHostController) {
    Log.d(TAG, "InfoScreen: Composable started")
    val context = LocalContext.current
    val sharedPreferences = context.getSharedPreferences("AppPreferences", Context.MODE_PRIVATE)

    val infoScreenShown = sharedPreferences.getBoolean("info_screen_shown", false)
    val consentGiven = sharedPreferences.getBoolean("consent_given", false)

    Log.d(TAG, "InfoScreen: infoScreenShown=$infoScreenShown, consentGiven=$consentGiven")

    LaunchedEffect(Unit) {
        if (infoScreenShown && consentGiven) {
            Log.i(TAG, "InfoScreen: Skipping to MapScreen")
            navController.navigate(Screen.RegisterDetails.route) {
                popUpTo(navController.graph.startDestinationId) { inclusive = true }
            }
        }
    }

    if (!infoScreenShown || !consentGiven) {
        var showConsent by remember { mutableStateOf(false) }

        if (showConsent) {
            Log.d(TAG, "InfoScreen: Showing consent screen")
            MarkdownViewerScreen(
                onConsentGiven = {
                    Log.i(TAG, "InfoScreen: Consent given")
                    sharedPreferences.edit()
                        .putBoolean("consent_given", true)
                        .apply()
                    Log.d(TAG, "InfoScreen: Navigating to MapScreen after consent")
                    navController.navigate(Screen.RegisterDetails.route) {
                        popUpTo(navController.graph.startDestinationId) { inclusive = true }
                    }
                },
                onDismiss = {
                    Log.d(TAG, "InfoScreen: Consent dismissed")
                    showConsent = false
                }
            )
        } else {
            Log.d(TAG, "InfoScreen: Showing info content")
            InfoContent(
                onContinue = {
                    Log.i(TAG, "InfoScreen: Continue button pressed")
                    sharedPreferences.edit()
                        .putBoolean("info_screen_shown", true)
                        .apply()
                    if (consentGiven) {
                        Log.d(TAG, "InfoScreen: Navigating to MapScreen")
                        navController.navigate(Screen.RegisterDetails.route) {
                            popUpTo(navController.graph.startDestinationId) { inclusive = true }
                        }
                    } else {
                        Log.d(TAG, "InfoScreen: Showing consent screen")
                        showConsent = true
                    }
                }
            )
        }
    }
}

@Composable
fun InfoContent(onContinue: () -> Unit) {
    Log.d(TAG, "InfoContent: Composable started")
    val context = LocalContext.current
    val markwon = remember {
        Markwon.builder(context)
            .usePlugin(StrikethroughPlugin.create())
            .usePlugin(TablePlugin.create(context))
            .build()
    }

    var infoText by remember { mutableStateOf("") }

    LaunchedEffect(Unit) {
        Log.d(TAG, "InfoContent: Loading info text")
        infoText = loadMarkdownFile(context, "info.md")
        Log.d(TAG, "InfoContent: Info text loaded, length: ${infoText.length}")
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.SpaceBetween,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        AndroidView(
            factory = { context ->
                TextView(context).apply {
                    Log.d(TAG, "InfoContent: Setting markdown in TextView")
                    markwon.setMarkdown(this, infoText)
                }
            },
            update = { textView ->
                Log.d(TAG, "InfoContent: Updating TextView with markdown")
                markwon.setMarkdown(textView, infoText)
            },
            modifier = Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState())
        )

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = {
            Log.d(TAG, "InfoContent: Continue button clicked")
            onContinue()
        }) {
            Text("Continue")
        }
    }
}