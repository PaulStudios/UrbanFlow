package org.paulstudios.urbanflow

import android.content.Context
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mockito.mock
import org.mockito.Mockito.verify
import org.paulstudios.urbanflow.utils.MarkdownViewerScreen
import org.paulstudios.urbanflow.utils.loadMarkdownFile

@RunWith(AndroidJUnit4::class)
class MarkdownViewerScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private lateinit var context: Context

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
    }

    @Test
    fun testLoadMarkdownFile_success() {
        val content = loadMarkdownFile(context, "test_file.md")
        assertEquals("Test content", content)
    }

    @Test
    fun testLoadMarkdownFile_failure() {
        val content = loadMarkdownFile(context, "non_existent_file.md")
        assertEquals("Error loading content.", content)
    }

    @Test
    fun testMarkdownViewerScreen_privacyPolicy() {
        val onConsentGiven = mock(Function0::class.java)
        val onDismiss = mock(Function0::class.java)

        composeTestRule.setContent {
            MarkdownViewerScreen(
                onConsentGiven = { onConsentGiven.invoke() },
                onDismiss = { onDismiss.invoke() }
            )
        }

        composeTestRule.onNodeWithText("Privacy Policy").assertExists()
        composeTestRule.onNodeWithText("Agree").performClick()

        composeTestRule.onNodeWithText("Terms and Conditions").assertExists()
        composeTestRule.onNodeWithText("Agree").performClick()

        verify(onConsentGiven).invoke()
    }

    @Test
    fun testMarkdownViewerScreen_dismiss() {
        val onDismiss = mock(Function0::class.java)

        composeTestRule.setContent {
            MarkdownViewerScreen(
                onConsentGiven = { },
                onDismiss = { onDismiss.invoke() }
            )
        }

        composeTestRule.onNodeWithText("Decline").performClick()
        verify(onDismiss).invoke()
    }
}