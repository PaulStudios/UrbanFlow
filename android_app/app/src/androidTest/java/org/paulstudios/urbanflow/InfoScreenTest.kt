package org.paulstudios.urbanflow

import android.content.Context
import android.content.SharedPreferences
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.navigation.testing.TestNavHostController
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mockito.anyBoolean
import org.mockito.Mockito.anyString
import org.mockito.Mockito.mock
import org.mockito.Mockito.verify
import org.mockito.Mockito.`when`
import org.paulstudios.urbanflow.data.models.Screen
import org.paulstudios.urbanflow.ui.screens.others.InfoScreen

@RunWith(AndroidJUnit4::class)
class InfoScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private lateinit var context: Context
    private lateinit var navController: TestNavHostController
    private lateinit var sharedPreferences: SharedPreferences
    private lateinit var editor: SharedPreferences.Editor

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        navController = TestNavHostController(context)
        sharedPreferences = mock(SharedPreferences::class.java)
        editor = mock(SharedPreferences.Editor::class.java)

        `when`(context.getSharedPreferences("AppPreferences", Context.MODE_PRIVATE)).thenReturn(sharedPreferences)
        `when`(sharedPreferences.edit()).thenReturn(editor)
        `when`(editor.putBoolean(anyString(), anyBoolean())).thenReturn(editor)
    }

    @Test
    fun testInfoScreen_firstTime() {
        `when`(sharedPreferences.getBoolean("info_screen_shown", false)).thenReturn(false)
        `when`(sharedPreferences.getBoolean("consent_given", false)).thenReturn(false)

        composeTestRule.setContent {
            InfoScreen(navController)
        }

        composeTestRule.onNodeWithText("Continue").assertExists()
        composeTestRule.onNodeWithText("Continue").performClick()

        verify(editor).putBoolean("info_screen_shown", true)
        composeTestRule.onNodeWithText("Privacy Policy").assertExists()
    }

    @Test
    fun testInfoScreen_consentGiven() {
        `when`(sharedPreferences.getBoolean("info_screen_shown", false)).thenReturn(true)
        `when`(sharedPreferences.getBoolean("consent_given", false)).thenReturn(true)

        composeTestRule.setContent {
            InfoScreen(navController)
        }

        verify(navController).navigate(Screen.MapScreen.route)
    }

    @Test
    fun testInfoScreen_consentNotGiven() {
        `when`(sharedPreferences.getBoolean("info_screen_shown", false)).thenReturn(true)
        `when`(sharedPreferences.getBoolean("consent_given", false)).thenReturn(false)

        composeTestRule.setContent {
            InfoScreen(navController)
        }

        composeTestRule.onNodeWithText("Privacy Policy").assertExists()
    }
}