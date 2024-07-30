package org.paulstudios.urbanflow

import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.navigation.NavHostController
import androidx.navigation.NavOptionsBuilder
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.firebase.auth.FirebaseUser
import kotlinx.coroutines.flow.MutableStateFlow
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mockito.any
import org.mockito.Mockito.eq
import org.mockito.Mockito.mock
import org.mockito.Mockito.verify
import org.mockito.Mockito.`when`
import org.paulstudios.urbanflow.data.models.Screen
import org.paulstudios.urbanflow.ui.screens.auth.LoginScreen
import org.paulstudios.urbanflow.viewmodels.AuthState
import org.paulstudios.urbanflow.viewmodels.AuthViewModel

@RunWith(AndroidJUnit4::class)
class LoginScreenTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    private lateinit var navController: NavHostController
    private lateinit var viewModel: AuthViewModel

    @Before
    fun setup() {
        navController = mock(NavHostController::class.java)
        viewModel = mock(AuthViewModel::class.java)
        `when`(viewModel.authState).thenReturn(MutableStateFlow(AuthState.Idle))
    }

    @Test
    fun testLoginScreen_initialState() {
        composeTestRule.setContent {
            LoginScreen(navController = navController, viewModel = viewModel)
        }

        composeTestRule.onNodeWithText("Login").assertExists()
        composeTestRule.onNodeWithText("Don't have an account? Register").assertExists()
    }

    @Test
    fun testLoginScreen_loginButtonClick() {
        composeTestRule.setContent {
            LoginScreen(navController = navController, viewModel = viewModel)
        }

        composeTestRule.onNodeWithText("Login").performClick()
        verify(viewModel).login(any(), any())
    }

    @Test
    fun testLoginScreen_registerNavigationClick() {
        composeTestRule.setContent {
            LoginScreen(navController = navController, viewModel = viewModel)
        }

        composeTestRule.onNodeWithText("Don't have an account? Register").performClick()
        verify(navController).navigate(Screen.Register.route)
    }

    @Test
    fun testLoginScreen_successState() {
        val mockUser = mock(FirebaseUser::class.java)
        `when`(viewModel.authState).thenReturn(MutableStateFlow(AuthState.Success(mockUser)))

        composeTestRule.setContent {
            LoginScreen(navController = navController, viewModel = viewModel)
        }

        verify(navController).navigate(eq(Screen.InfoScreen.route), any<NavOptionsBuilder.() -> Unit>())
    }
}