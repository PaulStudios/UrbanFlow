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
import org.paulstudios.urbanflow.ui.screens.auth.RegisterScreen
import org.paulstudios.urbanflow.viewmodels.AuthState
import org.paulstudios.urbanflow.viewmodels.AuthViewModel

@RunWith(AndroidJUnit4::class)
class RegisterScreenTest {

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
    fun testRegisterScreen_initialState() {
        composeTestRule.setContent {
            RegisterScreen(navController = navController, viewModel = viewModel)
        }

        composeTestRule.onNodeWithText("Register").assertExists()
        composeTestRule.onNodeWithText("Already have an account? Login").assertExists()
    }

    @Test
    fun testRegisterScreen_registerButtonClick() {
        composeTestRule.setContent {
            RegisterScreen(navController = navController, viewModel = viewModel)
        }

        composeTestRule.onNodeWithText("Register").performClick()
        verify(viewModel).register(any(), any())
    }

    @Test
    fun testRegisterScreen_loginNavigationClick() {
        composeTestRule.setContent {
            RegisterScreen(navController = navController, viewModel = viewModel)
        }

        composeTestRule.onNodeWithText("Already have an account? Login").performClick()
        verify(navController).navigate(Screen.Login.route)
    }

    @Test
    fun testRegisterScreen_successState() {
        val mockUser = mock(FirebaseUser::class.java)
        `when`(viewModel.authState).thenReturn(MutableStateFlow(AuthState.Success(mockUser)))

        composeTestRule.setContent {
            RegisterScreen(navController = navController, viewModel = viewModel)
        }

        verify(navController).navigate(eq(Screen.InfoScreen.route), any<NavOptionsBuilder.() -> Unit>())
    }
}