package org.paulstudios.urbanflow.ui.screens.auth

import org.paulstudios.urbanflow.MainActivity
import android.app.Activity
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.platform.LocalContext
import androidx.navigation.NavController
import org.paulstudios.datasurvey.ui.screens.auth.components.AuthScreen
import org.paulstudios.urbanflow.data.models.Screen
import org.paulstudios.urbanflow.viewmodels.AuthState
import org.paulstudios.urbanflow.viewmodels.AuthViewModel


@Composable
fun RegisterScreen(navController: NavController, viewModel: AuthViewModel) {
    val context = LocalContext.current
    val activity = context as? MainActivity

    val authState by viewModel.authState.collectAsState()

    AuthScreen(
        email = viewModel.email,
        onEmailChange = { viewModel.email = it },
        password = viewModel.password,
        onPasswordChange = { viewModel.password = it },
        buttonText = "Register",
        onSubmit = {
            viewModel.register(context) {
                navController.navigate(Screen.InfoScreen.route)
            }
        },
        secondaryButtonText = "Already have an account? Login",
        onSecondaryButtonClick = { navController.navigate(Screen.Login.route) },
        errorMessage = if (authState is AuthState.Error) (authState as AuthState.Error).message else "",
        isLoading = authState is AuthState.Loading,
        onGithubLogin = { (context as? MainActivity)?.signInWithGithub() },
        onGoogleLogin = { viewModel.signInWithGoogle(context as Activity) }
    )

    LaunchedEffect(authState) {
        when (authState) {
            is AuthState.Success -> {
                navController.navigate(Screen.InfoScreen.route) {
                    popUpTo(navController.graph.startDestinationId) { inclusive = true }
                }
            }
            else -> {} // Handle other states if needed
        }
    }
}