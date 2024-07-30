package org.paulstudios.urbanflow.ui.screens.auth

import android.app.Activity
import android.util.Log
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.platform.LocalContext
import androidx.navigation.NavController
import org.paulstudios.datasurvey.ui.screens.auth.components.AuthScreen
import org.paulstudios.urbanflow.MainActivity
import org.paulstudios.urbanflow.data.models.Screen
import org.paulstudios.urbanflow.viewmodels.AuthState
import org.paulstudios.urbanflow.viewmodels.AuthViewModel

@Composable
fun LoginScreen(navController: NavController, viewModel: AuthViewModel) {
    val context = LocalContext.current
    val activity = context as? Activity

    val authState by viewModel.authState.collectAsState()

    AuthScreen(
        email = viewModel.email,
        onEmailChange = { viewModel.email = it },
        password = viewModel.password,
        onPasswordChange = { viewModel.password = it },
        buttonText = "Login",
        onSubmit = {
            viewModel.login(context) {
                Log.d("LoginScreen", "Login successful")
                navController.navigate(Screen.InfoScreen.route)
            }
        },
        secondaryButtonText = "Don't have an account? Register",
        onSecondaryButtonClick = { navController.navigate(Screen.Register.route) },
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