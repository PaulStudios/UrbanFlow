package org.paulstudios.datasurvey.ui.screens.auth.components

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Email
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import org.paulstudios.urbanflow.R

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AuthScreen(
    email: String,
    onEmailChange: (String) -> Unit,
    password: String,
    onPasswordChange: (String) -> Unit,
    buttonText: String,
    onSubmit: () -> Unit,
    secondaryButtonText: String,
    onSecondaryButtonClick: () -> Unit,
    passwordStrengthComposable: @Composable (() -> Unit)? = null,
    errorMessage: String = "",
    isLoading: Boolean = false,
    onGithubLogin: () -> Unit,
    onGoogleLogin: () -> Unit
) {
    var emailError by remember { mutableStateOf("") }
    val snackbarHostState = remember { SnackbarHostState() }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues),
            contentAlignment = Alignment.Center
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Image(
                    painter = painterResource(id = R.drawable.logo),
                    contentDescription = "App Logo",
                    modifier = Modifier.size(100.dp)
                )

                Text(
                    text = buttonText,
                    style = MaterialTheme.typography.headlineLarge,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(bottom = 16.dp)
                )

                Spacer(modifier = Modifier.height(32.dp))

                OutlinedTextField(
                    value = email,
                    onValueChange = {
                        onEmailChange(it)
                        emailError = if (isValidEmail(it)) "" else "Invalid Email"
                    },
                    label = { Text("Email") },
                    leadingIcon = { Icon(Icons.Filled.Email, contentDescription = null) },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    isError = emailError.isNotEmpty()
                )
                if (emailError.isNotEmpty()) {
                    Text(emailError, color = MaterialTheme.colorScheme.error)
                }

                Spacer(modifier = Modifier.height(16.dp))

                OutlinedTextField(
                    value = password,
                    onValueChange = onPasswordChange,
                    label = { Text("Password") },
                    leadingIcon = { Icon(Icons.Filled.Lock, contentDescription = null) },
                    visualTransformation = PasswordVisualTransformation(),
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp)
                )

                passwordStrengthComposable?.invoke()

                Spacer(modifier = Modifier.height(24.dp))

                Button(
                    onClick = onSubmit,
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    enabled = !isLoading
                ) {
                    Text(buttonText)
                }

                Spacer(modifier = Modifier.height(16.dp))

                Button(onClick = onSecondaryButtonClick) {
                    Text(secondaryButtonText)
                }

                Spacer(modifier = Modifier.height(16.dp))

                Button(onClick = onGoogleLogin, modifier = Modifier.fillMaxWidth()) {
                    Text("Sign in with Google")
                }

                Spacer(modifier = Modifier.height(16.dp))

                Button(onClick = onGithubLogin, modifier = Modifier.fillMaxWidth()) {
                    Text("Sign in with GitHub")
                }
            }

            if (isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier
                        .size(50.dp)
                        .align(Alignment.Center)
                )
            }

            LaunchedEffect(errorMessage) {
                if (errorMessage.isNotEmpty()) {
                    snackbarHostState.showSnackbar(
                        message = errorMessage,
                        duration = SnackbarDuration.Long
                    )
                }
            }
        }
    }
}