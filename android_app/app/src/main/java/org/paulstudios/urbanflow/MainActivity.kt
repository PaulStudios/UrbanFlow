package org.paulstudios.urbanflow

import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.annotation.RequiresApi
import androidx.compose.animation.ExperimentalAnimationApi
import androidx.compose.runtime.LaunchedEffect
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.navigation.NavHostController
import com.google.accompanist.navigation.animation.rememberAnimatedNavController
import com.google.android.gms.auth.api.signin.GoogleSignIn
import com.google.android.gms.common.api.ApiException
import com.google.firebase.FirebaseApp
import com.google.firebase.analytics.FirebaseAnalytics
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.OAuthProvider
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import org.paulstudios.datasurvey.network.VehicleCreateRequest
import org.paulstudios.urbanflow.data.models.Screen
import org.paulstudios.urbanflow.network.SecureApiClient
import org.paulstudios.urbanflow.ui.theme.UrbanFlowTheme
import org.paulstudios.urbanflow.viewmodels.AuthState
import org.paulstudios.urbanflow.viewmodels.AuthViewModel
import org.paulstudios.urbanflow.viewmodels.ServerStatusViewModel

class MainActivity : ComponentActivity() {
    private lateinit var navController: NavHostController
    private lateinit var authViewModel: AuthViewModel
    private lateinit var serverStatusViewModel: ServerStatusViewModel
    private lateinit var secureApiClient: SecureApiClient
    lateinit var firebaseAnalytics: FirebaseAnalytics

    @RequiresApi(Build.VERSION_CODES.UPSIDE_DOWN_CAKE)
    @OptIn(ExperimentalAnimationApi::class, DelicateCoroutinesApi::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        FirebaseApp.initializeApp(this)
        firebaseAnalytics = FirebaseAnalytics.getInstance(this)
        secureApiClient = SecureApiClient("https://urbanflow.onrender.com/", context = this)

        // Perform key exchange when the app starts
        GlobalScope.launch {
            try {
                testEncryptedCommunication()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        authViewModel = ViewModelProvider(this, ViewModelProvider.AndroidViewModelFactory.getInstance(application))[AuthViewModel::class.java]
        serverStatusViewModel = ViewModelProvider(this)[ServerStatusViewModel::class.java]
        setContent {
            navController = rememberAnimatedNavController()
            LaunchedEffect(Unit) {
                checkAndNavigateIfLoggedIn()
            }
            UrbanFlowTheme {
                MyApp(navController, this)
            }
        }
        lifecycle.addObserver(serverStatusViewModel)
        observeAuthState()
    }

    private fun checkAndNavigateIfLoggedIn() {
        val (isLoggedIn, _) = authViewModel.checkLoginState()
        if (isLoggedIn) {
            navigateToMainScreen()
        }
    }

    private suspend fun testEncryptedCommunication() {
        try {
            // Send encrypted data to the server
            secureApiClient.sendData("Hello from Android!")

            // Receive encrypted data from the server
            val receivedData = secureApiClient.receiveData()
            println("Received data from server: $receivedData")

            // TODO: Remove debug code below

            val r = VehicleCreateRequest(
                type = "Ambulance",
                origin = "Madhyamgram, Barasat, IN",
                destination = "Barasat, West Bengal, IN"
            )

            secureApiClient.registerVehicle(r)

        } catch (e: Exception) {
            // Handle communication error
            e.printStackTrace()
        }
    }

    private fun observeAuthState() {
        lifecycleScope.launch {
            authViewModel.authState.collect { state ->
                when (state) {
                    is AuthState.Success -> navigateToMainScreen()
                    is AuthState.Error -> showErrorToast(state.message)
                    else -> {} // Handle other states if needed
                }
            }
        }
    }

    private fun navigateToMainScreen() {
        navController.navigate(Screen.InfoScreen.route) {
            popUpTo(navController.graph.startDestinationId) { inclusive = true }
        }
    }

    private fun showErrorToast(message: String) {
        Toast.makeText(this@MainActivity, message, Toast.LENGTH_LONG).show()
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == RC_SIGN_IN) {
            handleGoogleSignInResult(data)
        }
    }

    private fun handleGoogleSignInResult(data: Intent?) {
        try {
            val account = GoogleSignIn.getSignedInAccountFromIntent(data).getResult(ApiException::class.java)
            account.idToken?.let { authViewModel.firebaseAuthWithGoogle(it) }
        } catch (e: ApiException) {
            authViewModel.handleAuthError(e)
        }
    }

    fun signInWithGithub() {
        val provider = OAuthProvider.newBuilder("github.com")
        FirebaseAuth.getInstance().startActivityForSignInWithProvider(this, provider.build())
            .addOnCompleteListener { task ->
                authViewModel.handleGithubSignInResult(task)
            }
    }

    companion object {
        const val RC_SIGN_IN = 9001
    }

    override fun onDestroy() {
        super.onDestroy()
        lifecycle.removeObserver(serverStatusViewModel)
    }
}