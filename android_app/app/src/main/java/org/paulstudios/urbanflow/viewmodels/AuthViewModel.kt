package org.paulstudios.urbanflow.viewmodels

import android.app.Activity
import android.app.Application
import android.content.Context
import android.content.Intent
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.google.android.gms.auth.api.signin.GoogleSignIn
import com.google.android.gms.auth.api.signin.GoogleSignInOptions
import com.google.android.gms.common.api.ApiException
import com.google.android.gms.tasks.Task
import com.google.firebase.auth.AuthResult
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.FirebaseAuthInvalidCredentialsException
import com.google.firebase.auth.FirebaseAuthInvalidUserException
import com.google.firebase.auth.FirebaseAuthUserCollisionException
import com.google.firebase.auth.FirebaseUser
import com.google.firebase.auth.GoogleAuthProvider
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import org.paulstudios.urbanflow.R


class AuthViewModel(private val application: Application) : AndroidViewModel(application) {
    var email by mutableStateOf("")
    var password by mutableStateOf("")
    var errorMessage by mutableStateOf("")
    var isLoading by mutableStateOf(false)

    private val auth = FirebaseAuth.getInstance()
    private val _authState = MutableStateFlow<AuthState>(AuthState.Idle)
    val authState: StateFlow<AuthState> = _authState.asStateFlow()

    fun login(context: Context, onSuccess: () -> Unit) {
        isLoading = true
        viewModelScope.launch {
            try {
                auth.signInWithEmailAndPassword(email, password).await()
                handleSuccessfulLogin("email")
                onSuccess()
            } catch (e: Exception) {
                handleAuthError(e)
            } finally {
                isLoading = false
            }
        }
    }

    fun register(context: Context, onSuccess: () -> Unit) {
        isLoading = true
        viewModelScope.launch {
            try {
                auth.createUserWithEmailAndPassword(email, password).await()
                handleSuccessfulLogin("email")
                onSuccess()
            } catch (e: Exception) {
                handleAuthError(e)
            } finally {
                isLoading = false
            }
        }
    }

    fun signInWithGoogle(activity: Activity) {
        isLoading = true
        _authState.value = AuthState.Loading
        val googleSignInOptions = GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
            .requestIdToken(activity.getString(R.string.default_web_client_id))
            .requestEmail()
            .build()

        val googleSignInClient = GoogleSignIn.getClient(activity, googleSignInOptions)
        val signInIntent = googleSignInClient.signInIntent
        activity.startActivityForResult(signInIntent, RC_SIGN_IN)
    }

    fun handleGoogleSignInResult(data: Intent?) {
        try {
            val task = GoogleSignIn.getSignedInAccountFromIntent(data)
            val account = task.getResult(ApiException::class.java)
            firebaseAuthWithGoogle(account.idToken!!)
        } catch (e: ApiException) {
            _authState.value = AuthState.Error("Google sign in failed: ${e.message}")
        }
    }

    fun firebaseAuthWithGoogle(idToken: String) {
        val credential = GoogleAuthProvider.getCredential(idToken, null)
        auth.signInWithCredential(credential)
            .addOnCompleteListener { task ->
                if (task.isSuccessful) {
                    handleSuccessfulLogin("google")
                } else {
                    _authState.value =
                        AuthState.Error("Firebase auth failed: ${task.exception?.message}")
                }
            }
    }

    fun handleGithubSignInResult(task: Task<AuthResult>) {
        if (task.isSuccessful) {
            handleSuccessfulLogin("github")
        } else {
            _authState.value = AuthState.Error("GitHub sign in failed: ${task.exception?.message}")
        }
    }

    private fun handleSuccessfulLogin(provider: String) {
        saveLoginState(provider)
        _authState.value = AuthState.Success(auth.currentUser)
    }

    private fun saveLoginState(provider: String) {
        val sharedPreferences = application.getSharedPreferences("AuthPrefs", Context.MODE_PRIVATE)
        with(sharedPreferences.edit()) {
            putBoolean("isLoggedIn", true)
            putString("loginProvider", provider)
            apply()
        }
    }

    fun checkLoginState(): Pair<Boolean, String?> {
        val sharedPreferences = application.getSharedPreferences("AuthPrefs", Context.MODE_PRIVATE)
        val isLoggedIn = sharedPreferences.getBoolean("isLoggedIn", false)
        val provider = sharedPreferences.getString("loginProvider", null)
        return Pair(isLoggedIn, provider)
    }

    fun logout() {
        auth.signOut()
        clearLoginState()
        _authState.value = AuthState.Idle
    }

    private fun clearLoginState() {
        val sharedPreferences = application.getSharedPreferences("AuthPrefs", Context.MODE_PRIVATE)
        sharedPreferences.edit().clear().apply()
    }

    fun handleAuthError(exception: Exception) {
        errorMessage = when (exception) {
            is FirebaseAuthInvalidUserException -> "No account found with this email. Please register."
            is FirebaseAuthInvalidCredentialsException -> "Invalid email or password. Please try again."
            is FirebaseAuthUserCollisionException -> "An account already exists with this email. Please login."
            else -> "Authentication failed. Please check your internet connection and try again."
        }
        _authState.value = AuthState.Error(errorMessage)
    }

    fun clearError() {
        errorMessage = ""
    }

    companion object {
        const val RC_SIGN_IN = 9001
    }

    class Factory(private val application: Application) : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            if (modelClass.isAssignableFrom(AuthViewModel::class.java)) {
                @Suppress("UNCHECKED_CAST")
                return AuthViewModel(application) as T
            }
            throw IllegalArgumentException("Unknown ViewModel class")
        }
    }
}

sealed class AuthState {
    object Idle : AuthState()
    object Loading : AuthState()
    data class Success(val user: FirebaseUser?) : AuthState()
    data class Error(val message: String) : AuthState()
}