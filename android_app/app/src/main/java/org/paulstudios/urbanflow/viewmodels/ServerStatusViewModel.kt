package org.paulstudios.urbanflow.viewmodels

import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import org.paulstudios.urbanflow.network.RetrofitInstance
import java.io.IOException

class ServerStatusViewModel : ViewModel(), DefaultLifecycleObserver {
    private val _serverStatus = MutableStateFlow(false)
    val serverStatus: StateFlow<Boolean> = _serverStatus

    private val _statusMessage = MutableSharedFlow<String>()
    val statusMessage: SharedFlow<String> = _statusMessage

    private var statusCheckJob: Job? = null
    private val foregroundCheckInterval = 15000L // 15 seconds
    private val backgroundCheckInterval = 60000L // 1 minute

    init {
        startPeriodicStatusCheck(foregroundCheckInterval)
    }

    override fun onResume(owner: LifecycleOwner) {
        startPeriodicStatusCheck(foregroundCheckInterval)
    }

    override fun onPause(owner: LifecycleOwner) {
        startPeriodicStatusCheck(backgroundCheckInterval)
    }

    private fun startPeriodicStatusCheck(interval: Long) {
        statusCheckJob?.cancel()
        statusCheckJob = viewModelScope.launch {
            while (true) {
                checkServerStatus()
                delay(interval)
            }
        }
    }

    private fun checkServerStatus() {
        viewModelScope.launch {
            try {
                val response = RetrofitInstance.api.getServerStatus()
                _serverStatus.value = response.isSuccessful
            } catch (e: IOException) {
                _serverStatus.value = false
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        statusCheckJob?.cancel()
    }

    fun showStatusMessage() {
        viewModelScope.launch {
            val message = if (_serverStatus.value) {
                "Server is online"
            } else {
                "Server is offline"
            }
            _statusMessage.emit(message)
        }
    }
}