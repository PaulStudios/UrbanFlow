package org.paulstudios.urbanflow.utils

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.location.LocationManager
import android.util.Log
import androidx.localbroadcastmanager.content.LocalBroadcastManager

class GpsStatusReceiver : BroadcastReceiver() {
    private var listener: ((Boolean) -> Unit)? = null

    fun setListener(listener: (Boolean) -> Unit) {
        this.listener = listener
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == LocationManager.PROVIDERS_CHANGED_ACTION) {
            val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
            val isGpsEnabled = locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)

            listener?.invoke(isGpsEnabled)

            if (!isGpsEnabled) {
                Log.d("GpsStatusReceiver", "GPS disabled, sending broadcast")
                val disabledIntent = Intent("org.paulstudios.datasurvey.ACTION_GPS_DISABLED")
                LocalBroadcastManager.getInstance(context).sendBroadcast(disabledIntent)
                Log.d("GpsStatusReceiver", "Broadcast sent: ${disabledIntent.action}")
            }
        }
    }
}