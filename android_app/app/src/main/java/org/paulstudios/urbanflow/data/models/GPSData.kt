package org.paulstudios.urbanflow.data.models

import kotlinx.serialization.Serializable

@Serializable
data class GPSData(
    val latitude: Double,
    val longitude: Double,
    val timestamp: String
)
