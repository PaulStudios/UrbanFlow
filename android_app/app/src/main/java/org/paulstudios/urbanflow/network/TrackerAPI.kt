package org.paulstudios.datasurvey.network

import retrofit2.Response
import retrofit2.http.GET

interface TrackerAPI {
    @GET("/status")
    suspend fun getServerStatus(): Response<Unit>
}

