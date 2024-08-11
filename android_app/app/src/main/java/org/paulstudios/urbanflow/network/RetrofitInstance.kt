package org.paulstudios.urbanflow.network

import org.paulstudios.datasurvey.network.TrackerAPI
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

object RetrofitInstance {
    private const val BASE_URL = "https://urbanflow.onrender.com"

    val api: TrackerAPI by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(TrackerAPI::class.java)
    }
}