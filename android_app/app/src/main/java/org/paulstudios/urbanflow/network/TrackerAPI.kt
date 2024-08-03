package org.paulstudios.datasurvey.network

import retrofit2.Call
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Query

interface TrackerAPI {
    @GET("/status")
    suspend fun getServerStatus(): Response<Unit>

}

interface ApiService {
    @POST("api/encryption/exchange_key")
    suspend fun exchangeKey(@Body request: PublicKeyRequest): Response<String>

    @POST("api/encryption/send_data")
    suspend fun sendData(@Body request: EncryptedDataRequest): Response<Unit>

    @GET("api/encryption/receive_data")
    suspend fun receiveData(): EncryptedDataRequest
}

data class PublicKeyRequest(
    val public_key: String,
    val kdf: String
)
data class EncryptedDataRequest(
    val encrypted_data: String,
    val iv: String
)