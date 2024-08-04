package org.paulstudios.datasurvey.network

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Path

interface TrackerAPI {
    @GET("/status")
    suspend fun getServerStatus(): Response<Unit>

}

interface EncyptedApiService {
    @POST("api/encryption/exchange_key")
    suspend fun exchangeKey(@Body request: PublicKeyRequest): Response<String>

    @POST("api/encryption/send_data")
    suspend fun sendData(@Body request: EncryptedDataRequest): Response<Unit>

    @GET("api/encryption/receive_data/{clientId}")
    suspend fun receiveData(@Path("clientId") clientId: String): Response<EncryptedDataResponse>

    @GET("api/encryption/check_key_validity/{clientId}")
    suspend fun checkKeyValidity(@Path("clientId") clientId: String): Response<Unit>

    @POST("api/verify/user")
    suspend fun verifyUser(@Body request: EncryptedDataRequest): Response<EncryptedDataResponse>
}

data class PublicKeyRequest(
    val public_key: String,
    val kdf: String
)
data class EncryptedDataRequest(
    val client_id: String,
    val encrypted_data: String,
    val iv: String
)
data class EncryptedDataResponse(
    val encrypted_data: String,
    val iv: String
)
data class UserBase(
    val id: String,
    val name: String,
    val date_of_birth: String,  // Use ISO8601 format
    val mobile_number: String,
    val license_number: String,
    val vehicle_number: String,
    val aadhar_number: String,
    val permit_uri: String,
    val selfie_uri: String
)
data class VerifyResponse(
    val status: String,
    val checked_at: String,
)