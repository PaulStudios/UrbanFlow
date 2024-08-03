package org.paulstudios.urbanflow.network

import android.util.Base64
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.paulstudios.datasurvey.network.ApiService
import org.paulstudios.datasurvey.network.EncryptedDataRequest
import org.paulstudios.datasurvey.network.PublicKeyRequest
import org.paulstudios.urbanflow.utils.CryptoUtils
import org.paulstudios.urbanflow.utils.CryptoUtils.generateSharedSecret
import org.paulstudios.urbanflow.utils.CryptoUtils.getOrGenerateKeyPair
import org.spongycastle.crypto.digests.SHA256Digest
import org.spongycastle.crypto.generators.HKDFBytesGenerator
import org.spongycastle.crypto.params.HKDFParameters
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.security.KeyFactory
import java.security.KeyStore
import java.security.spec.X509EncodedKeySpec
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

private const val KEY_ALIAS = "MyAppKeyPair"

class SecureApiClient(baseUrl: String) {
    private val retrofit: Retrofit = Retrofit.Builder()
        .baseUrl(baseUrl)
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    private val apiService: ApiService = retrofit.create(ApiService::class.java)
    private var sharedKey: ByteArray? = null
    private val TAG = "SecureApiClient"


    suspend fun exchangeKey() {
        withContext(Dispatchers.IO) {
            try {
                val keyPair = getOrGenerateKeyPair()
                val publicKeyBase64 = Base64.encodeToString(keyPair.public.encoded, Base64.NO_WRAP)

                Log.d("SecureApiClient", "Sending public key: $publicKeyBase64")
                val response = apiService.exchangeKey(PublicKeyRequest(publicKeyBase64, "HKDF"))

                if (response.isSuccessful) {
                    val serverPublicKeyBase64 = response.body() ?: throw Exception("Empty response body")
                    val serverPublicKey = KeyFactory.getInstance("EC").generatePublic(
                        X509EncodedKeySpec(Base64.decode(serverPublicKeyBase64, Base64.NO_WRAP))
                    )

                    val privateKey = keyPair.private
                    val sharedSecret = generateSharedSecret(privateKey, serverPublicKey)
                    Log.d(TAG, "Shared secret (first 10 bytes): ${sharedSecret.toHex()}")

                    // Perform HKDF
                    sharedKey = performHKDF(sharedSecret)
                    Log.d(TAG, "Shared key (first 10 bytes): ${sharedKey!!.toHex()}")

                } else {
                    val errorBody = response.errorBody()?.string()
                    throw Exception("Key exchange failed: ${response.code()} - $errorBody")
                }
            } catch (e: Exception) {
                Log.e("SecureApiClient", "Key exchange error", e)
                throw e
            }
        }
    }

    fun performHKDF(sharedSecret: ByteArray, salt: ByteArray? = null, info: ByteArray = ByteArray(0), keyLen: Int = 32): ByteArray {
        // Extract
        val prk: ByteArray
        if (salt == null || salt.isEmpty()) {
            // If salt is null or empty, use a hash-length of zeroes
            prk = hmacSha256(ByteArray(32), sharedSecret)
        } else {
            prk = hmacSha256(salt, sharedSecret)
        }
        Log.d(TAG, "PRK: ${prk.toHex()}")

        // Expand
        val output = ByteArray(keyLen)
        var t = ByteArray(0)
        var offset = 0
        var i = 1
        while (offset < keyLen) {
            t = hmacSha256(prk, t + info + byteArrayOf(i.toByte()))
            val remaining = keyLen - offset
            System.arraycopy(t, 0, output, offset, minOf(t.size, remaining))
            offset += t.size
            i++
        }

        return output
    }

    fun hmacSha256(key: ByteArray, data: ByteArray): ByteArray {
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(key, "HmacSHA256"))
        return mac.doFinal(data)
    }


    suspend fun sendData(data: String) {
        withContext(Dispatchers.IO) {
            if (sharedKey == null) {
                exchangeKey()
            }

            val (encryptedData, iv) = CryptoUtils.encrypt(data.toByteArray(), sharedKey!!)
            val encryptedDataBase64 = Base64.encodeToString(encryptedData, Base64.NO_WRAP)
            val ivBase64 = Base64.encodeToString(iv, Base64.NO_WRAP)

            val request = EncryptedDataRequest(encryptedDataBase64, ivBase64)
            apiService.sendData(request)
        }
    }

    suspend fun receiveData(): String {
        return withContext(Dispatchers.IO) {
            if (sharedKey == null) {
                exchangeKey()
            }

            val (encryptedDataBase64, ivBase64) = apiService.receiveData()
            val encryptedData = Base64.decode(encryptedDataBase64, Base64.NO_WRAP)
            val iv = Base64.decode(ivBase64, Base64.NO_WRAP)

            val decryptedData = CryptoUtils.decrypt(encryptedData, sharedKey!!, iv)
            String(decryptedData)
        }
    }
}

fun ByteArray.toHex(): String = joinToString(separator = "") { eachByte -> "%02x".format(eachByte) }