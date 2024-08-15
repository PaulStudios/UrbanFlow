package org.paulstudios.urbanflow.network

import android.content.Context
import android.util.Base64
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.paulstudios.datasurvey.network.EncyptedApiService
import org.paulstudios.datasurvey.network.EncryptedDataRequest
import org.paulstudios.datasurvey.network.PublicKeyRequest
import org.paulstudios.datasurvey.network.UserBase
import org.paulstudios.urbanflow.utils.CryptoUtils
import org.paulstudios.urbanflow.utils.CryptoUtils.generateSharedSecret
import org.paulstudios.urbanflow.utils.CryptoUtils.getOrGenerateKeyPair
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.security.KeyFactory
import java.security.KeyStore
import java.security.spec.X509EncodedKeySpec
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import com.google.gson.Gson
import org.paulstudios.datasurvey.network.VehicleCreateRequest
import org.paulstudios.datasurvey.network.VerifyResponse
import org.paulstudios.urbanflow.utils.logDataReceived
import org.paulstudios.urbanflow.utils.logDataSent
import org.paulstudios.urbanflow.utils.logKeyExchange
import java.security.KeyPair
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.spec.GCMParameterSpec

private const val KEY_ALIAS = "MyAppKeyPair"

class SecureApiClient(baseUrl: String, private val context: Context) {
    private val retrofit: Retrofit = Retrofit.Builder()
        .baseUrl(baseUrl)
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    private val encyptedApiService: EncyptedApiService = retrofit.create(EncyptedApiService::class.java)
    private var sharedKey: ByteArray? = null
    private val TAG = "SecureApiClient"

    private val keyAlias = "SharedKeyAlias"
    private val sharedPrefsName = "SecureApiClientPrefs"

    private lateinit var clientId: String
    private lateinit var keyPair: KeyPair


    private fun saveSharedKey(key: ByteArray) {
        val keyGenerator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        val keyGenParameterSpec = KeyGenParameterSpec.Builder(keyAlias,
            KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
            .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
            .build()

        keyGenerator.init(keyGenParameterSpec)
        val secretKey = keyGenerator.generateKey()

        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, secretKey)

        val encryptedKey = cipher.doFinal(key)
        val iv = cipher.iv

        val combinedData = iv + encryptedKey

        val prefs = context.getSharedPreferences(sharedPrefsName, Context.MODE_PRIVATE)
        prefs.edit().putString("encrypted_shared_key", Base64.encodeToString(combinedData, Base64.DEFAULT)).apply()
    }

    private fun loadSharedKey(): ByteArray? {
        val prefs = context.getSharedPreferences(sharedPrefsName, Context.MODE_PRIVATE)
        val encodedCombinedData = prefs.getString("encrypted_shared_key", null) ?: return null

        val combinedData = Base64.decode(encodedCombinedData, Base64.DEFAULT)
        val iv = combinedData.slice(0 until 12).toByteArray()
        val encryptedKey = combinedData.slice(12 until combinedData.size).toByteArray()

        val keyStore = KeyStore.getInstance("AndroidKeyStore")
        keyStore.load(null)

        val secretKeyEntry = keyStore.getEntry(keyAlias, null) as? KeyStore.SecretKeyEntry
        val secretKey = secretKeyEntry?.secretKey ?: return null

        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, secretKey, GCMParameterSpec(128, iv))

        return cipher.doFinal(encryptedKey)
    }

    suspend fun ensureValidKey() {
        if (::clientId.isInitialized.not()) {
            Log.d(TAG, "Client ID is empty, fetching keys")
            getKeys()
        }
        if (sharedKey == null) {
            Log.d(TAG, "Shared key is null, performing key exchange")
            exchangeKey()
        } else {
            Log.d(TAG, "Checking key validity")
            val c = encyptedApiService.checkKeyValidity(clientId)
            if (c.isSuccessful) {
                Log.d(TAG, "Key is valid")
            } else {
                Log.d(TAG, "Key is invalid, performing a new key exchange")
                exchangeKey()
            }
        }
    }

    private suspend fun getKeys() {
        withContext(Dispatchers.IO) {
            try {
                // Try to load the existing shared key
                sharedKey = loadSharedKey()
                keyPair = getOrGenerateKeyPair()
                clientId = Base64.encodeToString(keyPair.public.encoded, Base64.URL_SAFE or Base64.NO_WRAP)
            } catch (e: Exception) {
                Log.e("SecureApiClient", "Error fetching Keys", e)
                throw e
            }
        }
    }

    suspend fun exchangeKey() {
        withContext(Dispatchers.IO) {
            try {
                val publicKeyBase64 = Base64.encodeToString(keyPair.public.encoded, Base64.NO_WRAP)

                Log.d("SecureApiClient", "Sending public key: $publicKeyBase64")
                val response = encyptedApiService.exchangeKey(PublicKeyRequest(publicKeyBase64, "HKDF"))

                if (response.isSuccessful) {
                    val serverPublicKeyBase64 = response.body() ?: throw Exception("Empty response body")
                    val serverPublicKey = KeyFactory.getInstance("EC").generatePublic(
                        X509EncodedKeySpec(Base64.decode(serverPublicKeyBase64, Base64.NO_WRAP))
                    )

                    val privateKey = keyPair.private
                    val sharedSecret = generateSharedSecret(privateKey, serverPublicKey)
                    Log.d(TAG, "Shared secret: ${sharedSecret.toHex()}")

                    // Perform HKDF
                    sharedKey = performHKDF(sharedSecret)
                    Log.d(TAG, "Shared key: ${sharedKey!!.toHex()}")

                    logKeyExchange(true, clientId)
                    saveSharedKey(sharedKey!!)
                } else {
                    logKeyExchange(false, clientId, response.errorBody()?.string())
                    throw Exception("Key exchange failed: ${response.code()} - ${response.errorBody()?.string()}")
                }
            } catch (e: Exception) {
                logKeyExchange(false, clientId, e.message)
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
            getKeys()
            ensureValidKey()
            try {
                val (encryptedData, iv) = CryptoUtils.encrypt(data.toByteArray(), sharedKey!!)
                val encryptedDataBase64 = Base64.encodeToString(encryptedData, Base64.NO_WRAP)
                val ivBase64 = Base64.encodeToString(iv, Base64.NO_WRAP)

                val request = EncryptedDataRequest(clientId, encryptedDataBase64, ivBase64)
                Log.d(TAG, "Sending data: ${request.encrypted_data}")
                encyptedApiService.sendData(request)
                logDataSent(true, clientId)
            } catch (e: Exception) {
                Log.e(TAG, "Error sending data", e)
                logDataSent(false, clientId, e.message)
            }
        }
    }

    suspend fun sendUserData(user: UserBase): VerifyResponse? {
        var verifyResponse: VerifyResponse? = null
        withContext(Dispatchers.IO) {
            ensureValidKey()

            try {
                val gson = Gson()
                val userJson = gson.toJson(user)

                val (encryptedData, iv) = CryptoUtils.encrypt(userJson.toByteArray(), sharedKey!!)
                val encryptedDataBase64 = Base64.encodeToString(encryptedData, Base64.NO_WRAP)
                val ivBase64 = Base64.encodeToString(iv, Base64.NO_WRAP)

                val request = EncryptedDataRequest(clientId, encryptedDataBase64, ivBase64)
                Log.d(TAG, "Sending user data: ${request.encrypted_data}")

                val response = encyptedApiService.verifyUser(request)
                logDataSent(true, clientId)

                if (response.isSuccessful) {
                    Log.d(TAG, "Received data: ${response.body()?.encrypted_data}")
                    val body = response.body() ?: throw Exception("Empty response body")
                    val encryptedData = Base64.decode(body.encrypted_data, Base64.NO_WRAP)
                    val iv = Base64.decode(body.iv, Base64.NO_WRAP)

                    val decryptedData = CryptoUtils.decrypt(encryptedData, sharedKey!!, iv)
                    val decryptedString = String(decryptedData)
                    Log.d(TAG, "Decrypted data: $decryptedString")
                    // Deserialize JSON string to VerifyResponse object
                    val gson = Gson()
                    verifyResponse = gson.fromJson(decryptedString, VerifyResponse::class.java)

                    logDataReceived(true, clientId)
                    verifyResponse
                } else {
                    logDataReceived(false, clientId, response.errorBody()?.string())
                    throw Exception("Failed to receive data: ${response.code()} - ${response.errorBody()?.string()}")
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error sending user data", e)
                logDataSent(false, clientId, e.message)
            }
        }
        return verifyResponse
    }

    suspend fun registerVehicle(vehicle: VehicleCreateRequest) {
        withContext(Dispatchers.IO) {
            ensureValidKey()

            try {
                val gson = Gson()
                val vehicleJson = gson.toJson(vehicle)

                val (encryptedData, iv) = CryptoUtils.encrypt(
                    vehicleJson.toByteArray(),
                    sharedKey!!
                )
                val encryptedDataBase64 = Base64.encodeToString(encryptedData, Base64.NO_WRAP)
                val ivBase64 = Base64.encodeToString(iv, Base64.NO_WRAP)

                val request = EncryptedDataRequest(clientId, encryptedDataBase64, ivBase64)
                Log.d(TAG, "Sending vehicle registration data: ${request.encrypted_data}")

                encyptedApiService.registerVehicle(request)
                logDataSent(true, clientId)
            } catch (e: Exception) {
                Log.e(TAG, "Error sending vehicle registration data", e)
                logDataSent(false, clientId, e.message)
            }
        }
    }

    suspend fun receiveData(): String {
        return withContext(Dispatchers.IO) {
            ensureValidKey()

            val response = encyptedApiService.receiveData(clientId)
            if (response.isSuccessful) {
                Log.d(TAG, "Received data: ${response.body()?.encrypted_data}")
                val body = response.body() ?: throw Exception("Empty response body")
                val encryptedData = Base64.decode(body.encrypted_data, Base64.NO_WRAP)
                val iv = Base64.decode(body.iv, Base64.NO_WRAP)

                val decryptedData = CryptoUtils.decrypt(encryptedData, sharedKey!!, iv)
                logDataReceived(true, clientId)
                String(decryptedData)
            } else {
                logDataReceived(false, clientId, response.errorBody()?.string())
                throw Exception("Failed to receive data: ${response.code()} - ${response.errorBody()?.string()}")
            }
        }
    }
}

fun ByteArray.toHex(): String = joinToString(separator = "") { eachByte -> "%02x".format(eachByte) }