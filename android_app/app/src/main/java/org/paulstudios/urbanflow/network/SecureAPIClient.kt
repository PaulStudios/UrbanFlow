package org.paulstudios.urbanflow.network

import android.content.Context
import android.util.Base64
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.paulstudios.datasurvey.network.ApiService
import org.paulstudios.datasurvey.network.EncryptedDataRequest
import org.paulstudios.datasurvey.network.PublicKeyRequest
import org.paulstudios.datasurvey.network.UserBase
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
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

private const val KEY_ALIAS = "MyAppKeyPair"

class SecureApiClient(baseUrl: String, private val context: Context) {
    private val retrofit: Retrofit = Retrofit.Builder()
        .baseUrl(baseUrl)
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    private val apiService: ApiService = retrofit.create(ApiService::class.java)
    private var sharedKey: ByteArray? = null
    private val TAG = "SecureApiClient"

    private val keyAlias = "SharedKeyAlias"
    private val sharedPrefsName = "SecureApiClientPrefs"

    private lateinit var clientId: String

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

    suspend fun exchangeKey() {
        withContext(Dispatchers.IO) {
            try {
                // Try to load the existing shared key
                sharedKey = loadSharedKey()
                val keyPair = getOrGenerateKeyPair()
                clientId = Base64.encodeToString(keyPair.public.encoded, Base64.URL_SAFE or Base64.NO_WRAP)

                if (sharedKey == null) {
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

                        saveSharedKey(sharedKey!!)
                    } else {
                        val errorBody = response.errorBody()?.string()
                        throw Exception("Key exchange failed: ${response.code()} - $errorBody")
                    }
                } else {
                    Log.d(TAG, "Loaded existing shared key")
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

            val request = EncryptedDataRequest(clientId, encryptedDataBase64, ivBase64)
            Log.d(TAG, "Sending data: ${request.encrypted_data}")
            apiService.sendData(request)
        }
    }

    suspend fun sendUserVerify(userData: UserBase) {
        withContext(Dispatchers.IO) {
            if (sharedKey == null) {
                exchangeKey()
            }

            val jsonData = Json.encodeToString(userData)
            val (encryptedData, iv) = CryptoUtils.encrypt(jsonData.toByteArray(), sharedKey!!)
            val encryptedDataBase64 = Base64.encodeToString(encryptedData, Base64.NO_WRAP)
            val ivBase64 = Base64.encodeToString(iv, Base64.NO_WRAP)

            val request = EncryptedDataRequest(clientId, encryptedDataBase64, ivBase64)
            apiService.sendData(request)
        }
    }

    suspend fun receiveData(): String {
        return withContext(Dispatchers.IO) {
            if (sharedKey == null) {
                exchangeKey()
            }

            val response = apiService.receiveData(clientId)
            if (response.isSuccessful) {
                val body = response.body() ?: throw Exception("Empty response body")
                val encryptedData = Base64.decode(body.encrypted_data, Base64.NO_WRAP)
                val iv = Base64.decode(body.iv, Base64.NO_WRAP)

                val decryptedData = CryptoUtils.decrypt(encryptedData, sharedKey!!, iv)
                String(decryptedData)
            } else {
                throw Exception("Failed to receive data: ${response.code()} - ${response.errorBody()?.string()}")
            }
        }
    }
}

fun ByteArray.toHex(): String = joinToString(separator = "") { eachByte -> "%02x".format(eachByte) }