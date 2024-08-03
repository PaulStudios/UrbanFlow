package org.paulstudios.urbanflow.utils

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Log
import org.spongycastle.jce.provider.BouncyCastleProvider
import java.security.KeyFactory
import java.security.KeyPair
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.spec.ECGenParameterSpec
import javax.crypto.Cipher
import javax.crypto.KeyAgreement
import javax.crypto.spec.IvParameterSpec
import javax.crypto.spec.SecretKeySpec
import java.security.PrivateKey
import java.security.PublicKey
import java.security.SecureRandom
import java.security.Security
import java.security.spec.PKCS8EncodedKeySpec

object CryptoUtils {
    private const val KEY_ALIAS = "MyAppKeyPair"
    private const val KEYSTORE_PROVIDER = "AndroidKeyStore"
    private const val CURVE = "secp256r1"
    private const val TAG = "CryptoUtils"

    init {
        Security.addProvider(BouncyCastleProvider())
        Log.d(TAG, "BouncyCastle provider added.")
    }

    fun getOrGenerateKeyPair(): KeyPair {
        val keyStore = KeyStore.getInstance(KEYSTORE_PROVIDER)
        keyStore.load(null)

        // Check if the key already exists
        if (keyStore.containsAlias(KEY_ALIAS)) {
            val entry = keyStore.getEntry(KEY_ALIAS, null) as? KeyStore.PrivateKeyEntry
            if (entry != null) {
                Log.d(TAG, "KeyPair retrieved from KeyStore.")
                return KeyPair(entry.certificate.publicKey, entry.privateKey)
            }
        }

        // If the key doesn't exist or couldn't be retrieved, generate a new one
        Log.d(TAG, "Generating new KeyPair.")
        return generateKeyPair()
    }

    private fun generateKeyPair(): KeyPair {
        val keyPairGenerator = KeyPairGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_EC,
            KEYSTORE_PROVIDER
        )
        val parameterSpec = KeyGenParameterSpec.Builder(
            KEY_ALIAS,
            KeyProperties.PURPOSE_AGREE_KEY
        ).apply {
            setAlgorithmParameterSpec(ECGenParameterSpec(CURVE))
        }.build()

        keyPairGenerator.initialize(parameterSpec)
        val keyPair = keyPairGenerator.generateKeyPair()
        Log.d(TAG, "New KeyPair generated.")
        return keyPair
    }

    fun getKeyPair(): KeyPair {
        val keyStore = KeyStore.getInstance(KEYSTORE_PROVIDER)
        keyStore.load(null)
        val entry = keyStore.getEntry(KEY_ALIAS, null) as KeyStore.PrivateKeyEntry
        Log.d(TAG, "KeyPair retrieved from KeyStore.")
        return KeyPair(entry.certificate.publicKey, entry.privateKey)
    }

    fun convertPrivateKeyToSpongyCastle(privateKey: PrivateKey): PrivateKey {
        val encoded = privateKey.encoded ?: throw IllegalArgumentException("Private key encoding is null")
        if (encoded.isEmpty()) throw IllegalArgumentException("Private key encoding is empty")

        try {
            val keyFactory = KeyFactory.getInstance("ECDH", "SC")
            return keyFactory.generatePrivate(PKCS8EncodedKeySpec(encoded))
        } catch (e: Exception) {
            throw IllegalArgumentException("Failed to convert private key: ${e.message}", e)
        }
    }

    fun generateSharedSecret(privateKey: PrivateKey, publicKey: PublicKey): ByteArray {
        val keyAgreement = KeyAgreement.getInstance("ECDH")
        keyAgreement.init(privateKey)
        keyAgreement.doPhase(publicKey, true)
        return keyAgreement.generateSecret()
    }

    fun encrypt(data: ByteArray, key: ByteArray): Pair<ByteArray, ByteArray> {
        val iv = generateIV()
        val cipher = Cipher.getInstance("AES/CBC/PKCS5Padding")
        val secretKeySpec = SecretKeySpec(key.copyOf(32), "AES")
        cipher.init(Cipher.ENCRYPT_MODE, secretKeySpec, IvParameterSpec(iv))
        val encryptedData = cipher.doFinal(data)
        return Pair(encryptedData, iv)
    }

    private fun generateIV(): ByteArray {
        val iv = ByteArray(16)
        SecureRandom().nextBytes(iv)
        return iv
    }

    fun decrypt(encryptedData: ByteArray, key: ByteArray, iv: ByteArray): ByteArray {
        try {
            val cipher = Cipher.getInstance("AES/CBC/PKCS7Padding")
            val secretKeySpec = SecretKeySpec(key, "AES")
            val ivParameterSpec = IvParameterSpec(iv)
            cipher.init(Cipher.DECRYPT_MODE, secretKeySpec, ivParameterSpec)
            val decryptedData = cipher.doFinal(encryptedData)
            Log.d(TAG, "Data decrypted.")
            return decryptedData
        } catch (e: Exception) {
            Log.e(TAG, "Error decrypting data.", e)
            throw e
        }
    }
}