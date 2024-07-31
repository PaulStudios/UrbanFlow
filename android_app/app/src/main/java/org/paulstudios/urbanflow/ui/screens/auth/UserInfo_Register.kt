package org.paulstudios.urbanflow.ui.screens.auth

import android.Manifest
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.storage.FirebaseStorage
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun UserInfoRegister(
    onRegistrationComplete: () -> Unit
) {
    var firstName by remember { mutableStateOf("") }
    var lastName by remember { mutableStateOf("") }
    var gender by remember { mutableStateOf("") }
    var dob by remember { mutableStateOf("") }
    var mobileNo by remember { mutableStateOf("") }
    var drivingLicenseNo by remember { mutableStateOf("") }
    var aadharNo by remember { mutableStateOf("") }
    var emergencyVehiclePermitUri by remember { mutableStateOf<Uri?>(null) }
    var selfieUri by remember { mutableStateOf<Uri?>(null) }

    val coroutineScope = rememberCoroutineScope()

    val context = LocalContext.current
    var tempImageUri by remember { mutableStateOf<Uri?>(null) }

    val cameraLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            // Image captured successfully
            selfieUri = tempImageUri
        }
    }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            // Permission granted, launch camera
            tempImageUri?.let { cameraLauncher.launch(it) }
        } else {
            // Show a message that camera permission is required
        }
    }

    val imagePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        emergencyVehiclePermitUri = uri
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        OutlinedTextField(
            value = firstName,
            onValueChange = { firstName = it },
            label = { Text("First Name") }
        )
        OutlinedTextField(
            value = lastName,
            onValueChange = { lastName = it },
            label = { Text("Last Name") }
        )
        OutlinedTextField(
            value = gender,
            onValueChange = { gender = it },
            label = { Text("Gender") }
        )
        OutlinedTextField(
            value = dob,
            onValueChange = { dob = it },
            label = { Text("Date of Birth") }
        )
        OutlinedTextField(
            value = mobileNo,
            onValueChange = { mobileNo = it },
            label = { Text("Mobile Number") }
        )
        OutlinedTextField(
            value = drivingLicenseNo,
            onValueChange = { drivingLicenseNo = it },
            label = { Text("Driving License Number") }
        )
        OutlinedTextField(
            value = aadharNo,
            onValueChange = { aadharNo = it },
            label = { Text("Aadhar Number") }
        )

        Button(onClick = { imagePicker.launch("image/*") }) {
            Text("Upload Emergency Vehicle Permit")
        }

        Button(onClick = {
            val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(
                Date()
            )
            val storageDir: File? = context.getExternalFilesDir(null)
            val imageFile = File.createTempFile(
                "JPEG_${timeStamp}_",
                ".jpg",
                storageDir
            )
            tempImageUri = FileProvider.getUriForFile(
                context,
                "${context.packageName}.fileprovider",
                imageFile
            )
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }) {
            Text("Take Selfie")
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = {
            coroutineScope.launch {
            submitUserInfo(
                firstName, lastName, gender, dob, mobileNo, drivingLicenseNo, aadharNo,
                emergencyVehiclePermitUri, selfieUri, onRegistrationComplete
            )
        }
             }) {
            Text("Submit")
        }
    }
}

private suspend fun submitUserInfo(
    firstName: String, lastName: String, gender: String, dob: String,
    mobileNo: String, drivingLicenseNo: String, aadharNo: String,
    emergencyVehiclePermitUri: Uri?, selfieUri: Uri?,
    onRegistrationComplete: () -> Unit
) {
    val userId = FirebaseAuth.getInstance().currentUser?.uid ?: return

    // Upload images to Firebase Storage
    val emergencyPermitUrl = uploadImage(emergencyVehiclePermitUri, "emergency_permits/$userId")
    val selfieUrl = uploadImage(selfieUri, "selfies/$userId")

    // Verify that the selfie shows a face (you'll need to implement this)
    if (!verifySelfieHasFace(selfieUri)) {
        // Show error message
        return
    }

    // Create user data object
    val userData = hashMapOf(
        "firstName" to firstName,
        "lastName" to lastName,
        "gender" to gender,
        "dob" to dob,
        "mobileNo" to mobileNo,
        "drivingLicenseNo" to drivingLicenseNo,
        "aadharNo" to aadharNo,
        "emergencyVehiclePermitUrl" to emergencyPermitUrl,
        "selfieUrl" to selfieUrl
    )

    // Send data to your API server
    sendDataToApiServer(userId, userData)

    // Create document in Firestore
    FirebaseFirestore.getInstance().collection("users").document(userId)
        .set(mapOf("registrationComplete" to true)).await()

    onRegistrationComplete()
}

private suspend fun uploadImage(uri: Uri?, path: String): String? {
    if (uri == null) return null
    val ref = FirebaseStorage.getInstance().reference.child(path)
    return ref.putFile(uri).await().storage.downloadUrl.await().toString()
}

private fun verifySelfieHasFace(selfieUri: Uri?): Boolean {
    // Implement face detection logic here
    // You might want to use ML Kit or another face detection library
    return true // Placeholder
}

private suspend fun sendDataToApiServer(userId: String, userData: Map<String, Any?>) {
    // Implement your API call here
    // You might want to use Retrofit or another networking library
}