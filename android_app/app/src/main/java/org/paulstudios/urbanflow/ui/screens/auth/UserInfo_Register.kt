package org.paulstudios.urbanflow.ui.screens.auth

import android.Manifest
import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.DateRange
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DatePicker
import androidx.compose.material3.DatePickerDefaults
import androidx.compose.material3.DatePickerDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberDatePickerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.OffsetMapping
import androidx.compose.ui.text.input.TransformedText
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import coil.compose.rememberImagePainter
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.storage.FirebaseStorage
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withContext
import org.paulstudios.datasurvey.network.UserBase
import org.paulstudios.urbanflow.network.SecureApiClient
import org.paulstudios.urbanflow.utils.CameraCapture
import org.paulstudios.urbanflow.utils.FaceDetectionState
import org.paulstudios.urbanflow.utils.logSelfieAttempt
import org.paulstudios.urbanflow.utils.logSelfieFaceDetectionResult
import java.time.LocalDate
import java.time.format.DateTimeFormatter

private const val TAG = "UserInfoRegister"


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun UserInfoRegister(navController: NavHostController) {
    var firstName by remember { mutableStateOf("") }
    var lastName by remember { mutableStateOf("") }
    var gender by remember { mutableStateOf("") }
    var dob by remember { mutableStateOf("") }
    var drivingLicenseNo by remember { mutableStateOf("") }
    var vehicleNo by remember { mutableStateOf("") }
    var aadharNo by remember { mutableStateOf("") }
    var mobileNo by remember { mutableStateOf("") }
    var emergencyVehiclePermitUri by remember { mutableStateOf<Uri?>(null) }
    var selfieUri by remember { mutableStateOf<Uri?>(null) }

    var showDatePicker by remember { mutableStateOf(false) }

    val coroutineScope = rememberCoroutineScope()

    var showCamera by remember { mutableStateOf(false) }

    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var faceDetected by remember { mutableStateOf(false) }
    var faceDetectionState by remember { mutableStateOf<FaceDetectionState>(FaceDetectionState.Initial) }
    var errorMessage by remember { mutableStateOf<String?>(null) }

    var selfieRetryCount by remember { mutableIntStateOf(0) }
    val maxRetries = 3

    val secureApiClient: SecureApiClient = SecureApiClient("http://10.0.2.2:8000/", context = context)

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            showCamera = true
        } else {
            // Show a message that camera permission is required
        }
    }

    val imagePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        emergencyVehiclePermitUri = uri
    }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        item {
            OutlinedTextField(
                value = firstName,
                onValueChange = { if (it.length <= 50) firstName = it },
                label = { Text("First Name") },
                isError = firstName.isBlank(),
                supportingText = { if (firstName.isBlank()) Text("Required") },
                modifier = Modifier.fillMaxWidth()
            )

            OutlinedTextField(
                value = lastName,
                onValueChange = { if (it.length <= 50) lastName = it },
                label = { Text("Last Name") },
                isError = lastName.isBlank(),
                supportingText = { if (lastName.isBlank()) Text("Required") },
                modifier = Modifier.fillMaxWidth()
            )

            OutlinedTextField(
                value = gender,
                onValueChange = { gender = it },
                label = { Text("Gender") },
                isError = gender !in listOf("Male", "Female"),
                supportingText = { Text("Enter Male or Female") },
                modifier = Modifier.fillMaxWidth()
            )

            OutlinedTextField(
                value = dob,
                onValueChange = { },
                label = { Text("Date of Birth") },
                readOnly = true,
                trailingIcon = {
                    IconButton(onClick = { showDatePicker = true }) {
                        Icon(Icons.Default.DateRange, "Select date")
                    }
                },
                modifier = Modifier.fillMaxWidth()
            )

            OutlinedTextField(
                value = mobileNo,
                onValueChange = { input ->
                    val digitsOnly = input.filter { it.isDigit() }
                    if (digitsOnly.length <= 10) mobileNo = digitsOnly
                },
                label = { Text("Mobile Number") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Phone),
                isError = mobileNo.length != 10,
                supportingText = { Text("Enter 10 digit number") },
                modifier = Modifier.fillMaxWidth(),
                visualTransformation = { text ->
                    val formattedText = if (text.text.length > 5) {
                        text.text.take(5) + " " + text.text.drop(5)
                    } else {
                        text.text
                    }
                    TransformedText(AnnotatedString(formattedText), OffsetMapping.Identity)
                }
            )

            OutlinedTextField(
                value = drivingLicenseNo,
                onValueChange = { input ->
                    val upperCaseInput = input.uppercase()
                    val formatted = buildString {
                        upperCaseInput.forEachIndexed { index, char ->
                            if (char.isLetterOrDigit()) {
                                append(char)
                                if (length in setOf(2, 5, 10) && length < 18) append('-')
                            }
                        }
                    }.take(18)
                    drivingLicenseNo = formatted
                },
                label = { Text("Driving License Number") },
                isError = !drivingLicenseNo.matches(Regex("^[A-Z]{2}-\\d{2}-\\d{4}-\\d{7}$")),
                supportingText = { Text("Format: AA-12-3456-1234567") },
                modifier = Modifier.fillMaxWidth()
            )

            OutlinedTextField(
                value = vehicleNo,
                onValueChange = { input ->
                    val upperCaseInput = input.uppercase()
                    val formatted = buildString {
                        upperCaseInput.forEachIndexed { index, char ->
                            if (char.isLetterOrDigit()) {
                                append(char)
                                if (length in setOf(2, 5) || (length == 7 && upperCaseInput.count { it.isLetter() } > 3)) {
                                    if (length < 13) append('-')
                                }
                            }
                        }
                    }.take(13)
                    vehicleNo = formatted
                },
                label = { Text("Vehicle Number") },
                isError = !vehicleNo.matches(Regex("^[A-Z]{2}-\\d{2}-[A-Z]{1,2}-\\d{1,4}$")),
                supportingText = { Text("Format: AA-12-A-1234 or AA-12-AA-1234") },
                modifier = Modifier.fillMaxWidth()
            )

            OutlinedTextField(
                value = aadharNo,
                onValueChange = { input ->
                    val digitsOnly = input.filter { it.isDigit() }
                    if (digitsOnly.length <= 12) aadharNo = digitsOnly
                },
                label = { Text("Aadhar Number") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                isError = !isValidAadhaar(aadharNo),
                supportingText = { Text("Enter 12 digit Aadhar number") },
                modifier = Modifier.fillMaxWidth(),
                visualTransformation = { text ->
                    val formattedText = text.text.chunked(4).joinToString(" ")
                    TransformedText(AnnotatedString(formattedText), OffsetMapping.Identity)
                }
            )

            Button(
                onClick = { imagePicker.launch("image/*") },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Upload Emergency Vehicle Permit")
            }

            if (selfieUri == null) {
                SelfieTips()
            }

            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Button(
                        onClick = {
                            if (selfieRetryCount < maxRetries) {
                                errorMessage = null
                                logSelfieAttempt()
                                selfieRetryCount++
                                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                            } else {
                                errorMessage = "Maximum retries reached. Please try again later or contact support."
                            }
                        },
                        enabled = faceDetectionState != FaceDetectionState.Loading && selfieRetryCount < maxRetries
                    ) {
                        Text(if (selfieUri == null) "Take Selfie" else "Retake Selfie")
                    }

                    Spacer(modifier = Modifier.width(8.dp))

                    when (faceDetectionState) {
                        FaceDetectionState.Loading -> {
                            CircularProgressIndicator(
                                modifier = Modifier.size(24.dp),
                                color = MaterialTheme.colorScheme.secondary
                            )
                        }
                        FaceDetectionState.Success -> {
                            Icon(
                                imageVector = Icons.Default.CheckCircle,
                                contentDescription = "Face detected",
                                tint = Color.Green
                            )
                        }
                        FaceDetectionState.Failure -> {
                            Icon(
                                imageVector = Icons.Default.Close,
                                contentDescription = "Face not detected",
                                tint = Color.Red
                            )
                        }
                        FaceDetectionState.Initial -> {
                            // No icon shown
                        }
                    }
                }

                errorMessage?.let { error ->
                    Text(
                        text = error,
                        color = Color.Red,
                        modifier = Modifier.padding(top = 8.dp)
                    )
                }

                if (faceDetectionState == FaceDetectionState.Failure) {
                    Button(
                        onClick = {
                            errorMessage = null
                            logSelfieAttempt()
                            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                        },
                        modifier = Modifier.padding(top = 8.dp)
                    ) {
                        Icon(Icons.Default.Refresh, contentDescription = "Retry")
                        Spacer(Modifier.width(4.dp))
                        Text("Retry Selfie")
                    }
                }

                selfieUri?.let { uri ->
                    Image(
                        painter = rememberImagePainter(uri),
                        contentDescription = "Selfie",
                        modifier = Modifier
                            .size(200.dp)
                            .padding(top = 16.dp)
                    )
                }
            }

            if (selfieRetryCount >= maxRetries) {
                Button(
                    onClick = {
                        // Implement a function to show help or contact support
                        showHelpOrContactSupport()
                    },
                    modifier = Modifier.padding(top = 8.dp)
                ) {
                    Text("Need Help?")
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            Button(
                onClick = {
                    coroutineScope.launch {
                        submitUserInfo(
                            firstName,
                            lastName,
                            gender,
                            dob,
                            mobileNo,
                            drivingLicenseNo,
                            vehicleNo,
                            aadharNo,
                            emergencyVehiclePermitUri,
                            selfieUri,
                            secureApiClient
                        )
                    }
                },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Submit")
            }
        }
    }

    if (showCamera) {
        CameraCapture(
            onImageCaptured = { uri ->
                selfieUri = uri
                showCamera = false
                faceDetectionState = FaceDetectionState.Loading
                coroutineScope.launch {
                    faceDetectionState = try {
                        if (detectFace(context, uri)) {
                            logSelfieFaceDetectionResult(true)
                            selfieRetryCount = 0
                            FaceDetectionState.Success
                        } else {
                            logSelfieFaceDetectionResult(false)
                            errorMessage = "No face detected. Please try again."
                            FaceDetectionState.Failure
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Face detection failed", e)
                        errorMessage = "An error occurred during face detection. Please try again."
                        FaceDetectionState.Failure
                    }
                }
            },
            onError = { exception ->
                Log.e(TAG, "Camera error", exception)
                showCamera = false
                errorMessage = "An error occurred while using the camera. Please try again."
            }
        )
    }

    if (showDatePicker) {
        val datePickerState = rememberDatePickerState(
            initialSelectedDateMillis = System.currentTimeMillis()
        )
        DatePickerDialog(
            onDismissRequest = { showDatePicker = false },
            confirmButton = {
                TextButton(onClick = {
                    datePickerState.selectedDateMillis?.let { millis ->
                        dob = LocalDate.ofEpochDay(millis / 86400000)
                            .format(DateTimeFormatter.ISO_DATE)
                    }
                    showDatePicker = false
                }) {
                    Text("OK")
                }
            }
        ) {
            DatePicker(
                state = datePickerState,
                dateFormatter = DatePickerDefaults.dateFormatter(),
                title = {
                    DatePickerDefaults.DatePickerTitle(
                        displayMode = datePickerState.displayMode,
                        modifier = Modifier.padding(16.dp)
                    )
                },
                headline = {
                    DatePickerDefaults.DatePickerHeadline(
                        selectedDateMillis = datePickerState.selectedDateMillis,
                        displayMode = datePickerState.displayMode,
                        dateFormatter = DatePickerDefaults.dateFormatter(),
                        modifier = Modifier.padding(16.dp)
                    )
                },
                showModeToggle = true,
                colors = DatePickerDefaults.colors()
            )
        }
    }
}

fun showHelpOrContactSupport() {
    TODO("Not yet implemented")
}

private suspend fun submitUserInfo(
    firstName: String, lastName: String, gender: String, dob: String,
    mobileNo: String, drivingLicenseNo: String, vehicleNo: String ,aadharNo: String,
    emergencyVehiclePermitUri: Uri?, selfieUri: Uri?, apiClient: SecureApiClient
) {
    val userId = FirebaseAuth.getInstance().currentUser?.uid ?: return

    // Upload images to Firebase Storage
    val emergencyPermitUrl = uploadImage(emergencyVehiclePermitUri, "emergency_permits/$userId")
    val selfieUrl = uploadImage(selfieUri, "selfies/$userId")


    // Create user data object
    val userData = UserBase(
        id = userId,
        name = "$firstName $lastName",
        date_of_birth = dob,
        mobile_number = mobileNo,
        license_number = drivingLicenseNo,
        vehicle_number = vehicleNo,
        aadhar_number = aadharNo,
        permit_uri = emergencyPermitUrl ?: "",
        selfie_uri = selfieUrl ?: ""
    )

    // Create document in Firestore
    FirebaseFirestore.getInstance().collection("users").document(userId)

    val verification = apiClient.sendUserData(userData)

    Log.d(TAG, "User data submitted: $verification")
}

private suspend fun uploadImage(uri: Uri?, path: String): String? {
    if (uri == null) return null
    val ref = FirebaseStorage.getInstance().reference.child(path)
    return ref.putFile(uri).await().storage.downloadUrl.await().toString()
}

suspend fun detectFace(context: Context, uri: Uri): Boolean = withContext(Dispatchers.Default) {
    val image = InputImage.fromFilePath(context, uri)
    val options = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
        .build()
    val detector = FaceDetection.getClient(options)

    try {
        val faces = detector.process(image).await()
        if (faces.isNotEmpty()) {
            val face = faces[0]
            val imageWidth = image.width.toFloat()
            val imageHeight = image.height.toFloat()
            val boundingBox = face.boundingBox

            // Check if face is centered
            val centerThreshold = 0.2f
            val isCentered = boundingBox.centerX().toFloat() in (imageWidth * (0.5f - centerThreshold))..(imageWidth * (0.5f + centerThreshold)) &&
                    boundingBox.centerY().toFloat() in (imageHeight * (0.5f - centerThreshold))..(imageHeight * (0.5f + centerThreshold))

            // Check if face is large enough
            val minFaceSize = 0.4f
            val isLargeEnough = boundingBox.width() > imageWidth * minFaceSize && boundingBox.height() > imageHeight * minFaceSize

            // Check if eyes are open
            val areEyesOpen = (face.leftEyeOpenProbability ?: 0f) > 0.8f && (face.rightEyeOpenProbability ?: 0f) > 0.8f

            return@withContext isCentered && isLargeEnough && areEyesOpen
        }
        false
    } catch (e: Exception) {
        Log.e(TAG, "Face detection failed", e)
        throw e
    } finally {
        detector.close()
    }
}

@Composable
fun SelfieTips() {
    Column(modifier = Modifier.padding(16.dp)) {
        Text("Tips for a good selfie:", style = MaterialTheme.typography.titleMedium)
        Spacer(modifier = Modifier.height(8.dp))
        Text("• Ensure your face is well-lit")
        Text("• Center your face in the frame")
        Text("• Keep your eyes open")
        Text("• Avoid wearing sunglasses or hats")
        Text("• Hold the camera at eye level")
    }
}

fun isValidAadhaar(aadhaar: String): Boolean {
    if (!aadhaar.matches(Regex("^[0-9]{12}$"))) return false

    val d = arrayOf(
        arrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        arrayOf(1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
        arrayOf(2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
        arrayOf(3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
        arrayOf(4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
        arrayOf(5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
        arrayOf(6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
        arrayOf(7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
        arrayOf(8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
        arrayOf(9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
    )

    val p = arrayOf(
        arrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        arrayOf(1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
        arrayOf(5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
        arrayOf(8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
        arrayOf(9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
        arrayOf(4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
        arrayOf(2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
        arrayOf(7, 0, 4, 6, 9, 1, 3, 2, 5, 8)
    )

    var c = 0
    for (i in 0..11) {
        c = d[c][p[i % 8][aadhaar[i].toString().toInt()]]
    }

    return c == 0
}