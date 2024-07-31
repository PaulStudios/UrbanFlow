package org.paulstudios.urbanflow.ui.screens.auth

import android.Manifest
import android.content.Context
import android.net.Uri
import android.util.Log
import android.view.ViewGroup
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
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
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
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
import org.paulstudios.urbanflow.utils.logSelfieAttempt
import org.paulstudios.urbanflow.utils.logSelfieFaceDetectionResult
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.Executor
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

private const val TAG = "UserInfoRegister"


@Composable
fun UserInfoRegister(navController: NavHostController) {
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

    var showCamera by remember { mutableStateOf(false) }

    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var faceDetected by remember { mutableStateOf(false) }
    var faceDetectionState by remember { mutableStateOf<FaceDetectionState>(FaceDetectionState.Initial) }
    var errorMessage by remember { mutableStateOf<String?>(null) }

    var selfieRetryCount by remember { mutableIntStateOf(0) }
    val maxRetries = 3

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

        Button(onClick = {
            coroutineScope.launch {
            submitUserInfo(
                firstName, lastName, gender, dob, mobileNo, drivingLicenseNo, aadharNo,
                emergencyVehiclePermitUri, selfieUri)
        }
             }) {
            Text("Submit")
        }
    }
}

fun showHelpOrContactSupport() {
    TODO("Not yet implemented")
}

private suspend fun submitUserInfo(
    firstName: String, lastName: String, gender: String, dob: String,
    mobileNo: String, drivingLicenseNo: String, aadharNo: String,
    emergencyVehiclePermitUri: Uri?, selfieUri: Uri?
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

@Composable
fun CameraCapture(
    onImageCaptured: (Uri) -> Unit,
    onError: (ImageCaptureException) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var lensFacing by remember { mutableStateOf(CameraSelector.LENS_FACING_FRONT) }
    val imageCapture: ImageCapture = remember {
        ImageCapture.Builder().build()
    }
    val cameraSelector = CameraSelector.Builder()
        .requireLensFacing(lensFacing)
        .build()

    LaunchedEffect(lensFacing) {
        val cameraProvider = context.getCameraProvider()
        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(
            lifecycleOwner,
            cameraSelector,
            imageCapture
        )
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { ctx ->
                PreviewView(ctx).apply {
                    this.scaleType = PreviewView.ScaleType.FILL_CENTER
                    layoutParams = ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                }
            },
            modifier = Modifier.fillMaxSize(),
        )

        Button(
            onClick = {
                takePhoto(
                    imageCapture = imageCapture,
                    outputDirectory = context.cacheDir,
                    executor = ContextCompat.getMainExecutor(context),
                    onImageCaptured = onImageCaptured,
                    onError = onError
                )
            },
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 16.dp)
        ) {
            Text("Take Photo")
        }
    }
}

private fun takePhoto(
    imageCapture: ImageCapture,
    outputDirectory: File,
    executor: Executor,
    onImageCaptured: (Uri) -> Unit,
    onError: (ImageCaptureException) -> Unit
) {
    val photoFile = File(
        outputDirectory,
        SimpleDateFormat("yyyy-MM-dd-HH-mm-ss-SSS", Locale.US)
            .format(System.currentTimeMillis()) + ".jpg"
    )

    val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

    imageCapture.takePicture(
        outputOptions,
        executor,
        object : ImageCapture.OnImageSavedCallback {
            override fun onError(exception: ImageCaptureException) {
                onError(exception)
            }

            override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                val savedUri = Uri.fromFile(photoFile)
                onImageCaptured(savedUri)
            }
        }
    )
}

suspend fun Context.getCameraProvider(): ProcessCameraProvider = suspendCoroutine { continuation ->
    ProcessCameraProvider.getInstance(this).also { cameraProvider ->
        cameraProvider.addListener({
            continuation.resume(cameraProvider.get())
        }, ContextCompat.getMainExecutor(this))
    }
}

sealed class FaceDetectionState {
    object Initial : FaceDetectionState()
    object Loading : FaceDetectionState()
    object Success : FaceDetectionState()
    object Failure : FaceDetectionState()
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