plugins {
    id("com.android.application")
    id("com.google.gms.google-services")
    kotlin("android")
    kotlin("plugin.serialization") version "2.0.0"
}
buildscript {
    repositories {
        google()
        mavenCentral()
    }
}

android {
    namespace = "org.paulstudios.urbanflow"
    compileSdk = 34

    defaultConfig {
        applicationId = "org.paulstudios.urbanflow"
        minSdk = 33
        targetSdk = 34
        versionCode = 1
        versionName = "0.3.1"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    buildFeatures {
        compose = true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.1"
    }
    packaging {
        resources {
            excludes += listOf(
                "META-INF/LICENSE-LGPL-3.txt",
                "META-INF/LICENSE-LGPL-2.1.txt",
                "META-INF/LICENSE-W3C-TEST",
                "META-INF/DEPENDENCIES",
                "META-INF/LICENSE",
                "META-INF/LICENSE.txt",
                "META-INF/license.txt",
                "META-INF/NOTICE",
                "META-INF/NOTICE.txt",
                "META-INF/notice.txt",
                "META-INF/ASL2.0",
                "/META-INF/{AL2.0,LGPL2.1}"
            )
        }
    }
}

dependencies {

    // Compose BOM
    implementation(platform(libs.androidx.compose.bom.v20230100))

    // Core Compose dependencies
    implementation(libs.ui)
    implementation(libs.ui.tooling.preview)
    implementation(libs.material3)
    implementation(libs.androidx.room.runtime)
    implementation(libs.androidx.runtime.livedata)
    implementation(libs.junit)
    implementation(libs.androidx.monitor)
    implementation(libs.androidx.junit.ktx)
    implementation(libs.androidx.ui.test.junit4.android)
    implementation(libs.androidx.lifecycle.viewmodel.compose)

    debugImplementation(libs.ui.tooling)
    debugImplementation(libs.ui.test.manifest)

    // Accompanist navigation animation
    implementation(libs.accompanist.navigation.animation.v0340)

    // Other dependencies
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.navigation.fragment.ktx)
    implementation(libs.androidx.navigation.ui.ktx)
    implementation(libs.androidx.navigation.compose)
    implementation(libs.androidx.navigation.dynamic.features.fragment)
    implementation(libs.retrofit)
    implementation(libs.converter.gson)
    implementation(libs.androidx.work.runtime.ktx)
    implementation(libs.play.services.location)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.play.services.auth)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.coil.compose)

    // Firebase implementation
    implementation(platform(libs.firebase.bom))
    implementation(libs.firebase.auth.ktx)
    implementation(libs.firebase.analytics.ktx)
    implementation(libs.firebase.firestore.ktx)
    implementation(libs.firebase.storage.ktx)

    // Markwon dependencies
    implementation(libs.core)
    implementation(libs.ext.strikethrough)
    implementation(libs.ext.tables)

    // Testing dependencies
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit.v115)
    androidTestImplementation(libs.androidx.espresso.core.v351)
    androidTestImplementation(platform(libs.androidx.compose.bom.v20230100))
    androidTestImplementation(libs.ui.test.junit4)
    debugImplementation(libs.androidx.compose.ui.ui.tooling)
    debugImplementation(libs.androidx.compose.ui.ui.test.manifest)

    // Mockito for mocking in tests
    testImplementation(libs.mockito.core)
    androidTestImplementation(libs.mockito.android)

    // Navigation Testing
    implementation(libs.androidx.navigation.fragment.ktx.v260)
    implementation(libs.androidx.navigation.ui.ktx.v260)
    androidTestImplementation(libs.androidx.navigation.testing.v260)

    // CameraX library
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)

    // Verification library
    implementation(libs.face.detection)

    implementation(libs.bctls.jdk15on)


}