package org.paulstudios.urbanflow.data.models

sealed class Screen(val route: String) {
    object Login : Screen("login")
    object Register : Screen("register")
    object InfoScreen : Screen("info_screen")
    object MapScreen : Screen("map_screen")
    object RegisterDetails : Screen("register_details")
}
