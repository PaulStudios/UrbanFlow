import sys

import requests
from datetime import datetime, timedelta, timezone
import uuid

base_url = "https://urbanflow.onrender.com/"
base_url = "http://localhost:8000/"

# Register a new user
register_data = {
    "username": "tester45",
    "password": "tester123"
}
register_response = requests.post(f"{base_url}/auth/register", json=register_data)
if register_response.status_code == 200:
    user_id = register_response.json().get("id")  # Assuming response contains user ID
    print("User registered successfully")
else:
    print("Failed to register user:", register_response.text)

# Log in the user to get credentials for the API key
login_data = {
    "username": "tester45",
    "password": "tester123"
}
login_response = requests.post(f"{base_url}/auth/token", data=login_data)
if login_response.status_code == 200:
    access_token = login_response.json().get("access_token")
    user_id = login_response.json().get("id")
    print("Access token obtained:", access_token)
    print("User ID:", user_id)
else:
    print("Failed to obtain access token:", login_response.json())

# Generate an API key for the user
headers = {
    "Authorization": f"Bearer {access_token}"
}

# Ensure you have a valid UUID for user_id
try:
    user_uuid = uuid.UUID(user_id)
except ValueError:
    print("Invalid UUID for user_id")
    user_uuid = None

if user_uuid:
    api_key_data = {
        "user_id": str(user_uuid),  # Valid UUID
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()  # Example: 30 days from now
    }
    api_key_response = requests.post(f"{base_url}/auth/api-key", json=api_key_data, headers=headers)
    if api_key_response.status_code == 200:
        api_key = api_key_response.json().get("api_key")
        print("API key generated:", api_key)
    else:
        print("Failed to generate API key:", api_key_response.json())
        sys.exit(1)


# Function to create a traffic signal
def create_traffic_signal(latitude, longitude, status="None"):
    create_signal_data = {
        "latitude": latitude,
        "longitude": longitude,
        "status": status
    }

    # Include the API key in the headers
    headers = {
        "X-API-Key": api_key
    }

    # Send a POST request to create the traffic signal
    signal_response = requests.post(f"{base_url}/signals/create", json=create_signal_data, headers=headers)
    if signal_response.status_code == 200:
        print("Traffic signal created:", signal_response.json())
    else:
        print("Failed to create traffic signal:", signal_response.json())

traffic_signals = [
    {"latitude": 22.69255749142857, "longitude": 88.46540731714285},
    {"latitude": 22.6913445, "longitude": 88.4651195},
    {"latitude": 22.68997736, "longitude": 88.46455227999999},
    {"latitude": 22.6859112, "longitude": 88.46335630000002},
    {"latitude": 22.68322355, "longitude": 88.462559275},
    {"latitude": 22.681934866666666, "longitude": 88.46182613333333},
    {"latitude": 22.6814273, "longitude": 88.46124904999999},
    {"latitude": 22.6808034, "longitude": 88.4602457},
    {"latitude": 22.680594, "longitude": 88.4610953},
    {"latitude": 22.679860385714285, "longitude": 88.45181904285714},
    {"latitude": 22.67969485, "longitude": 88.45036995000001},
    {"latitude": 22.678675100000003, "longitude": 88.44793014999999},
    {"latitude": 22.678232400000002, "longitude": 88.4474601},
    {"latitude": 22.676650199999997, "longitude": 88.44602760000001},
    {"latitude": 22.674807100000002, "longitude": 88.44546310000001},
    {"latitude": 22.673919866666665, "longitude": 88.4454377},
    {"latitude": 22.67023335, "longitude": 88.4449757},
    {"latitude": 22.66846162222222, "longitude": 88.44490293333332},
    {"latitude": 22.6605186, "longitude": 88.4418815},
    {"latitude": 22.65946066666667, "longitude": 88.4415466},
    {"latitude": 22.65808165, "longitude": 88.44085835},
    {"latitude": 22.6586344, "longitude": 88.4411879},
    {"latitude": 22.656322145454542, "longitude": 88.44027140909091},
    {"latitude": 22.652212590909087, "longitude": 88.437878},
    {"latitude": 22.6528745, "longitude": 88.4379234},
    {"latitude": 22.65116605, "longitude": 88.43698075},
    {"latitude": 22.6502368, "longitude": 88.43672989999999},
    {"latitude": 22.64832068275862, "longitude": 88.43605606896554},
    {"latitude": 22.6494502, "longitude": 88.4363874},
    {"latitude": 22.6410595173913, "longitude": 88.43246809565214},
    {"latitude": 22.633941322222224, "longitude": 88.4347337},
    {"latitude": 22.63238165, "longitude": 88.4348097},
    {"latitude": 22.626210712500004, "longitude": 88.43323863750001},
    {"latitude": 22.623741199999998, "longitude": 88.43277284999999},
    {"latitude": 22.6206545, "longitude": 88.4329727},
    {"latitude": 22.61731905, "longitude": 88.43198340000001},
    {"latitude": 22.613897, "longitude": 88.42991266},
    {"latitude": 22.612723250000002, "longitude": 88.42919764999999},
    {"latitude": 22.6119213, "longitude": 88.42919115000001},
    {"latitude": 22.607552933333334, "longitude": 88.42678604999999},
    {"latitude": 22.604747085714287, "longitude": 88.4255922},
    {"latitude": 22.6056893, "longitude": 88.4259215},
    {"latitude": 22.603781914285715, "longitude": 88.42438324999999},
    {"latitude": 22.602994244444446, "longitude": 88.42329808888888},
    {"latitude": 22.603125471428573, "longitude": 88.41854469999998},
    {"latitude": 22.603228924999996, "longitude": 88.41236492499999},
    {"latitude": 22.60236603333333, "longitude": 88.41102836666666},
    {"latitude": 22.595271750000002, "longitude": 88.39947275},
    {"latitude": 22.5926398, "longitude": 88.39511979},
    {"latitude": 22.5922195, "longitude": 88.3945846},
    {"latitude": 22.59125658749999, "longitude": 88.39337319},
    {"latitude": 22.5927552, "longitude": 88.39019625},
    {"latitude": 22.5931545375, "longitude": 88.3889312375},
    {"latitude": 22.59159607142857, "longitude": 88.38808180000001},
    {"latitude": 22.5912004, "longitude": 88.3875359},
    {"latitude": 22.591476795918364, "longitude": 88.38519223265304},
    # Additional fake entries
    {"latitude": 22.5935678, "longitude": 88.3867542},
    {"latitude": 22.5948901, "longitude": 88.3879023},
    {"latitude": 22.5962345, "longitude": 88.3890456},
    {"latitude": 22.5975789, "longitude": 88.3901234},
    {"latitude": 22.5989012, "longitude": 88.3912345},
    {"latitude": 22.6002345, "longitude": 88.3923456},
    {"latitude": 22.6015678, "longitude": 88.3934567},
    {"latitude": 22.6028901, "longitude": 88.3945678},
    {"latitude": 22.6042345, "longitude": 88.3956789},
    {"latitude": 22.6055678, "longitude": 88.3967890}
]

# Iterate over the sample data and create each traffic signal
for signal in traffic_signals:
    create_traffic_signal(signal["latitude"], signal["longitude"])

headers = {
    "X-API-Key": api_key,
}

response = requests.get(f"{base_url}/signals", headers=headers)
if response.status_code == 200:
    print(response.json())
    for signal in response.json():
        print(f"{signal['signal_id']}, {signal['status']}")
else:
    print("Failed to get traffic signals:", response.json())
