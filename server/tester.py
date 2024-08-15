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

traffic_signals = []

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
