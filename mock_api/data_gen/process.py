import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

from mock_api.data_gen.gen_license import gen_license
from mock_api.data_gen.gen_vehicle import gen_vehicle
from mock_api.data_gen.uploader import upload_license, upload_vehicle

load_dotenv("../../.env")

base_url = os.getenv("VEHICLE_API")
api_key = "<KEY>"

# Log in the user to get credentials for the API key
login_data = {
    "username": "new_user",
    "password": "secure_password"
}
login_response = requests.post(f"{base_url}/auth/token", data=login_data)
if login_response.status_code == 200:
    access_token = login_response.json().get("access_token")
    user_id = login_response.json().get("id")
    print("Access token obtained:", access_token)
    print("User ID:", user_id)
else:
    print("Failed to login:", login_response.text)
    print("Failed to obtain access token:", login_response.json())
    sys.exit(1)

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
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=60)).isoformat()  # Example: 30 days from now
    }
    api_key_response = requests.post(f"{base_url}/auth/api-key", json=api_key_data, headers=headers)
    if api_key_response.status_code == 200:
        api_key = api_key_response.json().get("api_key")
        print("API key generated:", api_key)
    else:
        print("Failed to generate API key:", api_key_response.json())
        sys.exit(1)

headers = {
        "X-API-Key": api_key
}

licenses = [gen_license() for _ in range(50)]
vehicles = [gen_vehicle() for _ in range(50)]

with ThreadPoolExecutor() as executor:
    # Create tasks for both uploads
    license_futures = [executor.submit(upload_license, license, headers) for license in licenses]
    vehicle_futures = [executor.submit(upload_vehicle, vehicle, headers) for vehicle in vehicles]

    # Wait for all uploads to complete
    for future in license_futures + vehicle_futures:
        try:
            future.result()  # This will raise exceptions if any occurred
        except Exception as e:
            print(f"An error occurred: {e}")
