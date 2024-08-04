import sys
import logging
import uuid
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)

def get_apikey(base_url):
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
        return None

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

    api_key = "<KEY>"
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
            return None

        return api_key
