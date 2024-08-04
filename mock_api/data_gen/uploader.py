import os
import logging

import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

load_dotenv("../../.env")

base_url = os.getenv("VEHICLE_API")
api_key = "<KEY>"


def upload_license(license: dict, headers: dict):
    try:
        r = requests.post(f'{base_url}/license', headers=headers, json=license)
        r.raise_for_status()
        logging.info(f"Uploaded license: {license}")

    except Exception as e:
        logging.error(f"Error uploading license: {e}")


def upload_vehicle(vehicle: dict, headers: dict):
    try:
        r = requests.post(f'{base_url}/vehicle', headers=headers, json=vehicle)
        r.raise_for_status()
        logging.info(f"Uploaded vehicle: {vehicle}")
    except Exception as e:
        logging.error(f"Error uploading vehicle: {e}")
