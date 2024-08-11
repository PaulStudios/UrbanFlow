import os
from datetime import datetime
import base64
import logging

import requests
from Crypto.Util.Padding import unpad, pad
from dotenv import load_dotenv
from fastapi import APIRouter, Response, Depends, HTTPException
from firebase_admin import auth as firebase_auth
from sqlalchemy.ext.asyncio import AsyncSession
from Crypto.Cipher import AES

from server.auth import auth
from server.database import get_db
from server.schemas import UserCreate, VerificationResponse, EncryptedDataRequest
from server.utils.api_access import get_apikey
from server.utils.encyption_key import load_shared_key
from server.utils.firebase.connection import firestore_db, firebase_bucket
from server.utils.serializers import deserialize_user, serialize_verify_response
from server.utils.utilities import create_directory

router = APIRouter(
    prefix="/api/verify",
    tags=["Android API"],
)

load_dotenv()
api_url = os.getenv("VEHICLE_API")

logger = logging.getLogger(__name__)

@router.get("/")
async def root(response: Response):
    return Response(status_code=200, content="Verification API for Android App")


@router.post("/user")
async def verify_user(
        request: EncryptedDataRequest,
        db: AsyncSession = Depends(get_db),
        # api_key: auth.models.APIKey = Depends(auth.get_api_key)
):
    logger.info(f"Received encrypted data: {request.encrypted_data}")
    try:
        # Load the shared key for the specific client
        shared_key = await load_shared_key(db, request.client_id)
        if not shared_key:
            raise HTTPException(status_code=400, detail="Client not registered")

        encrypted_data = base64.b64decode(request.encrypted_data)
        iv = base64.b64decode(request.iv)

        logger.info(f"Encrypted data length: {len(encrypted_data)}")

        cipher = AES.new(shared_key[:32], AES.MODE_CBC, iv)
        try:
            decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)
            user = deserialize_user(decrypted_data.decode())
            logger.info(f"Decryption successful for client {request.client_id}. Received user data")
            logger.info(f"Verifying user: {user.id}")

            user_id = user.id
            firebase_user = firebase_auth.get_user(user_id)

            name = user.name
            dob = user.date_of_birth
            mobile_number = user.mobile_number
            license_number = user.license_number
            vehicle_number = user.vehicle_number
            aadhar_number = user.aadhar_number
            permit_uri = user.permit_uri
            selfie_uri = user.selfie_uri

            user_doc = firestore_db.collection("users").document(user_id)

            response = VerificationResponse(
                status=False,
                checked_at=datetime.now(),
            )
            headers = {
                "X-API-Key": get_apikey(api_url)
            }
            license_response = requests.get(f"{api_url}/license/{license_number}", headers=headers).json()
            vehicle_response = requests.get(f"{api_url}/vehicle/{vehicle_number}", headers=headers).json()
            logger.info(f"Vehicle response: {vehicle_response}")
            logger.info(f"License response: {license_response}")
        except ValueError as ve:
            logger.error(f"Decryption failed for client {request.client_id}: {str(ve)}")
            raise HTTPException(status_code=400, detail="Decryption failed")
    except Exception as e:
        logger.error(f"Error in send_data for client {request.client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

    data = serialize_verify_response(response)
    logger.info(f"Verification response: {data}")
    cipher2 = AES.new(shared_key[:32], AES.MODE_CBC)
    encrypted_data = cipher2.encrypt(pad(data.encode(), AES.block_size))
    encrypted_data = base64.b64encode(encrypted_data).decode()
    logger.info(f"Response Encrypted data: {encrypted_data}")
    logger.info(f"Response Encrypted data length: {len(encrypted_data)}")

    return {
        "encrypted_data": encrypted_data,
        "iv": base64.b64encode(cipher2.iv).decode()
    }
