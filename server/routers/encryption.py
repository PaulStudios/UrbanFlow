import base64
import logging
from urllib.parse import unquote

from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Crypto.Protocol.KDF import HKDF
from Crypto.PublicKey import ECC
from Crypto.Util.Padding import pad, unpad
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from mock_api.auth import auth
from server.database import get_db
from server.models import ClientKey
from server.routers.auth import ADMIN_ACCESS_LIST
from server.schemas import PublicKeyRequest, EncryptedDataRequest
from server.utils.encyption_key import server_key, save_shared_key, perform_hkdf, load_shared_key, \
    generate_or_load_server_key

router = APIRouter(
    prefix="/api/encryption",
    tags=["Authentication"],
)

logger = logging.getLogger(__name__)


# Define the HKDF callable
def hkdf_function(shared_secret: bytes) -> bytes:
    return HKDF(shared_secret, 32, b"", SHA256)


@router.post("/exchange_key")
async def exchange_key(request: PublicKeyRequest, db: AsyncSession = Depends(get_db)):
    logger.debug(f"Received public key: {request.public_key}")
    logger.debug(f"Received kdf: {request.kdf}")
    try:
        # Decode client's public key
        client_public_key = ECC.import_key(base64.b64decode(request.public_key))

        # Perform ECDH key exchange
        shared_secret_point = client_public_key.pointQ * server_key.d
        shared_secret_bytes = int(shared_secret_point.x).to_bytes((int(shared_secret_point.x).bit_length() + 7) // 8,
                                                                  byteorder='big')
        logger.debug(f"Shared secret: {shared_secret_bytes.hex()}")

        # Derive the final shared key using HKDF
        shared_key = perform_hkdf(shared_secret_bytes)
        logger.debug(f"Generated shared key: {shared_key.hex()}")

        # Save the shared key along with client id
        client_id = base64.urlsafe_b64encode(client_public_key.export_key(format='DER')).decode()
        await save_shared_key(db, client_id, shared_key)

        # Return server's public key
        server_public_key = server_key.public_key().export_key(format='DER')
        return base64.b64encode(server_public_key).decode()
    except Exception as e:
        logger.error(f"Key exchange error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/send_data")
async def send_data(request: EncryptedDataRequest, db: AsyncSession = Depends(get_db)):
    logger.debug(f"Received encrypted data: {request.encrypted_data}")
    try:
        # Load the shared key for the specific client
        shared_key = await load_shared_key(db, request.client_id)
        if not shared_key:
            raise HTTPException(status_code=400, detail="Client not registered or Client Key is invalid")

        encrypted_data = base64.b64decode(request.encrypted_data)
        iv = base64.b64decode(request.iv)

        cipher = AES.new(shared_key[:32], AES.MODE_CBC, iv)
        try:
            decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)
            logger.info(f"Decryption successful. Received data: {decrypted_data.decode()}")
        except ValueError as ve:
            logger.error(f"Decryption failed: {str(ve)}")
            raise HTTPException(status_code=400, detail="Decryption failed")

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error in send_data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/receive_data/{client_id}")
async def receive_data(client_id: str, db: AsyncSession = Depends(get_db)):
    decoded_client_id = unquote(client_id)
    shared_key = await load_shared_key(db, decoded_client_id)
    if not shared_key:
        raise HTTPException(status_code=400, detail="Client not registered or Client Key is invalid")

    data = "Hello from the server!"

    cipher = AES.new(shared_key[:32], AES.MODE_CBC)
    encrypted_data = cipher.encrypt(pad(data.encode(), AES.block_size))

    return {
        "encrypted_data": base64.b64encode(encrypted_data).decode(),
        "iv": base64.b64encode(cipher.iv).decode()
    }


@router.get("/check_key_validity/{client_id}")
async def check_key_validity(client_id: str, db: AsyncSession = Depends(get_db)):
    shared_key = await load_shared_key(db, client_id)
    if not shared_key:
        raise HTTPException(status_code=401, detail="Key invalid or not found")
    return {"status": "valid"}


@router.post("/invalidate_keys")
async def invalidate_keys(
        request: Request,
        db: AsyncSession = Depends(get_db, ),
        api_key: auth.models.APIKey = Depends(auth.get_api_key)
):
    client_host = request.client.host
    if client_host not in ADMIN_ACCESS_LIST:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access Restricted"
        )
    try:
        generate_or_load_server_key(del_key=True)
        await db.execute(update(ClientKey).values(is_valid=False))
        await db.commit()
        logger.info("Server keypair rotated and all client keys invalidated")
        return {"status": "success", "message": "Server keypair rotated and all keys invalidated"}
    except Exception as e:
        logger.error(f"Error rotating server keypair and invalidating keys: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
