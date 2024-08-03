import base64
import logging

from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Crypto.Protocol.KDF import HKDF
from Crypto.PublicKey import ECC
from Crypto.Util.Padding import pad, unpad
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from server.database import get_db
from server.models import ClientKey
from server.schemas import PublicKeyRequest, EncryptedDataRequest
from server.utils.encyption_key import server_key, save_shared_key, perform_hkdf

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
        shared_secret_bytes = int(shared_secret_point.x).to_bytes((int(shared_secret_point.x).bit_length() + 7) // 8, byteorder='big')
        logger.debug(f"Shared secret: {shared_secret_bytes.hex()}")

        # Derive the final shared key using HKDF
        shared_key = perform_hkdf(shared_secret_bytes)
        logger.debug(f"Generated shared key: {shared_key.hex()}")


        # Save shared key
        client_id = base64.b64encode(client_public_key.export_key(format='DER')).decode()
        await save_shared_key(db, client_id, shared_key)

        # Return server's public key
        server_public_key = server_key.public_key().export_key(format='DER')
        return base64.b64encode(server_public_key).decode()
    except Exception as e:
        logger.error(f"Key exchange error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/send_data")
async def send_data(request: EncryptedDataRequest, db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(select(ClientKey))
        client_key = result.scalar_one_or_none()
        if not client_key:
            raise HTTPException(status_code=400, detail="No clients registered")

        shared_key = client_key.shared_key
        logger.debug(f"Shared key (first 10 bytes): {shared_key[:10].hex()}")

        encrypted_data = base64.b64decode(request.encrypted_data)
        iv = base64.b64decode(request.iv)

        logger.debug(f"IV: {iv.hex()}")
        logger.debug(f"Encrypted data length: {len(encrypted_data)}")

        cipher = AES.new(shared_key[:32], AES.MODE_CBC, iv)
        try:
            decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)
            logger.info(f"Decryption successful. Received data: {decrypted_data.decode()}")
        except ValueError as ve:
            logger.error(f"Decryption failed: {str(ve)}")
            raise HTTPException(status_code=400, detail="Decryption failed")

        # Process the decrypted data as needed

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error in send_data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/receive_data")
async def receive_data(db: AsyncSession = Depends(get_db)):
    # In a production environment, you would need to identify the client
    # For simplicity, we'll use the first client in our storage
    result = await db.execute(select(ClientKey))
    client_key = result.scalar_one_or_none()
    if not client_key:
        raise HTTPException(status_code=400, detail="No clients registered")

    shared_key = client_key.shared_key

    # Example data to send
    data = "Hello from the server!"

    cipher = AES.new(shared_key[:32], AES.MODE_CBC)
    encrypted_data = cipher.encrypt(pad(data.encode(), AES.block_size))

    return {
        "encrypted_data": base64.b64encode(encrypted_data).decode(),
        "iv": base64.b64encode(cipher.iv).decode()
    }
