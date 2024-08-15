import base64
import logging
from typing import Tuple, Union

from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad, pad
from sqlalchemy.ext.asyncio import AsyncSession

from server.utils.encyption_key import load_shared_key

logger = logging.getLogger(__name__)


async def decode_input(data: str, iv: str, client_id: str, db: AsyncSession) -> Tuple[Union[str, bool], int]:
    """
    Decrypts the provided encrypted data using the client's shared key.

    Args:
        data (str): Base64 encoded encrypted data to be decrypted.
        iv (str): Base64 encoded initialization vector for the decryption.
        client_id (str): The ID of the client whose shared key will be used for decryption.
        db (AsyncSession): Asynchronous database session used to load the shared key.

    Returns:
        Tuple[Union[str, bool], int]:
            - The decrypted data as a string if successful, or False if decryption fails.
            - An integer code representing the result of the operation:
                - 0: Success
                - 1: Shared key not found (client not registered)
                - 2: Decryption failed (invalid data or key)
                - 3: Other error encountered during the process
    """
    logger.info(f"Received encrypted data: {data}")
    try:
        # Load the shared key for the specific client
        shared_key = await load_shared_key(db, client_id)
        if not shared_key:
            logger.error("No shared key found. Client not registered")
            return False, 1

        encrypted_data = base64.b64decode(data)
        iv = base64.b64decode(iv)

        logger.info(f"Encrypted data length: {len(encrypted_data)}")

        cipher = AES.new(shared_key[:32], AES.MODE_CBC, iv)
        try:
            decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)
        except ValueError as ve:
            logger.error(f"Decryption failed for client {client_id}: {str(ve)}")
            return False, 2
    except Exception as e:
        logger.error(f"Error in send_data for client {client_id}: {str(e)}")
        return False, 3

    return decrypted_data.decode(), 0


async def encode_input(data: str, client_id: str, db: AsyncSession) -> Tuple[Union[dict, bool], int]:
    try:
        # Load the shared key for the specific client
        shared_key = await load_shared_key(db, client_id)
        if not shared_key:
            logger.error("No shared key found. Client not registered")
            return False, 1

        cipher = AES.new(shared_key[:32], AES.MODE_CBC)
        encrypted_data = cipher.encrypt(pad(data.encode(), AES.block_size))
        encrypted_data = base64.b64encode(encrypted_data).decode()
        logger.info(f"Response Encrypted data: {encrypted_data}")
        logger.info(f"Response Encrypted data length: {len(encrypted_data)}")

        return {
            "encrypted_data": encrypted_data,
            "iv": base64.b64encode(cipher.iv).decode()
        }, 0

    except ValueError as ve:
        logger.error(f"Encryption failed for client {client_id}: {str(ve)}")
        return False, 2
