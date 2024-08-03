import hashlib
import hmac
import logging
import os

from Crypto.PublicKey import ECC
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from server.models import ClientKey

SERVER_KEY_FILE = "server_key.pem"
logger = logging.getLogger(__name__)


def generate_or_load_server_key():
    if os.path.exists(SERVER_KEY_FILE):
        with open(SERVER_KEY_FILE, "rt") as f:
            return ECC.import_key(f.read())
    else:
        key = ECC.generate(curve='P-256')
        with open(SERVER_KEY_FILE, "wt") as f:
            f.write(key.export_key(format='PEM'))
        return key


server_key = generate_or_load_server_key()


async def save_shared_key(db: AsyncSession, client_id: str, shared_key: bytes):
    print("here")
    client_key = ClientKey(client_id=client_id, shared_key=shared_key)
    db.add(client_key)
    await db.commit()


async def load_shared_key(db: AsyncSession, client_id: str):
    result = await db.execute(select(ClientKey).filter(ClientKey.client_id == client_id))
    client_key = result.scalar_one_or_none()
    return client_key.shared_key if client_key else None


def perform_hkdf(shared_secret, salt=None, info=b'', key_len=32):
    # Extract
    if salt is None or len(salt) == 0:
        salt = b'\x00' * 32
    prk = hmac.new(salt, shared_secret, hashlib.sha256).digest()
    logger.debug(f"PRK: {prk.hex()}")

    # Expand
    t = b""
    okm = b""
    for i in range(1, (key_len // 32) + 2):
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t

    return okm[:key_len]