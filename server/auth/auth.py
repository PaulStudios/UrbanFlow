from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from server.auth import models, schemas, jwt_handler
from server.database import get_db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
api_key_header = APIKeyHeader(name="X-API-Key")

async def authenticate_user(db: AsyncSession, username: str, password: str):
    user = await db.execute(select(models.User).filter(models.User.username == username))
    user = user.scalar_one_or_none()
    if not user or not jwt_handler.verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, jwt_handler.SECRET_KEY, algorithms=[jwt_handler.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = await db.execute(select(models.User).filter(models.User.username == token_data.username))
    user = user.scalar_one_or_none()
    if user is None:
        raise credentials_exception
    return user

async def create_api_key(db: AsyncSession, user_id: uuid4, expires_delta: timedelta):
    api_key = str(uuid4())
    expires_at = datetime.now(timezone.utc) + expires_delta
    db_api_key = models.APIKey(user_id=user_id, api_key=api_key, expires_at=expires_at)
    db.add(db_api_key)
    await db.commit()
    await db.refresh(db_api_key)
    return db_api_key

async def get_api_key(
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(api_key_header)
):
    db_api_key = await db.execute(
        select(models.APIKey).filter(models.APIKey.api_key == api_key)
    )
    db_api_key = db_api_key.scalar_one_or_none()

    if not db_api_key or db_api_key.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key"
        )
    return db_api_key