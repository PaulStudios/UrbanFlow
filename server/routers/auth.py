from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import UUID4
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from server import models, schemas, database, cache
from server.auth import auth
from typing import List
from datetime import datetime, timedelta, timezone
from fastapi.security import OAuth2PasswordRequestForm

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


@router.post("/register", response_model=auth.schemas.User)
async def register_user(
        user: auth.schemas.UserCreate,
        db: AsyncSession = Depends(database.get_db)
):
    # Check if the user already exists
    result = await db.execute(select(auth.models.User).filter_by(username=user.username))
    existing_user = result.scalars().first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )

    # Create a new user if not exists
    db_user = auth.models.User(username=user.username,
                               hashed_password=auth.jwt_handler.get_password_hash(user.password))
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user


@router.post("/token", response_model=auth.schemas.Token)
async def login_for_access_token(
        form_data: OAuth2PasswordRequestForm = Depends(),
        db: AsyncSession = Depends(database.get_db)
):
    user = await auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.jwt_handler.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.jwt_handler.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "id": user.id}


@router.post("/api-key", response_model=auth.schemas.APIKey)
async def create_api_key(
        api_key: auth.schemas.APIKeyCreate,
        db: AsyncSession = Depends(database.get_db),
        current_user: auth.models.User = Depends(auth.get_current_user)
):
    db_api_key = await auth.create_api_key(db, api_key.user_id, api_key.expires_at - datetime.now(timezone.utc))
    return db_api_key
