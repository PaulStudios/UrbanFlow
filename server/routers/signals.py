from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import UUID4
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from server import models, schemas, database, cache
from server.auth import auth
from typing import List
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordRequestForm

from server.schemas import TrafficSignal

router = APIRouter(
    prefix="/signals",
    tags=["Traffic Signals"],
)


@router.post("/create", response_model=schemas.TrafficSignal)
async def create_signal(
        signal: schemas.TrafficSignalCreate,
        db: AsyncSession = Depends(database.get_db),
        api_key: auth.models.APIKey = Depends(auth.get_api_key)
):
    db_signal = models.TrafficSignal(**signal.dict())
    db.add(db_signal)

    try:
        await db.commit()
        await db.refresh(db_signal)
    except IntegrityError:
        await db.rollback()  # Rollback the transaction in case of an error
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A traffic signal with this latitude and longitude already exists."
        )

    return db_signal


@router.put("/{signal_id}", response_model=schemas.TrafficSignal)
async def update_signal(
        signal_id: UUID4,
        signal: schemas.TrafficSignalUpdate,
        db: AsyncSession = Depends(database.get_db),
        api_key: auth.models.APIKey = Depends(auth.get_api_key)
):
    result = await db.execute(select(models.TrafficSignal).filter(models.TrafficSignal.signal_id == signal_id))
    db_signal = result.scalar_one_or_none()
    if db_signal is None:
        raise HTTPException(status_code=404, detail="Signal not found")
    update_data = signal.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_signal, key, value)
    await db.commit()
    await db.refresh(db_signal)
    return db_signal


@router.get("/", response_model=List[schemas.TrafficSignal])
async def read_signals(
        db: AsyncSession = Depends(database.get_db),
        api_key: auth.models.APIKey = Depends(auth.get_api_key)
):
    updated_ids = cache.get_updated_signal_ids()

    # Fetch updated signals from the database
    updated_signals = await db.execute(
        select(models.TrafficSignal).filter(models.TrafficSignal.signal_id.in_(updated_ids)))
    updated_signals = updated_signals.scalars().all()

    # Fetch remaining signals from cache
    cached_signals = [signal_id for signal_id in cache.all_signal_ids_cache.values() if signal_id not in updated_ids]
    cached_signals = await db.execute(
        select(models.TrafficSignal).filter(models.TrafficSignal.signal_id.in_(cached_signals)))
    cached_signals = cached_signals.scalars().all()

    return updated_signals + cached_signals
