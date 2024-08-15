import logging
import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import UUID4
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from server import models, schemas
from server.auth import auth
from server.cache import cache_manager
from server.database import get_db
from server.schemas import TrafficSignalCreate, BatchSignalUpdate

router = APIRouter(
    prefix="/signals",
    tags=["Traffic Signals"],
)

logger = logging.getLogger(__name__)


@router.put("/batch", response_model=List[schemas.TrafficSignal])
async def batch_update_signals(
        updates: BatchSignalUpdate,
        background_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_db),
        api_key: auth.models.APIKey = Depends(auth.get_api_key)
):
    """Update multiple traffic signals' status in the cache."""
    logger.debug(f"Batch update request received for {len(updates.updates)} signals")
    try:
        updated_signals = []
        for update in updates.updates:
            signal_id = update['signal_id']
            status = update['status']
            logger.debug(f"Updating signal {signal_id} with status {status}")
            cache_manager.update_signal_in_cache(str(signal_id), status)
            updated_signals.append(schemas.TrafficSignal(signal_id=signal_id, status=status))

        logger.debug("Scheduling background task to update database from cache")
        background_tasks.add_task(cache_manager.update_db_from_cache, db)

        logger.info(f"Successfully updated {len(updated_signals)} signals in cache")
        return updated_signals
    except ValueError as ve:
        logger.error(f"Error in batch update: {str(ve)}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in batch update: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/", response_model=List[schemas.TrafficSignal])
async def read_signals(
        db: AsyncSession = Depends(get_db),
        api_key: auth.models.APIKey = Depends(auth.get_api_key)
):
    """Fetch all traffic signals from the cache with database fallback."""
    logger.debug("Received request to fetch all signals")
    try:
        start_time = time.time()
        logger.debug("Attempting to fetch signals from cache")
        signals = await cache_manager.get_all_signals_from_cache(db)
        end_time = time.time()
        logger.info(f"Fetched {len(signals)} signals in {end_time - start_time:.2f} seconds")
        return [
            schemas.TrafficSignal(
                signal_id=signal['signal_id'],
                latitude=float(signal['latitude']),
                longitude=float(signal['longitude']),
                status=signal['status'],
            )
            for signal in signals
        ]
    except Exception as e:
        logger.error(f"Error fetching signals: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/create", response_model=schemas.TrafficSignal)
async def create_signal(
        signal: TrafficSignalCreate,
        db: AsyncSession = Depends(get_db),
        api_key: auth.models.APIKey = Depends(auth.get_api_key)
):
    """Create a new traffic signal and add it to both the database and cache."""
    logger.debug(f"Received request to create new signal: {signal}")
    db_signal = models.TrafficSignal(**signal.dict())
    db.add(db_signal)
    try:
        logger.debug("Committing new signal to database")
        await db.commit()
        await db.refresh(db_signal)
        logger.debug(f"Adding new signal {db_signal.signal_id} to cache")
        cache_manager.add_signal_to_cache(
            str(db_signal.signal_id),
            db_signal.latitude,
            db_signal.longitude,
            db_signal.status
        )
        logger.info(f"Created new traffic signal with ID {db_signal.signal_id}.")
        return db_signal
    except IntegrityError:
        logger.error("A traffic signal with this latitude and longitude already exists.")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A traffic signal with this latitude and longitude already exists."
        )
    except Exception as e:
        logger.error(f"Error creating new signal: {str(e)}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{signal_id}", response_model=schemas.TrafficSignal)
async def update_signal(
        signal_id: UUID4,
        signal: schemas.TrafficSignalUpdate,
        db: AsyncSession = Depends(get_db),
        api_key: auth.models.APIKey = Depends(auth.get_api_key)
):
    """Update a traffic signal's status in the cache."""
    logger.debug(f"Received request to update signal {signal_id} with status {signal.status}")
    try:
        # Check if signal exists in database
        stmt = select(models.TrafficSignal).where(models.TrafficSignal.signal_id == signal_id)
        result = await db.execute(stmt)
        db_signal = result.scalar_one_or_none()

        if not db_signal:
            logger.error(f"Signal {signal_id} not found in database")
            raise HTTPException(status_code=404, detail="Signal not found")

        cache_manager.update_signal_in_cache(str(signal_id), signal.status)
        logger.info(f"Updated traffic signal with ID {signal_id} to status {signal.status} in cache.")
        return schemas.TrafficSignal(signal_id=signal_id, status=signal.status)
    except ValueError as ve:
        logger.error(f"Error updating signal {signal_id}: {str(ve)}")
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Error updating signal {signal_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
