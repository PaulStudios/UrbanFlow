import asyncio
import logging
import os
from functools import wraps
from typing import List, Dict

from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from server import models

logger = logging.getLogger(__name__)

# Environment variables
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

redis_pool = Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    decode_responses=True,
    socket_timeout=5,
    socket_connect_timeout=5,
    retry_on_timeout=True,
    health_check_interval=30
)


def with_redis_fallback(fallback_func):
    """
    Decorator to implement fallback mechanism for Redis operations.
    If Redis is unavailable, it will call the fallback function.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Redis error in {func.__name__}: {str(e)}")
                logger.info(f"Falling back to database for {func.__name__}")
                return await fallback_func(*args, **kwargs)

        return wrapper

    return decorator


class CacheManager:
    def __init__(self, redis_pool: Redis):
        self.redis = redis_pool

    async def initialize_cache(self, db: AsyncSession):
        """Initialize the cache with all signal IDs and statuses from the database."""
        try:
            signals = await db.execute(select(models.TrafficSignal))
            signals = signals.scalars().all()

            pipe = self.redis.pipeline()
            for signal in signals:
                pipe.hset(f"signal:{signal.signal_id}", mapping={
                    "latitude": signal.latitude,
                    "longitude": signal.longitude,
                    "status": signal.status
                })
                pipe.sadd("all_signal_ids", str(signal.signal_id))
            pipe.execute()

            logger.info(f"Initialized cache with {len(signals)} signals from database")
        except Exception as e:
            logger.error(f"Error initializing cache: {str(e)}")
            raise

    def update_signal_in_cache(self, signal_id: str, status: str):
        """Update the status of a signal in the cache."""
        try:
            self.redis.hset(f"signal:{signal_id}", "status", status)
            self.redis.sadd("updated_signal_ids", signal_id)
            logger.info(f"Updated signal {signal_id} in cache with status: {status}")
        except Exception as e:
            logger.error(f"Error updating signal {signal_id} in cache: {str(e)}")
            raise

    def add_signal_to_cache(self, signal_id: str, latitude: float, longitude: float, status: str):
        """Add a new signal to the cache."""
        try:
            self.redis.hset(f"signal:{signal_id}", mapping={
                "latitude": latitude,
                "longitude": longitude,
                "status": status
            })
            self.redis.sadd("all_signal_ids", str(signal_id))
            logger.info(f"Added new signal {signal_id} to cache")
        except Exception as e:
            logger.error(f"Error adding signal {signal_id} to cache: {str(e)}")
            raise

    @with_redis_fallback(fallback_func=lambda self, db: self.get_all_signals_from_db(db))
    async def get_all_signals_from_cache(self, db: AsyncSession) -> List[Dict[str, str]]:
        """Fetch all signals from the cache."""
        all_ids = self.redis.smembers("all_signal_ids")
        pipe = self.redis.pipeline()
        for signal_id in all_ids:
            pipe.hgetall(f"signal:{signal_id}")
        signals = pipe.execute()
        return [{"signal_id": id, **signal} for id, signal in zip(all_ids, signals) if signal]

    async def get_all_signals_from_db(self, db: AsyncSession) -> List[Dict[str, str]]:
        """Fallback method to fetch all signals from the database."""
        signals = await db.execute(select(models.TrafficSignal))
        signals = signals.scalars().all()
        return [
            {
                "signal_id": str(signal.signal_id),
                "latitude": str(signal.latitude),
                "longitude": str(signal.longitude),
                "status": signal.status
            }
            for signal in signals
        ]

    async def update_db_from_cache(self, db: AsyncSession):
        """Update the database with changes from the cache every 30 seconds."""
        try:
            updated_ids = self.redis.smembers("updated_signal_ids")
            if updated_ids:
                for signal_id in updated_ids:
                    status = self.redis.hget(f"signal:{signal_id}", "status")

                    stmt = select(models.TrafficSignal).where(models.TrafficSignal.signal_id == signal_id)
                    result = await db.execute(stmt)
                    db_signal = result.scalar_one_or_none()

                    if db_signal:
                        db_signal.status = status

                await db.commit()
                self.redis.delete("updated_signal_ids")
                logger.info(f"Updated {len(updated_ids)} signals in the database")
        except Exception as e:
            logger.error(f"Error updating database from cache: {str(e)}")
            await db.rollback()

    async def warm_cache(self, db: AsyncSession):
        """Warm the cache by pre-populating it with data from the database."""
        logger.info("Starting cache warming process")
        await self.initialize_cache(db)
        logger.info("Cache warming completed")


cache_manager = CacheManager(redis_pool)