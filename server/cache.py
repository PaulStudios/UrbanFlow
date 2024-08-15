import asyncio
import json
import logging
import os
import signal
import sys
from functools import wraps
from typing import List, Dict, Optional

from cachecontrol.caches import RedisCache
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
        """
        Initialize the CacheManager with a Redis connection pool.

        Args:
            redis_pool (Redis): A Redis connection pool object.
        """
        self.redis = redis_pool
        logger.info("CacheManager initialized with Redis pool")

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
                await self.redis.delete("updated_signal_ids")
                logger.info(f"Updated {len(updated_ids)} signals in the database")
        except Exception as e:
            logger.error(f"Error updating database from cache: {str(e)}")
            await db.rollback()

    async def warm_cache(self, db: AsyncSession):
        """Warm the cache by pre-populating it with data from the database."""
        logger.info("Starting cache warming process")
        await self.initialize_cache(db)
        logger.info("Cache warming completed")

    def save_route_data(self, key: str, route: Optional[List], filtered_intersections: Optional[List],
                        all_intersections: Optional[List]) -> bool:
        """
        Save route data to Redis with a provided key.

        Args:
            key (str): The key under which to store the data.
            route (List, optional): A list representing the route.
            filtered_intersections (List, optional): A list of filtered intersections.
            all_intersections (List, optional): A list of all intersections.

        Returns:
            bool: True if the save was successful, False otherwise.
        """
        if not route and not all_intersections and not filtered_intersections:
            logger.error("No route or intersections provided")
            return False

        pipe = self.redis.pipeline()

        # Store route data
        for index, point in enumerate(route):
            pipe.hset(f"{key}:route:{index}", mapping={
                "latitude": point[0],
                "longitude": point[1]
            })

        # Store filtered intersections
        for index, intersection in enumerate(filtered_intersections or []):
            pipe.hset(f"{key}:filtered_intersection:{index}", mapping={
                "latitude": intersection[0],
                "longitude": intersection[1]
            })

        # Store all intersections
        for index, intersection in enumerate(all_intersections or []):
            pipe.hset(f"{key}:intersection:{index}", mapping={
                "latitude": intersection[0],
                "longitude": intersection[1]
            })
        try:
            pipe.execute()
            logger.info(f"Route data saved with key: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to save route data with key: {key}, error: {e}")
            return False

    def edit_route_data(self, key: str, route: Optional[List] = None,
                        filtered_intersections: Optional[List] = None,
                        all_intersections: Optional[List] = None) -> bool:
        """
        Edit existing route data in Redis.

        Args:
            key (str): The key of the data to edit.
            route (List, optional): New route data. If None, the existing route data is not modified.
            filtered_intersections (List, optional): New filtered intersections data. If None, the existing filtered intersections data is not modified.
            all_intersections (List, optional): New all intersections data. If None, the existing all intersections data is not modified.

        Returns:
            bool: True if the edit was successful, False otherwise.
        """
        pipe = self.redis.pipeline()

        if route is not None:
            for i, point in enumerate(route):
                pipe.hset(f"{key}:route:{i}", mapping={"latitude": point[0], "longitude": point[1]})

        if filtered_intersections is not None:
            for i, intersection in enumerate(filtered_intersections):
                pipe.hset(f"{key}:filtered_intersection:{i}",
                          mapping={"latitude": intersection[0], "longitude": intersection[1]})

        if all_intersections is not None:
            for i, intersection in enumerate(all_intersections):
                pipe.hset(f"{key}:intersection:{i}",
                          mapping={"latitude": intersection[0], "longitude": intersection[1]})

        try:
            pipe.execute()
            logger.info(f"Route data updated for key: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to update route data for key: {key}, error: {e}")
            return False

    def delete_route_data(self, key: str) -> bool:
        """
        Delete route data from Redis using a provided key.

        Args:
            key (str): The key under which the data is stored.

        Returns:
            bool: True if the deletion was successful, False otherwise.
        """
        pipe = self.redis.pipeline()

        # Get all keys associated with the route data
        route_keys = self.redis.keys(f"{key}:route:*")
        filtered_intersection_keys = self.redis.keys(f"{key}:filtered_intersection:*")
        intersection_keys = self.redis.keys(f"{key}:intersection:*")

        # Add delete commands to the pipeline
        for route_key in route_keys:
            pipe.delete(route_key)
        for filtered_key in filtered_intersection_keys:
            pipe.delete(filtered_key)
        for intersection_key in intersection_keys:
            pipe.delete(intersection_key)

        try:
            pipe.execute()
            logger.info(f"Route data deleted with key: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete route data with key: {key}, error: {e}")
            return False

    def get_route_data(self, key: str) -> Optional[dict]:
        """
        Retrieve route data from Redis using a provided key.

        Args:
            key (str): The key under which the data is stored.

        Returns:
            dict: A dictionary containing the route, filtered intersections, and all intersections.
            None: If the data retrieval fails.
        """
        pipe = self.redis.pipeline()

        # Retrieve route data
        route_keys = self.redis.keys(f"{key}:route:*")
        for route_key in route_keys:
            pipe.hgetall(route_key)

        # Retrieve filtered intersections
        filtered_intersection_keys = self.redis.keys(f"{key}:filtered_intersection:*")
        for filtered_key in filtered_intersection_keys:
            pipe.hgetall(filtered_key)

        # Retrieve all intersections
        intersection_keys = self.redis.keys(f"{key}:intersection:*")
        for intersection_key in intersection_keys:
            pipe.hgetall(intersection_key)

        try:
            results = pipe.execute()

            # Process results to structure them properly
            route = [(float(res['latitude']), float(res['longitude'])) for res in results[:len(route_keys)]]
            filtered_intersections = [(float(res['latitude']), float(res['longitude'])) for res in
                                      results[len(route_keys):len(route_keys) + len(filtered_intersection_keys)]]
            all_intersections = [(float(res['latitude']), float(res['longitude'])) for res in
                                 results[-len(intersection_keys):]]

            return {
                "route": route,
                "filtered_intersections": filtered_intersections,
                "all_intersections": all_intersections
            }

        except Exception as e:
            logger.error(f"Failed to retrieve route data with key: {key}, error: {e}")
            return None

    def set_result(self, task_id: str, result: str):
        """
        Set the result of a task in the Redis cache.

        Args:
            task_id (str): The UUID of the task.
            result (str): The result of the task.
        """
        pipe = self.redis.pipeline()
        pipe.hset(f"{task_id}:result", mapping={"result": result})
        try:
            pipe.execute()
            logger.info(f"Task {task_id} result set to {result}")
            return True
        except Exception as e:
            logger.error(f"Failed to set result for task {task_id}, error: {e}")
            return False

    def get_result(self, task_id: str):
        """
        Get the result of a task from the Redis cache.

        Args:
            task_id (str): The UUID of the task.

        Returns:
            str: The result of the task if it exists, otherwise None.
        """
        pipe = self.redis.pipeline()
        key = f"{task_id}:result"
        pipe.hgetall(key)

        try:
            result = pipe.execute()
            logger.info(f"Task {task_id} result fetched.")
            return result
        except Exception as e:
            logger.error(f"Failed to retrieve result with key: {key}, error: {e}")
            return None


logger.info(f"Checking cache connection status")
try:
    redis_pool.ping()
except Exception as e:
    logger.error(f"Check failed: {str(e)}")
    os.kill(os.getpid(), signal.SIGINT)

cache_manager = CacheManager(redis_pool)
