import asyncio
import logging

from cachetools import TTLCache
from threading import Lock

from sqlalchemy import text

from server.database import get_db

all_signal_ids_cache = {}
updated_signal_ids_cache = TTLCache(maxsize=1000, ttl=30)

all_cache_lock = Lock()
updated_cache_lock = Lock()


async def update_all_signal_ids_cache(db):
    while True:
        async for session in get_db():
            signals = await session.execute(text("SELECT signal_id FROM traffic_signals"))
            signal_ids = [str(row[0]) for row in signals]

            with all_cache_lock:
                all_signal_ids_cache.clear()
                all_signal_ids_cache.update({signal_id: signal_id for signal_id in signal_ids})
        logging.info("Updated all signals cache")
        print("Updated all signals cache")
        await asyncio.sleep(30)


def add_updated_signal_id(signal_id):
    with updated_cache_lock:
        updated_signal_ids_cache[str(signal_id)] = True


def get_updated_signal_ids():
    with updated_cache_lock:
        return list(updated_signal_ids_cache.keys())