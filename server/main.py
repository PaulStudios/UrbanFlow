import asyncio
import logging

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from sqlalchemy import text

from server.cache import cache_manager, redis_pool
from server.database import engine, Base, get_db
from server.routers import signals, auth, encryption, android_api

app = FastAPI()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def periodic_db_update(cache_manager, get_db):
    while True:
        async for db in get_db():
            await cache_manager.update_db_from_cache(db)
            break
        await asyncio.sleep(30)


async def startup_event():
    logger.debug("Initializing FastAPI Cache")
    redis = RedisBackend(redis_pool)
    FastAPICache.init(redis, prefix="fastapi-cache")
    async for db in get_db():
        await cache_manager.warm_cache(db)
        break
    asyncio.create_task(periodic_db_update(cache_manager, get_db))
    logger.debug("FastAPI Cache initialized")


app.add_event_handler("startup", startup_event)


# Graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down FastAPI Cache")
    redis_pool.close()
    logger.info("FastAPI Cache closed.")


app.include_router(signals.router)
app.include_router(auth.router)
app.include_router(encryption.router)
app.include_router(android_api.router)


@app.get("/status", response_class=PlainTextResponse)
async def healthcheck(request: Request):
    return "OK"


@app.get("/health")
async def health_check():
    logger.debug("Health check requested")
    try:
        # Check Redis connection
        logger.debug("Checking Redis connection")
        redis_pool.ping()

        # Check database connection
        logger.debug("Checking database connection")
        async for db in get_db():
            await db.execute(text("SELECT 1"))
            break

        logger.info("Health check passed")
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=503, detail="Service unavailable")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting URBANFLOW Main Server")
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
