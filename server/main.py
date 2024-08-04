import asyncio
import logging

from fastapi import FastAPI

from server.cache import update_all_signal_ids_cache
from server.database import engine, Base
from server.routers import signals, auth, encryption, android_api

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Start background task to update all_signal_ids_cache
    asyncio.create_task(update_all_signal_ids_cache(engine))


app.include_router(signals.router)
app.include_router(auth.router)
app.include_router(encryption.router)
app.include_router(android_api.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)