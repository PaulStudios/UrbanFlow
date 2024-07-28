from fastapi import FastAPI
from server.routers import signals, auth
from server.database import engine, Base
from server.cache import update_all_signal_ids_cache
import asyncio

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Start background task to update all_signal_ids_cache
    asyncio.create_task(update_all_signal_ids_cache(engine))


app.include_router(signals.router)
app.include_router(auth.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)