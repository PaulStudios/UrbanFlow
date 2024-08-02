import os
import sys

from fastapi import FastAPI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mock_api.database import engine, Base
from mock_api.routers import auth, license, vehicles

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

app.include_router(auth.router)
app.include_router(license.router)
app.include_router(vehicles.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=8001, reload=True)