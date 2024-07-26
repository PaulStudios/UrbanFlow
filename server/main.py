from fastapi import FastAPI

from server.database import engine, Base
from server.routers import traffic_signals

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(traffic_signals.router)
