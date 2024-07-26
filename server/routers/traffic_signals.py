from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import models, schemas, database

router = APIRouter(
    prefix="/traffic-signals",
    tags=["Traffic Signals"],
)


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/create")
def create_traffic_signal(signal: schemas.TrafficSignalCreate, db: Session = Depends(get_db)):
    db_signal = models.TrafficSignal(**signal.dict())
    db.add(db_signal)
    db.commit()
    db.refresh(db_signal)
    return db_signal
