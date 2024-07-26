import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas, database
from fastapi.responses import JSONResponse, HTMLResponse
import datetime
import cachetools
import threading
import time
from ..utils.auth import verify_token, oauth2_scheme

router = APIRouter(
    prefix="/traffic-signals",
    tags=["Traffic Signals"],
)

# Initialize caches
all_signal_ids_cache = cachetools.Cache(maxsize=1000)
updated_signal_ids_cache = cachetools.TTLCache(maxsize=1000, ttl=30)

# Locks for thread safety
all_cache_lock = threading.Lock()
updated_cache_lock = threading.Lock()


def update_all_signal_ids_cache():
    while True:
        db = database.SessionLocal()
        try:
            with all_cache_lock:
                signals = db.query(models.TrafficSignal.signal_id).all()
                for signal in signals:
                    all_signal_ids_cache[signal.signal_id] = signal.signal_id
            logging.info("Updated all signals cache")
            time.sleep(30)  # Update every 30 seconds
        finally:
            db.close()


# Start background thread for updating all_signal_ids_cache
threading.Thread(target=update_all_signal_ids_cache, daemon=True).start()


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/create")
def create_traffic_signal(signal: schemas.TrafficSignalCreate, db: Session = Depends(get_db),
                          token: str = Depends(oauth2_scheme)):
    verify_token(token)
    try:
        db_signal = models.TrafficSignal(**signal.dict())
        db.add(db_signal)
        db.commit()
        db.refresh(db_signal)
        with all_cache_lock:
            all_signal_ids_cache[signal.signal_id] = signal.signal_id
        with updated_cache_lock:
            updated_signal_ids_cache[signal.signal_id] = signal.signal_id
        return db_signal
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred while creating the traffic signal: {str(e)}")


@router.get("/create", response_class=HTMLResponse)
async def create_signal_form():
    return """
    <html>
        <body>
            <h2>Create New Traffic Signal</h2>
            <form method="post">
                <input type="text" name="signal_id" placeholder="Signal ID" required><br>
                <input type="text" name="status" placeholder="Status" required><br>
                <input type="number" step="0.000001" name="longitude" placeholder="Longitude" required><br>
                <input type="number" step="0.000001" name="latitude" placeholder="Latitude" required><br>
                <input type="submit" value="Create Signal">
            </form>
        </body>
    </html>
    """


@router.put("/{signal_id}")
def update_traffic_signal(signal_id: str, status: str, db: Session = Depends(get_db),
                          token: str = Depends(oauth2_scheme)):
    verify_token(token)
    try:
        db_signal = db.query(models.TrafficSignal).filter(models.TrafficSignal.signal_id == signal_id).first()
        if db_signal is None:
            raise HTTPException(status_code=404, detail="Signal not found")
        db_signal.status = status
        db_signal.time_from_last_change = datetime.datetime.now(datetime.timezone.utc)
        db.commit()
        db.refresh(db_signal)
        with updated_cache_lock:
            updated_signal_ids_cache[signal_id] = signal_id
        return db_signal
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred while updating the traffic signal: {str(e)}")


@router.get("/", response_class=JSONResponse)
def get_all_signals(db: Session = Depends(get_db)):
    signals = []
    try:
        with updated_cache_lock:
            updated_ids = list(updated_signal_ids_cache.keys())
        # Load updated signals from the database
        if updated_ids:
            signals += db.query(models.TrafficSignal).filter(models.TrafficSignal.signal_id.in_(updated_ids)).all()
        # Load remaining signals from the cache
        with all_cache_lock:
            for signal_id in all_signal_ids_cache:
                if signal_id not in updated_ids:
                    db_signal = db.query(models.TrafficSignal).filter(
                        models.TrafficSignal.signal_id == signal_id).first()
                    if db_signal:
                        signals.append(db_signal)
        return {
            "signals": [
                {
                    "signal_id": signal.signal_id,
                    "status": signal.status,
                    "longitude": signal.longitude,
                    "latitude": signal.latitude,
                }
                for signal in signals
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while retrieving the signals: {str(e)}")
