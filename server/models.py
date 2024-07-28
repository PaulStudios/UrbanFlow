from sqlalchemy import Column, String, Float, Integer, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from .database import Base
import uuid

class TrafficSignal(Base):
    __tablename__ = "traffic_signals"

    signal_id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid.uuid4)
    status = Column(String, index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    time_from_last_change = Column(Integer)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())