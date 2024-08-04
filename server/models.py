from sqlalchemy import Column, String, Float, Integer, DateTime, func, UniqueConstraint, LargeBinary, Boolean
from sqlalchemy.dialects.postgresql import UUID
from server.database import Base
import uuid


class TrafficSignal(Base):
    __tablename__ = "traffic_signals"

    signal_id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid.uuid4)
    status = Column(String, index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Add a unique constraint on the combination of latitude and longitude
    __table_args__ = (
        UniqueConstraint('latitude', 'longitude', name='uix_latitude_longitude'),
    )


class ClientKey(Base):
    __tablename__ = "client_keys"

    client_id = Column(String, primary_key=True, index=True)
    shared_key = Column(LargeBinary)
    is_valid = Column(Boolean, default=True)
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
