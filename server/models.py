import datetime

from sqlalchemy import Column, String, DateTime, Float, Integer

from .database import Base


class TrafficSignal(Base):
    __tablename__ = "traffic_signal_status"

    signal_id = Column(String, unique=True, primary_key=True, index=True)
    status = Column(String)
    time_from_last_change = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))
    longitude = Column(Float, nullable=False)
    latitude = Column(Float, nullable=False)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
