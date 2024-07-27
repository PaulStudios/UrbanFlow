import datetime
import secrets
import uuid

from sqlalchemy import Column, String, DateTime, Float, Integer, ForeignKey
from sqlalchemy.orm import relationship

from .database import Base


class TrafficSignal(Base):
    __tablename__ = "traffic_signal_status"

    signal_id = Column(String, unique=True, primary_key=True, index=True, default=uuid.uuid4)
    status = Column(String)
    time_from_last_change = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))
    longitude = Column(Float, nullable=False)
    latitude = Column(Float, nullable=False)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    api_keys = relationship("APIKey", back_populates="user")


class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True, default=lambda: secrets.token_urlsafe(32))
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="api_keys")
