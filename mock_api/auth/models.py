from sqlalchemy import Column, String, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from mock_api.database import Base
import uuid

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid.uuid4)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), index=True)
    api_key = Column(String, unique=True, index=True)
    expires_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())