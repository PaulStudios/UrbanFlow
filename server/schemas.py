from typing import Dict, List

from pydantic import BaseModel, UUID4
from datetime import datetime


class TrafficSignalBase(BaseModel):
    """
    Base schema for traffic signals.
    """
    status: str
    latitude: float
    longitude: float


class TrafficSignalCreate(TrafficSignalBase):
    """
    Schema for creating a new traffic signal.
    """

    class Config:
        schema_extra = {
            "example": {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "status": "green"
            }
        }


class TrafficSignalUpdate(BaseModel):
    """
    Schema for updating an existing traffic signal's status.
    """
    status: str


class TrafficSignal(TrafficSignalBase):
    """
    Schema representing a traffic signal with an ID and updated_at timestamp.
    """
    signal_id: UUID4

    class Config:
        orm_mode = True


class BatchSignalUpdate(BaseModel):
    updates: List[Dict[str, str]]


class UserBase(BaseModel):
    id: str
    name: str
    date_of_birth: datetime
    mobile_number: str
    license_number: str
    vehicle_number: str
    aadhar_number: str
    permit_uri: str
    selfie_uri: str


class UserCreate(UserBase):
    pass


class VerificationResponse(BaseModel):
    status: bool
    checked_at: datetime

    class Config:
        orm_mode = True


class PublicKeyRequest(BaseModel):
    public_key: str
    kdf: str = "hkdf"


class EncryptedDataRequest(BaseModel):
    client_id: str
    encrypted_data: str
    iv: str
