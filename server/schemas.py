from pydantic import BaseModel, UUID4
from datetime import datetime


class TrafficSignalBase(BaseModel):
    status: str
    latitude: float
    longitude: float


class TrafficSignalCreate(TrafficSignalBase):
    pass


class TrafficSignalUpdate(BaseModel):
    status: str


class TrafficSignal(TrafficSignalBase):
    signal_id: UUID4
    status: str
    updated_at: datetime

    class Config:
        orm_mode = True


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
