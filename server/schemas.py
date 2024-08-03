from datetime import datetime

from pydantic import BaseModel, UUID4


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


class PublicKeyRequest(BaseModel):
    public_key: str
    kdf: str = "hkdf"


class EncryptedDataRequest(BaseModel):
    encrypted_data: str
    iv: str
