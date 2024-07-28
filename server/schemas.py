from pydantic import BaseModel, UUID4
from datetime import datetime


class TrafficSignalBase(BaseModel):
    status: str
    latitude: float
    longitude: float
    time_from_last_change: int


class TrafficSignalCreate(TrafficSignalBase):
    pass


class TrafficSignalUpdate(BaseModel):
    status: str
    time_from_last_change: int


class TrafficSignal(TrafficSignalBase):
    signal_id: UUID4
    status: str
    updated_at: datetime

    class Config:
        orm_mode = True
