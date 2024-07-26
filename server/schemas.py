from pydantic import BaseModel


class TrafficSignalCreate(BaseModel):
    signal_id: str
    status: str
    time_from_last_change: int
    longitude: float
    latitude: float
