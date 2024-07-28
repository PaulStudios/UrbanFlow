from pydantic import BaseModel


class TrafficSignalCreate(BaseModel):
    signal_id: str
    status: str
    time_from_last_change: int
    longitude: float
    latitude: float


class UserBase(BaseModel):
    username: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int

    class Config:
        orm_mode = True


class APIKeyCreate(BaseModel):
    user_id: int


class APIKey(APIKeyCreate):
    id: int
    key: str

    class Config:
        orm_mode = True