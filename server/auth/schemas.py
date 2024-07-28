from pydantic import BaseModel, UUID4
from datetime import datetime

class UserCreate(BaseModel):
    username: str
    password: str

class User(BaseModel):
    id: UUID4
    username: str
    created_at: datetime

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class APIKeyCreate(BaseModel):
    user_id: UUID4
    expires_at: datetime

class APIKey(BaseModel):
    id: UUID4
    user_id: UUID4
    api_key: str
    expires_at: datetime
    created_at: datetime

    class Config:
        orm_mode = True