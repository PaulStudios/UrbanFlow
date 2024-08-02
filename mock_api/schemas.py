from datetime import datetime
from typing import Optional

from pydantic import BaseModel, validator

from mock_api.models import Type


class DrivingLicenseBase(BaseModel):
    name: str
    age: int
    gender: str
    language: str
    type: str
    issued_by: str
    number: str


class DrivingLicenseCreate(DrivingLicenseBase):
    @validator('type')
    def validate_type(cls, v):
        if v not in Type.__members__:
            raise ValueError(f"Invalid type: {v}. Expected one of {list(Type.__members__.keys())}")
        return v

    class Config:
        orm_mode = True


class DrivingLicense(DrivingLicenseBase):
    expiry_date: datetime
    status: bool

    @validator('type', pre=True, always=True)
    def convert_type_to_abbreviation(cls, v):
        for member in Type:
            if member.value == v:
                return member.name
        return v

    class Config:
        orm_mode = True


class VehicleRegistrationBase(BaseModel):
    name: str
    type: str
    language: str
    issued_by: str
    number: str


class VehicleRegistrationCreate(VehicleRegistrationBase):
    @validator('type')
    def validate_type(cls, v):
        if v not in Type.__members__:
            raise ValueError(f"Invalid type: {v}. Expected one of {list(Type.__members__.keys())}")
        return v

    class Config:
        orm_mode = True


class VehicleRegistration(VehicleRegistrationBase):
    expiry_date: datetime
    status: bool

    @validator('type', pre=True, always=True)
    def convert_type_to_abbreviation(cls, v):
        for member in Type:
            if member.value == v:
                return member.name
        return v

    class Config:
        orm_mode = True
