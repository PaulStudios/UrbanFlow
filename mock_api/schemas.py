from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from mock_api.models import Type


class DrivingLicenseBase(BaseModel):
    name: str
    age: int
    gender: str
    language: str
    type: Type
    issued_by: str


class DrivingLicenseCreate(DrivingLicenseBase):
    number: str


class DrivingLicense(DrivingLicenseCreate):
    number: str
    expiry_date: datetime
    name: str
    age: int
    status: bool

    class Config:
        orm_mode = True


class VehicleRegistrationBase(BaseModel):
    name: str
    type: Type
    language: str
    issued_by: str


class VehicleRegistrationCreate(VehicleRegistrationBase):
    number: str


class VehicleRegistration(VehicleRegistrationCreate):
    number: str
    expiry_date: datetime
    name: str
    age: int
    status: bool

    class Config:
        orm_mode = True
