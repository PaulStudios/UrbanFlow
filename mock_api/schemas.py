from datetime import datetime
from typing import Optional
import re

from pydantic import BaseModel, validator, constr
from pydantic.types import Annotated
import pydantic

from mock_api.models import Type

valid_state_codes = {"KA", "MH", "DL", "TN", "UP", "WB", "RJ", "PB", "KL", "AP", "GJ", "MP", "CH", "JH", "AS", "OR", "HR", "HP", "TR", "GA", "UT", "MN", "SK", "ML", "MZ", "AR", "NL", "JK", "LD", "AN"}


class DrivingLicenseBase(BaseModel):
    name: str
    age: int
    gender: str
    language: str
    type: str
    issued_by: str
    number: str

    @validator('age')
    def validate_age(cls, v):
        if v < 18:
            raise ValueError('Driver must be at least 18 years old')
        return v

    @validator('issued_by')
    def validate_issued_by(cls, v):
        if v not in valid_state_codes:
            raise ValueError('Invalid issuing state code')
        return v

    @validator('number')
    def validate_number_format(cls, v):
        if not re.match(r'^[A-Z]{2}-\d{2}-\d{4}-\d{7}$', v):
            raise ValueError("Invalid number format. Expected format: XX-XX-XXXX-XXXXXXX")
        return v

    @validator('number')
    def validate_license_number(cls, v):
        # Split the license number into its components
        state_code, rto_code, year_issued, unique_id = v.split('-')

        # Validate the state code
        if state_code not in valid_state_codes:
            raise ValueError('Invalid state code')

        # Validate the year issued
        current_year = datetime.now().year
        year_issued = int(year_issued)
        if year_issued < 1960 or year_issued > current_year:
            raise ValueError('Invalid year of issuance')

        return v

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

    @validator('number')
    def validate_number_format(cls, v):
        # Regular expression to match XX-XX-X-XXXX or XX-XX-XX-XXXX
        if not re.match(r'^[A-Z]{2}-\d{2}-[A-Z]{1,2}-\d{1,4}$', v):
            raise ValueError("Invalid number format. Expected format: XX-XX-X-XXXX or XX-XX-XX-XXXX")
        return v

    @validator('issued_by')
    def validate_issued_by(cls, v):
        if v not in valid_state_codes:
            raise ValueError('Invalid issuing state code')
        return v

    @validator('number')
    def validate_license_number(cls, v):
        # Split the license number into its components
        state_code = v.split('-')[0]

        # Validate the state code
        if state_code not in valid_state_codes:
            raise ValueError('Invalid state code')
        return v


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
