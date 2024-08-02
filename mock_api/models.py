from sqlalchemy import Column, String, Integer, DateTime, func, Interval, Boolean, ForeignKey, Enum
from sqlalchemy.sql import expression
from enum import Enum as Enum_Base

from mock_api.database import Base


class Type(Enum_Base):
    LMV = "Light Motor Vehicle - Cars and bikes"
    MCWG = "Motorcycle with Gear - Motorcycles with gear"
    MCWOG = "Motorcycle without Gear - Scooters and mopeds"
    HPMV = "Heavy Passenger Motor Vehicle - Buses and coaches"
    HTV = "Heavy Transport Vehicle - Trucks and trailers"
    LMV_NT = "Light Motor Vehicle-Non Transport - Private vehicles only"
    LMV_TR = "Light Motor Vehicle-Transport - Commercial transport vehicles"
    HMV = "Heavy Motor Vehicle - Large commercial vehicles"
    MGV = "Medium Goods Vehicle - Smaller trucks, goods"
    MPMV = "Medium Passenger Motor Vehicle - Minibuses, small passenger"
    FVG = "Forklift Vehicle Goods - Forklifts and loaders"
    AR = "Auto Rickshaw - Three-wheeled auto rickshaws"
    RDR = "Road Roller - Road rollers only"
    HP = "Heavy Passenger Vehicle - Buses, passenger carriers"
    GCD = "General Crane Driver - Operating cranes"

    @classmethod
    def get_description(cls, abbreviation):
        return cls[abbreviation].value


class DrivingLicense(Base):
    __tablename__ = 'licenses'
    __table_args__ = {'extend_existing': True}

    number = Column(String, primary_key=True, index=True)
    name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    language = Column(String)
    type = Column(Enum(Type, name='license_type_enum'), nullable=False)
    issued_date = Column(DateTime(timezone=True), server_default=func.now())
    issued_by = Column(String)
    expiry_date = Column(DateTime(timezone=True))
    status = Column(Boolean)


class VehicleRegistration(Base):
    __tablename__ = 'vehicles'
    __table_args__ = {'extend_existing': True}

    number = Column(String, primary_key=True, index=True)
    name = Column(String)
    type = Column(Enum(Type, name='vehicle_type_enum'), nullable=False)
    language = Column(String)
    issued_date = Column(DateTime(timezone=True), server_default=func.now())
    valid_from_date = Column(DateTime(timezone=True), server_default=func.now())
    issued_by = Column(String)
    expiry_date = Column(DateTime(timezone=True))
    status = Column(Boolean)
