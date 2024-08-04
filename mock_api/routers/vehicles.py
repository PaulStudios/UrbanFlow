import re

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime, timedelta

from mock_api import schemas, database, models
from mock_api.auth import auth

router = APIRouter(
    prefix="/vehicle",
    tags=["Vehicle Registration"],
)

@router.post("/", response_model=schemas.VehicleRegistration)
async def create_vehicle_registration(
        vehicle: schemas.VehicleRegistrationCreate,
        db: AsyncSession = Depends(database.get_db),
        api_key: auth.models.APIKey = Depends(auth.get_api_key)
):
    # Check if vehicle number already exists
    query = select(models.VehicleRegistration).where(models.VehicleRegistration.number == vehicle.number)
    result = await db.execute(query)
    db_vehicle = result.scalar_one_or_none()

    if db_vehicle:
        raise HTTPException(status_code=400, detail="Vehicle number already registered")

    # Convert type abbreviation to full description
    vehicle_dict = vehicle.dict()

    # Create new vehicle registration
    new_vehicle = models.VehicleRegistration(**vehicle_dict)
    new_vehicle.status = True
    new_vehicle.issued_date = datetime.now()
    new_vehicle.valid_from_date = datetime.now()
    new_vehicle.expiry_date = datetime.now() + timedelta(days=365.25 * 15)  # Set expiry to 15 years from now

    db.add(new_vehicle)
    await db.commit()
    await db.refresh(new_vehicle)

    return new_vehicle


@router.get("/{vehicle_number}", response_model=schemas.VehicleRegistration)
async def get_vehicle_registration(
        vehicle_number: str,
        db: AsyncSession = Depends(database.get_db),
        api_key: auth.models.APIKey = Depends(auth.get_api_key)
):
    if not re.match(r'^[A-Z]{2}-\d{2}-[A-Z]{1,2}-\d{1,4}$', vehicle_number):
        raise HTTPException(status_code=400, detail="Invalid vehicle number")
    query = select(models.VehicleRegistration).where(models.VehicleRegistration.number == vehicle_number)
    result = await db.execute(query)
    vehicle = result.scalar_one_or_none()

    if vehicle is None:
        raise HTTPException(status_code=404, detail="Vehicle registration not found")

    return vehicle