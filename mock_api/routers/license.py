from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from mock_api import schemas, database, models
from mock_api.auth import auth
from mock_api.models import Type

router = APIRouter(
    prefix="/license",
    tags=["License"],
)


@router.post("/", response_model=schemas.DrivingLicense)
async def create_license(
        driving_license: schemas.DrivingLicenseCreate,
        db: AsyncSession = Depends(database.get_db),
        api_key: auth.models.APIKey = Depends(auth.get_api_key)
):
    # Check if license number already exists
    query = select(models.DrivingLicense).where(models.DrivingLicense.number == driving_license.number)
    result = await db.execute(query)
    db_license = result.scalar_one_or_none()

    if db_license:
        raise HTTPException(status_code=400, detail="License number already registered")

    driving_license_dict = driving_license.dict()

    # Create new license
    new_license = models.DrivingLicense(**driving_license_dict)
    new_license.status = True
    new_license.issued_date = datetime.now()
    new_license.expiry_date = datetime.now() + timedelta(days=365.25 * 5)  # Set expiry to 5 years from now

    db.add(new_license)
    await db.commit()
    await db.refresh(new_license)

    return new_license


@router.get("/{license_number}", response_model=schemas.DrivingLicense)
async def get_driving_license(
        license_number: str,
        db: AsyncSession = Depends(database.get_db),
        api_key: auth.models.APIKey = Depends(auth.get_api_key)
):
    query = select(models.DrivingLicense).where(models.DrivingLicense.number == license_number)
    result = await db.execute(query)
    license = result.scalar_one_or_none()

    if license is None:
        raise HTTPException(status_code=404, detail="Driving license not found")

    return license
