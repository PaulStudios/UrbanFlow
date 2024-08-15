import asyncio
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from server.cache import cache_manager
from server.database import get_db
from server.schemas import EncryptedDataRequest, VehicleRegistrationResponse
from server.utils import data_encryption
from server.utils.priority.basic import get_coordinates, calculate_distance_km, calculate_approx_time
from server.utils.priority.route_finding.manager import get_vehicle_route
from server.utils.serializers import deserialize_vehicle_create, serialize_vehicle_response

router = APIRouter(
    prefix="/api/vehicle",
    tags=["Vehicle API"],
)

logger = logging.getLogger(__name__)


@router.post("/create")
async def create_vehicle(
        request: EncryptedDataRequest,
        background_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_db),
        # api_key: auth.models.APIKey = Depends(auth.get_api_key)
):
    decrypted_data, result_code = await data_encryption.decode_input(request.encrypted_data, request.iv,
                                                                     request.client_id, db)

    if result_code == 0:
        logger.info(f"Decryption successful for client {request.client_id}.")
    elif result_code == 1:
        raise HTTPException(status_code=400, detail="Client not registered")
    elif result_code == 2:
        raise HTTPException(status_code=400, detail="Decryption failed. Invalid credentials")
    elif result_code == 3:
        raise HTTPException(status_code=500, detail="Internal server error")

    data = deserialize_vehicle_create(decrypted_data)
    logger.info(f"Received vehicle data: {data}")

    origin = data.origin
    destination = data.destination

    vehicle_id = uuid.uuid4()

    origin_coordinates = get_coordinates(origin)
    destination_coordinates = get_coordinates(destination)

    logger.info(f"Origin coordinates: {origin_coordinates}")
    logger.info(f"Destination coordinates: {destination_coordinates}")

    if (origin_coordinates and destination_coordinates) and (origin_coordinates != destination_coordinates):
        distance = calculate_distance_km(origin_coordinates, destination_coordinates)
        eta = calculate_approx_time(distance)
        logger.info(f"Calculated distance: {distance} Kilometres")
        logger.info(f"Calculated calculation time: {eta} minutes")

        background_tasks.add_task(lambda: asyncio.run(get_vehicle_route(origin, destination, vehicle_id, db)))

        response_data = VehicleRegistrationResponse(
            id=uuid.uuid4(),
            origin=origin,
            destination=destination,
            status=True,
            detail=str(eta)
        )

        response, error_code = await data_encryption.encode_input(serialize_vehicle_response(response_data),
                                                                  request.client_id, db)

        if error_code == 0:
            return response_data
        elif error_code == 1:
            raise HTTPException(status_code=400, detail="Client not registered")
        elif error_code == 2:
            raise HTTPException(status_code=400, detail="Encryption failed.")
    else:
        response_data = VehicleRegistrationResponse(
            id=uuid.uuid4(),
            origin=origin,
            destination=destination,
            status=False,
            detail="Invalid coordinates",
        )
        logger.warning(f"Invalid coordinates submitted for client {request.client_id}")

        response, error_code = await data_encryption.encode_input(serialize_vehicle_response(response_data),
                                                                  request.client_id, db)

        if error_code == 0:
            return response_data
        elif error_code == 1:
            raise HTTPException(status_code=400, detail="Client not registered")
        elif error_code == 2:
            raise HTTPException(status_code=400, detail="Encryption failed.")


@router.get("/registration-status/{vehicle_id}")
async def get_vehicle_registration_status(
        vehicle_id: str,
        db: AsyncSession = Depends(get_db),
        # api_key: auth.models.APIKey = Depends(auth.get_api_key)
):
    # Use asyncio to efficiently wait for the result
    result = cache_manager.get_result(vehicle_id)[0]['result']
    logger.info(f"Vehicle registration status: {result}")
    if result:
        if result == "DONE":
            reg_response = cache_manager.get_route_data(vehicle_id)
            return reg_response
        elif result == "PROGRESS":
            raise HTTPException(status_code=202, detail="Processing in progress. Please try again later.")
        elif result == "ERROR":
            raise HTTPException(status_code=400, detail="Vehicle not registered")
    raise HTTPException(status_code=408, detail="Vehicle not found")
