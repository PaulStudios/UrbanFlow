from datetime import datetime
import json

from server.schemas import UserBase, VerificationResponse, VehicleBase, VehicleCreate, VehicleRegistrationResponse


def serialize_user(user: UserBase) -> str:
    user_dict = user.dict()
    user_dict['date_of_birth'] = user_dict['date_of_birth'].isoformat()
    return json.dumps(user_dict)


def deserialize_user(user_json: str) -> UserBase:
    user_dict = json.loads(user_json)
    user_dict['date_of_birth'] = datetime.fromisoformat(user_dict['date_of_birth'])
    return UserBase(**user_dict)


def serialize_verify_response(response: VerificationResponse) -> str:
    verify_response_dict = response.dict()
    verify_response_dict['checked_at'] = verify_response_dict['checked_at'].isoformat()
    return json.dumps(verify_response_dict)


def deserialize_verify_response(response_json: str) -> VerificationResponse:
    verify_response_dict = json.loads(response_json)
    verify_response_dict['checked_at'] = datetime.fromisoformat(verify_response_dict['checked_at'])
    return VerificationResponse(**verify_response_dict)


def deserialize_vehicle_create(response: str) -> VehicleCreate:
    vehicle_response_dict = json.loads(response)
    return VehicleCreate(**vehicle_response_dict)


def serialize_vehicle_response(vehicle: VehicleRegistrationResponse) -> str:
    vehicle_response_dict = vehicle.dict()
    vehicle_response_dict['id'] = str(vehicle_response_dict['id'])  # Convert UUID to string
    return json.dumps(vehicle_response_dict)
