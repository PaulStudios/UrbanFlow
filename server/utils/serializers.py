from datetime import datetime
import json

from server.schemas import UserBase


def serialize_user(user: UserBase) -> str:
    user_dict = user.dict()
    user_dict['date_of_birth'] = user_dict['date_of_birth'].isoformat()
    return json.dumps(user_dict)

def deserialize_user(user_json: str) -> UserBase:
    user_dict = json.loads(user_json)
    user_dict['date_of_birth'] = datetime.fromisoformat(user_dict['date_of_birth'])
    return UserBase(**user_dict)