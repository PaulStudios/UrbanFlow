from faker import Faker
from faker.generator import random
from faker.providers import python

from mock_api.data_gen.valid_codes import valid_state_codes, valid_types

fake = Faker()
fake.add_provider(python)


def gen_vehicle_number():
    state_code = random.choice(valid_state_codes)
    rto_code = fake.pyint(max_value=99, min_value=10)
    series_code = f"{fake.pystr(max_chars=1)}{fake.pystr(max_chars=1)}".upper()
    unique_id = fake.pyint(min_value=1001, max_value=9999)
    return f"{state_code}-{rto_code}-{series_code}-{unique_id}", state_code

def gen_vehicle():
    number, state_code = gen_vehicle_number()
    r = {
        "name": fake.name(),
        "language": "English",
        "number": number,
        "issued_by": state_code,
        "type": random.choice(valid_types),
    }
    return r
