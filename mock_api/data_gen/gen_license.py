from faker import Faker
from faker.generator import random
from faker.providers import python

from mock_api.data_gen.valid_codes import valid_state_codes, valid_types, valid_genders

fake = Faker()
fake.add_provider(python)


def gen_license_number():
    state_code = random.choice(valid_state_codes)
    rto_code = fake.pyint(max_value=99, min_value=10)
    year = fake.random.randint(1955, 2002)
    unique_id = fake.pyint(max_value=9999999, min_value=1000000)
    return f"{state_code}-{rto_code}-{year}-{unique_id}", state_code


def gen_license():
    number, state_code = gen_license_number()
    r = {
        "name": fake.name(),
        "gender": random.choice(valid_genders),
        "age": fake.pyint(min_value=16, max_value=60),
        "language": "English",
        "number": number,
        "issued_by": state_code,
        "type": random.choice(valid_types),
    }
    return r
