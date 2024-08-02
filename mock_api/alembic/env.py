import os
import sys
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, create_engine
from sqlalchemy import pool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mock_api.models import Base, DrivingLicense, VehicleRegistration
from mock_api.auth.models import User, APIKey

load_dotenv()

# Load the config file for logging
fileConfig(context.config.config_file_name)

# Get the sync database URL
DATABASE_URL = os.getenv('MOCK_DATABASE')

# Set up the synchronous SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=True, future=True)


def run_migrations_offline():
    url = DATABASE_URL
    context.configure(url=url, target_metadata=Base.metadata, literal_binds=True)

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=Base.metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
