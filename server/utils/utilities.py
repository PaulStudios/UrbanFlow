import logging
import os
from pathlib import Path


def create_directory(directory):
    """
    Create a directory if it does not already exist.

    Args:
        directory (str): The path of the directory to create.
    """
    if not os.path.exists(directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        logging.info(f'Directory {directory} created.')
