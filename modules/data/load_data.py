import pandas as pd

from modules.data.features import add_temporal_features
from modules.utilities import logging


def load_data(csv_file):
    """
    Load data from a CSV file and add temporal features.

    Args:
        csv_file (str): The path to the CSV file.

    Returns:
        DataFrame: The loaded data with additional temporal features.
    """
    logging.info("Loading data from CSV file.")
    data = pd.read_csv(csv_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(['upload_id', 'timestamp'])
    data = add_temporal_features(data)  # Add temporal features
    logging.info("Data loaded and sorted.")
    return data
