import json

from modules.utilities import logging


def save_hyperparameters(params, model_name):
    """
    Save hyperparameters to a JSON file.

    Args:
        params (dict): Hyperparameters to save.
        model_name (str): Name of the model to associate with the hyperparameters.

    Returns:
        None
    """
    filename = f'outputs/hyperparameters/best_hyperparameters_{model_name}.json'
    with open(filename, 'w') as f:
        json.dump(params, f)
    logging.info(f"Best hyperparameters for {model_name} saved to {filename}")


def load_hyperparameters(model_name):
    """
    Load hyperparameters from a JSON file.

    Args:
        model_name (str): Name of the model to load the hyperparameters for.

    Returns:
        dict or None: Loaded hyperparameters if the file exists, None otherwise.
    """
    filename = f'outputs/hyperparameters/best_hyperparameters_{model_name}.json'
    try:
        with open(filename, 'r') as f:
            params = json.load(f)
        logging.info(f"Loaded hyperparameters for {model_name} from {filename}")
        return params
    except FileNotFoundError:
        logging.info(f"No saved hyperparameters found for {model_name} at {filename}")
        return None
