import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from route_prediction_ai.modules.utilities import logging


def add_temporal_features(data):
    """
    Add temporal features to the data.

    Args:
        data (DataFrame): The input data with a 'timestamp' column.

    Returns:
        DataFrame: The data with additional temporal features.
    """
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    return data


def add_velocity_acceleration(X):
    """
    Add velocity and acceleration features to the data.

    Args:
        X (ndarray): The input data.

    Returns:
        ndarray: The data with additional velocity and acceleration features.
    """
    logging.info("Adding velocity and acceleration features.")
    if len(X) < 3:
        velocity = np.zeros((len(X), 2))
        acceleration = np.zeros((len(X), 2))
    else:
        velocity = np.diff(X[:, :2], axis=0)
        acceleration = np.diff(velocity, axis=0)
        velocity = np.vstack([np.zeros((1, 2)), velocity])
        acceleration = np.vstack([np.zeros((2, 2)), acceleration])
    return np.hstack([X, velocity, acceleration])


def add_domain_specific_features(X):
    """
    Add domain-specific features to the dataset.

    Args:
        X (np.ndarray): Input data.

    Returns:
        np.ndarray: Dataset with added domain-specific features.
    """
    speed = np.sqrt(np.sum(np.diff(X[:, :, :2], axis=1) ** 2, axis=2))
    acceleration = np.diff(speed, axis=1)
    return np.dstack([X, speed[:, :, np.newaxis], acceleration[:, :, np.newaxis]])


def add_polynomial_features(X, degree=2):
    """
    Add polynomial features to the dataset.

    Args:
        X (np.ndarray): Input data.
        degree (int): Degree of polynomial features.

    Returns:
        np.ndarray: Dataset with added polynomial features.
    """
    poly = PolynomialFeatures(degree, include_bias=False)
    return poly.fit_transform(X)


def add_interaction_terms(X):
    """
    Add interaction terms to the dataset.

    Args:
        X (np.ndarray): Input data.

    Returns:
        np.ndarray: Dataset with added interaction terms.
    """
    return np.column_stack([X, X[:, :, 0] * X[:, :, 1], X[:, :, 0] * X[:, :, 2], X[:, :, 1] * X[:, :, 2]])
