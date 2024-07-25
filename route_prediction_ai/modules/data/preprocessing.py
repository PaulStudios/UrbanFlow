import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from sklearn.preprocessing import StandardScaler

from route_prediction_ai.modules.utilities import logging


def create_sequences(data, seq_length):
    """
    Create sequences of a specified length from the data.

    Args:
        data (ndarray): The input data.
        seq_length (int): The length of the sequences to create.

    Returns:
        ndarray: The created sequences.
    """
    logging.info("Creating sequences of length %d.", seq_length)
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)


def prepare_data(grouped, seq_length=30):
    """
    Prepare data by creating sequences and adding temporal features.

    Args:
        grouped (GroupBy): The grouped data.
        seq_length (int, optional): The length of the sequences to create. Defaults to 30.

    Returns:
        tuple: A tuple containing the input and output sequences.
    """
    logging.info("Preparing data.")
    X_groups, y_groups = [], []
    for _, group in grouped:
        if len(group) < seq_length:
            continue  # Skip this group if it's too small

        group = group.sort_values('timestamp')

        # Convert timestamp to Unix timestamp (seconds since epoch)
        group['timestamp_seconds'] = group['timestamp'].astype('int64') / 10 ** 9

        X = group[['longitude', 'latitude', 'timestamp_seconds', 'hour', 'day_of_week', 'is_weekend']].values
        y = group[['longitude', 'latitude']].values

        # Create sequences
        X_seq = create_sequences(X, seq_length)
        y_seq = create_sequences(y, seq_length)

        if len(X_seq) > 0 and len(y_seq) > 0:
            X_groups.append(X_seq)
            y_groups.append(y_seq)

    if not X_groups or not y_groups:
        raise ValueError("No valid sequences found. Ensure that the sequence length is appropriate for the data.")

    logging.info("Data preparation complete.")
    return X_groups, y_groups


def split_data(X_groups, y_groups, test_size=0.2):
    """
    Split data into training and testing sets.

    Args:
        X_groups (list): The input sequences.
        y_groups (list): The output sequences.
        test_size (float, optional): The proportion of the data to use as the test set. Defaults to 0.2.

    Returns:
        tuple: A tuple containing the training and testing sets.
    """
    logging.info("Splitting data into training and testing sets.")
    X_train_groups, X_test_groups, y_train_groups, y_test_groups = [], [], [], []
    for X, y in zip(X_groups, y_groups):
        if len(X) > 5:
            split_index = int(len(X) * (1 - test_size))
            X_train_groups.append(X[:split_index])
            X_test_groups.append(X[split_index:])
            y_train_groups.append(y[:split_index])
            y_test_groups.append(y[split_index:])
        else:
            X_train_groups.append(X)
            X_test_groups.append([])
            y_train_groups.append(y)
            y_test_groups.append([])
    logging.info("Data splitting complete.")
    return X_train_groups, X_test_groups, y_train_groups, y_test_groups


def apply_kalman_filter(X_train, X_test, Q_var=0.01, R_var=0.1):
    """
    Apply a Kalman filter to the training and testing data.

    Args:
        X_train (ndarray): The training data.
        X_test (ndarray): The testing data.
        Q_var (float, optional): The process noise variance. Defaults to 0.01.
        R_var (float, optional): The measurement noise variance. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the filtered training and testing data.
    """
    logging.info("Applying Kalman filter.")
    kf = KalmanFilter(dim_x=6, dim_z=3)
    dt = np.mean(X_train[:, 3])
    kf.F = np.array([
        [1, 0, dt, 0, 0.5 * dt ** 2, 0],
        [0, 1, 0, dt, 0, 0.5 * dt ** 2],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    kf.x = np.array([X_train[0, 0], X_train[0, 1], 0, 0, 0, 0])
    kf.P *= 1000
    kf.R = np.eye(3) * R_var
    kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=Q_var, block_size=2)

    predictions_train = []
    for x in X_train:
        kf.predict()
        kf.update(x[:3])
        predictions_train.append(kf.x[:2])

    predictions_test = []
    for x in X_test:
        kf.predict()
        kf.update(x[:3])
        predictions_test.append(kf.x[:2])

    if len(predictions_test) < 30:
        predictions_test.extend([predictions_test[-1]] * (30 - len(predictions_test)))
    else:
        predictions_test = predictions_test[:30]

    logging.info("Kalman filter application complete.")
    return np.array(predictions_train), np.array(predictions_test)


def create_padded_sequences(X_groups, y_groups, seq_length):
    """
    Create padded sequences of a specified length from the data.

    Args:
        X_groups (list): The input sequences.
        y_groups (list): The output sequences.
        seq_length (int): The length of the sequences to create.

    Returns:
        tuple: A tuple containing the padded input and output sequences.
    """
    logging.info("Creating padded sequences of length %d.", seq_length)
    X_sequences, y_sequences = [], []
    for X, y in zip(X_groups, y_groups):
        if len(X) >= seq_length and len(y) >= seq_length:
            X_seq = create_sequences(X, seq_length)
            y_seq = create_sequences(y, seq_length)
            if len(X_seq) > 0 and len(y_seq) > 0:
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)

    if not X_sequences or not y_sequences:
        raise ValueError("No valid sequences found. Ensure that the sequence length is appropriate for the data.")

    X_padded = np.vstack(X_sequences)
    y_padded = np.vstack(y_sequences)

    # Ensure the padded arrays have the same number of samples
    min_samples = min(X_padded.shape[0], y_padded.shape[0])
    X_padded = X_padded[:min_samples]
    y_padded = y_padded[:min_samples]

    logging.info("Padded sequences creation complete.")
    return X_padded, y_padded


def normalize_data(X_groups, y_groups):
    """
    Normalize the input and output data using StandardScaler.

    Args:
        X_groups (list): The input sequences.
        y_groups (list): The output sequences.

    Returns:
        tuple: A tuple containing the normalized input and output sequences,
               and the scalers used for normalization.
    """
    logging.info("Normalizing data.")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_normalized = []
    y_normalized = []

    for X_group, y_group in zip(X_groups, y_groups):
        if len(X_group) == 0 or len(y_group) == 0:
            continue  # Skip empty groups

        X_flat = np.vstack(X_group)
        y_flat = np.vstack(y_group)

        X_norm_flat = scaler_X.fit_transform(X_flat)
        y_norm_flat = scaler_y.fit_transform(y_flat)

        X_norm = X_norm_flat.reshape(X_group.shape)
        y_norm = y_norm_flat.reshape(y_group.shape)

        X_normalized.append(X_norm)
        y_normalized.append(y_norm)
    logging.info("Data normalization complete.")
    return X_normalized, y_normalized, scaler_X, scaler_y
