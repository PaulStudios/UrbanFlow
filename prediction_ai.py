import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import concurrent.futures
import logging
import download_data

download_data.run()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_data(csv_file):
    """
    Load and preprocess data from a CSV file.

    Parameters:
    csv_file (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Preprocessed data.
    """
    logging.info("Loading data from CSV file.")
    data = pd.read_csv(csv_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(['upload_id', 'timestamp'])
    logging.info("Data loaded and sorted.")
    return data

def add_velocity_acceleration(X):
    """
    Add velocity and acceleration features to the data.

    Parameters:
    X (np.ndarray): Input data array with columns [longitude, latitude, timestamp_seconds].

    Returns:
    np.ndarray: Data array with added velocity and acceleration features.
    """
    logging.info("Adding velocity and acceleration features.")
    if len(X) < 3:
        # Handle cases with insufficient data points
        velocity = np.zeros((len(X), 2))
        acceleration = np.zeros((len(X), 2))
    else:
        velocity = np.diff(X[:, :2], axis=0)
        acceleration = np.diff(velocity, axis=0)
        velocity = np.vstack([np.zeros((1, 2)), velocity])
        acceleration = np.vstack([np.zeros((2, 2)), acceleration])
    return np.hstack([X, velocity, acceleration])

def prepare_data(grouped):
    """
    Prepare data by grouping and adding features.

    Parameters:
    grouped (pd.core.groupby.generic.DataFrameGroupBy): Grouped DataFrame.

    Returns:
    list: List of feature arrays for each group.
    list: List of target arrays for each group.
    """
    logging.info("Preparing data.")
    X_groups, y_groups = [], []
    for _, group in grouped:
        group = group.sort_values('timestamp')
        group['timestamp_seconds'] = group['timestamp'].astype('int64') / 10 ** 9
        X = group[['longitude', 'latitude', 'timestamp_seconds']].values
        y = group[['longitude', 'latitude']].values
        time_diff = group['timestamp_seconds'].diff().fillna(0).values.reshape(-1, 1)
        X = np.column_stack((X, time_diff))
        X = add_velocity_acceleration(X)
        X_groups.append(X)
        y_groups.append(y)
    logging.info("Data preparation complete.")
    return X_groups, y_groups

def create_sequences(data, seq_length):
    """
    Create sequences of a given length from the data.

    Parameters:
    data (np.ndarray): Input data array.
    seq_length (int): Length of sequences.

    Returns:
    np.ndarray: Array of sequences.
    """
    logging.info("Creating sequences of length %d.", seq_length)
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

def split_data(X_groups, y_groups, test_size=0.2):
    """
    Split data into training and testing sets.

    Parameters:
    X_groups (list): List of feature arrays for each group.
    y_groups (list): List of target arrays for each group.
    test_size (float): Proportion of the data to use for testing.

    Returns:
    list: List of training feature arrays.
    list: List of testing feature arrays.
    list: List of training target arrays.
    list: List of testing target arrays.
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
    Apply Kalman filter to the data.

    Parameters:
    X_train (np.ndarray): Training feature array.
    X_test (np.ndarray): Testing feature array.
    Q_var (float): Process noise covariance.
    R_var (float): Measurement noise covariance.

    Returns:
    np.ndarray: Kalman filter predictions for training data.
    np.ndarray: Kalman filter predictions for testing data.
    """
    logging.info("Applying Kalman filter.")
    kf = KalmanFilter(dim_x=6, dim_z=3)
    dt = np.mean(X_train[:, 3])
    # State transition matrix
    kf.F = np.array([
        [1, 0, dt, 0, 0.5 * dt ** 2, 0],
        [0, 1, 0, dt, 0, 0.5 * dt ** 2],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    # Measurement function
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    # Initial state estimate
    kf.x = np.array([X_train[0, 0], X_train[0, 1], 0, 0, 0, 0])
    kf.P *= 1000  # Initial state covariance
    kf.R = np.eye(3) * R_var  # Measurement noise covariance
    kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=Q_var, block_size=2)  # Process noise covariance

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
    Create padded sequences for LSTM training.

    Parameters:
    X_groups (list): List of feature arrays for each group.
    y_groups (list): List of target arrays for each group.
    seq_length (int): Length of sequences.

    Returns:
    np.ndarray: Padded feature sequences.
    np.ndarray: Padded target sequences.
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

    min_samples = min(X_padded.shape[0], y_padded.shape[0])
    X_padded = X_padded[:min_samples]
    y_padded = y_padded[:min_samples]

    logging.info("Padded sequences creation complete.")
    return X_padded, y_padded

def normalize_data(X_groups, y_groups):
    """
    Normalize the data.

    Parameters:
    X_groups (list): List of feature arrays for each group.
    y_groups (list): List of target arrays for each group.

    Returns:
    list: List of normalized feature arrays.
    list: List of normalized target arrays.
    StandardScaler: Scaler for features.
    StandardScaler: Scaler for targets.
    """
    logging.info("Normalizing data.")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_normalized = [scaler_X.fit_transform(X) for X in X_groups]
    y_normalized = [scaler_y.fit_transform(y) for y in y_groups]
    logging.info("Data normalization complete.")
    return X_normalized, y_normalized, scaler_X, scaler_y

def build_lstm_model(input_shape):
    """
    Build and compile the LSTM model.

    Parameters:
    input_shape (tuple): Shape of the input data.

    Returns:
    tf.keras.models.Sequential: Compiled LSTM model.
    """
    logging.info("Building LSTM model.")
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape),
        LSTM(32, activation='tanh', return_sequences=True),
        TimeDistributed(Dense(16, activation='relu')),
        TimeDistributed(Dense(2))
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=scaled_mse)
    logging.info("LSTM model built and compiled.")
    return model

def scaled_mse(y_true, y_pred):
    """
    Custom scaled mean squared error loss function.

    Parameters:
    y_true (tf.Tensor): True values.
    y_pred (tf.Tensor): Predicted values.

    Returns:
    tf.Tensor: Scaled mean squared error.
    """
    return tf.reduce_mean(tf.square((y_true - y_pred) * tf.constant([0.0001, 0.0001])))

def evaluate_models(actual_groups, kalman_groups, lstm_groups):
    """
    Evaluate the performance of the models.

    Parameters:
    actual_groups (list): List of actual target arrays for each group.
    kalman_groups (list): List of Kalman filter predictions for each group.
    lstm_groups (list): List of LSTM predictions for each group.
    """
    logging.info("Evaluating models.")
    kalman_mae, lstm_mae = [], []
    for actual, kalman, lstm in zip(actual_groups, kalman_groups, lstm_groups):
        min_length = min(len(actual), len(kalman), len(lstm))
        actual = actual[:min_length]
        kalman = kalman[:min_length]
        lstm = lstm[:min_length]
        kalman_mae.append(mean_absolute_error(actual, kalman))
        lstm_mae.append(mean_absolute_error(actual, lstm))
    logging.info(f"Average Kalman Filter MAE: {np.mean(kalman_mae)}")
    logging.info(f"Average LSTM MAE: {np.mean(lstm_mae)}")

def plot_predictions(group_index, actual, kalman, lstm):
    """
    Plot the predictions of the models.

    Parameters:
    group_index (int): Index of the group to plot.
    actual (np.ndarray): Actual target array.
    kalman (np.ndarray): Kalman filter predictions.
    lstm (np.ndarray): LSTM predictions.
    """
    logging.info(f"Plotting predictions for group {group_index}.")
    min_length = min(len(actual), len(kalman), len(lstm))
    actual = actual[:min_length]
    kalman = kalman[:min_length]
    lstm = lstm[:min_length]

    plt.figure(figsize=(12, 8))
    plt.plot(actual[:, 0], actual[:, 1], marker='o', markersize=6, linestyle='-', label='Actual Position', color='blue')
    plt.plot(kalman[:, 0], kalman[:, 1], marker='x', markersize=8, linestyle='--', label='Kalman Filter Prediction', color='red')
    plt.plot(lstm[:, 0], lstm[:, 1], marker='s', markersize=4, linestyle=':', label='LSTM Prediction', color='green')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Vehicle Route Prediction - Group {group_index}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    logging.info(f"Prediction plot for group {group_index} complete.")

def main(csv_file):
    """
    Main function to load data, train models, and evaluate predictions.

    Parameters:
    csv_file (str): Path to the CSV file.
    """
    # Load and prepare the data
    data = load_data(csv_file)
    grouped = data.groupby('upload_id')
    X_groups, y_groups = prepare_data(grouped)
    X_train_groups, X_test_groups, y_train_groups, y_test_groups = split_data(X_groups, y_groups)

    # Normalize the data
    X_train_groups, y_train_groups, scaler_X, scaler_y = normalize_data(X_train_groups, y_train_groups)
    X_test_groups, y_test_groups, _, _ = normalize_data(X_test_groups, y_test_groups)

    # Apply Kalman filter
    with concurrent.futures.ThreadPoolExecutor() as executor:
        kalman_results = list(executor.map(apply_kalman_filter, X_train_groups, X_test_groups))
    kalman_train_groups, kalman_test_groups = zip(*kalman_results)

    # Sequence length for LSTM
    seq_length = 30

    # Create padded sequences for LSTM
    X_train_padded, y_train_padded = create_padded_sequences(X_train_groups, y_train_groups, seq_length)
    X_test_padded, y_test_padded = create_padded_sequences(X_test_groups, y_test_groups, seq_length)

    X_train_padded = X_train_padded.reshape((-1, seq_length, X_train_padded.shape[-1]))
    X_test_padded = X_test_padded.reshape((-1, seq_length, X_test_padded.shape[-1]))
    y_train_padded = y_train_padded.reshape((-1, seq_length, y_train_padded.shape[-1]))
    y_test_padded = y_test_padded.reshape((-1, seq_length, y_test_padded.shape[-1]))

    # Build and train LSTM model
    lstm_model = build_lstm_model((X_train_padded.shape[1], X_train_padded.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint_path = 'checkpoints/epoch-{epoch:02d}.keras'
    checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=False, save_freq='epoch')

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_padded, y_train_padded))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_test_padded, y_test_padded))
    val_dataset = val_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    latest_checkpoint = tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))
    if (latest_checkpoint):
        lstm_model.load_weights(latest_checkpoint)

    history = lstm_model.fit(
        train_dataset,
        epochs=5000,
        validation_data=val_dataset,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    lstm_model.save('final_lstm_model.keras')

    lstm_predictions_groups = lstm_model.predict(X_test_padded)

    # Evaluate the models
    evaluate_models(y_test_padded, kalman_test_groups, lstm_predictions_groups)

    # Plot the predictions
    for i in range(min(5, len(y_test_padded))):
        if len(y_test_padded[i]) > 0:
            plot_predictions(i, y_test_padded[i], kalman_test_groups[i], lstm_predictions_groups[i])

def load_and_predict(model_path, X_data):
    """
    Load a pre-trained model and make predictions.

    Parameters:
    model_path (str): Path to the saved model.
    X_data (np.ndarray): Input data array for prediction.

    Returns:
    np.ndarray: Predictions made by the model.
    """
    logging.info(f"Loading model from {model_path} and making predictions.")
    model = load_model(model_path, custom_objects={'scaled_mse': scaled_mse})
    predictions = model.predict(X_data)
    logging.info("Predictions made.")
    return predictions

if __name__ == "__main__":
    csv_file = 'final_data.csv'
    main(csv_file)

    # Example test data and predictions
    test_data = np.random.rand(1, 30, 8)
    predictions = load_and_predict('final_lstm_model.keras', test_data)
