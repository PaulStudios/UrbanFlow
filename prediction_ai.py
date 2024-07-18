import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import tensorflow as tf
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, TimeDistributed
import matplotlib.pyplot as plt
import os
import concurrent.futures
from functools import partial
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

training_rounds = 5000


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


def build_bidirectional_lstm_model(input_shape):
    """
    Build and compile a bidirectional LSTM model.

    Parameters:
    input_shape (tuple): Shape of the input data.

    Returns:
    tf.keras.models.Sequential: Compiled bidirectional LSTM model.
    """
    logging.info("Building bidirectional LSTM model.")
    model = Sequential([
        Bidirectional(LSTM(64, activation='tanh', return_sequences=True), input_shape=input_shape),
        Bidirectional(LSTM(32, activation='tanh', return_sequences=True)),
        TimeDistributed(Dense(16, activation='relu')),
        TimeDistributed(Dense(2))
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=scaled_mse)
    logging.info("Bidirectional LSTM model built and compiled.")
    return model


def build_stacked_lstm_model(input_shape):
    """
    Build and compile a stacked LSTM model.

    Parameters:
    input_shape (tuple): Shape of the input data.

    Returns:
    tf.keras.models.Sequential: Compiled stacked LSTM model.
    """
    logging.info("Building stacked LSTM model.")
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape),
        LSTM(32, activation='tanh', return_sequences=True),
        LSTM(16, activation='tanh', return_sequences=True),
        TimeDistributed(Dense(8, activation='relu')),
        TimeDistributed(Dense(2))
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=scaled_mse)
    logging.info("Stacked LSTM model built and compiled.")
    return model


def train_model(model, train_dataset, val_dataset, model_name):
    """
    Train a single model.

    Parameters:
    model (tf.keras.Model): The model to train
    train_dataset (tf.data.Dataset): Training dataset
    val_dataset (tf.data.Dataset): Validation dataset
    model_name (str): Name of the model for saving

    Returns:
    tf.keras.Model: Trained model
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        train_dataset,
        epochs=training_rounds,
        validation_data=val_dataset,
        callbacks=[early_stopping],
        verbose=1
    )

    model.save(f'final_{model_name}_model.keras')
    return model


def model_averaging(models, X):
    """
    Perform model averaging on the given models.

    Parameters:
    models (list): List of trained models
    X (np.ndarray): Input data

    Returns:
    np.ndarray: Averaged predictions
    """
    predictions = [model.predict(X) for model in models]
    return np.mean(predictions, axis=0)


def weighted_average_predictions(lstm_preds, bilstm_preds, lstm_weight=0.4, bilstm_weight=0.6):
    return lstm_weight * lstm_preds + bilstm_weight * bilstm_preds


def train_stacking_model(lstm_preds, bilstm_preds, y_true):
    # Reshape the predictions
    n_samples, n_timesteps, n_features = lstm_preds.shape
    lstm_preds_reshaped = lstm_preds.reshape(n_samples * n_timesteps, n_features)
    bilstm_preds_reshaped = bilstm_preds.reshape(n_samples * n_timesteps, n_features)

    # Combine the predictions
    X_meta = np.hstack([lstm_preds_reshaped, bilstm_preds_reshaped])

    # Reshape the true values
    y_true_reshaped = y_true.reshape(n_samples * n_timesteps, n_features)

    meta_learner = LinearRegression()
    meta_learner.fit(X_meta, y_true_reshaped)
    return meta_learner

# For making predictions:
def stacked_predict(stacking_model, lstm_preds, bilstm_preds):
    n_samples, n_timesteps, n_features = lstm_preds.shape
    lstm_preds_reshaped = lstm_preds.reshape(n_samples * n_timesteps, n_features)
    bilstm_preds_reshaped = bilstm_preds.reshape(n_samples * n_timesteps, n_features)
    X_meta = np.hstack([lstm_preds_reshaped, bilstm_preds_reshaped])
    stacked_preds = stacking_model.predict(X_meta)
    return stacked_preds.reshape(n_samples, n_timesteps, n_features)


class SequenceEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        return self  # LSTM models are already trained

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return np.mean(predictions, axis=0)


def evaluate_model_cv(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv)
    print(f'Cross-validated scores: {scores}')
    print(f'Mean score: {scores.mean()}, Std: {scores.std()}')


def train_regularized_stacking_model(lstm_preds, bilstm_preds, y_true, alpha=1.0):
    n_samples, n_timesteps, n_features = lstm_preds.shape
    lstm_preds_reshaped = lstm_preds.reshape(n_samples * n_timesteps, n_features)
    bilstm_preds_reshaped = bilstm_preds.reshape(n_samples * n_timesteps, n_features)
    X_meta = np.hstack([lstm_preds_reshaped, bilstm_preds_reshaped])
    y_true_reshaped = y_true.reshape(n_samples * n_timesteps, n_features)

    meta_learner = Ridge(alpha=alpha)
    meta_learner.fit(X_meta, y_true_reshaped)
    return meta_learner

def regularized_stacked_predict(reg_stacking_model, lstm_preds, bilstm_preds):
    n_samples, n_timesteps, n_features = lstm_preds.shape
    lstm_preds_reshaped = lstm_preds.reshape(n_samples * n_timesteps, n_features)
    bilstm_preds_reshaped = bilstm_preds.reshape(n_samples * n_timesteps, n_features)
    X_meta = np.hstack([lstm_preds_reshaped, bilstm_preds_reshaped])
    reg_stacked_preds = reg_stacking_model.predict(X_meta)
    return reg_stacked_preds.reshape(n_samples, n_timesteps, n_features)


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


def evaluate_models(actual_groups, lstm_groups, bilstm_groups, weighted_avg_groups, stacked_groups, ensemble_groups,
                    regularized_groups):
    """
    Evaluate the performance of the models.

    Parameters:
    actual_groups (list): List of actual target arrays for each group.
    lstm_groups (list): List of LSTM predictions for each group.
    bilstm_groups (list): List of Bidirectional LSTM predictions for each group.
    weighted_avg_groups (list): List of Weighted Average predictions for each group.
    stacked_groups (list): List of Stacked model predictions for each group.
    ensemble_groups (list): List of Ensemble model predictions for each group.
    regularized_groups (list): List of Regularized Stacking model predictions for each group.
    """
    logging.info("Evaluating models.")
    lstm_mae, bilstm_mae, weighted_avg_mae, stacked_mae, ensemble_mae, regularized_mae = [], [], [], [], [], []

    for actual, lstm, bilstm, weighted_avg, stacked, ensemble, regularized in zip(
            actual_groups, lstm_groups, bilstm_groups, weighted_avg_groups,
            stacked_groups, ensemble_groups, regularized_groups):
        min_length = min(len(actual), len(lstm), len(bilstm), len(weighted_avg),
                         len(stacked), len(ensemble), len(regularized))

        actual = actual[:min_length]
        lstm = lstm[:min_length]
        bilstm = bilstm[:min_length]
        weighted_avg = weighted_avg[:min_length]
        stacked = stacked[:min_length]
        ensemble = ensemble[:min_length]
        regularized = regularized[:min_length]

        lstm_mae.append(mean_absolute_error(actual, lstm))
        bilstm_mae.append(mean_absolute_error(actual, bilstm))
        weighted_avg_mae.append(mean_absolute_error(actual, weighted_avg))
        stacked_mae.append(mean_absolute_error(actual, stacked))
        ensemble_mae.append(mean_absolute_error(actual, ensemble))
        regularized_mae.append(mean_absolute_error(actual, regularized))

    logging.info(f"Average LSTM MAE: {np.mean(lstm_mae)}")
    logging.info(f"Average Bidirectional LSTM MAE: {np.mean(bilstm_mae)}")
    logging.info(f"Average Weighted Average MAE: {np.mean(weighted_avg_mae)}")
    logging.info(f"Average Stacked Model MAE: {np.mean(stacked_mae)}")
    logging.info(f"Average Ensemble Model MAE: {np.mean(ensemble_mae)}")
    logging.info(f"Average Regularized Stacking Model MAE: {np.mean(regularized_mae)}")


def plot_predictions(group_index, actual, lstm, bilstm, weighted_avg, stacked, ensemble, regularized):
    """
    Plot the predictions of the models.

    Parameters:
    group_index (int): Index of the group to plot.
    actual (np.ndarray): Actual target array.
    lstm (np.ndarray): LSTM predictions.
    bilstm (np.ndarray): Bidirectional LSTM predictions.
    weighted_avg (np.ndarray): Weighted Average predictions.
    stacked (np.ndarray): Stacked model predictions.
    ensemble (np.ndarray): Ensemble model predictions.
    regularized (np.ndarray): Regularized Stacking model predictions.
    """
    logging.info(f"Plotting predictions for group {group_index}.")
    min_length = min(len(actual), len(lstm), len(bilstm), len(weighted_avg),
                     len(stacked), len(ensemble), len(regularized))

    actual = actual[:min_length]
    lstm = lstm[:min_length]
    bilstm = bilstm[:min_length]
    weighted_avg = weighted_avg[:min_length]
    stacked = stacked[:min_length]
    ensemble = ensemble[:min_length]
    regularized = regularized[:min_length]

    plt.figure(figsize=(12, 8))
    plt.plot(actual[:, 0], actual[:, 1], marker='o', markersize=6, linestyle='-', label='Actual Position', color='blue')
    plt.plot(lstm[:, 0], lstm[:, 1], marker='s', markersize=4, linestyle=':', label='LSTM Prediction', color='green')
    plt.plot(bilstm[:, 0], bilstm[:, 1], marker='^', markersize=4, linestyle='-.',
             label='Bidirectional LSTM Prediction', color='purple')
    plt.plot(weighted_avg[:, 0], weighted_avg[:, 1], marker='*', markersize=6, linestyle=':',
             label='Weighted Average Prediction', color='cyan')
    plt.plot(stacked[:, 0], stacked[:, 1], marker='D', markersize=4, linestyle='--',
             label='Stacked Model Prediction', color='orange')
    plt.plot(ensemble[:, 0], ensemble[:, 1], marker='x', markersize=6, linestyle='-',
             label='Ensemble Model Prediction', color='red')
    plt.plot(regularized[:, 0], regularized[:, 1], marker='p', markersize=6, linestyle='-.',
             label='Regularized Stacking Prediction', color='magenta')

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

    # Prepare datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_padded, y_train_padded))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_test_padded, y_test_padded))
    val_dataset = val_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    # Build models
    lstm_model = build_lstm_model((X_train_padded.shape[1], X_train_padded.shape[2]))
    bilstm_model = build_bidirectional_lstm_model((X_train_padded.shape[1], X_train_padded.shape[2]))
    stacked_lstm_model = build_stacked_lstm_model((X_train_padded.shape[1], X_train_padded.shape[2]))

    # Prepare arguments for concurrent training
    models = [lstm_model, bilstm_model, stacked_lstm_model]
    model_names = ['lstm', 'bilstm', 'stacked_lstm']

    # Train models concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        trained_models = list(executor.map(
            lambda m, n: train_model(m, train_dataset, val_dataset, n),
            models, model_names
        ))

    # Make individual predictions
    predictions = [model.predict(X_test_padded) for model in trained_models]
    lstm_predictions, bilstm_predictions, stacked_lstm_predictions = predictions

    # Perform model averaging
    averaged_predictions = model_averaging(trained_models, X_test_padded)
    weighted_avg_predictions = weighted_average_predictions(lstm_predictions, bilstm_predictions)
    stacking_model = train_stacking_model(lstm_predictions, bilstm_predictions, y_test_padded)
    stacked_predictions = stacked_predict(stacking_model, lstm_predictions, bilstm_predictions)
    ensemble_model = SequenceEnsemble([lstm_model, bilstm_model])
    ensemble_predictions = ensemble_model.predict(X_test_padded)
    regularized_stacking_model = train_regularized_stacking_model(lstm_predictions, bilstm_predictions, y_test_padded)
    regularized_predictions = regularized_stacked_predict(regularized_stacking_model, lstm_predictions,
                                                          bilstm_predictions)

    # Evaluate the models
    evaluate_models(y_test_padded, lstm_predictions, bilstm_predictions,
                    weighted_avg_predictions, stacked_predictions,
                    ensemble_predictions, regularized_predictions)

    # Plot predictions for a few groups
    for i in range(min(5, len(y_test_padded))):
        if len(y_test_padded[i]) > 0:
            plot_predictions(i, y_test_padded[i], lstm_predictions[i],
                             bilstm_predictions[i], weighted_avg_predictions[i],
                             stacked_predictions[i], ensemble_predictions[i],
                             regularized_predictions[i])

    # Save the final models
    for model, name in zip(trained_models, model_names):
        model.save(f'final_{name}_model.keras')

    logging.info("Training and evaluation complete.")


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
