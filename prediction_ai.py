"""
Author: Indradip Paul [HilFing]
Purpose: This file is part of the UrbanFlow project, designed to manage traffic lights to prioritize emergency vehicles and improve overall traffic flow.
         Specifically, this file focuses on the implementation and evaluation of various machine learning models for predicting vehicle routes.
Release Date: 23-07-2024

Additional Information:
- This project leverages both traditional machine learning models and advanced deep learning architectures to provide robust predictions.
- The models evaluated include Random Forest, XGBoost, Ensemble, GRU, CNN-LSTM, Stacked LSTM, Optimized LSTM, BiLSTM, SVM, k-NN, Regularized Stacking, and Weighted Average.
- Performance metrics used for evaluation include Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Coefficient of Determination (R²), Explained Variance Score (EVS), and Mean Absolute Percentage Error (MAPE).
- The goal is to identify the most accurate and reliable models for real-time traffic management and route prediction.
"""

import concurrent.futures
import gc
import json
import logging
import os
import platform
import signal
import threading
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from optuna.samplers import TPESampler
from prophet import Prophet
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, TimeDistributed, Dropout, GRU, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from xgboost import XGBRegressor

import download_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Global variables
optimization_lock = threading.Lock()
training_rounds = 200
hyperparamemter_rounds = 100

def create_directory(directory):
    """
    Create a directory if it does not already exist.

    Args:
        directory (str): The path of the directory to create.
    """
    if not os.path.exists(directory):
        Path(directory).mkdir(parents=True, exist_ok=True)

def timeout_handler(signum, frame):
    """
    Signal handler for function call timeouts.

    Args:
        signum: The signal number.
        frame: The current stack frame.
    """
    raise TimeoutError("Function call timed out")

def run_with_timeout(func, args=(), kwargs={}, timeout_duration=300):
    """
    Run a function with a timeout. If the function does not complete within the specified
    duration, a TimeoutError is raised.

    Args:
        func (callable): The function to run.
        args (tuple): Positional arguments to pass to the function.
        kwargs (dict): Keyword arguments to pass to the function.
        timeout_duration (int): The maximum time (in seconds) to allow the function to run.

    Returns:
        The result of the function call.
    """
    if platform.system() != 'Windows':
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_duration)
        try:
            result = func(*args, **kwargs)
        finally:
            signal.alarm(0)
    else:
        timer = threading.Timer(timeout_duration, timeout_handler, args=(None, None))
        timer.start()
        try:
            result = func(*args, **kwargs)
        finally:
            timer.cancel()
    return result

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


class ProgressMonitor(tf.keras.callbacks.Callback):
    """
    Custom Keras callback to monitor training progress and implement early stopping.

    Args:
        patience (int): Number of epochs to wait for improvement before stopping training.
        min_delta (float): Minimum change to qualify as an improvement.
    """

    def __init__(self, patience=5, min_delta=1e-5):
        super(ProgressMonitor, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.no_improvement_count = 0

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to check for improvement.

        Args:
            epoch (int): The current epoch number.
            logs (dict): The logs from the current epoch.
        """
        current_loss = logs.get('val_loss')
        if current_loss is None:
            return

        if (self.best_loss - current_loss) > self.min_delta:
            self.best_loss = current_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            logging.info(f"Stopping training due to lack of improvement after {self.patience} epochs")
            self.model.stop_training = True

    def on_train_batch_end(self, batch, logs=None):
        """
        Called at the end of each training batch to log progress.

        Args:
            batch (int): The current batch number.
            logs (dict): The logs from the current batch.
        """
        if batch % 10 == 0:  # Log every 10 batches
            logging.info(f"Batch {batch}: loss = {logs['loss']:.6f}")


def build_lstm_model(input_shape):
    """
    Build an LSTM model for time series prediction.

    Args:
        input_shape (tuple): The shape of the input data.

    Returns:
        Model: The compiled LSTM model.
    """
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        LSTM(32, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        TimeDistributed(Dense(16, activation='relu')),
        TimeDistributed(Dense(2))
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=scaled_mse)
    return model


def build_bidirectional_lstm_model(input_shape):
    """
    Build a Bidirectional LSTM model for time series prediction.

    Args:
        input_shape (tuple): The shape of the input data.

    Returns:
        Model: The compiled Bidirectional LSTM model.
    """
    model = Sequential([
        Bidirectional(LSTM(64, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.01)),
                      input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(32, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.01))),
        Dropout(0.2),
        TimeDistributed(Dense(16, activation='relu')),
        TimeDistributed(Dense(2))
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=scaled_mse)
    return model


def build_stacked_lstm_model(input_shape):
    """
    Build a stacked LSTM model for time series prediction.

    Args:
        input_shape (tuple): The shape of the input data.

    Returns:
        Model: The compiled stacked LSTM model.
    """
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape),
        LSTM(32, activation='tanh', return_sequences=True),
        LSTM(16, activation='tanh', return_sequences=True),
        TimeDistributed(Dense(8, activation='relu')),
        TimeDistributed(Dense(2))
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=scaled_mse)
    return model


def build_cnn_lstm_model(input_shape):
    """
    Build a CNN-LSTM model for time series prediction.

    Args:
        input_shape (tuple): The shape of the input data.

    Returns:
        Model: The compiled CNN-LSTM model.
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        LSTM(50, return_sequences=True),
        LSTM(50, return_sequences=True),
        TimeDistributed(Dense(16, activation='relu')),
        TimeDistributed(Dense(2))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def build_gru_model(input_shape):
    """
    Build a GRU model for time series prediction.

    Args:
        input_shape (tuple): The shape of the input data.

    Returns:
        Model: The compiled GRU model.
    """
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        GRU(32, return_sequences=True),
        TimeDistributed(Dense(2))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def lr_schedule(epoch):
    """
    Learning rate schedule function.

    Args:
        epoch (int): The current epoch number.

    Returns:
        float: The learning rate for the given epoch.
    """
    initial_lr = 0.001
    if epoch < 10:
        return initial_lr
    else:
        return float(initial_lr * tf.math.exp(0.1 * (10 - epoch)))


def train_model(model, train_dataset, val_dataset, model_name):
    """
    Train the model with early stopping, learning rate scheduling, and progress monitoring.

    Args:
        model (tf.keras.Model): The model to be trained.
        train_dataset (tf.data.Dataset): The training dataset.
        val_dataset (tf.data.Dataset): The validation dataset.
        model_name (str): The name of the model.

    Returns:
        tf.keras.Model: The trained model, or None if training fails.
    """
    # Check model output shape
    for x, y in train_dataset.take(1):
        output = model(x)
        if output.shape[1:] != y.shape[1:]:
            raise ValueError(f"Model {model_name} output shape {output.shape} does not match target shape {y.shape}")

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, min_delta=1e-5)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()
    progress_monitor = ProgressMonitor(patience=5, min_delta=1e-5)

    try:
        def train_with_timeout():
            history = model.fit(
                train_dataset,
                epochs=training_rounds,
                validation_data=val_dataset,
                callbacks=[early_stopping, lr_scheduler, progress_monitor, terminate_on_nan],
                verbose=1
            )
            return history

        history = run_with_timeout(train_with_timeout, timeout_duration=7200)  # 2 hour timeout

        # Save training history to a JSON file
        with open(f'outputs/train_history/{model_name}_history.json', 'w') as f:
            json.dump(history.history, f)

        return model
    except TimeoutError:
        logging.error(f"Training for {model_name} timed out")
        return None
    except Exception as e:
        logging.error(f"Error during training of {model_name}: {str(e)}")
        return None


def bagged_lstm_predict(X, y, n_models=5):
    """
    Perform prediction using bagged LSTM models.

    Args:
        X (np.array): The input data.
        y (np.array): The target data.
        n_models (int): The number of LSTM models to use.

    Returns:
        np.array: The average predictions of the models.
    """
    predictions = []
    for _ in range(n_models):
        model = build_lstm_model(X.shape[1:])
        model.fit(X, y, epochs=training_rounds, verbose=0)
        predictions.append(model.predict(X))
    return np.mean(predictions, axis=0)


def model_averaging(models, X):
    """
    Perform model averaging on predictions.

    Args:
        models (list): The list of models to use for prediction.
        X (np.array): The input data.

    Returns:
        np.array: The average predictions of the models.
    """
    predictions = [model.predict(X) for model in models]
    return np.mean(predictions, axis=0)


def weighted_average_predictions(rf_preds, stack_preds, xgb_preds, gru_preds):
    """
    Calculate weighted average predictions based on inverse mean squared error (MSE).

    Args:
        rf_preds (np.array): Predictions from the random forest model.
        stack_preds (np.array): Predictions from the stacked model.
        xgb_preds (np.array): Predictions from the XGBoost model.
        gru_preds (np.array): Predictions from the GRU model.

    Returns:
        np.array: The weighted average predictions.
    """
    # Model MSE values
    mse_rf = 0.000935
    mse_stack = 0.000991
    mse_xgb = 0.005670
    mse_gru = 0.004105

    # Calculate inverse MSE
    inv_mse_rf = 1 / mse_rf
    inv_mse_stack = 1 / mse_stack
    inv_mse_xgb = 1 / mse_xgb
    inv_mse_gru = 1 / mse_gru

    # Calculate the sum of inverse MSEs
    total_inv_mse = inv_mse_rf + inv_mse_stack + inv_mse_xgb + inv_mse_gru

    # Normalize to get weights
    weight_rf = inv_mse_rf / total_inv_mse
    weight_stack = inv_mse_stack / total_inv_mse
    weight_xgb = inv_mse_xgb / total_inv_mse
    weight_gru = inv_mse_gru / total_inv_mse

    # Calculate weighted average predictions
    weighted_avg_preds = (
            weight_rf * rf_preds +
            weight_stack * stack_preds +
            weight_xgb * xgb_preds +
            weight_gru * gru_preds
    )

    return weighted_avg_preds


def train_stacking_model(lstm_preds, bilstm_preds, y_true):
    """
    Train a stacking model using LSTM and BiLSTM predictions.

    Args:
        lstm_preds (np.array): Predictions from the LSTM model.
        bilstm_preds (np.array): Predictions from the BiLSTM model.
        y_true (np.array): The true target values.

    Returns:
        tuple: The trained meta-learner, meta features, and reshaped true values.
    """
    n_samples, n_timesteps, n_features = lstm_preds.shape
    lstm_preds_reshaped = lstm_preds.reshape(n_samples * n_timesteps, n_features)
    bilstm_preds_reshaped = bilstm_preds.reshape(n_samples * n_timesteps, n_features)
    X_meta = np.hstack([lstm_preds_reshaped, bilstm_preds_reshaped])
    y_true_reshaped = y_true.reshape(n_samples * n_timesteps, n_features)
    meta_learner = LinearRegression()
    meta_learner.fit(X_meta, y_true_reshaped)
    return meta_learner, X_meta, y_true_reshaped


def stacked_predict(stacking_model, lstm_preds, bilstm_preds):
    """
    Make predictions using a trained stacking model.

    Args:
        stacking_model (sklearn.base.BaseEstimator): The trained stacking model.
        lstm_preds (np.array): Predictions from the LSTM model.
        bilstm_preds (np.array): Predictions from the BiLSTM model.

    Returns:
        np.array: The stacked predictions.
    """
    n_samples, n_timesteps, n_features = lstm_preds.shape
    lstm_preds_reshaped = lstm_preds.reshape(n_samples * n_timesteps, n_features)
    bilstm_preds_reshaped = bilstm_preds.reshape(n_samples * n_timesteps, n_features)
    X_meta = np.hstack([lstm_preds_reshaped, bilstm_preds_reshaped])
    stacked_preds = stacking_model.predict(X_meta)
    return stacked_preds.reshape(n_samples, n_timesteps, n_features)


class SequenceEnsemble(BaseEstimator, RegressorMixin):
    """
    Ensemble model for sequence predictions.

    Args:
        models (list): List of models to be used in the ensemble.
        weights (list): List of weights for each model. If None, equal weights are used.
    """

    def __init__(self, models, weights=None):
        self.models = models
        if weights is None:
            self.weights = [1 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models.")
            self.weights = weights

    def fit(self, X, y):
        """
        Fit the ensemble model to the training data.

        Args:
            X (np.array): The input data.
            y (np.array): The target data.

        Returns:
            SequenceEnsemble: The fitted ensemble model.
        """
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict using the ensemble model.

        Args:
            X (np.array): The input data.

        Returns:
            np.array: The ensemble predictions.
        """
        predictions = []
        for model in self.models:
            if isinstance(model, (MultiOutputRegressor)):
                # Reshape 3D input to 2D for non-sequential models
                X_2d = X.reshape(-1, X.shape[-1])
                pred = model.predict(X_2d)
                # Reshape back to 3D
                pred = pred.reshape(X.shape[0], X.shape[1], -1)
            else:
                pred = model.predict(X)

            predictions.append(pred)

        # Ensure all predictions have the same shape before averaging
        shapes = [p.shape for p in predictions]
        if len(set(shapes)) > 1:
            raise ValueError(f"Inconsistent prediction shapes: {shapes}")

        weighted_preds = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_preds += weight * pred

        return weighted_preds


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


class SequenceStackingRegressor(BaseEstimator, RegressorMixin):
    """
    A stacking regressor that stacks sequence models.

    Args:
        estimators (list): List of (name, estimator) tuples for base estimators.
        final_estimator (object): Estimator used to combine the base estimators.
        cv (int): Number of cross-validation folds.

    Methods:
        fit(X, y): Fit the model to the data.
        predict(X): Predict using the fitted model.
    """
    def __init__(self, estimators, final_estimator, cv=3):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv

    def fit(self, X, y):
        """
        Fit the stacking model to the data.

        Args:
            X (array-like): Training data.
            y (array-like): Target values.

        Returns:
            self: Fitted estimator.
        """
        self.base_models_ = []
        meta_features = []

        for name, model in self.estimators:
            if isinstance(model, KerasRegressor):
                model.fit(X, y)
                pred = model.predict(X)
            else:
                X_2d = X.reshape(-1, X.shape[-1])
                y_2d = y.reshape(-1, y.shape[-1])
                model.fit(X_2d, y_2d)
                pred = model.predict(X_2d).reshape(X.shape[0], X.shape[1], -1)

            self.base_models_.append((name, model))
            meta_features.append(pred)

        self.meta_features_ = np.concatenate(meta_features, axis=-1)
        meta_features_2d = self.meta_features_.reshape(-1, self.meta_features_.shape[-1])
        y_2d = y.reshape(-1, y.shape[-1])

        self.final_estimator.fit(meta_features_2d, y_2d)
        return self

    def predict(self, X):
        """
        Predict using the fitted stacking model.

        Args:
            X (array-like): Data to predict on.

        Returns:
            array-like: Predicted values.
        """
        meta_features = []
        for name, model in self.base_models_:
            if isinstance(model, KerasRegressor):
                pred = model.predict(X)
            else:
                X_2d = X.reshape(-1, X.shape[-1])
                pred = model.predict(X_2d).reshape(X.shape[0], X.shape[1], -1)
            meta_features.append(pred)

        meta_features = np.concatenate(meta_features, axis=-1)
        meta_features_2d = meta_features.reshape(-1, meta_features.shape[-1])

        y_pred_2d = self.final_estimator.predict(meta_features_2d)
        return y_pred_2d.reshape(X.shape[0], X.shape[1], -1)


class RegularizedSequenceStackingRegressor(SequenceStackingRegressor):
    """
    A regularized stacking regressor that stacks sequence models.

    Args:
        estimators (list): List of (name, estimator) tuples for base estimators.
        final_estimator (object): Estimator used to combine the base estimators.
        cv (int): Number of cross-validation folds.
        alpha (float): Regularization strength.
        l1_ratio (float): The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.

    Methods:
        fit(X, y): Fit the model to the data.
    """
    def __init__(self, estimators, final_estimator, cv=3, alpha=1.0, l1_ratio=0.5):
        super().__init__(estimators, final_estimator, cv)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        """
        Fit the regularized stacking model to the data.

        Args:
            X (array-like): Training data.
            y (array-like): Target values.

        Returns:
            self: Fitted estimator.
        """
        super().fit(X, y)
        self.final_estimator = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)
        meta_features_2d = self.meta_features_.reshape(-1, self.meta_features_.shape[-1])
        y_2d = y.reshape(-1, y.shape[-1])
        self.final_estimator.fit(meta_features_2d, y_2d)
        return self


class KerasRegressor(BaseEstimator, RegressorMixin):
    """
    A wrapper for Keras regressors.

    Args:
        model (object): A Keras model.

    Methods:
        fit(X, y, **kwargs): Fit the Keras model to the data.
        predict(X): Predict using the fitted Keras model.
        get_params(deep): Get parameters for this estimator.
        set_params(**parameters): Set the parameters of this estimator.
    """
    def __init__(self, model):
        self.model = model

    def fit(self, X, y, **kwargs):
        """
        Fit the Keras model to the data.

        Args:
            X (array-like): Training data.
            y (array-like): Target values.
            **kwargs: Additional arguments for Keras model fit method.

        Returns:
            History: Keras model training history.
        """
        # Ensure X and y are 3D
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        if y.ndim == 2:
            y = y.reshape(y.shape[0], 1, y.shape[1])
        return self.model.fit(X, y, **kwargs)

    def predict(self, X):
        """
        Predict using the fitted Keras model.

        Args:
            X (array-like): Data to predict on.

        Returns:
            array-like: Predicted values.
        """
        # Ensure X is 3D
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        return self.model.predict(X)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and contained subobjects.

        Returns:
            dict: Parameters for this estimator.
        """
        return {"model": self.model}

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.

        Args:
            **parameters: Estimator parameters.

        Returns:
            self: Estimator instance.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def objective_xgb(trial, X_train_groups, y_train_groups):
    """
    Objective function for optimizing XGBoost model hyperparameters.

    Args:
        trial (optuna.trial.Trial): A trial object for hyperparameter suggestions.
        X_train_groups (list): List of training data groups.
        y_train_groups (list): List of training target groups.

    Returns:
        float: Mean squared error of the predictions.
    """
    with optimization_lock:
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }
    model = MultiOutputRegressor(XGBRegressor(**params))

    # Reshape the data
    X_train_flat = np.vstack([group.reshape(-1, group.shape[-1]) for group in X_train_groups])
    y_train_flat = np.vstack([group.reshape(-1, group.shape[-1]) for group in y_train_groups])

    model.fit(X_train_flat, y_train_flat)
    predictions = model.predict(X_train_flat)
    return mean_squared_error(y_train_flat, predictions)


def objective_rf(trial, X_train_groups, y_train_groups):
    """
    Objective function for optimizing Random Forest model hyperparameters.

    Args:
        trial (optuna.trial.Trial): A trial object for hyperparameter suggestions.
        X_train_groups (list): List of training data groups.
        y_train_groups (list): List of training target groups.

    Returns:
        float: Mean squared error of the predictions.
    """
    with optimization_lock:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 1, 32),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        }
    model = MultiOutputRegressor(RandomForestRegressor(**params))

    # Flatten the grouped data for Random Forest
    X_train_flat = np.vstack([group.reshape(-1, group.shape[-1]) for group in X_train_groups])
    y_train_flat = np.vstack([group.reshape(-1, group.shape[-1]) for group in y_train_groups])

    model.fit(X_train_flat, y_train_flat)
    predictions = model.predict(X_train_flat)
    return mean_squared_error(y_train_flat, predictions)


def objective_stacking(trial, X_train_padded, y_train_padded, base_models):
    """
    Objective function for optimizing stacking model hyperparameters.

    Args:
        trial (optuna.trial.Trial): A trial object for hyperparameter suggestions.
        X_train_padded (array-like): Padded training data.
        y_train_padded (array-like): Padded training target values.
        base_models (list): List of base models for stacking.

    Returns:
        float: Mean squared error of the predictions.
    """
    with optimization_lock:
        meta_model = Ridge(alpha=trial.suggest_float('alpha', 1e-3, 1e3, log=True))

    base_predictions = []
    for name, model in base_models:
        if isinstance(model, (KerasRegressor,)):
            pred = model.predict(X_train_padded)
        else:
            X_2d = X_train_padded.reshape(-1, X_train_padded.shape[-1])
            pred = model.predict(X_2d)
            pred = pred.reshape(X_train_padded.shape[0], X_train_padded.shape[1], -1)
        base_predictions.append(pred.reshape(X_train_padded.shape[0], -1))

    X_meta = np.hstack(base_predictions)
    y_meta = y_train_padded.reshape(y_train_padded.shape[0], -1)

    stacking_model = MultiOutputRegressor(meta_model)
    stacking_model.fit(X_meta, y_meta)

    stacked_predictions = stacking_model.predict(X_meta)
    mse = mean_squared_error(y_meta, stacked_predictions)

    return mse


def objective_reg_stacking(trial, X_train_padded, y_train_padded, base_models):
    """
    Objective function for optimizing regularized stacking model hyperparameters.

    Args:
        trial (optuna.trial.Trial): A trial object for hyperparameter suggestions.
        X_train_padded (array-like): Padded training data.
        y_train_padded (array-like): Padded training target values.
        base_models (list): List of base models for stacking.

    Returns:
        float: Mean squared error of the predictions.
    """
    with optimization_lock:
        alpha = trial.suggest_float('alpha', 1e-4, 1, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0, 1)
        max_iter = trial.suggest_int('max_iter', 2000, 10000)
        tol = trial.suggest_float('tol', 1e-5, 1e-2, log=True)

    meta_model = ElasticNetCV(l1_ratio=[l1_ratio], alphas=[alpha], max_iter=max_iter, tol=tol, cv=5)

    base_predictions = []
    for name, model in base_models:
        if isinstance(model, (KerasRegressor,)):
            pred = model.predict(X_train_padded)
        else:
            X_2d = X_train_padded.reshape(-1, X_train_padded.shape[-1])
            pred = model.predict(X_2d)
            pred = pred.reshape(X_train_padded.shape[0], X_train_padded.shape[1], -1)

        # Flatten the predictions
        pred_flat = pred.reshape(pred.shape[0], -1)
        base_predictions.append(pred_flat)

    X_meta = np.hstack(base_predictions)
    y_meta = y_train_padded.reshape(y_train_padded.shape[0], -1)

    stacking_model = MultiOutputRegressor(meta_model)
    stacking_model.fit(X_meta, y_meta)

    stacked_predictions = stacking_model.predict(X_meta)
    mse = mean_squared_error(y_meta, stacked_predictions)

    return mse



def create_study(name, objective):
    """
    Create or load an Optuna study.

    Args:
        name (str): The name of the study.
        objective (callable): The objective function for optimization.

    Returns:
        tuple: A tuple containing the study object and the objective function.
    """
    saved_params = load_hyperparameters(name)
    if saved_params is None:
        return optuna.create_study(direction='minimize', sampler=TPESampler()), objective
    return None, None


def optimize_model(name, objective, X_train_groups, y_train_groups, X_train_padded, y_train_padded, base_models,
                   timeout):
    """
    Optimize hyperparameters for a specific model using Optuna.

    Args:
        name (str): The name of the model.
        objective (callable): The objective function for optimization.
        X_train_groups (list): List of training data groups.
        y_train_groups (list): List of training target groups.
        X_train_padded (array-like): Padded training data.
        y_train_padded (array-like): Padded training target values.
        base_models (list): List of base models for stacking.
        timeout (int): Timeout for optimization in seconds.

    Returns:
        tuple: The model name and the best hyperparameters.
    """
    study, func = create_study(name, objective)
    if study is not None and func is not None:
        try:
            def optimize_with_timeout():
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    study.optimize(func, n_trials=hyperparamemter_rounds, timeout=timeout, n_jobs=-1,
                                   show_progress_bar=True)

            run_with_timeout(optimize_with_timeout, timeout_duration=timeout)
            return name, study.best_params
        except TimeoutError:
            logging.error(f"Optimization for {name} timed out")
            return name, None
        except Exception as exc:
            logging.error(f'{name} generated an exception: {exc}')
            return name, None
    else:
        return name, load_hyperparameters(name)


def optimize_hyperparameters(X_train_groups, y_train_groups, X_train_padded, y_train_padded, base_models, timeout=3600):
    """
    Optimize hyperparameters for multiple models in parallel.

    Args:
        X_train_groups (list): List of training data groups.
        y_train_groups (list): List of training target groups.
        X_train_padded (array-like): Padded training data.
        y_train_padded (array-like): Padded training target values.
        base_models (list): List of base models for stacking.
        timeout (int): Timeout for optimization in seconds.

    Returns:
        dict: Best hyperparameters for each model.
    """
    models_to_optimize = [
        ('xgb', lambda trial: objective_xgb(trial, X_train_groups, y_train_groups)),
        ('rf', lambda trial: objective_rf(trial, X_train_groups, y_train_groups)),
        ('stacking', lambda trial: objective_stacking(trial, X_train_padded, y_train_padded, base_models)),
        ('reg_stacking', lambda trial: objective_reg_stacking(trial, X_train_padded, y_train_padded, base_models))
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda x: optimize_model(*x, X_train_groups, y_train_groups, X_train_padded, y_train_padded, base_models,
                                     timeout), models_to_optimize))

    params = {}
    for name, best_params in results:
        if best_params is not None and isinstance(best_params, dict):
            params[name] = best_params
            save_hyperparameters(best_params, name)
        else:
            params[name] = load_hyperparameters(name)

        # Ensure params[name] is a dictionary
        if not isinstance(params[name], dict):
            raise ValueError(f"Invalid parameters for {name}.")

    # Ensure all parameters are loaded
    for name in ['xgb', 'rf', 'stacking', 'reg_stacking']:
        if name not in params or params[name] is None:
            params[name] = load_hyperparameters(name)
        if params[name] is None:
            logging.warning(f"No parameters found for {name}. Using default parameters.")
            params[name] = {}
    xgb = params['xgb']
    rf = params['rf']
    stacking = params['stacking']
    reg_stacking = params['reg_stacking']
    return xgb, rf, stacking, reg_stacking


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for optimizing an LSTM model.

    Args:
        trial (optuna.trial.Trial): A trial object for hyperparameter suggestions.
        X_train (array-like): Training data.
        y_train (array-like): Training target values.
        X_val (array-like): Validation data.
        y_val (array-like): Validation target values.

    Returns:
        float: Validation loss.
    """
    with optimization_lock:
        lstm_units = trial.suggest_int('lstm_units', 32, 128)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)

    input_shape = (X_train.shape[1], X_train.shape[2])

    model = Sequential([
        LSTM(lstm_units, activation='tanh', return_sequences=True, input_shape=input_shape,
             kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        LSTM(lstm_units // 2, activation='tanh', return_sequences=True, kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        TimeDistributed(Dense(16, activation='relu')),
        TimeDistributed(Dense(2))
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=scaled_mse)

    history = model.fit(
        X_train, y_train,
        epochs=training_rounds,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0
    )

    val_loss = history.history['val_loss'][-1]

    trial.set_user_attr('model_json', model.to_json())
    trial.set_user_attr('model_weights', model.get_weights())

    logging.info(f"Trial {trial.number} completed with value: {val_loss}")

    return val_loss


def optimize_hyperparameters_base(X_train, y_train, X_val, y_val, timeout=3600):
    """
    Optimize hyperparameters for the base LSTM model using Optuna.

    Args:
        X_train (array-like): Training data.
        y_train (array-like): Training target values.
        X_val (array-like): Validation data.
        y_val (array-like): Validation target values.
        timeout (int): Timeout for optimization in seconds.

    Returns:
        dict: Best hyperparameters.
    """
    logging.info("Starting hyperparameter optimization")

    study = optuna.create_study(direction='minimize', sampler=TPESampler())

    try:
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val),
                       n_trials=hyperparamemter_rounds,
                       timeout=timeout,
                       n_jobs=-1,
                       show_progress_bar=True)
    except KeyboardInterrupt:
        logging.info("Optimization interrupted by user.")

    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best value: {study.best_value}")
    logging.info(f"Best hyperparameters: {study.best_params}")

    return study.best_params


def compute_metrics(y_true, y_pred):
    """
    Compute various performance metrics.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        dict: A dictionary of computed metrics.
    """
    y_true_reshaped = y_true.reshape(-1, y_true.shape[-1])
    y_pred_reshaped = y_pred.reshape(-1, y_pred.shape[-1])

    mse = mean_squared_error(y_true_reshaped, y_pred_reshaped)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_reshaped, y_pred_reshaped)
    r2 = r2_score(y_true_reshaped, y_pred_reshaped)
    evs = explained_variance_score(y_true_reshaped, y_pred_reshaped)
    mape = np.mean(np.abs((y_true_reshaped - y_pred_reshaped) / y_true_reshaped)) * 100

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'EVS': evs,
        'MAPE': mape
    }


def evaluate_model_cv(model, X, y, cv=5):
    """
    Evaluate a model using cross-validation.

    Args:
        model (object): The model to evaluate.
        X (array-like): Input data.
        y (array-like): Target values.
        cv (int): Number of cross-validation folds.

    Returns:
        None
    """
    scores = cross_val_score(model, X, y, cv=cv)
    logging.info(f'Cross-validated scores: {scores}')
    logging.info(f'Mean score: {scores.mean()}, Std: {scores.std()}')



def train_regularized_stacking_model(lstm_preds, bilstm_preds, y_true, alpha=1.0):
    """
    Train a regularized stacking model using predictions from LSTM and BiLSTM models.

    Args:
        lstm_preds (np.ndarray): Predictions from LSTM model.
        bilstm_preds (np.ndarray): Predictions from BiLSTM model.
        y_true (np.ndarray): True target values.
        alpha (float): Regularization strength for Ridge regression.

    Returns:
        tuple: Trained Ridge model, reshaped meta features, and reshaped true values.
    """
    n_samples, n_timesteps, n_features = lstm_preds.shape
    lstm_preds_reshaped = lstm_preds.reshape(n_samples * n_timesteps, n_features)
    bilstm_preds_reshaped = bilstm_preds.reshape(n_samples * n_timesteps, n_features)
    X_meta = np.hstack([lstm_preds_reshaped, bilstm_preds_reshaped])
    y_true_reshaped = y_true.reshape(n_samples * n_timesteps, n_features)

    meta_learner = Ridge(alpha=alpha)
    meta_learner.fit(X_meta, y_true_reshaped)
    return meta_learner, X_meta, y_true_reshaped


def regularized_stacked_predict(reg_stacking_model, lstm_preds, bilstm_preds):
    """
    Make predictions using a regularized stacking model.

    Args:
        reg_stacking_model (Ridge): Trained Ridge model.
        lstm_preds (np.ndarray): Predictions from LSTM model.
        bilstm_preds (np.ndarray): Predictions from BiLSTM model.

    Returns:
        np.ndarray: Stacked model predictions reshaped to original dimensions.
    """
    n_samples, n_timesteps, n_features = lstm_preds.shape
    lstm_preds_reshaped = lstm_preds.reshape(n_samples * n_timesteps, n_features)
    bilstm_preds_reshaped = bilstm_preds.reshape(n_samples * n_timesteps, n_features)
    X_meta = np.hstack([lstm_preds_reshaped, bilstm_preds_reshaped])
    reg_stacked_preds = reg_stacking_model.predict(X_meta)

    return reg_stacked_preds.reshape(n_samples, n_timesteps, n_features)


def cross_validate_stacking_model(X, y, meta_learner, cv=5):
    """
    Cross-validate a stacking model.

    Args:
        X (np.ndarray): Meta features.
        y (np.ndarray): True target values.
        meta_learner (object): Meta learner model.
        cv (int): Number of cross-validation folds.

    Returns:
        float: Mean cross-validated MAE score.
    """
    scores = cross_val_score(meta_learner, X, y, cv=cv, scoring='neg_mean_absolute_error')
    mean_score = -scores.mean()
    std_score = scores.std()
    logging.info(f'Cross-validated MAE scores: {-scores}')
    logging.info(f'Mean Cross-validated MAE: {mean_score:.4f} ± {std_score:.4f}')
    return mean_score


def test_meta_learners(X, y):
    """
    Test different meta learners and evaluate their cross-validated performance.

    Args:
        X (np.ndarray): Meta features.
        y (np.ndarray): True target values.

    Returns:
        dict: Cross-validated MAE scores for each meta learner.
    """
    meta_learners = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'RandomForest': RandomForestRegressor(n_estimators=100),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1)
    }

    results = {}
    for name, learner in meta_learners.items():
        cv_mae = cross_validate_stacking_model(X, y, learner)
        results[name] = cv_mae

    return results


@tf.keras.utils.register_keras_serializable()
def scaled_mse(y_true, y_pred):
    """
    Custom scaled mean squared error loss function.

    Args:
        y_true (tf.Tensor): True target values.
        y_pred (tf.Tensor): Predicted values.

    Returns:
        tf.Tensor: Scaled MSE loss.
    """
    return tf.reduce_mean(tf.square((y_true - y_pred) * tf.constant([0.0001, 0.0001])))


def evaluate_models(y_true, predictions, model_names):
    """
    Evaluate models using various performance metrics.

    Args:
        y_true (np.ndarray): True target values.
        predictions (list): List of predicted values from different models.
        model_names (list): List of model names corresponding to predictions.

    Returns:
        dict: Evaluation results for each model.
    """
    logging.info("Evaluating models.")
    results = {}

    for name, pred in zip(model_names, predictions):
        y_true_reshaped = y_true.reshape(-1, y_true.shape[-1])
        pred_reshaped = pred.reshape(-1, pred.shape[-1])

        mse = mean_squared_error(y_true_reshaped, pred_reshaped)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_reshaped, pred_reshaped)
        r2 = r2_score(y_true_reshaped, pred_reshaped)
        evs = explained_variance_score(y_true_reshaped, pred_reshaped)
        mape = np.mean(np.abs((y_true_reshaped - pred_reshaped) / (y_true_reshaped + 1e-8))) * 100

        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'EVS': evs,
            'MAPE': mape
        }

        logging.info(f"Metrics for {name}:")
        logging.info(f"  MSE: {mse:.4f}")
        logging.info(f"  RMSE: {rmse:.4f}")
        logging.info(f"  MAE: {mae:.4f}")
        logging.info(f"  R²: {r2:.4f}")
        logging.info(f"  EVS: {evs:.4f}")
        logging.info(f"  MAPE: {mape:.4f}%")

    with open('outputs/results/evaluation_results.json', 'w') as f:
        json.dump(results, f)

    return results


def plot_evaluation_results(results):
    """
    Plot evaluation results for different models.

    Args:
        results (dict): Evaluation results for each model.
    """
    metrics = list(next(iter(results.values())).keys())
    models = list(results.keys())

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 5 * len(metrics)))
    fig.suptitle('Model Evaluation Results')

    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        axes[i].bar(models, values)
        axes[i].set_title(metric)
        axes[i].set_xticklabels(models, rotation=45, ha='right')
        axes[i].set_ylabel('Value')

    plt.tight_layout()
    plt.savefig('outputs/results/evaluation_results.png')
    plt.show()


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



def train_arima_model(y):
    """
    Train an ARIMA model.

    Args:
        y (np.ndarray): Time series data.

    Returns:
        ARIMAResultsWrapper: Trained ARIMA model.
    """
    model = ARIMA(y, order=(1, 1, 1))
    return model.fit()


def train_prophet_model(df):
    """
    Train a Prophet model.

    Args:
        df (pd.DataFrame): DataFrame containing time series data with 'ds' and 'y' columns.

    Returns:
        Prophet: Trained Prophet model.
    """
    model = Prophet()
    model.fit(df)
    return model


def dynamic_weighted_average(predictions, weights):
    """
    Compute the dynamic weighted average of predictions.

    Args:
        predictions (np.ndarray): Array of predictions.
        weights (np.ndarray): Array of weights.

    Returns:
        np.ndarray: Weighted average of predictions.
    """
    return np.average(predictions, axis=0, weights=weights)


def neural_network_meta_model(X_meta, y_meta):
    """
    Train a neural network meta-model.

    Args:
        X_meta (np.ndarray): Meta features.
        y_meta (np.ndarray): True target values.

    Returns:
        MLPRegressor: Trained neural network meta-model.
    """
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
    model.fit(X_meta, y_meta)
    return model


def nested_cross_validation(X, y, model, param_grid):
    """
    Perform nested cross-validation.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Target values.
        model (object): Model to train.
        param_grid (dict): Parameter grid for GridSearchCV.

    Returns:
        list: Nested cross-validation scores.
    """
    outer_cv = TimeSeriesSplit(n_splits=5)
    inner_cv = TimeSeriesSplit(n_splits=3)

    nested_scores = []

    for train_index, test_index in outer_cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv)
        clf.fit(X_train, y_train)
        nested_scores.append(clf.score(X_test, y_test))

    return nested_scores


def build_final_model(best_params, input_shape):
    """
    Build the final LSTM model with the best hyperparameters.

    Args:
        best_params (dict): Best hyperparameters.
        input_shape (tuple): Shape of the input data.

    Returns:
        Sequential: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(best_params['lstm_units'], activation='tanh', return_sequences=True, input_shape=input_shape,
             kernel_regularizer=l2(best_params['l2_reg'])),
        Dropout(best_params['dropout_rate']),
        LSTM(best_params['lstm_units'] // 2, activation='tanh', return_sequences=True,
             kernel_regularizer=l2(best_params['l2_reg'])),
        Dropout(best_params['dropout_rate']),
        TimeDistributed(Dense(16, activation='relu')),
        TimeDistributed(Dense(2))
    ])
    model.compile(optimizer=Adam(learning_rate=max(best_params['learning_rate'], 1e-5)), loss=scaled_mse)
    return model


def plot_predictions(group_index, actual, lstm, bilstm, stacked_lstm, cnn_lstm, gru, svm, knn, ensemble, xgb, rf,
                     stacked, reg_stacked):
    """
    Plot the predictions of the models.

    Parameters:
    group_index (int): Index of the group to plot.
    actual (np.ndarray): Actual target array.
    lstm (np.ndarray): LSTM predictions.
    bilstm (np.ndarray): Bidirectional LSTM predictions.
    stacked_lstm (np.ndarray): Stacked LSTM predictions.
    cnn_lstm (np.ndarray): CNN-LSTM predictions.
    gru (np.ndarray): GRU predictions.
    svm (np.ndarray): SVM predictions.
    knn (np.ndarray): KNN predictions.
    ensemble (np.ndarray): Ensemble model predictions.
    xgb (np.ndarray): XGBoost predictions.
    rf (np.ndarray): Random Forest predictions.
    stacked (np.ndarray): Optimized Stacked LSTM predictions.
    reg_stacked (np.ndarray): Optimized Regular Stacked LSTM predictions.

    """
    logging.info(f"Plotting predictions for group {group_index}.")

    # Ensure all arrays have the same length
    min_length = min(len(actual), len(lstm), len(bilstm), len(stacked_lstm),
                     len(cnn_lstm), len(gru), len(svm), len(knn), len(xgb),
                     len(ensemble), len(rf), len(stacked), len(reg_stacked))

    actual = actual[:min_length]
    lstm = lstm[:min_length]
    bilstm = bilstm[:min_length]
    stacked_lstm = stacked_lstm[:min_length]
    cnn_lstm = cnn_lstm[:min_length]
    gru = gru[:min_length]
    svm = svm[:min_length]
    knn = knn[:min_length]
    ensemble = ensemble[:min_length]
    xgb = xgb[:min_length]
    rf = rf[:min_length]
    stacked = stacked[:min_length]
    reg_stacked = reg_stacked[:min_length]

    plt.figure(figsize=(12, 8))
    plt.plot(actual[:, 0], actual[:, 1], marker='o', markersize=6, linestyle='-', label='Actual Position', color='blue')
    plt.plot(lstm[:, 0], lstm[:, 1], marker='s', markersize=4, linestyle=':', label='LSTM', color='green')
    plt.plot(bilstm[:, 0], bilstm[:, 1], marker='^', markersize=4, linestyle='-.', label='BiLSTM', color='purple')
    plt.plot(stacked_lstm[:, 0], stacked_lstm[:, 1], marker='D', markersize=4, linestyle='--', label='Stacked LSTM',
             color='orange')
    plt.plot(cnn_lstm[:, 0], cnn_lstm[:, 1], marker='*', markersize=6, linestyle=':', label='CNN-LSTM', color='cyan')
    plt.plot(gru[:, 0], gru[:, 1], marker='x', markersize=6, linestyle='-', label='GRU', color='red')
    plt.plot(svm[:, 0], svm[:, 1], marker='p', markersize=6, linestyle='-.', label='SVM', color='magenta')
    plt.plot(knn[:, 0], knn[:, 1], marker='h', markersize=6, linestyle=':', label='KNN', color='brown')
    plt.plot(ensemble[:, 0], ensemble[:, 1], marker='2', markersize=6, linestyle='-', label='Ensemble', color='black')
    plt.plot(xgb[:, 0], xgb[:, 1], marker='v', markersize=6, linestyle='--', label='XGBoost', color='darkblue')
    plt.plot(rf[:, 0], rf[:, 1], marker='1', markersize=6, linestyle='-.', label='Random Forest', color='darkgreen')
    plt.plot(stacked[:, 0], stacked[:, 1], marker='|', markersize=6, linestyle='-', label='Stacked', color='darkred')
    plt.plot(reg_stacked[:, 0], reg_stacked[:, 1], marker='_', markersize=6, linestyle=':', label='Reg Stacked',
             color='darkorange')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Vehicle Route Prediction - Group {group_index}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'outputs/results/group_{group_index}_predictions.png')
    plt.show()
    logging.info(f"Prediction plot for group {group_index} complete.")


def predict_with_proper_shape(model, X):
    """
    Make predictions with the proper input shape for different models.

    Args:
        model (object): Trained model.
        X (np.ndarray): Input data.

    Returns:
        np.ndarray: Predictions.
    """
    if isinstance(model, MultiOutputRegressor):
        if len(X.shape) == 3:  # If input is 3D
            original_shape = X.shape
            X_reshaped = X.reshape(-1, X.shape[-1])
            predictions = model.predict(X_reshaped)
            # Reshape predictions back to 3D
            return predictions.reshape(original_shape[0], original_shape[1], -1)
        else:  # If input is already 2D
            return model.predict(X)
    elif isinstance(model, (SVR, KNeighborsRegressor)):
        # These models expect 2D input
        if len(X.shape) == 3:
            X_reshaped = X.reshape(-1, X.shape[-1])
            predictions = model.predict(X_reshaped)
            return predictions.reshape(X.shape[0], X.shape[1], -1)
        else:
            return model.predict(X)
    else:
        # Keep 3D shape for other models
        return model.predict(X)


def main():
    """
    The main function to run the script
    """
    # Create the required output directories
    create_directory("outputs/train_history")
    create_directory("outputs/models")
    create_directory("outputs/results")
    create_directory("outputs/hyperparameters")

    # Download training data
    download_data.run()

    # Load and prepare the data
    data = load_data("final_data.csv")
    grouped = data.groupby('upload_id')
    X_groups, y_groups = prepare_data(grouped, seq_length=30)

    if not X_groups or not y_groups:
        logging.error("No valid sequences found in the data. Please check your data or reduce the sequence length.")
        return

    X_train_groups, X_test_groups, y_train_groups, y_test_groups = split_data(X_groups, y_groups)

    if not X_train_groups or not y_train_groups:
        logging.error("No training data available after splitting. Please check your data or adjust the split ratio.")
        return

    # Normalize the data
    X_train_groups, y_train_groups, scaler_X, scaler_y = normalize_data(X_train_groups, y_train_groups)
    X_test_groups, y_test_groups, _, _ = normalize_data(X_test_groups, y_test_groups)

    # Combine groups into a single array
    X_train_padded = np.vstack(X_train_groups)
    y_train_padded = np.vstack(y_train_groups)
    X_test_padded = np.vstack(X_test_groups)
    y_test_padded = np.vstack(y_test_groups)

    # Prepare datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_padded, y_train_padded))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_test_padded, y_test_padded))
    val_dataset = val_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    input_shape = (X_train_padded.shape[1], X_train_padded.shape[2])
    print(f"Input shape: {input_shape}")

    lstm_model = build_lstm_model(input_shape)
    bilstm_model = build_bidirectional_lstm_model(input_shape)
    stacked_lstm_model = build_stacked_lstm_model(input_shape)
    cnn_lstm_model = build_cnn_lstm_model(input_shape)
    gru_model = build_gru_model(input_shape)

    # Prepare arguments for concurrent training
    models = [lstm_model, bilstm_model, stacked_lstm_model, cnn_lstm_model, gru_model]
    model_names = ['lstm', 'bilstm', 'stacked_lstm', 'cnn_lstm', 'gru']
    gc.collect()
    # Train models concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        trained_models = list(executor.map(
            lambda m, n: train_model(m, train_dataset, val_dataset, n),
            models, model_names
        ))
    gc.collect()
    # Load saved hyperparameters if they exist
    saved_params = load_hyperparameters('optimized_lstm')

    if saved_params:
        logging.info("Using saved hyperparameters")
        best_params = saved_params
    else:
        # Split the data into train and validation sets for hyperparameter tuning
        train_size = int(0.8 * len(X_train_padded))
        X_train, X_val = X_train_padded[:train_size], X_train_padded[train_size:]
        y_train, y_val = y_train_padded[:train_size], y_train_padded[train_size:]

        # Hyperparameter tuning
        best_params = optimize_hyperparameters_base(X_train, y_train, X_val, y_val)
        logging.info(f"Best hyperparameters: {best_params}")

        # Save the best hyperparameters
        save_hyperparameters(best_params, 'optimized_lstm')

    # Build final model with best hyperparameters
    input_shape = (X_train_padded.shape[1], X_train_padded.shape[2])
    final_model = build_final_model(best_params, input_shape)

    # Train final model
    final_model = train_model(final_model, train_dataset, val_dataset, 'optimized_lstm')

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_index, test_index in tscv.split(X_train_padded):
        X_train_cv, X_test_cv = X_train_padded[train_index], X_train_padded[test_index]
        y_train_cv, y_test_cv = y_train_padded[train_index], y_train_padded[test_index]

        model_cv = build_lstm_model((X_train_cv.shape[1], X_train_cv.shape[2]))
        model_cv.fit(X_train_cv, y_train_cv, epochs=training_rounds, verbose=0)

        y_pred_cv = model_cv.predict(X_test_cv)

        y_test_cv_reshaped = y_test_cv.reshape(-1, y_test_cv.shape[-1])
        y_pred_cv_reshaped = y_pred_cv.reshape(-1, y_pred_cv.shape[-1])

        cv_scores.append(mean_squared_error(y_test_cv_reshaped, y_pred_cv_reshaped))

    logging.info(f"Cross-validation MSE scores: {cv_scores}")
    logging.info(f"Mean CV MSE: {np.mean(cv_scores)}, Std: {np.std(cv_scores)}")

    # Train final model
    final_model = train_model(final_model, train_dataset, val_dataset, 'optimized_lstm')

    # Make predictions with all models
    lstm_predictions = final_model.predict(X_test_padded)
    bilstm_predictions = trained_models[1].predict(X_test_padded)
    stacked_lstm_predictions = trained_models[2].predict(X_test_padded)
    cnn_lstm_predictions = trained_models[3].predict(X_test_padded)
    gru_predictions = trained_models[4].predict(X_test_padded)

    # Reshape data for non-sequential models
    X_train_reshaped = X_train_padded.reshape(-1, X_train_padded.shape[-1])
    X_test_reshaped = X_test_padded.reshape(-1, X_test_padded.shape[-1])
    y_train_reshaped = y_train_padded.reshape(-1, y_train_padded.shape[-1])
    y_test_reshaped = y_test_padded.reshape(-1, y_test_padded.shape[-1])

    print(f"Reshaped training data shape: X: {X_train_reshaped.shape}, y: {y_train_reshaped.shape}")
    print(f"Reshaped testing data shape: X: {X_test_reshaped.shape}, y: {y_test_reshaped.shape}")

    # Train models
    X_train_2d = X_train_padded.reshape(-1, X_train_padded.shape[-1])
    y_train_2d = y_train_padded.reshape(-1, y_train_padded.shape[-1])

    svm_model = MultiOutputRegressor(SVR(kernel='rbf'))
    knn_model = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5))

    svm_model.fit(X_train_2d, y_train_2d)
    knn_model.fit(X_train_2d, y_train_2d)

    base_models = [
        ('lstm', KerasRegressor(lstm_model)),
        ('bilstm', KerasRegressor(bilstm_model)),
        ('optimized_lstm', KerasRegressor(final_model)),
        ('cnn_lstm', KerasRegressor(cnn_lstm_model)),
        ('gru', KerasRegressor(gru_model)),
        ('svm', svm_model),
        ('knn', knn_model),
    ]
    gc.collect()
    # Optimize hyperparameters
    xgb_params, rf_params, stacking_params, reg_stacking_params = optimize_hyperparameters(
        X_train_groups, y_train_groups, X_train_padded, y_train_padded, base_models
    )
    print(f"xgb_params: {xgb_params}")
    # Create models with optimized hyperparameters
    xgb_model = MultiOutputRegressor(XGBRegressor(**xgb_params or {}))
    rf_model = MultiOutputRegressor(RandomForestRegressor(**(rf_params or {})))

    # Add XGBoost and Random Forest to base models
    base_models.extend([('xgb', xgb_model), ('rf', rf_model)])

    stacking_model = SequenceStackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=stacking_params['alpha']),
        cv=TimeSeriesSplit(n_splits=3)
    )

    reg_stacking_model = RegularizedSequenceStackingRegressor(
        estimators=base_models,
        final_estimator=ElasticNet(alpha=reg_stacking_params['alpha'], l1_ratio=reg_stacking_params['l1_ratio']),
        cv=TimeSeriesSplit(n_splits=3),
        alpha=reg_stacking_params['alpha'],
        l1_ratio=reg_stacking_params['l1_ratio']
    )
    gc.collect()
    # Train models
    X_train_flat = np.vstack(X_train_groups)
    y_train_flat = np.vstack(y_train_groups)
    n_samples, seq_length, n_features = X_train_flat.shape
    X_train_flat_2d = X_train_flat.reshape(n_samples * seq_length, n_features)
    n_samples, seq_length, n_outputs = y_train_flat.shape
    y_train_flat_2d = y_train_flat.reshape(n_samples * seq_length, n_outputs)

    xgb_model.fit(X_train_flat_2d, y_train_flat_2d)
    rf_model.fit(X_train_flat_2d, y_train_flat_2d)
    n_samples, seq_length, n_features_y = y_train_padded.shape
    n_features_x = X_train_padded.shape[2]

    y_train_padded_reshaped = y_train_padded.reshape(n_samples * seq_length, n_features_y)
    X_train_padded_reshaped = X_train_padded.reshape(n_samples * seq_length, n_features_x)

    print(f"Original X shape: {X_train_padded.shape}")
    print(f"Original y shape: {y_train_padded.shape}")
    print(f"Reshaped X shape: {X_train_padded_reshaped.shape}")
    print(f"Reshaped y shape: {y_train_padded_reshaped.shape}")

    # Fit the stacking model
    stacking_model.fit(X_train_padded, y_train_padded)
    reg_stacking_model.fit(X_train_padded, y_train_padded)
    gc.collect()
    # Make predictions
    X_test_flat = np.vstack(X_test_groups)
    X_test_flat_2d = X_test_flat.reshape(-1, X_test_flat.shape[-1])
    print(f"X_train_flat shape: {X_train_flat.shape}")
    print(f"X_test_flat shape: {X_test_flat.shape}")
    print(f"y_train_flat shape: {y_train_flat.shape}")
    n_samples, seq_length, n_features = X_test_flat.shape
    X_test_flat_2d = X_test_flat.reshape(n_samples * seq_length, n_features)
    xgb_predictions = xgb_model.predict(X_test_flat_2d)
    print(f"xgb_predictions shape after predict: {xgb_predictions.shape}")

    # Assuming xgb_predictions is now (n_samples * seq_length, n_outputs)
    n_outputs = xgb_predictions.shape[1]
    xgb_predictions = xgb_predictions.reshape(n_samples, seq_length, n_outputs)
    print(f"xgb_predictions final shape: {xgb_predictions.shape}")
    rf_predictions = rf_model.predict(X_test_flat_2d)
    rf_predictions = rf_predictions.reshape(n_samples, seq_length, -1)
    stacked_predictions = stacking_model.predict(X_test_padded)
    reg_stacked_predictions = reg_stacking_model.predict(X_test_padded)

    # Make predictions
    X_test_2d = X_test_padded.reshape(-1, X_test_padded.shape[-1])
    svm_predictions = svm_model.predict(X_test_2d).reshape(X_test_padded.shape[0], X_test_padded.shape[1], -1)
    knn_predictions = knn_model.predict(X_test_2d).reshape(X_test_padded.shape[0], X_test_padded.shape[1], -1)

    # Reshape predictions back to 3D
    seq_length = X_test_padded.shape[1]
    svm_predictions = svm_predictions.reshape(-1, seq_length, 2)
    knn_predictions = knn_predictions.reshape(-1, seq_length, 2)

    print(f"SVM predictions shape: {svm_predictions.shape}")
    print(f"KNN predictions shape: {knn_predictions.shape}")

    # Ensemble methods
    weighted_avg_predictions = weighted_average_predictions(rf_predictions, stacked_predictions,
                                                            xgb_predictions, gru_predictions)
    gc.collect()

    weights = {
        "Optimized LSTM": 0.02916539001225107,
        "BiLSTM": 0.02916536807642109,
        "Stacked LSTM": 0.02919162412461098,
        "CNN-LSTM": 0.2760862067248823,
        "GRU": 0.4550166050010763,
        "SVM": 0.12030917662800869,
        "k-NN": 0.1303153870039379,
        "XGBoost": 0.38980594500194685,
        "Random Forest": 0.9578702863455163,
        "Stacking": 0.9308839514145453,
        "Regularized Stacking": 0.2656866625093892,
    }
    # Normalize weights
    total_weight = sum(weights.values())
    normalized_weights = {model: weight / total_weight for model, weight in weights.items()}
    ensemble_model = SequenceEnsemble([
        final_model, trained_models[1], trained_models[2], trained_models[3], trained_models[4],
        svm_model, knn_model, xgb_model, rf_model, stacking_model, reg_stacking_model
    ], weights=list(normalized_weights.values()))
    ensemble_predictions = ensemble_model.predict(X_test_padded)

    # Evaluate all models
    models = [
        final_model, bilstm_model, stacked_lstm_model, cnn_lstm_model, gru_model,
        svm_model, knn_model, xgb_model, rf_model, stacking_model, reg_stacking_model, ensemble_model
    ]
    model_names = [
        'Optimized LSTM', 'BiLSTM', 'Stacked LSTM', 'CNN-LSTM', 'GRU',
        'SVM', 'k-NN', 'XGBoost', 'Random Forest', 'Stacking', 'Regularized Stacking', 'Ensemble'
    ]
    predictions = [
        lstm_predictions, bilstm_predictions, stacked_lstm_predictions,
        cnn_lstm_predictions, gru_predictions, svm_predictions,
        knn_predictions, xgb_predictions, rf_predictions, stacked_predictions, reg_stacked_predictions,
        weighted_avg_predictions, ensemble_predictions
    ]
    prediction_names = [
        'Optimized LSTM', 'BiLSTM', 'Stacked LSTM', 'CNN-LSTM', 'GRU',
        'SVM', 'k-NN', 'XGBoost', 'Random Forest', 'Stacking', 'Regularized Stacking',
        'Weighted Average', 'Ensemble'
    ]
    gc.collect()
    # Print model summaries and output shapes
    sample_input_2d = np.random.random((1, input_shape[-1]))  # For SVM and other 2D models
    sample_input_3d = np.random.random((1,) + input_shape)  # For 3D models

    for model, name in zip(models, model_names):
        print(f"\n{name} Model Summary:")
        if isinstance(model, (MultiOutputRegressor, SVR, KNeighborsRegressor)):
            output = predict_with_proper_shape(model, sample_input_2d)
        else:
            output = predict_with_proper_shape(model, sample_input_3d)
        print(f"Output shape for {name}: {output.shape}")

    evaluation_results = evaluate_models(y_test_padded, predictions, prediction_names)

    # Plot evaluation metrics
    plot_evaluation_results(evaluation_results)

    # Plot predictions
    for i in range(min(5, len(y_test_padded))):
        plot_predictions(i, y_test_padded[i], lstm_predictions[i],
                         bilstm_predictions[i], stacked_lstm_predictions[i],
                         cnn_lstm_predictions[i], gru_predictions[i],
                         svm_predictions[i], knn_predictions[i],
                         ensemble_predictions[i], xgb_predictions[i], rf_predictions[i],
                         stacked_predictions[i], reg_stacked_predictions[i])

    # Save the final models
    for model, name in zip(models, model_names):  # Save only the base models
        if isinstance(model, tf.keras.Model):
            model.save(f'outputs/models/final_{name}_model.keras')
        else:
            joblib.dump(model, f'outputs/models/final_{name}_model.joblib')

    logging.info("Training and evaluation complete.")


if __name__ == "__main__":
    main()
