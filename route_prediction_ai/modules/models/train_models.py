import gc
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from prophet import Prophet
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

from route_prediction_ai.modules.config import training_rounds
from route_prediction_ai.modules.models.progress import ProgressMonitor
from route_prediction_ai.modules.utilities import run_with_timeout, logging


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
    gc.collect()

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
        with open(f'route_prediction_ai/outputs/train_history/{model_name}_history.json', 'w') as f:
            json.dump(history.history, f)

        return model
    except TimeoutError:
        logging.error(f"Training for {model_name} timed out")
        return None
    except Exception as e:
        logging.error(f"Error during training of {model_name}: {str(e)}")
        return None


def reshape_dataset(lstm_preds, bilstm_preds, y_true):
    """
    Reshapes Training data for the stacking based models

    Args:
        lstm_preds (np.array): Predictions from the LSTM model.
        bilstm_preds (np.array): Predictions from the BiLSTM model.
        y_true (np.array): The true target values.

    Returns:
        X_meta (np.array): Training data for the stacking based models.
        y_true_reshaped (np.array): Training data for the stacking based models.

    """
    n_samples, n_timesteps, n_features = lstm_preds.shape
    lstm_preds_reshaped = lstm_preds.reshape(n_samples * n_timesteps, n_features)
    bilstm_preds_reshaped = bilstm_preds.reshape(n_samples * n_timesteps, n_features)

    X_meta = np.hstack([lstm_preds_reshaped, bilstm_preds_reshaped])
    y_true_reshaped = y_true.reshape(n_samples * n_timesteps, n_features)

    return X_meta, y_true_reshaped


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
    gc.collect()
    X_meta, y_true_reshaped = reshape_dataset(lstm_preds, bilstm_preds, y_true)

    meta_learner = LinearRegression()
    meta_learner.fit(X_meta, y_true_reshaped)
    return meta_learner, X_meta, y_true_reshaped


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
    gc.collect()
    X_meta, y_true_reshaped = reshape_dataset(lstm_preds, bilstm_preds, y_true)

    meta_learner = Ridge(alpha=alpha)
    meta_learner.fit(X_meta, y_true_reshaped)
    return meta_learner, X_meta, y_true_reshaped


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
