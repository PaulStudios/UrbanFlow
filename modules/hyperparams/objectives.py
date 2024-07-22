import threading

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from xgboost import XGBRegressor

from modules.config import training_rounds
from modules.hyperparams.custom_classes import KerasRegressor
from modules.utilities import logging, scaled_mse

optimization_lock = threading.Lock()


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
