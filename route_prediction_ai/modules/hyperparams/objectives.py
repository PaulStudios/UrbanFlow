import gc
import hashlib
import json
import threading

import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from route_prediction_ai.modules.hyperparams.custom_classes import KerasRegressor

optimization_lock = threading.Lock()
cache = {}


def get_cache_key(hp):
    # Create a string representation of the hyperparameters
    hp_str = json.dumps(hp, sort_keys=True)
    # Create a hash of the string
    return hashlib.md5(hp_str.encode()).hexdigest()


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
    gc.collect()
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
    gc.collect()
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

    gc.collect()
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

    gc.collect()
    return mse

