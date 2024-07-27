import json

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor

from route_prediction_ai.modules.utilities import logging


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

    with open('route_prediction_ai/outputs/results/evaluation_results.json', 'w') as f:
        json.dump(results, f)

    return results


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
