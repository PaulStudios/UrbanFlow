import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from route_prediction_ai.modules.config import training_rounds
from route_prediction_ai.modules.models.build_models import build_lstm_model


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
