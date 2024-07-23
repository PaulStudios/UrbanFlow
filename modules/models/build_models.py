from tensorflow.keras.layers import Bidirectional, LSTM, Dense, TimeDistributed, Dropout, GRU, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from modules.utilities import scaled_mse


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


def build_lstm_model_custom():
    return None
