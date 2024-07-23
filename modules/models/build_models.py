from keras import Model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, TimeDistributed, Dropout, GRU, Conv1D
from tensorflow.keras.layers import Input
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
    inputs = Input(shape=input_shape)
    x = LSTM(64, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.01))(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(32, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)
    x = TimeDistributed(Dense(16, activation='relu'))(x)
    outputs = TimeDistributed(Dense(2))(x)

    model = Model(inputs=inputs, outputs=outputs)
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
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.01)))(inputs)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(32, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.01)))(x)
    x = Dropout(0.2)(x)
    x = TimeDistributed(Dense(16, activation='relu'))(x)
    outputs = TimeDistributed(Dense(2))(x)

    model = Model(inputs=inputs, outputs=outputs)
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
    inputs = Input(shape=input_shape)
    x = LSTM(64, activation='tanh', return_sequences=True)(inputs)
    x = LSTM(32, activation='tanh', return_sequences=True)(x)
    x = LSTM(16, activation='tanh', return_sequences=True)(x)
    x = TimeDistributed(Dense(8, activation='relu'))(x)
    outputs = TimeDistributed(Dense(2))(x)

    model = Model(inputs=inputs, outputs=outputs)
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
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = LSTM(50, return_sequences=True)(x)
    x = LSTM(50, return_sequences=True)(x)
    x = TimeDistributed(Dense(16, activation='relu'))(x)
    outputs = TimeDistributed(Dense(2))(x)

    model = Model(inputs=inputs, outputs=outputs)
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
    inputs = Input(shape=input_shape)
    x = GRU(64, return_sequences=True)(inputs)
    x = GRU(32, return_sequences=True)(x)
    outputs = TimeDistributed(Dense(2))(x)

    model = Model(inputs=inputs, outputs=outputs)
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
    inputs = Input(shape=input_shape)
    x = LSTM(best_params['lstm_units'], activation='tanh', return_sequences=True,
             kernel_regularizer=l2(best_params['l2_reg']))(inputs)
    x = Dropout(best_params['dropout_rate'])(x)
    x = LSTM(best_params['lstm_units'] // 2, activation='tanh', return_sequences=True,
             kernel_regularizer=l2(best_params['l2_reg']))(x)
    x = Dropout(best_params['dropout_rate'])(x)
    x = TimeDistributed(Dense(16, activation='relu'))(x)
    outputs = TimeDistributed(Dense(2))(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=max(best_params['learning_rate'], 1e-5)), loss=scaled_mse)
    return model

