import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from datetime import datetime


def load_data(csv_file):
    """Load and preprocess CSV data."""
    data = pd.read_csv(csv_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(['upload_id', 'timestamp'])
    return data


def add_velocity_acceleration(X):
    velocity = np.diff(X[:, :2], axis=0)
    acceleration = np.diff(velocity, axis=0)

    velocity = np.vstack([np.zeros((1, 2)), velocity])
    acceleration = np.vstack([np.zeros((2, 2)), acceleration])

    return np.hstack([X, velocity, acceleration])

def prepare_data(grouped):
    """Prepare feature and target arrays for each group."""
    X_groups, y_groups = [], []
    for _, group in grouped:
        group = group.sort_values('timestamp')

        # Convert timestamp to seconds since epoch
        group['timestamp_seconds'] = group['timestamp'].astype('int64') / 10 ** 9

        X = group[['longitude', 'latitude', 'timestamp_seconds']].values
        y = group[['longitude', 'latitude']].values

        # Add time difference as a feature
        time_diff = group['timestamp_seconds'].diff().fillna(0).values.reshape(-1, 1)
        X = np.column_stack((X, time_diff))

        X_groups.append(X)
        y_groups.append(y)
    return X_groups, y_groups


def split_data(X_groups, y_groups, test_size=0.2):
    """Split each group into training and testing sets."""
    X_train_groups, X_test_groups, y_train_groups, y_test_groups = [], [], [], []
    for X, y in zip(X_groups, y_groups):
        if len(X) > 5:  # Only split if we have enough data points
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
    return X_train_groups, X_test_groups, y_train_groups, y_test_groups


def apply_kalman_filter(X_train, X_test, Q_var=0.01, R_var=0.1):
    """Apply Kalman filter to both training and testing data."""
    kf = KalmanFilter(dim_x=6, dim_z=3)
    dt = np.mean(X_train[:, 3])  # Use mean time difference

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

    return np.array(predictions_train), np.array(predictions_test)


def create_padded_sequences(X_groups, y_groups):
    """Create padded sequences for LSTM input."""
    max_len = max(len(X) for X in X_groups)
    X_padded = pad_sequences(X_groups, maxlen=max_len, padding='post', dtype='float32')
    y_padded = pad_sequences(y_groups, maxlen=max_len, padding='post', dtype='float32')
    return X_padded, y_padded


def normalize_data(X_groups, y_groups):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_normalized = [scaler_X.fit_transform(X) for X in X_groups]
    y_normalized = [scaler_y.fit_transform(y) for y in y_groups]

    return X_normalized, y_normalized, scaler_X, scaler_y

def build_lstm_model(input_shape):
    """Build and compile LSTM model."""
    model = Sequential([
        LSTM(100, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=True),
        Dropout(0.2),
        TimeDistributed(Dense(20, activation='relu')),
        TimeDistributed(Dense(2))
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def evaluate_models(actual_groups, kalman_groups, lstm_groups):
    """Evaluate Kalman filter and LSTM models using MSE."""
    kalman_mse, lstm_mse = [], []
    for actual, kalman, lstm in zip(actual_groups, kalman_groups, lstm_groups):
        if len(actual) > 0:
            kalman_mse.append(mean_squared_error(actual, kalman[:len(actual)]))
            lstm_mse.append(mean_squared_error(actual, lstm[:len(actual)]))

    print(f"Average Kalman Filter MSE: {np.mean(kalman_mse)}")
    print(f"Average LSTM MSE: {np.mean(lstm_mse)}")


def plot_predictions(group_index, actual, kalman, lstm):
    plt.figure(figsize=(12, 8))
    plt.plot(actual[:, 0], actual[:, 1], marker='o', markersize=8, linestyle='-', label='Actual Position', color='blue')
    plt.plot(kalman[:len(actual), 0], kalman[:len(actual), 1], marker='x', markersize=10, linestyle='--',
             label='Kalman Filter Prediction', color='red')
    plt.plot(lstm[:len(actual), 0], lstm[:len(actual), 1], marker='s', markersize=6, linestyle=':',
             label='LSTM Prediction', color='green')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Vehicle Route Prediction - Group {group_index}')
    plt.legend()
    plt.grid(True)
    plt.show()


def main(csv_file):
    try:
        # Data Preparation
        data = load_data(csv_file)
        grouped = data.groupby('upload_id')
        X_groups, y_groups = prepare_data(grouped)
        X_train_groups, X_test_groups, y_train_groups, y_test_groups = split_data(X_groups, y_groups)

        # Apply Kalman filter
        kalman_train_groups, kalman_test_groups = zip(*[apply_kalman_filter(X_train, X_test)
                                                        for X_train, X_test in zip(X_train_groups, X_test_groups)])

        print(f"Kalman predictions shape: {kalman_test_groups[0].shape}")
        print(f"First few Kalman predictions: {kalman_test_groups[0][:5]}")
        # Prepare data for LSTM
        X_train_padded, y_train_padded = create_padded_sequences(X_train_groups, y_train_groups)
        X_test_padded, y_test_padded = create_padded_sequences(X_test_groups, y_test_groups)
        X_train_lstm = X_train_padded.reshape((X_train_padded.shape[0], X_train_padded.shape[1], 4))
        X_test_lstm = X_test_padded.reshape((X_test_padded.shape[0], X_test_padded.shape[1], 4))

        # Build and train LSTM model
        lstm_model = build_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Convert numpy arrays to TensorFlow tensors
        X_train_tensor = tf.convert_to_tensor(X_train_lstm, dtype=tf.float32)
        y_train_tensor = tf.convert_to_tensor(y_train_padded, dtype=tf.float32)

        lstm_model.fit(X_train_tensor, y_train_tensor, epochs=200, batch_size=32, validation_split=0.2,
                       callbacks=[early_stopping], verbose=1)

        # Predict with LSTM model
        X_test_tensor = tf.convert_to_tensor(X_test_lstm, dtype=tf.float32)
        lstm_predictions_groups = lstm_model.predict(X_test_tensor)

        # Evaluate models
        evaluate_models(y_test_groups, kalman_test_groups, lstm_predictions_groups)

        # Plot predictions for each group
        for i in range(min(5, len(y_test_groups))):
            if len(y_test_groups[i]) > 0:
                print(f"Group {i} shapes:")
                print(f"  Actual: {y_test_groups[i].shape}")
                print(f"  Kalman: {kalman_test_groups[i].shape}")
                print(f"  LSTM: {lstm_predictions_groups[i].shape}")
                plot_predictions(i, y_test_groups[i], kalman_test_groups[i], lstm_predictions_groups[i])

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        csv_file = 'final_data.csv'  # Replace with the path to your CSV file
        main(csv_file)
    except FileNotFoundError:
        print("Error: Data file not found. Please ensure the file path is correct.")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty. Please provide a valid data file.")
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        import traceback

        traceback.print_exc()