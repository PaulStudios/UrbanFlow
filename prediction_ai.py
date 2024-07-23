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

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

import download_data
from modules.config import training_rounds, parallel_workers
from modules.data.load_data import load_data
from modules.data.preprocessing import prepare_data, split_data, normalize_data
from modules.evaluation.evaluate import evaluate_models
from modules.evaluation.plotting import plot_evaluation_results, plot_predictions
from modules.hyperparams.custom_classes import (SequenceStackingRegressor, RegularizedSequenceStackingRegressor,
                                                KerasRegressor)
from modules.hyperparams.optimisers import optimize_hyperparameters
from modules.models.build_models import (build_lstm_model, build_bidirectional_lstm_model, build_stacked_lstm_model,
                                         build_cnn_lstm_model, build_gru_model, build_final_model)
from modules.models.ensemble import SequenceEnsemble
from modules.models.predict import weighted_average_predictions, predict_with_proper_shape
from modules.models.train_models import train_model
from modules.utilities import create_directory
from modules.utilities import logging

# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


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
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        trained_models = list(executor.map(
            lambda m, n: train_model(m, train_dataset, val_dataset, n),
            models, model_names
        ))
    gc.collect()

    best_params = {"lstm_units": 97,
                   "dropout_rate": 0.14648565863246926,
                   "learning_rate": 0.0007653631599876039,
                   "l2_reg": 0.00013107209632797685}

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
