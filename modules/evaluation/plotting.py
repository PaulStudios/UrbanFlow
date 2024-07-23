from matplotlib import pyplot as plt

from modules.utilities import logging


def plot_evaluation_results(results):
    """
    Plot evaluation results for different models.

    Args:
        results (dict): Evaluation results for each model.
    """
    metrics = list(next(iter(results.values())).keys())
    models = list(results.keys())

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 5 * len(metrics)))

    # Ensure axes is always a list, even for a single subplot
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        x = range(len(models))
        axes[i].bar(x, values)
        axes[i].set_title(metric)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(models, rotation=45, ha='right')
        axes[i].set_ylabel('Value')

    # Adjust the layout to make room for the suptitle
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Add suptitle after adjusting the layout
    fig.suptitle('Model Evaluation Results', fontsize=16, y=0.98)

    plt.savefig('outputs/results/evaluation_results.png')
    plt.show()


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
