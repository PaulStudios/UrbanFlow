import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.multioutput import MultiOutputRegressor


class SequenceEnsemble(BaseEstimator, RegressorMixin):
    """
    Ensemble model for sequence predictions.

    Args:
        models (list): List of models to be used in the ensemble.
        weights (list): List of weights for each model. If None, equal weights are used.
    """

    def __init__(self, models, weights=None):
        self.models = models
        if weights is None:
            self.weights = [1 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models.")
            self.weights = weights

    def fit(self, X, y):
        """
        Fit the ensemble model to the training data.

        Args:
            X (np.array): The input data.
            y (np.array): The target data.

        Returns:
            SequenceEnsemble: The fitted ensemble model.
        """
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict using the ensemble model.

        Args:
            X (np.array): The input data.

        Returns:
            np.array: The ensemble predictions.
        """
        predictions = []
        for model in self.models:
            if isinstance(model, (MultiOutputRegressor)):
                # Reshape 3D input to 2D for non-sequential models
                X_2d = X.reshape(-1, X.shape[-1])
                pred = model.predict(X_2d)
                # Reshape back to 3D
                pred = pred.reshape(X.shape[0], X.shape[1], -1)
            else:
                pred = model.predict(X)

            predictions.append(pred)

        # Ensure all predictions have the same shape before averaging
        shapes = [p.shape for p in predictions]
        if len(set(shapes)) > 1:
            raise ValueError(f"Inconsistent prediction shapes: {shapes}")

        weighted_preds = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_preds += weight * pred

        return weighted_preds
