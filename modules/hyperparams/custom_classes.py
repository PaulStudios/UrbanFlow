import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet


class SequenceStackingRegressor(BaseEstimator, RegressorMixin):
    """
    A stacking regressor that stacks sequence models.

    Args:
        estimators (list): List of (name, estimator) tuples for base estimators.
        final_estimator (object): Estimator used to combine the base estimators.
        cv (int): Number of cross-validation folds.

    Methods:
        fit(X, y): Fit the model to the data.
        predict(X): Predict using the fitted model.
    """

    def __init__(self, estimators, final_estimator, cv=3):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv

    def fit(self, X, y):
        """
        Fit the stacking model to the data.

        Args:
            X (array-like): Training data.
            y (array-like): Target values.

        Returns:
            self: Fitted estimator.
        """
        self.base_models_ = []
        meta_features = []

        for name, model in self.estimators:
            if isinstance(model, KerasRegressor):
                model.fit(X, y)
                pred = model.predict(X)
            else:
                X_2d = X.reshape(-1, X.shape[-1])
                y_2d = y.reshape(-1, y.shape[-1])
                model.fit(X_2d, y_2d)
                pred = model.predict(X_2d).reshape(X.shape[0], X.shape[1], -1)

            self.base_models_.append((name, model))
            meta_features.append(pred)

        self.meta_features_ = np.concatenate(meta_features, axis=-1)
        meta_features_2d = self.meta_features_.reshape(-1, self.meta_features_.shape[-1])
        y_2d = y.reshape(-1, y.shape[-1])

        self.final_estimator.fit(meta_features_2d, y_2d)
        return self

    def predict(self, X):
        """
        Predict using the fitted stacking model.

        Args:
            X (array-like): Data to predict on.

        Returns:
            array-like: Predicted values.
        """
        meta_features = []
        for name, model in self.base_models_:
            if isinstance(model, KerasRegressor):
                pred = model.predict(X)
            else:
                X_2d = X.reshape(-1, X.shape[-1])
                pred = model.predict(X_2d).reshape(X.shape[0], X.shape[1], -1)
            meta_features.append(pred)

        meta_features = np.concatenate(meta_features, axis=-1)
        meta_features_2d = meta_features.reshape(-1, meta_features.shape[-1])

        y_pred_2d = self.final_estimator.predict(meta_features_2d)
        return y_pred_2d.reshape(X.shape[0], X.shape[1], -1)


class RegularizedSequenceStackingRegressor(SequenceStackingRegressor):
    """
    A regularized stacking regressor that stacks sequence models.

    Args:
        estimators (list): List of (name, estimator) tuples for base estimators.
        final_estimator (object): Estimator used to combine the base estimators.
        cv (int): Number of cross-validation folds.
        alpha (float): Regularization strength.
        l1_ratio (float): The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.

    Methods:
        fit(X, y): Fit the model to the data.
    """

    def __init__(self, estimators, final_estimator, cv=3, alpha=1.0, l1_ratio=0.5):
        super().__init__(estimators, final_estimator, cv)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        """
        Fit the regularized stacking model to the data.

        Args:
            X (array-like): Training data.
            y (array-like): Target values.

        Returns:
            self: Fitted estimator.
        """
        super().fit(X, y)
        self.final_estimator = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)
        meta_features_2d = self.meta_features_.reshape(-1, self.meta_features_.shape[-1])
        y_2d = y.reshape(-1, y.shape[-1])
        self.final_estimator.fit(meta_features_2d, y_2d)
        return self


class KerasRegressor(BaseEstimator, RegressorMixin):
    """
    A wrapper for Keras regressors.

    Args:
        model (object): A Keras model.

    Methods:
        fit(X, y, **kwargs): Fit the Keras model to the data.
        predict(X): Predict using the fitted Keras model.
        get_params(deep): Get parameters for this estimator.
        set_params(**parameters): Set the parameters of this estimator.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y, **kwargs):
        """
        Fit the Keras model to the data.

        Args:
            X (array-like): Training data.
            y (array-like): Target values.
            **kwargs: Additional arguments for Keras model fit method.

        Returns:
            History: Keras model training history.
        """
        # Ensure X and y are 3D
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        if y.ndim == 2:
            y = y.reshape(y.shape[0], 1, y.shape[1])
        return self.model.fit(X, y, **kwargs)

    def predict(self, X):
        """
        Predict using the fitted Keras model.

        Args:
            X (array-like): Data to predict on.

        Returns:
            array-like: Predicted values.
        """
        # Ensure X is 3D
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        return self.model.predict(X)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and contained subobjects.

        Returns:
            dict: Parameters for this estimator.
        """
        return {"model": self.model}

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.

        Args:
            **parameters: Estimator parameters.

        Returns:
            self: Estimator instance.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
