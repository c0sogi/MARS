import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.covariance import OAS
from library.utils import set_seed


class LDAWrapper:
    """
    A wrapper for Linear Discriminant Analysis that supports specific shrinkage configurations
    required by the ensemble strategy, including OAS (Oracle Approximating Shrinkage) and
    fixed float shrinkage values.

    Attributes:
        solver (str): Solver to use ('svd', 'lsqr', 'eigen').
        shrinkage (float, str, or None): Shrinkage parameter.
            - If 'oas': Uses sklearn.covariance.OAS as the covariance estimator.
            - If 'auto': Uses Ledoit-Wolf lemma.
            - If float: Fixed shrinkage coefficient.
            - If None: No shrinkage.
        random_state (int): Seed for reproducibility.
    """

    def __init__(self, solver="lsqr", shrinkage=None, random_state=42):
        self.solver = solver
        self.shrinkage = shrinkage
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        """
        Fits the LDA model.

        Args:
            X (array-like): Training data, shape (n_samples, n_features).
            y (array-like): Target values, shape (n_samples,).
        """
        set_seed(self.random_state)

        # Ensure float64 precision
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        # Configure the underlying sklearn model based on shrinkage strategy
        if self.shrinkage == "oas":
            # Use OAS covariance estimator
            # Note: covariance_estimator is supported in sklearn >= 1.2
            # solver must be 'lsqr' or 'eigen' for custom covariance
            solver_to_use = self.solver if self.solver in ["lsqr", "eigen"] else "lsqr"
            self.model = LinearDiscriminantAnalysis(
                solver=solver_to_use,
                shrinkage=None,  # shrinkage is ignored when covariance_estimator is provided
                covariance_estimator=OAS(),
                store_covariance=False,
            )
        else:
            # Standard shrinkage (float, 'auto', or None)
            self.model = LinearDiscriminantAnalysis(
                solver=self.solver, shrinkage=self.shrinkage, store_covariance=False
            )

        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X (array-like): Input data.

        Returns:
            np.ndarray: Class probabilities, shape (n_samples, n_classes).
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        X = np.array(X, dtype=np.float64)
        return self.model.predict_proba(X)

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X (array-like): Input data.

        Returns:
            np.ndarray: Predicted labels.
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        X = np.array(X, dtype=np.float64)
        return self.model.predict(X)


class QDAWrapper:
    """
    A wrapper for Quadratic Discriminant Analysis that exposes regularization
    parameters to stabilize covariance estimation, particularly for the
    Morphometric view.

    Attributes:
        reg_param (float): Regularization parameter for the covariance estimate.
                           Regularizes the covariance estimate as:
                           (1 - reg_param) * Sigma + reg_param * I
        random_state (int): Seed for reproducibility.
    """

    def __init__(self, reg_param=0.0, random_state=42):
        self.reg_param = reg_param
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        """
        Fits the QDA model.

        Args:
            X (array-like): Training data, shape (n_samples, n_features).
            y (array-like): Target values, shape (n_samples,).
        """
        set_seed(self.random_state)

        # Ensure float64 precision
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        self.model = QuadraticDiscriminantAnalysis(
            reg_param=self.reg_param, store_covariance=False
        )

        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X (array-like): Input data.

        Returns:
            np.ndarray: Class probabilities, shape (n_samples, n_classes).
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        X = np.array(X, dtype=np.float64)
        return self.model.predict_proba(X)

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X (array-like): Input data.

        Returns:
            np.ndarray: Predicted labels.
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        X = np.array(X, dtype=np.float64)
        return self.model.predict(X)
