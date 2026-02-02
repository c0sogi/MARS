import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import OAS
from library.config import FLOAT_PRECISION


class LDAExpert:
    """
    Linear Discriminant Analysis Expert wrapper.

    This class wraps sklearn's LinearDiscriminantAnalysis to provide:
    1. Support for 'oas' (Oracle Approximating Shrinkage) estimation, which is not
       natively exposed as a string argument in sklearn's LDA.
    2. Strict enforcement of float64 precision for numerical stability.
    3. Unified interface for the selection strategy defined in Idea 40.
    """

    def __init__(self, shrinkage="auto"):
        """
        Initialize the LDA Expert.

        Args:
            shrinkage (str or float): The shrinkage strategy to use.
                - 'oas': Computes the shrinkage coefficient using the OAS estimator on the input data.
                - 'ledoit_wolf': Uses sklearn's 'auto' mode which implements Ledoit-Wolf shrinkage.
                - float: A fixed shrinkage coefficient between 0 and 1.
        """
        self.shrinkage = shrinkage
        self.model = None
        self.shrinkage_val_ = None

    def fit(self, X, y):
        """
        Fit the LDA model to the training data.

        Args:
            X (array-like): Training feature matrix.
            y (array-like): Target labels.

        Returns:
            self: Returns the instance itself.
        """
        # Enforce strict double precision
        X = np.array(X, dtype=FLOAT_PRECISION)
        y = np.array(y, dtype=int)

        # LDA with shrinkage requires 'lsqr' or 'eigen' solver.
        # 'lsqr' is generally preferred for high-dimensional classification.
        solver = "lsqr"
        shrinkage_param = None

        if self.shrinkage == "oas":
            # Explicitly estimate shrinkage using the OAS covariance estimator.
            # We calculate the optimal shrinkage for the global covariance structure
            # and pass this coefficient to the LDA solver.
            oas = OAS()
            oas.fit(X)
            shrinkage_param = oas.shrinkage_
            self.shrinkage_val_ = shrinkage_param
        elif self.shrinkage == "ledoit_wolf":
            # sklearn's LDA implements Ledoit-Wolf when shrinkage is set to 'auto'
            shrinkage_param = "auto"
        else:
            # Handle fixed float values or other valid sklearn shrinkage arguments
            shrinkage_param = self.shrinkage
            if isinstance(shrinkage_param, (float, int)):
                self.shrinkage_val_ = shrinkage_param

        # Initialize and fit the underlying sklearn model
        self.model = LinearDiscriminantAnalysis(
            solver=solver, shrinkage=shrinkage_param
        )
        self.model.fit(X, y)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X (array-like): Input feature matrix.

        Returns:
            np.ndarray: Matrix of class probabilities (n_samples, n_classes) in float64.
        """
        if self.model is None:
            raise RuntimeError("LDAExpert must be fitted before calling predict_proba.")

        # Enforce precision
        X = np.array(X, dtype=FLOAT_PRECISION)

        # Predict and cast back to ensure consistency
        probs = self.model.predict_proba(X)
        return probs.astype(FLOAT_PRECISION)

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X (array-like): Input feature matrix.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if self.model is None:
            raise RuntimeError("LDAExpert must be fitted before calling predict.")

        # Enforce precision
        X = np.array(X, dtype=FLOAT_PRECISION)
        return self.model.predict(X)
