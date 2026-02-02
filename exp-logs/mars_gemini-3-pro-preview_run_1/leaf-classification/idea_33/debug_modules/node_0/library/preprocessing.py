import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler


class Float64Preprocessor:
    """
    A high-precision preprocessor that applies Yeo-Johnson Power Transformation
    followed by Standard Scaling, strictly maintaining float64 precision.

    This class is designed to mitigate floating-point associativity noise and
    numerical degradation by enforcing 64-bit precision at every step of the
    feature engineering pipeline.
    """

    def __init__(self):
        """
        Initialize the preprocessor with PowerTransformer and StandardScaler.

        The PowerTransformer is configured with standardize=False because
        StandardScaler is applied explicitly as the subsequent step.
        """
        self.pt = PowerTransformer(method="yeo-johnson", standardize=False)
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        """
        Fits the preprocessing pipeline on the provided data.

        Args:
            X (array-like): The training data to fit.
            y (ignored): Argument for compatibility with sklearn API.

        Returns:
            self: The fitted instance.
        """
        # Enforce float64 precision immediately upon entry
        X = np.array(X, dtype=np.float64)

        # 1. Fit PowerTransformer (Yeo-Johnson)
        self.pt.fit(X)

        # 2. Transform X to get the intermediate representation for the Scaler
        #    We do not use fit_transform on the pipeline components sequentially
        #    to ensure we control the data flow and precision if needed,
        #    though here we trust the internal float64 handling of sklearn
        #    once inputs are cast.
        X_pt = self.pt.transform(X)

        # 3. Fit StandardScaler on the power-transformed data
        self.scaler.fit(X_pt)

        return self

    def transform(self, X):
        """
        Applies the learned transformations to new data.

        Args:
            X (array-like): The data to transform.

        Returns:
            np.ndarray: The transformed data in float64 precision.
        """
        # Enforce float64 precision immediately
        X = np.array(X, dtype=np.float64)

        # 1. Apply PowerTransformer
        X_pt = self.pt.transform(X)

        # 2. Apply StandardScaler
        X_scaled = self.scaler.transform(X_pt)

        # 3. Explicitly return float64 to prevent any downstream degradation
        return X_scaled.astype(np.float64)

    def fit_transform(self, X, y=None):
        """
        Fits the pipeline to X and returns the transformed version.

        Args:
            X (array-like): The training data.
            y (ignored): Argument for compatibility.

        Returns:
            np.ndarray: The transformed data in float64 precision.
        """
        self.fit(X, y)
        return self.transform(X)
