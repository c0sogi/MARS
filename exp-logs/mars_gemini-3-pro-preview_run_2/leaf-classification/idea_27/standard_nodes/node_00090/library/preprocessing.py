import os
import numpy as np
import joblib
from sklearn.preprocessing import PowerTransformer
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("preprocessing")


class RobustPreprocessor:
    """
    Handles statistical transformation of features using PowerTransformer (Yeo-Johnson)
    and ensures data type precision for numerical stability in LDA/QDA models.

    This class is designed to Gaussianize input features, which is a key assumption
    or at least a beneficial property for Linear and Quadratic Discriminant Analysis.
    It also strictly enforces float64 precision.
    """

    def __init__(self, method=Config.POWER_TRANSFORM_METHOD):
        """
        Initialize the preprocessor.

        Args:
            method (str): The power transform method ('yeo-johnson' or 'box-cox').
                          Defaults to Config.POWER_TRANSFORM_METHOD.
        """
        self.method = method
        # standardize=True ensures zero mean and unit variance after power transform
        self.transformer = PowerTransformer(method=self.method, standardize=True)
        self.is_fitted = False

    def _validate_input(self, X):
        """
        Validates input array, handling NaNs/Infs and ensuring float64 type.

        Args:
            X (array-like): Input data.

        Returns:
            np.ndarray: Cleaned input data as float64.
        """
        # Ensure input is a numpy array of float64
        X = np.array(X, dtype=np.float64)

        # Check for NaNs or Infs
        if np.isnan(X).any() or np.isinf(X).any():
            nan_inf_count = np.isnan(X).sum() + np.isinf(X).sum()
            # Log warning if issues are found (though unlikely with this dataset)
            # We use a debug level or warning depending on frequency, here warning is appropriate
            # as it might indicate upstream extraction issues.
            if nan_inf_count > 0:
                logger.warning(
                    f"Input contains {nan_inf_count} NaN/Inf values. "
                    "Replacing with 0.0 before transformation."
                )
            # Replace NaNs and Infs with 0.0
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X

    def fit(self, X):
        """
        Fits the PowerTransformer to the data X.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            self: The fitted instance.
        """
        X = self._validate_input(X)

        logger.info(
            f"Fitting RobustPreprocessor (method='{self.method}') on data with shape {X.shape}..."
        )

        try:
            self.transformer.fit(X)
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Failed to fit PowerTransformer: {e}")
            raise e

        return self

    def transform(self, X):
        """
        Applies the transformation to X.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data in float64.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "RobustPreprocessor must be fitted before calling transform."
            )

        X = self._validate_input(X)

        # Apply transformation
        # PowerTransformer returns float64 by default, but we enforce it explicitly
        X_trans = self.transformer.transform(X)

        return X_trans.astype(np.float64)

    def fit_transform(self, X):
        """
        Fits and transforms X in a single step.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data in float64.
        """
        self.fit(X)
        return self.transform(X)

    def save(self, filepath):
        """
        Saves the fitted preprocessor to disk using joblib.

        Args:
            filepath (str): Path to save the object.
        """
        if not self.is_fitted:
            logger.warning("Saving an unfitted RobustPreprocessor.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            joblib.dump(self, filepath)
            logger.info(f"RobustPreprocessor saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save preprocessor to {filepath}: {e}")
            raise e

    @staticmethod
    def load(filepath):
        """
        Loads a preprocessor from disk.

        Args:
            filepath (str): Path to the saved object.

        Returns:
            RobustPreprocessor: The loaded instance.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")

        try:
            preprocessor = joblib.load(filepath)
            logger.info(f"RobustPreprocessor loaded from {filepath}")
            return preprocessor
        except Exception as e:
            logger.error(f"Failed to load preprocessor from {filepath}: {e}")
            raise e
