import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer
from library.config import Config
from library.utils import setup_logger
from library.data_loader import NUMERIC_COLS

# Initialize Logger
logger = setup_logger("feature_meta")


class MetadataScaler(BaseEstimator, TransformerMixin):
    """
    Transformer that selects numerical metadata features and applies RankGauss (QuantileTransformer).
    This ensures that the numerical features are robust to outliers and normally distributed,
    facilitating fusion with embeddings and topic vectors.
    """

    def __init__(self, random_state: int = Config.RANDOM_SEED):
        """
        Args:
            random_state (int): Seed for reproducibility.
        """
        self.random_state = random_state
        self.scaler = QuantileTransformer(
            output_distribution="normal", random_state=self.random_state
        )
        self.numeric_cols = NUMERIC_COLS

    def _extract_data(self, X):
        """
        Extracts numerical data from DataFrame or returns array as is.
        Ensures only the defined NUMERIC_COLS are used if X is a DataFrame containing them.
        """
        if isinstance(X, pd.DataFrame):
            # Check if we can select by name (Robust selection)
            if all(col in X.columns for col in self.numeric_cols):
                return X[self.numeric_cols].values
            else:
                # If columns missing, assume X is already the numeric matrix
                # (e.g. passed from a ColumnTransformer that stripped names)
                # But warn if shape doesn't match expected feature count
                if X.shape[1] != len(self.numeric_cols):
                    logger.warning(
                        f"Input DataFrame has {X.shape[1]} columns, expected {len(self.numeric_cols)}. "
                        "Proceeding with all columns, but this may be incorrect."
                    )
                return X.values
        return X

    def fit(self, X, y=None):
        """
        Fits the QuantileTransformer to the numerical data.
        Adjusts n_quantiles automatically for small datasets.

        Args:
            X: DataFrame or numpy array containing numerical features.
            y: Ignored.

        Returns:
            self
        """
        data = self._extract_data(X)
        n_samples = data.shape[0]

        # Adjust n_quantiles if sample size is small
        # QuantileTransformer requires n_quantiles <= n_samples
        default_quantiles = 1000
        if n_samples < default_quantiles:
            logger.info(
                f"Sample size ({n_samples}) is smaller than default n_quantiles ({default_quantiles}). "
                f"Adjusting n_quantiles to {n_samples}."
            )
            self.scaler.n_quantiles = n_samples
        else:
            self.scaler.n_quantiles = default_quantiles

        logger.info(f"Fitting MetadataScaler on {n_samples} samples...")
        self.scaler.fit(data)
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted QuantileTransformer.

        Args:
            X: DataFrame or numpy array containing numerical features.

        Returns:
            np.ndarray: Transformed features (RankGauss scaled).
        """
        data = self._extract_data(X)
        return self.scaler.transform(data)

    def get_feature_names_out(self, input_features=None):
        """
        Returns the names of the numerical features.
        """
        return np.array(self.numeric_cols)
