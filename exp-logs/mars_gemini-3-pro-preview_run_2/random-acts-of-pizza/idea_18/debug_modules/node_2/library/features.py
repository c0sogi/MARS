import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, QuantileTransformer
from sklearn.impute import SimpleImputer
from library.config import Config


class UserPersonaTransformer(BaseEstimator, TransformerMixin):
    """
    View 2: Latent User Persona (LSA).

    Pipelines:
    1. Extraction of subreddit text string.
    2. TF-IDF Vectorization.
    3. TruncatedSVD (Latent Semantic Analysis).
    4. L2 Normalization.
    """

    def __init__(
        self,
        subreddit_col=Config.SUBREDDIT_COL,
        n_components=Config.LSA_N_COMPONENTS,
        min_df=2,
        random_state=Config.SEED,
    ):
        self.subreddit_col = subreddit_col
        self.n_components = n_components
        self.min_df = min_df
        self.random_state = random_state

        # Initialize components
        # token_pattern matches alphanumeric strings (subreddits)
        self.tfidf = TfidfVectorizer(
            min_df=self.min_df, max_df=0.9, token_pattern=r"(?u)\b\w+\b"
        )
        self.svd = TruncatedSVD(
            n_components=self.n_components, random_state=self.random_state
        )

    def fit(self, X, y=None):
        """
        Fits the TF-IDF and SVD models on the subreddit strings.

        Args:
            X (pd.DataFrame): Input dataframe containing 'subreddit_col'.
            y (None): Ignored.
        """
        # Extract text column
        if isinstance(X, pd.DataFrame):
            text_data = X[self.subreddit_col].fillna("").astype(str)
        else:
            # Assuming X is already the series/array of strings
            text_data = X

        # Fit TF-IDF
        # We transform immediately to pass to SVD
        tfidf_matrix = self.tfidf.fit_transform(text_data)

        # Fit SVD
        # Ensure n_components <= n_features
        n_features = tfidf_matrix.shape[1]
        if n_features <= self.n_components:
            # Adjust components if vocabulary is too small (rare edge case)
            self.svd.n_components = max(1, n_features - 1)
        else:
            self.svd.n_components = self.n_components

        self.svd.fit(tfidf_matrix)

        return self

    def transform(self, X):
        """
        Transforms the input data into the latent persona vector.

        Args:
            X (pd.DataFrame): Input dataframe.

        Returns:
            np.ndarray: L2-normalized latent vectors.
        """
        if isinstance(X, pd.DataFrame):
            text_data = X[self.subreddit_col].fillna("").astype(str)
        else:
            text_data = X

        # TF-IDF Transform
        tfidf_matrix = self.tfidf.transform(text_data)

        # SVD Transform
        latent_matrix = self.svd.transform(tfidf_matrix)

        # L2 Normalization (Project to hypersphere)
        # Handles zero-vectors gracefully (remains zero)
        normalized_matrix = normalize(latent_matrix, norm="l2", axis=1)

        return normalized_matrix


class MetadataTransformer(BaseEstimator, TransformerMixin):
    """
    View 3: Robust Metadata.

    Pipelines:
    1. Selection of numerical columns.
    2. Imputation (Safety check).
    3. QuantileTransformer (RankGauss) to normalize distributions.
    """

    def __init__(self, numerical_cols=None, random_state=Config.SEED):
        # Use default from Config if not provided, but allow override for flexibility
        self.numerical_cols = (
            numerical_cols if numerical_cols is not None else Config.NUMERICAL_COLS
        )
        self.random_state = random_state

        self.imputer = SimpleImputer(strategy="constant", fill_value=0.0)
        self.scaler = QuantileTransformer(
            output_distribution="normal", random_state=self.random_state
        )

    def fit(self, X, y=None):
        """
        Fits the Imputer and QuantileTransformer.

        Args:
            X (pd.DataFrame): Input dataframe.
            y (None): Ignored.
        """
        # Select columns
        X_num = X[self.numerical_cols]

        # Fit Imputer
        X_imputed = self.imputer.fit_transform(X_num)

        # Fit Scaler
        self.scaler.fit(X_imputed)

        return self

    def transform(self, X):
        """
        Transforms the numerical metadata.

        Args:
            X (pd.DataFrame): Input dataframe.

        Returns:
            np.ndarray: Transformed features.
        """
        # Select columns
        # Handle case where columns might be missing (fill with 0)
        # This adds robustness if used on raw test data with missing cols
        X_num = pd.DataFrame(index=X.index)
        for col in self.numerical_cols:
            if col in X.columns:
                X_num[col] = X[col]
            else:
                X_num[col] = 0.0

        # Impute
        X_imputed = self.imputer.transform(X_num)

        # Scale
        X_scaled = self.scaler.transform(X_imputed)

        return X_scaled
