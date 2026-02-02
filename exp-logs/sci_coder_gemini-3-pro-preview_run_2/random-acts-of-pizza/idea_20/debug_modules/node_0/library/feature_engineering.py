import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from library.config import Config


class SubredditPLSProjector(BaseEstimator, TransformerMixin):
    """
    Implements View 2: Supervised User Persona.
    Projects subreddit history into a low-dimensional dense space using
    TF-IDF followed by Partial Least Squares (PLS) Regression.
    """

    def __init__(self, n_components=10):
        self.n_components = n_components
        # TF-IDF: Unigrams, min_df=5 to prune rare subreddits
        self.tfidf = TfidfVectorizer(
            min_df=5,
            binary=False,
            use_idf=True,
            norm="l2",
            token_pattern=r"(?u)\b\w+\b",  # Simple word tokenization
        )
        # PLS: Supervised dimensionality reduction
        self.pls = PLSRegression(n_components=n_components, scale=True)
        # Scaler: Normalize components
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """
        Fits the TF-IDF -> PLS -> Scaler pipeline.

        Args:
            X (pd.Series or list): List of space-separated subreddit strings.
            y (array-like): Target labels. Required for PLS.
        """
        if y is None:
            raise ValueError(
                "SubredditPLSProjector requires target 'y' for fitting (Supervised)."
            )

        # 1. TF-IDF Vectorization
        X_tfidf = self.tfidf.fit_transform(X)

        # 2. Convert to Dense (PLS requirement)
        # Given 220GB RAM, converting ~20k features x ~3k rows to dense is safe.
        X_dense = X_tfidf.toarray()

        # 3. Fit PLS
        self.pls.fit(X_dense, y)

        # 4. Fit Scaler on the projected training data
        X_projected = self.pls.transform(X_dense)
        self.scaler.fit(X_projected)

        return self

    def transform(self, X):
        """
        Applies the transformation pipeline.

        Args:
            X (pd.Series or list): List of space-separated subreddit strings.

        Returns:
            np.ndarray: Scaled PLS components.
        """
        # 1. TF-IDF
        X_tfidf = self.tfidf.transform(X)

        # 2. Dense
        X_dense = X_tfidf.toarray()

        # 3. PLS Projection
        X_projected = self.pls.transform(X_dense)

        # 4. Scaling
        X_scaled = self.scaler.transform(X_projected)

        return X_scaled


class MetadataScaler(BaseEstimator, TransformerMixin):
    """
    Implements View 3: Robust Metadata.
    Applies RankGauss (QuantileTransformer) to numerical metadata.
    """

    def __init__(self):
        # Output distribution normal handles outliers and scales to ~N(0,1)
        self.qt = QuantileTransformer(
            output_distribution="normal", random_state=Config.SEED
        )

    def fit(self, X, y=None):
        """
        Fits the QuantileTransformer.

        Args:
            X (pd.DataFrame or np.ndarray): Numerical features.
            y: Ignored.
        """
        self.qt.fit(X)
        return self

    def transform(self, X):
        """
        Transforms the data.
        """
        return self.qt.transform(X)


def assemble_feature_matrix(sbert_embeddings, pls_features, metadata_features):
    """
    Concatenates the processed features from all three views into a single matrix.

    Args:
        sbert_embeddings (np.ndarray): View 1 (Text Semantics)
        pls_features (np.ndarray): View 2 (User Persona)
        metadata_features (np.ndarray): View 3 (Metadata)

    Returns:
        np.ndarray: Fused feature matrix.
    """
    # Validate inputs
    if not isinstance(sbert_embeddings, np.ndarray):
        raise TypeError("sbert_embeddings must be a numpy array")
    if not isinstance(pls_features, np.ndarray):
        raise TypeError("pls_features must be a numpy array")
    if not isinstance(metadata_features, np.ndarray):
        raise TypeError("metadata_features must be a numpy array")

    # Check sample alignment
    n_samples = sbert_embeddings.shape[0]
    if pls_features.shape[0] != n_samples:
        raise ValueError(
            f"Sample count mismatch: SBERT={n_samples}, PLS={pls_features.shape[0]}"
        )
    if metadata_features.shape[0] != n_samples:
        raise ValueError(
            f"Sample count mismatch: SBERT={n_samples}, Meta={metadata_features.shape[0]}"
        )

    # Early Fusion: Concatenate features
    fused_matrix = np.hstack([sbert_embeddings, pls_features, metadata_features])

    return fused_matrix
