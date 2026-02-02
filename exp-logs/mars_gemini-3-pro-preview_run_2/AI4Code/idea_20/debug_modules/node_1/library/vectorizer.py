import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from library.config import Config


class TextVectorizer:
    """
    Wrapper class for TF-IDF and Truncated SVD vectorization.
    Manages the lifecycle of the vectorizers and provides methods for
    transforming text into sparse (lexical) and dense (latent) representations.
    """

    def __init__(self):
        self.tfidf = TfidfVectorizer(**Config.TFIDF_PARAMS)
        self.svd = TruncatedSVD(**Config.SVD_PARAMS)
        self.is_fitted = False

    def fit(self, texts):
        """
        Fits the TF-IDF vectorizer on the provided texts, then fits the
        Truncated SVD on the resulting sparse matrix.

        Args:
            texts (iterable): An iterable of text strings (corpus).
        """
        print("Fitting TF-IDF Vectorizer...")
        tfidf_matrix = self.tfidf.fit_transform(texts)

        print(f"Fitting Truncated SVD on TF-IDF matrix (shape={tfidf_matrix.shape})...")
        self.svd.fit(tfidf_matrix)

        self.is_fitted = True
        print("Vectorizers fitted successfully.")

    def transform(self, texts):
        """
        Transforms texts into a sparse TF-IDF matrix.

        Args:
            texts (iterable): An iterable of text strings.

        Returns:
            scipy.sparse.csr_matrix: The sparse TF-IDF representation.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "TextVectorizer must be fitted before calling transform."
            )
        return self.tfidf.transform(texts)

    def transform_svd(self, texts):
        """
        Transforms texts into a dense SVD (latent) matrix.
        First applies TF-IDF transform, then SVD transform.

        Args:
            texts (iterable): An iterable of text strings.

        Returns:
            np.ndarray: The dense latent representation.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "TextVectorizer must be fitted before calling transform_svd."
            )

        # Transform to TF-IDF first
        tfidf_matrix = self.tfidf.transform(texts)
        # Transform to SVD space
        return self.svd.transform(tfidf_matrix)

    def save(self, base_path):
        """
        Saves the fitted vectorizers to disk.

        Args:
            base_path (str): Base path (without extension) to save the models.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted TextVectorizer.")

        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        tfidf_path = f"{base_path}_tfidf.joblib"
        svd_path = f"{base_path}_svd.joblib"

        joblib.dump(self.tfidf, tfidf_path)
        joblib.dump(self.svd, svd_path)
        print(f"Vectorizers saved to {tfidf_path} and {svd_path}")

    def load(self, base_path):
        """
        Loads fitted vectorizers from disk.

        Args:
            base_path (str): Base path (without extension) from where to load.
        """
        tfidf_path = f"{base_path}_tfidf.joblib"
        svd_path = f"{base_path}_svd.joblib"

        if not os.path.exists(tfidf_path) or not os.path.exists(svd_path):
            raise FileNotFoundError(f"Model files not found at {base_path}_*.joblib")

        self.tfidf = joblib.load(tfidf_path)
        self.svd = joblib.load(svd_path)
        self.is_fitted = True
        print(f"Vectorizers loaded from {base_path}")


def get_svd_features(
    df: pd.DataFrame,
    vectorizer: TextVectorizer,
    split_name: str,
    load_cached_data: bool = True,
) -> np.ndarray:
    """
    Generates or loads SVD features for a given DataFrame.
    Implements caching using .npy files.

    Args:
        df (pd.DataFrame): DataFrame containing a 'source' column with text.
        vectorizer (TextVectorizer): A fitted TextVectorizer instance.
        split_name (str): Name of the split (e.g., 'train', 'val', 'test') for cache naming.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: The dense SVD features matrix.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache path
    cache_filename = f"{split_name}_svd_features.npy"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached SVD features for {split_name} from {cache_path}")
        try:
            features = np.load(cache_path)
            if len(features) == len(df):
                return features
            else:
                print(
                    f"Cache size mismatch ({len(features)} vs {len(df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Computing SVD features for {split_name}...")

    # Handle missing source values
    texts = df["source"].fillna("").astype(str).tolist()

    features = vectorizer.transform_svd(texts)

    # 3. Save to cache
    print(f"Saving SVD features to {cache_path}")
    np.save(cache_path, features)

    return features
