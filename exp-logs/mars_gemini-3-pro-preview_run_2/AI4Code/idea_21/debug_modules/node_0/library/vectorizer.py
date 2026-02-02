import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from library.config import Config


class DualVectorizer:
    """
    Manages Multi-Resolution Vectorization for the Stacked Hybrid Ranking model.

    Resolution A: Lexical (Sparse TF-IDF) - Captures exact keyword matches.
    Resolution B: Latent (Dense SVD) - Captures semantic/topic matches.
    """

    def __init__(self):
        """
        Initializes the vectorizers using configurations from library.config.
        """
        self.tfidf = TfidfVectorizer(**Config.TFIDF_PARAMS)
        self.svd = TruncatedSVD(**Config.SVD_PARAMS)
        self.is_fitted = False

    def fit(self, raw_documents):
        """
        Fits the TF-IDF vectorizer on the documents, then fits the SVD on the
        resulting TF-IDF matrix.

        Args:
            raw_documents (iterable): An iterable of string documents.

        Returns:
            self
        """
        print("Fitting TF-IDF Vectorizer...")
        tfidf_matrix = self.tfidf.fit_transform(raw_documents)

        print(f"Fitting Truncated SVD (n_components={self.svd.n_components})...")
        self.svd.fit(tfidf_matrix)

        self.is_fitted = True
        return self

    def transform(self, raw_documents):
        """
        Transforms documents into both Sparse TF-IDF and Dense SVD representations.

        Args:
            raw_documents (iterable): An iterable of string documents.

        Returns:
            tuple: (tfidf_matrix, svd_matrix)
                - tfidf_matrix: scipy.sparse.csr_matrix
                - svd_matrix: numpy.ndarray
        """
        if not self.is_fitted:
            raise ValueError("Vectorizers are not fitted. Call fit() or load() first.")

        tfidf_matrix = self.tfidf.transform(raw_documents)
        svd_matrix = self.svd.transform(tfidf_matrix)

        return tfidf_matrix, svd_matrix

    def save(self, directory):
        """
        Saves the fitted vectorizers to the specified directory using joblib.

        Args:
            directory (str): Path to the directory where models should be saved.
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted vectorizers.")

        os.makedirs(directory, exist_ok=True)

        tfidf_path = os.path.join(directory, "text_vectorizer_tfidf.joblib")
        svd_path = os.path.join(directory, "text_vectorizer_svd.joblib")

        joblib.dump(self.tfidf, tfidf_path)
        joblib.dump(self.svd, svd_path)
        print(f"Vectorizers saved to {directory}")

    def load(self, directory):
        """
        Loads the vectorizers from the specified directory.

        Args:
            directory (str): Path to the directory containing the model files.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        tfidf_path = os.path.join(directory, "text_vectorizer_tfidf.joblib")
        svd_path = os.path.join(directory, "text_vectorizer_svd.joblib")

        if os.path.exists(tfidf_path) and os.path.exists(svd_path):
            print(f"Loading vectorizers from {directory}...")
            self.tfidf = joblib.load(tfidf_path)
            self.svd = joblib.load(svd_path)
            self.is_fitted = True
            return True

        return False

    def fit_or_load(self, raw_documents, cache_dir=None, load_cached_data=True):
        """
        Implements the strict caching logic:
        1. Try to load from cache if enabled.
        2. If fail or disabled, fit on data.
        3. Save to cache.

        Args:
            raw_documents (iterable): Documents to fit on if cache is missing.
            cache_dir (str, optional): Directory to cache models. Defaults to Config.WORKING_DIR.
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        if cache_dir is None:
            cache_dir = Config.WORKING_DIR

        # 1. Try to load
        if load_cached_data:
            if self.load(cache_dir):
                print("Vectorizers loaded successfully.")
                return self
            else:
                print("Cached vectorizers not found. Proceeding to fit...")
        else:
            print("Cache loading disabled. Proceeding to fit...")

        # 2. Fit
        self.fit(raw_documents)

        # 3. Save
        self.save(cache_dir)

        return self
