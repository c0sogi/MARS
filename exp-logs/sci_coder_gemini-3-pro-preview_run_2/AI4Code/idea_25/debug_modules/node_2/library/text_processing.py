import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from library.config import (
    TFIDF_PARAMS,
    SVD_N_COMPONENTS,
    SVD_RANDOM_STATE,
    WORKING_DIR,
    RANDOM_STATE,
)
from library.utils import seed_everything


class TextVectorizer:
    """
    Manages the decoupled vectorization of text data using TF-IDF (Lexical View)
    and Truncated SVD (Latent View).
    """

    def __init__(self):
        seed_everything(RANDOM_STATE)

        # Initialize model parameters from config
        self.tfidf_params = TFIDF_PARAMS.copy()
        self.svd_n_components = SVD_N_COMPONENTS
        self.svd_random_state = SVD_RANDOM_STATE

        # Instantiate models
        self.tfidf_model = TfidfVectorizer(**self.tfidf_params)
        self.svd_model = TruncatedSVD(
            n_components=self.svd_n_components, random_state=self.svd_random_state
        )

        # Define cache paths
        self.tfidf_path = os.path.join(WORKING_DIR, "tfidf_vectorizer.joblib")
        self.svd_path = os.path.join(WORKING_DIR, "svd_model.joblib")

        self.is_fitted = False

    def fit(self, texts, save_models=True):
        """
        Fits the TF-IDF and SVD models on the provided texts.

        Args:
            texts (pd.Series or list): The text data to fit on (typically markdown cells).
            save_models (bool): Whether to save the fitted models to disk.

        Returns:
            self
        """
        print("Fitting TfidfVectorizer...")
        # Ensure input is string and handle NaNs
        texts = pd.Series(texts).fillna("").astype(str)

        # Fit TF-IDF
        tfidf_matrix = self.tfidf_model.fit_transform(texts)

        print(f"Fitting TruncatedSVD (n_components={self.svd_n_components})...")
        # Fit SVD on the TF-IDF matrix
        self.svd_model.fit(tfidf_matrix)

        self.is_fitted = True

        if save_models:
            self._save_models()

        return self

    def transform(self, texts):
        """
        Transforms the input texts into TF-IDF and SVD representations.

        Args:
            texts (pd.Series or list): The text data to transform.

        Returns:
            tuple: (tfidf_matrix (scipy.sparse.csr_matrix), svd_matrix (numpy.ndarray))
        """
        if not self.is_fitted:
            # Attempt to load if not explicitly fitted in this session
            if not self._load_models():
                raise RuntimeError(
                    "Models are not fitted and could not be loaded from cache."
                )

        # Ensure input is string and handle NaNs
        texts = pd.Series(texts).fillna("").astype(str)

        # Transform using TF-IDF
        tfidf_matrix = self.tfidf_model.transform(texts)

        # Transform using SVD (project into latent space)
        svd_matrix = self.svd_model.transform(tfidf_matrix)

        return tfidf_matrix, svd_matrix

    def fit_transform(self, texts, save_models=True):
        """
        Fits and transforms the data in one step.
        """
        self.fit(texts, save_models=save_models)
        return self.transform(texts)

    def _save_models(self):
        """Saves models to disk using joblib."""
        try:
            os.makedirs(WORKING_DIR, exist_ok=True)
            joblib.dump(self.tfidf_model, self.tfidf_path)
            joblib.dump(self.svd_model, self.svd_path)
            print(f"Models saved to {WORKING_DIR}")
        except Exception as e:
            print(f"Failed to save models: {e}")

    def _load_models(self):
        """Loads models from disk if they exist."""
        if os.path.exists(self.tfidf_path) and os.path.exists(self.svd_path):
            try:
                self.tfidf_model = joblib.load(self.tfidf_path)
                self.svd_model = joblib.load(self.svd_path)
                self.is_fitted = True
                print(f"Models loaded from {WORKING_DIR}")
                return True
            except Exception as e:
                print(f"Failed to load models: {e}")
                return False
        return False


def get_vectorizer(train_texts=None, load_cached=True):
    """
    Factory function to retrieve a ready-to-use vectorizer.
    Implements the caching logic: Load if available, else fit and save.

    Args:
        train_texts (pd.Series, optional): Text data to fit on if loading fails.
        load_cached (bool): Whether to attempt loading from disk.

    Returns:
        TextVectorizer: A fitted vectorizer instance.
    """
    vectorizer = TextVectorizer()

    loaded = False
    if load_cached:
        loaded = vectorizer._load_models()

    if not loaded:
        if train_texts is None:
            raise ValueError(
                "Cached models not found and no training data provided to fit."
            )

        # Fit on provided data and save
        vectorizer.fit(train_texts, save_models=True)

    return vectorizer
