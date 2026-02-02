import os
import joblib
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from library.config import Config


class TextPipeline:
    """
    Manages text transformation pipelines including TF-IDF and Truncated SVD (LSA).
    """

    def __init__(self):
        """
        Initialize the pipeline with configurations.
        """
        self.tfidf = TfidfVectorizer(**Config.TFIDF_PARAMS)
        self.svd = TruncatedSVD(
            n_components=Config.SVD_N_COMPONENTS, random_state=Config.SVD_RANDOM_STATE
        )
        self.is_fitted = False

    def fit(self, corpus: list, load_cached_models: bool = True):
        """
        Fits the TF-IDF and SVD models on the provided corpus.
        Implements caching to avoid re-computation.

        Args:
            corpus (list): List of text strings (typically markdown cells).
            load_cached_models (bool): Whether to attempt loading from cache.
        """
        tfidf_path = Config.CACHE_TFIDF_VECTORIZER
        svd_path = Config.CACHE_SVD_MODEL

        # 1. Try Loading from Cache
        if (
            load_cached_models
            and os.path.exists(tfidf_path)
            and os.path.exists(svd_path)
        ):
            print(f"Loading TF-IDF model from {tfidf_path}")
            self.tfidf = joblib.load(tfidf_path)

            print(f"Loading SVD model from {svd_path}")
            self.svd = joblib.load(svd_path)

            self.is_fitted = True
            return

        # 2. Fit from Scratch
        print("Fitting TF-IDF and SVD models from scratch...")

        # Fit TF-IDF
        print(f"Fitting TF-IDF on corpus of size {len(corpus)}...")
        X_tfidf = self.tfidf.fit_transform(corpus)
        print(f"TF-IDF Vocabulary size: {len(self.tfidf.vocabulary_)}")

        # Fit SVD on the TF-IDF matrix
        print(f"Fitting SVD (n_components={Config.SVD_N_COMPONENTS})...")
        self.svd.fit(X_tfidf)
        explained_variance = self.svd.explained_variance_ratio_.sum()
        print(f"SVD Explained Variance Ratio: {explained_variance}")

        self.is_fitted = True

        # 3. Save to Cache
        print(f"Saving TF-IDF model to {tfidf_path}")
        os.makedirs(os.path.dirname(tfidf_path), exist_ok=True)
        joblib.dump(self.tfidf, tfidf_path)

        print(f"Saving SVD model to {svd_path}")
        os.makedirs(os.path.dirname(svd_path), exist_ok=True)
        joblib.dump(self.svd, svd_path)

    def transform(self, text_list: list):
        """
        Transforms the input text list into both sparse (TF-IDF) and dense (SVD) vectors.

        Args:
            text_list (list): List of text strings to transform.

        Returns:
            tuple: (scipy.sparse.csr_matrix, np.ndarray)
                   - Sparse TF-IDF features
                   - Dense SVD features
        """
        if not self.is_fitted:
            raise RuntimeError("TextPipeline must be fitted before calling transform.")

        # Transform to TF-IDF (Sparse)
        X_tfidf = self.tfidf.transform(text_list)

        # Transform to SVD (Dense)
        X_svd = self.svd.transform(X_tfidf)

        return X_tfidf, X_svd
