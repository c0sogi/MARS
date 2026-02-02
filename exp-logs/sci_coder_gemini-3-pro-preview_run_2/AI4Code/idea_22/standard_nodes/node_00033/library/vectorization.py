import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from library.config import Config
from library.utils import save_artifacts


class TextPipeline:
    """
    Manages the text vectorization pipeline using TF-IDF and Truncated SVD (LSA).
    Handles fitting, transforming, and persistence of the vectorizer models.
    """

    def __init__(self):
        """
        Initializes the vectorizer and SVD models with configuration parameters.
        """
        self.tfidf = TfidfVectorizer(**Config.TFIDF_PARAMS)
        self.svd = TruncatedSVD(**Config.SVD_PARAMS)
        self.is_fitted = False

    def fit_transform_corpus(self, df, load_cached_model=True):
        """
        Fits the TF-IDF and SVD models on the markdown cells of the training corpus.
        Implements caching to avoid re-fitting if models already exist.

        Args:
            df (pd.DataFrame): The training dataframe containing 'cell_type' and 'source' columns.
            load_cached_model (bool): If True, attempts to load models from disk.

        Returns:
            self: Returns the instance itself.
        """
        tfidf_path = Config.TFIDF_MODEL_PATH
        svd_path = Config.SVD_MODEL_PATH

        # 1. Try to load from cache
        if (
            load_cached_model
            and os.path.exists(tfidf_path)
            and os.path.exists(svd_path)
        ):
            print(f"Loading cached text models from {tfidf_path} and {svd_path}")
            try:
                self.tfidf = joblib.load(tfidf_path)
                self.svd = joblib.load(svd_path)
                self.is_fitted = True
                return self
            except Exception as e:
                print(f"Failed to load cached models: {e}. Re-fitting...")

        # 2. Fit from scratch
        print("Fitting TF-IDF and SVD on markdown corpus...")

        if df is None:
            raise ValueError(
                "Dataframe cannot be None when fitting models from scratch."
            )

        # Extract markdown cells only
        mask_md = df["cell_type"] == "markdown"
        markdown_corpus = df.loc[mask_md, "source"].astype(str).fillna("")

        print(f"Training on {len(markdown_corpus)} markdown cells...")

        # Fit TF-IDF
        X_tfidf = self.tfidf.fit_transform(markdown_corpus)

        # Fit SVD on the TF-IDF matrix
        self.svd.fit(X_tfidf)

        self.is_fitted = True
        print(
            f"Explained Variance Ratio (Sum): {self.svd.explained_variance_ratio_.sum():.6f}"
        )

        # 3. Save models
        print(f"Saving models to {tfidf_path} and {svd_path}")
        save_artifacts(self.tfidf, tfidf_path)
        save_artifacts(self.svd, svd_path)

        return self

    def transform_cells(self, text_series):
        """
        Projects a series of text (code or markdown) into the Latent Semantic Space (SVD).

        Args:
            text_series (pd.Series or list): The text content to transform.

        Returns:
            np.ndarray: The dense SVD vectors (n_samples, n_components).
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Pipeline must be fitted before calling transform_cells."
            )

        # Ensure string format and handle NaNs
        text_clean = pd.Series(text_series).astype(str).fillna("")

        # Transform to TF-IDF sparse matrix
        X_tfidf = self.tfidf.transform(text_clean)

        # Project to SVD dense matrix
        X_svd = self.svd.transform(X_tfidf)

        return X_svd

    def transform_tfidf(self, text_series):
        """
        Transforms text to TF-IDF sparse vectors (used for Stage 1 Ridge).

        Args:
            text_series (pd.Series or list): The text content to transform.

        Returns:
            scipy.sparse.csr_matrix: The sparse TF-IDF matrix.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Pipeline must be fitted before calling transform_tfidf."
            )

        text_clean = pd.Series(text_series).astype(str).fillna("")
        return self.tfidf.transform(text_clean)
