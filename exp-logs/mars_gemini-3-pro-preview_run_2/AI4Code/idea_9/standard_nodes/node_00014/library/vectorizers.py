import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from library.config import Config


class SemanticSpace:
    """
    Manages the vectorization of text data into explicit (TF-IDF) and
    latent (SVD/LSA) representations.
    """

    def __init__(self):
        self.tfidf = None
        self.svd = None

        # Define paths for model persistence
        self.tfidf_path = os.path.join(Config.WORKING_DIR, "tfidf_vectorizer.joblib")
        self.svd_path = os.path.join(Config.WORKING_DIR, "svd_model.joblib")

    def fit(self, df=None, load_cached_models=True):
        """
        Fits the TF-IDF and SVD models on the markdown cells of the provided DataFrame.
        If cached models exist and load_cached_models is True, loads them instead.

        Args:
            df (pd.DataFrame, optional): DataFrame containing 'cell_type' and 'source' columns.
                                         Required if models are not cached.
            load_cached_models (bool): Whether to attempt loading from disk.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Attempt to load from cache
        if (
            load_cached_models
            and os.path.exists(self.tfidf_path)
            and os.path.exists(self.svd_path)
        ):
            print(f"Loading cached TF-IDF and SVD models from {Config.WORKING_DIR}...")
            try:
                self.tfidf = joblib.load(self.tfidf_path)
                self.svd = joblib.load(self.svd_path)
                return
            except Exception as e:
                print(
                    f"Failed to load cached models: {e}. Proceeding to fit from scratch."
                )

        if df is None:
            raise ValueError(
                "DataFrame 'df' is required to fit models when cache is missing or ignored."
            )

        print("Fitting TF-IDF and SVD models from scratch...")

        # Filter for markdown cells only for vocabulary and topic modeling
        # We convert to string and fillna to handle potential data issues
        markdown_mask = df["cell_type"] == "markdown"
        markdown_corpus = df.loc[markdown_mask, "source"].astype(str).fillna("")

        print(f"Training on {len(markdown_corpus)} markdown cells...")

        # 1. Fit TF-IDF
        self.tfidf = TfidfVectorizer(**Config.TFIDF_PARAMS)
        tfidf_matrix = self.tfidf.fit_transform(markdown_corpus)
        print(f"TF-IDF Vocabulary Size: {len(self.tfidf.vocabulary_)}")

        # 2. Fit SVD (Latent Semantic Analysis)
        self.svd = TruncatedSVD(
            n_components=Config.SVD_N_COMPONENTS, random_state=Config.SVD_RANDOM_STATE
        )
        self.svd.fit(tfidf_matrix)
        explained_variance = self.svd.explained_variance_ratio_.sum()
        print(
            f"SVD Explained Variance (n={Config.SVD_N_COMPONENTS}): {explained_variance:.6f}"
        )

        # 3. Save models
        print(f"Saving models to {Config.WORKING_DIR}...")
        joblib.dump(self.tfidf, self.tfidf_path)
        joblib.dump(self.svd, self.svd_path)

    def transform_tfidf(self, text_data):
        """
        Transforms text data into Sparse TF-IDF vectors.

        Args:
            text_data (pd.Series or list): Text content to transform.

        Returns:
            scipy.sparse.csr_matrix: Sparse TF-IDF representation.
        """
        if self.tfidf is None:
            raise RuntimeError("TF-IDF model is not fitted. Call fit() first.")

        # Ensure input is string and handle NaNs
        if isinstance(text_data, pd.Series):
            text_data = text_data.astype(str).fillna("")
        else:
            text_data = [str(t) if t is not None else "" for t in text_data]

        return self.tfidf.transform(text_data)

    def transform_svd(self, text_data):
        """
        Transforms text data into Dense SVD (LSA) vectors.
        First applies TF-IDF transform, then SVD projection.

        Args:
            text_data (pd.Series or list): Text content to transform.

        Returns:
            np.ndarray: Dense SVD representation.
        """
        if self.svd is None:
            raise RuntimeError("SVD model is not fitted. Call fit() first.")

        # Get sparse TF-IDF representation first
        tfidf_matrix = self.transform_tfidf(text_data)

        # Project into latent space
        return self.svd.transform(tfidf_matrix)
