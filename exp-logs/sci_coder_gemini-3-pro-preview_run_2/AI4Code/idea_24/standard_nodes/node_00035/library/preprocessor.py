import os
import joblib
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from library.config import (
    CACHE_DIR,
    VOCAB_SIZE,
    NGRAM_RANGE,
    SUBLINEAR_TF,
    USE_IDF,
    STRIP_ACCENTS,
    SVD_COMPONENTS,
    SVD_RANDOM_STATE,
)


class VectorizationPipeline:
    """
    Wraps TfidfVectorizer and TruncatedSVD to provide a unified text processing pipeline.
    Manages both the sparse 'Lexical' view and the dense 'Latent' view.
    """

    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=VOCAB_SIZE,
            ngram_range=NGRAM_RANGE,
            sublinear_tf=SUBLINEAR_TF,
            use_idf=USE_IDF,
            strip_accents=STRIP_ACCENTS,
            min_df=2,
            token_pattern=r"(?u)\b\w\w+\b",
        )
        self.svd = TruncatedSVD(
            n_components=SVD_COMPONENTS, random_state=SVD_RANDOM_STATE
        )
        self.is_fitted = False

    def fit(self, corpus):
        """
        Fits the TF-IDF and SVD models on the provided text corpus.
        """
        print("Fitting TF-IDF Vectorizer...")
        tfidf_matrix = self.tfidf.fit_transform(corpus)

        print(f"Fitting Truncated SVD on matrix shape {tfidf_matrix.shape}...")
        self.svd.fit(tfidf_matrix)

        self.is_fitted = True
        return self

    def transform(self, corpus):
        """
        Transforms text into both sparse TF-IDF and dense SVD representations.
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transformation.")

        # Lexical View (Sparse)
        tfidf_matrix = self.tfidf.transform(corpus)

        # Latent View (Dense)
        svd_matrix = self.svd.transform(tfidf_matrix)

        return tfidf_matrix, svd_matrix

    def save(self, cache_dir):
        """
        Saves the fitted models to the cache directory using joblib.
        """
        os.makedirs(cache_dir, exist_ok=True)
        tfidf_path = os.path.join(cache_dir, "tfidf_vectorizer.joblib")
        svd_path = os.path.join(cache_dir, "svd_model.joblib")

        joblib.dump(self.tfidf, tfidf_path)
        joblib.dump(self.svd, svd_path)
        print(f"Vectorization pipeline saved to {cache_dir}")

    def load(self, cache_dir):
        """
        Attempts to load fitted models from the cache directory.
        Returns True if successful, False otherwise.
        """
        tfidf_path = os.path.join(cache_dir, "tfidf_vectorizer.joblib")
        svd_path = os.path.join(cache_dir, "svd_model.joblib")

        if os.path.exists(tfidf_path) and os.path.exists(svd_path):
            try:
                self.tfidf = joblib.load(tfidf_path)
                self.svd = joblib.load(svd_path)
                self.is_fitted = True
                print(f"Vectorization pipeline loaded from {cache_dir}")
                return True
            except Exception as e:
                print(f"Failed to load cached models: {e}")
                return False
        return False


def fit_transform_corpus(
    df_train: pd.DataFrame, load_cached_models: bool = True
) -> VectorizationPipeline:
    """
    Orchestrates the fitting of the vectorization pipeline.
    Uses the markdown cells of the training dataframe to define the vector space.

    Args:
        df_train: DataFrame containing 'cell_type' and 'source' columns.
        load_cached_models: If True, attempts to load models from CACHE_DIR.

    Returns:
        VectorizationPipeline: The fitted pipeline object.
    """
    pipeline = VectorizationPipeline()

    # 1. Try Loading from Cache
    if load_cached_models:
        if pipeline.load(CACHE_DIR):
            return pipeline

    # 2. Fit from Scratch
    print("Training vectorization pipeline from scratch...")

    # Filter for markdown cells only to build the vocabulary/concept space
    # Code cells will be projected into this space later
    markdown_mask = df_train["cell_type"] == "markdown"
    corpus = df_train.loc[markdown_mask, "source"].astype(str).fillna("")

    if len(corpus) == 0:
        raise ValueError("No markdown cells found in training data to fit vectorizer.")

    pipeline.fit(corpus)

    # 3. Save to Cache
    pipeline.save(CACHE_DIR)

    return pipeline


def transform_cells(df: pd.DataFrame, pipeline: VectorizationPipeline):
    """
    Transforms the 'source' text of the provided dataframe using the fitted pipeline.
    Handles both code and markdown cells.

    Args:
        df: DataFrame containing a 'source' column.
        pipeline: A fitted VectorizationPipeline instance.

    Returns:
        tuple: (tfidf_matrix (sparse), svd_matrix (dense))
    """
    # Ensure all inputs are strings and handle NaNs
    corpus = df["source"].astype(str).fillna("")
    return pipeline.transform(corpus)
