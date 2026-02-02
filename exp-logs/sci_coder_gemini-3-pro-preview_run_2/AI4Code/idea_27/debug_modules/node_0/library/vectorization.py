import os
import gc
import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from library.config import Config


class TextProcessor:
    def __init__(self, config=None):
        """
        Initialize the TextProcessor with configuration settings.
        """
        self.config = config if config else Config()
        self.model_dir = self.config.WORKING_DIR
        os.makedirs(self.model_dir, exist_ok=True)

        self.tfidf = TfidfVectorizer(
            max_features=self.config.TFIDF_MAX_FEATURES,
            ngram_range=self.config.TFIDF_NGRAM_RANGE,
            sublinear_tf=self.config.TFIDF_SUBLINEAR_TF,
            use_idf=self.config.TFIDF_USE_IDF,
            strip_accents="unicode",
        )

        self.svd = TruncatedSVD(
            n_components=self.config.SVD_COMPONENTS,
            random_state=self.config.SVD_RANDOM_STATE,
        )

    def fit_pipeline(self, df_corpus, load_cached_models=True):
        """
        Fits the TF-IDF and SVD models on the markdown cells of the training corpus.

        Args:
            df_corpus (pd.DataFrame): Training corpus containing 'cell_type' and 'source'.
            load_cached_models (bool): Whether to load models from disk if available.
        """
        tfidf_path = os.path.join(self.model_dir, "tfidf_model.joblib")
        svd_path = os.path.join(self.model_dir, "svd_model.joblib")

        # Try loading cached models
        if (
            load_cached_models
            and os.path.exists(tfidf_path)
            and os.path.exists(svd_path)
        ):
            try:
                self.tfidf = joblib.load(tfidf_path)
                self.svd = joblib.load(svd_path)
                return
            except Exception:
                # If loading fails, proceed to refit
                pass

        # Filter for markdown cells only for vocabulary learning
        # We only learn the vocabulary/latent space from the prose, not the code
        md_mask = df_corpus["cell_type"] == "markdown"
        md_sources = df_corpus.loc[md_mask, "source"].astype(str).fillna("").tolist()

        # Fit TF-IDF
        self.tfidf.fit(md_sources)

        # Transform for SVD fitting
        tfidf_mat = self.tfidf.transform(md_sources)

        # Fit SVD
        self.svd.fit(tfidf_mat)

        # Save models
        joblib.dump(self.tfidf, tfidf_path)
        joblib.dump(self.svd, svd_path)

        # Cleanup to free memory
        del md_sources, tfidf_mat
        gc.collect()

    def transform_cells(self, df_corpus, mode="train", load_cached_data=True):
        """
        Transforms the corpus into TF-IDF and SVD representations.
        Projects both code and markdown cells into the shared vector spaces.

        Args:
            df_corpus (pd.DataFrame): Corpus to transform.
            mode (str): Mode identifier (train/val/test) for caching.
            load_cached_data (bool): Whether to load data from disk if available.

        Returns:
            tuple: (tfidf_matrix, svd_matrix)
        """
        tfidf_cache_path = os.path.join(self.model_dir, f"{mode}_tfidf.npz")
        svd_cache_path = os.path.join(self.model_dir, f"{mode}_svd.npy")

        # Try loading cached data
        if (
            load_cached_data
            and os.path.exists(tfidf_cache_path)
            and os.path.exists(svd_cache_path)
        ):
            try:
                tfidf_mat = scipy.sparse.load_npz(tfidf_cache_path)
                svd_mat = np.load(svd_cache_path)
                # Verify shape matches current dataframe
                if tfidf_mat.shape[0] == len(df_corpus):
                    return tfidf_mat, svd_mat
            except Exception:
                # If loading fails or shape mismatch, proceed to recompute
                pass

        # Transform all cells (code and markdown)
        sources = df_corpus["source"].astype(str).fillna("").tolist()

        # Transform to Sparse TF-IDF
        tfidf_mat = self.tfidf.transform(sources)

        # Transform to Dense SVD
        svd_mat = self.svd.transform(tfidf_mat)

        # Save to cache
        scipy.sparse.save_npz(tfidf_cache_path, tfidf_mat)
        np.save(svd_cache_path, svd_mat)

        return tfidf_mat, svd_mat
