import os
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

from library.config import Config
from library.utils import timer


class FeatureEngineer:
    """
    Handles feature engineering for the Hex-View Stacking Ensemble.
    Generates Lexical, Behavioral, Semantic, Interaction, and Metadata views.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.meta_cols = []

        # Initialize Transformers
        self.text_vectorizer = TfidfVectorizer(**Config.TEXT_VEC_PARAMS)
        self.subreddit_vectorizer = TfidfVectorizer(**Config.SUBREDDIT_VEC_PARAMS)

        self.svd_text = TruncatedSVD(
            n_components=Config.SVD_N_COMPONENTS_TEXT, random_state=Config.RANDOM_SEED
        )
        self.svd_history = TruncatedSVD(
            n_components=Config.SVD_N_COMPONENTS_HISTORY,
            random_state=Config.RANDOM_SEED,
        )

        self.scaler = StandardScaler()

        # Sentence Transformer model (loaded lazily or in transform to be safe)
        self.embedding_model = None

    def _get_embedding_model(self):
        if self.embedding_model is None:
            self.logger.info(f"Loading SentenceTransformer: {Config.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        return self.embedding_model

    def fit(self, df: pd.DataFrame):
        """
        Fits the vectorizers, scalers, and SVD models on the training data.
        """
        self.logger.info("Fitting feature extractors...")

        # 1. Identify Metadata Columns
        # Exclude non-numeric and special columns handled by other views
        exclude = set(
            [Config.ID_COL, Config.TARGET_COL, "text_combined", "subreddit_text"]
        )
        self.meta_cols = [c for c in df.columns if c not in exclude]

        # 2. Fit Scaler on Metadata
        self.logger.info(
            f"Fitting StandardScaler on {len(self.meta_cols)} metadata columns."
        )
        self.scaler.fit(df[self.meta_cols])

        # 3. Fit Text Vectorizer (Lexical)
        self.logger.info("Fitting Text TfidfVectorizer...")
        self.text_vectorizer.fit(df["text_combined"])

        # 4. Fit Subreddit Vectorizer (Behavioral)
        self.logger.info("Fitting Subreddit TfidfVectorizer...")
        self.subreddit_vectorizer.fit(df["subreddit_text"])

        # 5. Fit SVDs (Interaction)
        # We need to transform first to fit SVD on the sparse output
        self.logger.info("Fitting SVD for Interaction View...")
        X_text_sparse = self.text_vectorizer.transform(df["text_combined"])
        self.svd_text.fit(X_text_sparse)

        X_sub_sparse = self.subreddit_vectorizer.transform(df["subreddit_text"])
        self.svd_history.fit(X_sub_sparse)

        self.logger.info("FeatureEngineer fitting complete.")

    def transform(
        self, df: pd.DataFrame, split: str, load_cached_data: bool = True
    ) -> dict:
        """
        Transforms the data into the 5 feature views.
        Implements caching to disk.

        Args:
            df: Dataframe to transform.
            split: 'train', 'val', or 'test' (used for cache naming).
            load_cached_data: Whether to load from cache if available.

        Returns:
            Dictionary containing 'view_lexical', 'view_behavioral', 'view_semantic',
            'view_interaction', 'view_meta', and 'y'.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Define cache file paths
        cache_files = {
            "view_lexical": os.path.join(Config.WORKING_DIR, f"X_{split}_lexical.npz"),
            "view_behavioral": os.path.join(
                Config.WORKING_DIR, f"X_{split}_behavioral.npz"
            ),
            "view_semantic": os.path.join(
                Config.WORKING_DIR, f"X_{split}_semantic.npy"
            ),
            "view_interaction": os.path.join(
                Config.WORKING_DIR, f"X_{split}_interaction.npy"
            ),
            "view_meta": os.path.join(Config.WORKING_DIR, f"X_{split}_meta.npy"),
            "y": os.path.join(Config.WORKING_DIR, f"y_{split}.npy"),
        }

        # Check if all cache files exist
        all_cached = all(os.path.exists(p) for p in cache_files.values())

        if load_cached_data and all_cached:
            self.logger.info(f"Loading cached features for split: {split}")
            data = {}
            data["view_lexical"] = sp.load_npz(cache_files["view_lexical"])
            data["view_behavioral"] = sp.load_npz(cache_files["view_behavioral"])
            data["view_semantic"] = np.load(cache_files["view_semantic"])
            data["view_interaction"] = np.load(cache_files["view_interaction"])
            data["view_meta"] = np.load(cache_files["view_meta"])
            data["y"] = np.load(cache_files["y"])
            return data

        self.logger.info(f"Computing features for split: {split}")

        with timer(f"Transforming {split} data"):
            # 1. Lexical View (Sparse)
            X_lexical = self.text_vectorizer.transform(df["text_combined"])

            # 2. Behavioral View (Sparse)
            X_behavioral = self.subreddit_vectorizer.transform(df["subreddit_text"])

            # 3. Metadata View (Dense, Scaled)
            # Ensure columns match fit time
            if not self.meta_cols:
                # Fallback if fit wasn't called or meta_cols not set (should not happen in proper pipeline)
                exclude = set(
                    [
                        Config.ID_COL,
                        Config.TARGET_COL,
                        "text_combined",
                        "subreddit_text",
                    ]
                )
                self.meta_cols = [c for c in df.columns if c not in exclude]

            X_meta = self.scaler.transform(df[self.meta_cols])

            # 4. Semantic View (Dense Embeddings)
            model = self._get_embedding_model()
            # Encode returns numpy array
            X_semantic = model.encode(
                df["text_combined"].tolist(),
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # 5. Interaction View (Dense Low-Rank + Meta)
            # Project sparse views to low-rank dense
            X_text_svd = self.svd_text.transform(X_lexical)
            X_hist_svd = self.svd_history.transform(X_behavioral)

            # Concatenate: SVD_Text + SVD_History + Scaled_Meta
            X_interaction = np.hstack([X_text_svd, X_hist_svd, X_meta])

            # 6. Target
            if Config.TARGET_COL in df.columns:
                y = df[Config.TARGET_COL].values
            else:
                # For test set, create dummy y if not present
                y = np.zeros(len(df))

        # Save to cache
        self.logger.info(f"Saving features to cache for split: {split}")
        sp.save_npz(cache_files["view_lexical"], X_lexical)
        sp.save_npz(cache_files["view_behavioral"], X_behavioral)
        np.save(cache_files["view_semantic"], X_semantic)
        np.save(cache_files["view_interaction"], X_interaction)
        np.save(cache_files["view_meta"], X_meta)
        np.save(cache_files["y"], y)

        return {
            "view_lexical": X_lexical,
            "view_behavioral": X_behavioral,
            "view_semantic": X_semantic,
            "view_interaction": X_interaction,
            "view_meta": X_meta,
            "y": y,
        }
