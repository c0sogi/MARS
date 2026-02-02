import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import (
    WORKING_DIR,
    NUMERICAL_COLS,
    TEXT_COL,
    SUBREDDIT_LIST_COL,
    TEXT_TFIDF_PARAMS,
    SUBREDDIT_TFIDF_PARAMS,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    RANDOM_SEED,
)


class FeaturePipeline:
    """
    Implements the feature engineering pipeline for the Dual-Topology Stacking model.
    Generates Metadata, Text (Sparse/Dense), and Behavioral (Sparse/Dense) views.
    """

    def __init__(self):
        # Initialize Scikit-Learn transformers
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.text_tfidf = TfidfVectorizer(**TEXT_TFIDF_PARAMS)
        self.beh_tfidf = TfidfVectorizer(**SUBREDDIT_TFIDF_PARAMS)

        # Placeholder for the heavy embedding model (Lazy Loading)
        self.embedding_model = None

    def _load_embedding_model(self):
        """Lazy loads the SentenceTransformer model."""
        if self.embedding_model is None:
            # We assume the model is available or can be downloaded/cached
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def _get_time_features(self, df):
        """Extracts Hour and Day of Week from UTC timestamp."""
        # Convert timestamp to datetime
        # Handle potential NaNs if any (though unlikely in this dataset for timestamp)
        dt = pd.to_datetime(df["unix_timestamp_of_request_utc"], unit="s")

        # Extract features
        hour = dt.dt.hour.values.reshape(-1, 1)
        day = dt.dt.dayofweek.values.reshape(-1, 1)

        return hour, day

    def fit(self, df):
        """
        Fits the transformers on the training data.
        """
        print("Fitting feature pipeline...")

        # 1. Metadata Fitting
        # Identify numerical columns excluding the timestamp (which is transformed)
        num_cols = [c for c in NUMERICAL_COLS if c != "unix_timestamp_of_request_utc"]
        X_num = df[num_cols].values

        # Get derived time features
        hour, day = self._get_time_features(df)

        # Combine for fitting
        X_meta_raw = np.hstack([X_num, hour, day])

        # Fit Imputer and Scaler
        X_imputed = self.imputer.fit_transform(X_meta_raw)
        self.scaler.fit(X_imputed)

        # 2. Text Fitting
        texts = df[TEXT_COL].fillna("").astype(str)
        self.text_tfidf.fit(texts)

        # 3. Behavioral Fitting
        # Convert list of subreddits to space-separated string
        subs = (
            df[SUBREDDIT_LIST_COL]
            .apply(lambda x: " ".join(x) if isinstance(x, (list, np.ndarray)) else "")
            .fillna("")
            .astype(str)
        )
        self.beh_tfidf.fit(subs)

        return self

    def transform(self, df, prefix=None, load_cached_data=True):
        """
        Transforms the data into the 5 feature views.
        Uses caching if 'prefix' is provided.
        """
        # Define cache filenames
        cache_files = {}
        if prefix:
            cache_files = {
                "meta": f"{prefix}_meta.npy",
                "text_sparse": f"{prefix}_text_sparse.npz",
                "text_dense": f"{prefix}_text_dense.npy",
                "beh_sparse": f"{prefix}_beh_sparse.npz",
                "beh_dense": f"{prefix}_beh_dense.npy",
            }

        # Attempt to load from cache
        if prefix and load_cached_data:
            all_exist = all(
                os.path.exists(os.path.join(WORKING_DIR, f))
                for f in cache_files.values()
            )
            if all_exist:
                print(f"Loading cached features for '{prefix}' from {WORKING_DIR}...")
                data = {}
                data["meta"] = np.load(os.path.join(WORKING_DIR, cache_files["meta"]))
                data["text_sparse"] = sparse.load_npz(
                    os.path.join(WORKING_DIR, cache_files["text_sparse"])
                )
                data["text_dense"] = np.load(
                    os.path.join(WORKING_DIR, cache_files["text_dense"])
                )
                data["beh_sparse"] = sparse.load_npz(
                    os.path.join(WORKING_DIR, cache_files["beh_sparse"])
                )
                data["beh_dense"] = np.load(
                    os.path.join(WORKING_DIR, cache_files["beh_dense"])
                )
                return data

        print(f"Generating features for '{prefix if prefix else 'data'}'...")

        # --- 1. Metadata View ---
        num_cols = [c for c in NUMERICAL_COLS if c != "unix_timestamp_of_request_utc"]
        X_num = df[num_cols].values
        hour, day = self._get_time_features(df)
        X_meta_raw = np.hstack([X_num, hour, day])
        # Impute and Scale
        X_meta = self.scaler.transform(self.imputer.transform(X_meta_raw))

        # --- 2. Text View ---
        texts = df[TEXT_COL].fillna("").astype(str).tolist()

        # Sparse (Lexical)
        X_text_sparse = self.text_tfidf.transform(texts)

        # Dense (Semantic) - Requires Model
        self._load_embedding_model()
        X_text_dense = self.embedding_model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # --- 3. Behavioral View ---
        subs = (
            df[SUBREDDIT_LIST_COL]
            .apply(lambda x: " ".join(x) if isinstance(x, (list, np.ndarray)) else "")
            .fillna("")
            .astype(str)
            .tolist()
        )

        # Sparse (Community)
        X_beh_sparse = self.beh_tfidf.transform(subs)

        # Dense (Persona)
        # We embed the string of subreddits to capture semantic clusters of interest
        X_beh_dense = self.embedding_model.encode(
            subs,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Assemble Data Dictionary
        data = {
            "meta": X_meta,
            "text_sparse": X_text_sparse,
            "text_dense": X_text_dense,
            "beh_sparse": X_beh_sparse,
            "beh_dense": X_beh_dense,
        }

        # Save to Cache
        if prefix:
            print(f"Saving features to cache for '{prefix}'...")
            os.makedirs(WORKING_DIR, exist_ok=True)
            np.save(os.path.join(WORKING_DIR, cache_files["meta"]), data["meta"])
            sparse.save_npz(
                os.path.join(WORKING_DIR, cache_files["text_sparse"]),
                data["text_sparse"],
            )
            np.save(
                os.path.join(WORKING_DIR, cache_files["text_dense"]), data["text_dense"]
            )
            sparse.save_npz(
                os.path.join(WORKING_DIR, cache_files["beh_sparse"]), data["beh_sparse"]
            )
            np.save(
                os.path.join(WORKING_DIR, cache_files["beh_dense"]), data["beh_dense"]
            )

        return data

    def fit_transform(self, df, prefix=None, load_cached_data=True):
        """
        Convenience method to fit and then transform.
        """
        self.fit(df)
        return self.transform(df, prefix, load_cached_data)
