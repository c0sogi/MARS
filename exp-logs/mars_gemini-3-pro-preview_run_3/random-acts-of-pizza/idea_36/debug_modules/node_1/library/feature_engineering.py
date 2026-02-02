import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
import torch

from library.config import Config
from library.utils import get_logger, Timer

logger = get_logger(__name__)


class CommunityProfiler:
    """
    Implements Bayesian Target Encoding for subreddit histories.
    Calculates a 'Community Success Score' based on the historical success rate
    of the subreddits a user participates in.
    """

    def __init__(self, vocab_size=1000, smoothing=10):
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.subreddit_map = {}
        self.global_mean = 0.0

    def fit(self, subreddits_series, y):
        """
        Fits the profiler on a series of subreddit lists and target labels.
        """
        # Explode the lists to get one row per subreddit-user interaction
        # We need to align y with the exploded series
        df = pd.DataFrame({"subreddit": subreddits_series, "target": y})

        # Explode
        df_exploded = df.explode("subreddit")

        # Calculate global mean
        self.global_mean = df["target"].mean()

        # Filter for top K subreddits by frequency
        top_subs = (
            df_exploded["subreddit"].value_counts().nlargest(self.vocab_size).index
        )
        df_filtered = df_exploded[df_exploded["subreddit"].isin(top_subs)]

        # Calculate stats per subreddit
        stats = df_filtered.groupby("subreddit")["target"].agg(["count", "mean"])

        # Bayesian Smoothing: (n * mean + m * global_mean) / (n + m)
        # n = count, m = smoothing factor
        stats["smoothed_score"] = (
            (stats["count"] * stats["mean"]) + (self.smoothing * self.global_mean)
        ) / (stats["count"] + self.smoothing)

        self.subreddit_map = stats["smoothed_score"].to_dict()
        return self

    def transform(self, subreddits_series):
        """
        Transforms a series of subreddit lists into community success scores.
        """

        def get_score(sub_list):
            if not isinstance(sub_list, list) or not sub_list:
                return self.global_mean

            scores = []
            for sub in sub_list:
                # Use mapped score if available, else global mean (neutral prior)
                scores.append(self.subreddit_map.get(sub, self.global_mean))

            return np.mean(scores) if scores else self.global_mean

        return subreddits_series.apply(get_score).values


class TextProcessor:
    """
    Handles text cleaning and concatenation.
    """

    @staticmethod
    def process(df, text_cols):
        """
        Concatenates specified text columns into a single string.
        """
        # Fill NaNs with empty string
        filled_df = df[text_cols].fillna("")

        # Concatenate with a space separator
        # If multiple columns, join them. If single, just return it.
        if len(text_cols) > 1:
            return filled_df.apply(lambda row: " ".join(row.values.astype(str)), axis=1)
        else:
            return filled_df[text_cols[0]].astype(str)


class MetadataExtractor:
    """
    Extracts, imputes, and scales numerical metadata.
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_names = []

    def fit_transform(self, df, dense_cols):
        """
        Fits imputer and scaler on training data and transforms it.
        """
        self.feature_names = dense_cols
        X = df[dense_cols].copy()

        # Handle potential non-numeric types (though parquet should be strict)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # Impute
        X_imputed = self.imputer.fit_transform(X)

        # Scale
        X_scaled = self.scaler.fit_transform(X_imputed)

        return X_scaled

    def transform(self, df):
        """
        Transforms validation/test data using fitted imputer and scaler.
        """
        X = df[self.feature_names].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        return X_scaled


class VectorizationManager:
    """
    Manages TF-IDF and Dense Embedding generation.
    """

    def __init__(self):
        self.tfidf = TfidfVectorizer(**Config.TFIDF_PARAMS)
        self.embedding_model = None

    def get_tfidf(self, train_text, val_text, test_text):
        """
        Fits TF-IDF on train, transforms all.
        """
        logger.info("Generating TF-IDF features...")
        train_tfidf = self.tfidf.fit_transform(train_text)
        val_tfidf = self.tfidf.transform(val_text)
        test_tfidf = self.tfidf.transform(test_text)
        return train_tfidf, val_tfidf, test_tfidf

    def get_embeddings(self, train_text, val_text, test_text):
        """
        Generates dense embeddings using SentenceTransformer.
        """
        logger.info(f"Generating Dense Embeddings ({Config.EMBEDDING_MODEL})...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(
            Config.EMBEDDING_MODEL, device=device
        )

        # Encode in batches
        train_emb = self.embedding_model.encode(
            train_text.tolist(),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        val_emb = self.embedding_model.encode(
            val_text.tolist(),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        test_emb = self.embedding_model.encode(
            test_text.tolist(),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return train_emb, val_emb, test_emb


def process_data(load_cached_data=True, debug_sample_size=None):
    """
    Main orchestration function for feature engineering.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug_sample_size (int, optional): If set, subsamples data for debugging.

    Returns:
        dict: Dictionary containing processed datasets for train, val, test.
    """
    Config.create_dirs()

    # Cache File Paths
    cache_files = {
        "train_meta": os.path.join(Config.WORKING_DIR, "train_meta.parquet"),
        "val_meta": os.path.join(Config.WORKING_DIR, "val_meta.parquet"),
        "test_meta": os.path.join(Config.WORKING_DIR, "test_meta.parquet"),
        "train_tfidf": os.path.join(Config.WORKING_DIR, "train_tfidf.npz"),
        "val_tfidf": os.path.join(Config.WORKING_DIR, "val_tfidf.npz"),
        "test_tfidf": os.path.join(Config.WORKING_DIR, "test_tfidf.npz"),
        "train_emb": os.path.join(Config.WORKING_DIR, "train_emb.npy"),
        "val_emb": os.path.join(Config.WORKING_DIR, "val_emb.npy"),
        "test_emb": os.path.join(Config.WORKING_DIR, "test_emb.npy"),
        "y_train": os.path.join(Config.WORKING_DIR, "y_train.npy"),
        "y_val": os.path.join(Config.WORKING_DIR, "y_val.npy"),
        "test_ids": os.path.join(Config.WORKING_DIR, "test_ids.npy"),
    }

    # 1. Check Cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_files.values())
        if all_exist:
            logger.info("Loading cached features from disk...")
            with Timer("Loading Cache"):
                data = {
                    "train": {
                        "metadata": pd.read_parquet(cache_files["train_meta"]),
                        "tfidf": sp.load_npz(cache_files["train_tfidf"]),
                        "embeddings": np.load(cache_files["train_emb"]),
                        "y": np.load(cache_files["y_train"]),
                    },
                    "val": {
                        "metadata": pd.read_parquet(cache_files["val_meta"]),
                        "tfidf": sp.load_npz(cache_files["val_tfidf"]),
                        "embeddings": np.load(cache_files["val_emb"]),
                        "y": np.load(cache_files["y_val"]),
                    },
                    "test": {
                        "metadata": pd.read_parquet(cache_files["test_meta"]),
                        "tfidf": sp.load_npz(cache_files["test_tfidf"]),
                        "embeddings": np.load(cache_files["test_emb"]),
                        "ids": np.load(cache_files["test_ids"], allow_pickle=True),
                    },
                    "CommunityProfiler": CommunityProfiler,  # Return class for external use
                }
                return data
        else:
            logger.info("Cache incomplete or missing. Regenerating features...")

    # 2. Load Raw Metadata
    with Timer("Loading Raw Data"):
        train_df = pd.read_parquet(Config.TRAIN_DATA_PATH)
        val_df = pd.read_parquet(Config.VAL_DATA_PATH)
        test_df = pd.read_parquet(Config.TEST_DATA_PATH)

        if debug_sample_size:
            logger.info(f"DEBUG MODE: Subsampling to {debug_sample_size} rows.")
            train_df = train_df.iloc[:debug_sample_size]
            val_df = val_df.iloc[:debug_sample_size]
            test_df = test_df.iloc[:debug_sample_size]

    # 3. Process Text
    with Timer("Text Processing"):
        tp = TextProcessor()
        # Train/Val use standard columns
        train_text = tp.process(train_df, Config.TRAIN_TEXT_COLS)
        val_text = tp.process(val_df, Config.TRAIN_TEXT_COLS)
        # Test uses edit-aware columns to prevent leakage
        test_text = tp.process(test_df, Config.TEST_TEXT_COLS)

    # 4. Vectorization (TF-IDF & Embeddings)
    with Timer("Vectorization"):
        vm = VectorizationManager()
        train_tfidf, val_tfidf, test_tfidf = vm.get_tfidf(
            train_text, val_text, test_text
        )
        train_emb, val_emb, test_emb = vm.get_embeddings(
            train_text, val_text, test_text
        )

    # 5. Metadata Processing
    with Timer("Metadata Processing"):
        me = MetadataExtractor()

        # Fit on Train, Transform Val/Test
        # Note: We do NOT include the Community Score here yet.
        # That is handled by the model loop using CommunityProfiler to avoid leakage.
        # We only process the dense numerical features defined in Config.

        train_meta_dense = me.fit_transform(train_df, Config.METADATA_DENSE_FEATURES)
        val_meta_dense = me.transform(val_df)
        test_meta_dense = me.transform(test_df)

        # Convert back to DataFrame to hold the data alongside raw subreddits
        # We need to preserve 'requester_subreddits_at_request' for the CommunityProfiler

        def create_meta_df(original_df, dense_matrix, dense_cols):
            meta_df = pd.DataFrame(
                dense_matrix, columns=dense_cols, index=original_df.index
            )
            # Add back the raw subreddits for the Profiler
            meta_df[Config.COMMUNITY_COL] = original_df[Config.COMMUNITY_COL]
            return meta_df

        train_meta_final = create_meta_df(
            train_df, train_meta_dense, Config.METADATA_DENSE_FEATURES
        )
        val_meta_final = create_meta_df(
            val_df, val_meta_dense, Config.METADATA_DENSE_FEATURES
        )
        test_meta_final = create_meta_df(
            test_df, test_meta_dense, Config.METADATA_DENSE_FEATURES
        )

    # 6. Extract Targets and IDs
    y_train = train_df[Config.TARGET_COL].values.astype(int)
    y_val = val_df[Config.TARGET_COL].values.astype(int)
    test_ids = test_df[Config.ID_COL].values

    # 7. Save to Cache
    with Timer("Saving Cache"):
        # Metadata (Parquet handles lists)
        train_meta_final.to_parquet(cache_files["train_meta"])
        val_meta_final.to_parquet(cache_files["val_meta"])
        test_meta_final.to_parquet(cache_files["test_meta"])

        # TF-IDF (Sparse)
        sp.save_npz(cache_files["train_tfidf"], train_tfidf)
        sp.save_npz(cache_files["val_tfidf"], val_tfidf)
        sp.save_npz(cache_files["test_tfidf"], test_tfidf)

        # Embeddings (Numpy)
        np.save(cache_files["train_emb"], train_emb)
        np.save(cache_files["val_emb"], val_emb)
        np.save(cache_files["test_emb"], test_emb)

        # Targets/IDs
        np.save(cache_files["y_train"], y_train)
        np.save(cache_files["y_val"], y_val)
        np.save(cache_files["test_ids"], test_ids)

    # 8. Return Data
    return {
        "train": {
            "metadata": train_meta_final,
            "tfidf": train_tfidf,
            "embeddings": train_emb,
            "y": y_train,
        },
        "val": {
            "metadata": val_meta_final,
            "tfidf": val_tfidf,
            "embeddings": val_emb,
            "y": y_val,
        },
        "test": {
            "metadata": test_meta_final,
            "tfidf": test_tfidf,
            "embeddings": test_emb,
            "ids": test_ids,
        },
        "CommunityProfiler": CommunityProfiler,
    }
