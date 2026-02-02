import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch

from library.config import (
    TFIDF_TEXT_PARAMS,
    TFIDF_SUBREDDIT_PARAMS,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    METADATA_COLS,
    TARGET_COL,
    ID_COL,
    SUBREDDIT_COL,
    SEED,
    CACHE_DIR,
)
from library.utils import get_logger, save_to_cache, load_from_cache, set_seed
from library.preprocessor import Preprocessor

logger = get_logger("feature_factory")


class FeatureFactory:
    """
    Orchestrates the generation of feature views for the Stacking Ensemble.
    Generates:
    1. Lexical View (Sparse TF-IDF on Text)
    2. Community View (Sparse TF-IDF on Subreddit History)
    3. Semantic View (Dense Embeddings on Text)
    4. Metadata View (Dense User Stats + Community Score)
    """

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Define cache filenames
        self.cache_files = {
            "lexical": "X_lexical.npz",
            "community": "X_community.npz",
            "semantic": "X_semantic.npz",
            "metadata": "X_metadata.npz",
            "targets": "y_targets.npz",
            "ids": "ids.npz",
        }

    def _check_cache(self):
        """Checks if all required cache files exist."""
        for fname in self.cache_files.values():
            if load_from_cache(fname, format="npz") is None:
                return False
        return True

    def _load_all_from_cache(self):
        """Loads all feature views from cache."""
        logger.info("Loading features from cache...")
        data = {}

        # Helper to unpack npz dicts
        def unpack(name, npz_data):
            for split in ["train", "val", "test"]:
                key = f"X_{split}_{name}"
                # Handle sparse matrices saved in npz (0-d array object)
                arr = npz_data[split]
                if arr.dtype == object and not isinstance(arr.item(), dict):
                    # Likely a sparse matrix wrapped in 0-d array
                    try:
                        data[key] = arr.item()
                    except:
                        data[key] = arr
                else:
                    data[key] = arr

        # Load Features
        lexical = load_from_cache(self.cache_files["lexical"], format="npz")
        unpack("lexical", lexical)

        community = load_from_cache(self.cache_files["community"], format="npz")
        unpack("community", community)

        semantic = load_from_cache(self.cache_files["semantic"], format="npz")
        unpack("semantic", semantic)

        metadata = load_from_cache(self.cache_files["metadata"], format="npz")
        unpack("metadata", metadata)

        # Load Targets
        targets = load_from_cache(self.cache_files["targets"], format="npz")
        data["y_train"] = targets["train"]
        data["y_val"] = targets["val"]

        # Load IDs
        ids = load_from_cache(self.cache_files["ids"], format="npz")
        data["id_train"] = ids["train"]
        data["id_val"] = ids["val"]
        data["id_test"] = ids["test"]

        return data

    def run(self, load_cached_data=True):
        """
        Main execution method.
        """
        set_seed(SEED)

        # 1. Check Cache
        if load_cached_data and self._check_cache():
            return self._load_all_from_cache()

        logger.info("Cache miss or force reload. Generating features...")

        # 2. Get Processed Data
        train_df, val_df, test_df = self.preprocessor.run(
            load_cached_data=load_cached_data
        )

        # Initialize output container
        output = {}

        # Extract IDs
        output["id_train"] = train_df[ID_COL].values
        output["id_val"] = val_df[ID_COL].values
        output["id_test"] = test_df[ID_COL].values

        # Extract Targets
        output["y_train"] = train_df[TARGET_COL].values
        output["y_val"] = val_df[TARGET_COL].values

        # --- View 1: Lexical (Sparse TF-IDF) ---
        logger.info("Generating Lexical View...")
        tfidf_text = TfidfVectorizer(**TFIDF_TEXT_PARAMS)

        # Fit on Train, Transform All
        X_train_lex = tfidf_text.fit_transform(train_df["text_concat"])
        X_val_lex = tfidf_text.transform(val_df["text_concat"])
        X_test_lex = tfidf_text.transform(test_df["text_concat"])

        output["X_train_lexical"] = X_train_lex
        output["X_val_lexical"] = X_val_lex
        output["X_test_lexical"] = X_test_lex

        # --- View 2: Community (Sparse TF-IDF) ---
        logger.info("Generating Community View...")

        def join_subreddits(series):
            # Handle potential non-list values or NaNs gracefully
            return series.apply(
                lambda x: " ".join(x) if isinstance(x, (list, np.ndarray)) else ""
            )

        train_subs = join_subreddits(train_df[SUBREDDIT_COL])
        val_subs = join_subreddits(val_df[SUBREDDIT_COL])
        test_subs = join_subreddits(test_df[SUBREDDIT_COL])

        tfidf_comm = TfidfVectorizer(**TFIDF_SUBREDDIT_PARAMS)

        X_train_comm = tfidf_comm.fit_transform(train_subs)
        X_val_comm = tfidf_comm.transform(val_subs)
        X_test_comm = tfidf_comm.transform(test_subs)

        output["X_train_community"] = X_train_comm
        output["X_val_community"] = X_val_comm
        output["X_test_community"] = X_test_comm

        # --- View 3: Semantic (Dense Embeddings) ---
        logger.info(f"Generating Semantic View using {EMBEDDING_MODEL}...")
        model = SentenceTransformer(EMBEDDING_MODEL, device=self.device)

        # Encode
        # Note: encode returns numpy array by default
        X_train_sem = model.encode(
            train_df["text_concat"].tolist(),
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
        )
        X_val_sem = model.encode(
            val_df["text_concat"].tolist(),
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
        )
        X_test_sem = model.encode(
            test_df["text_concat"].tolist(),
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
        )

        output["X_train_semantic"] = X_train_sem
        output["X_val_semantic"] = X_val_sem
        output["X_test_semantic"] = X_test_sem

        # --- View 4: Metadata (Dense) ---
        logger.info("Generating Metadata View...")

        # Identify columns: Configured Metadata Cols + Community Score
        # Note: Preprocessor has already scaled METADATA_COLS.
        # Community score is in 'community_generosity_score'.

        # Filter METADATA_COLS to ensure they exist in DF (safety check)
        valid_meta_cols = [c for c in METADATA_COLS if c in train_df.columns]
        score_col = "community_generosity_score"

        final_cols = valid_meta_cols + [score_col]

        X_train_meta = train_df[final_cols].values.astype(np.float32)
        X_val_meta = val_df[final_cols].values.astype(np.float32)
        X_test_meta = test_df[final_cols].values.astype(np.float32)

        output["X_train_metadata"] = X_train_meta
        output["X_val_metadata"] = X_val_meta
        output["X_test_metadata"] = X_test_meta

        # --- Save to Cache ---
        logger.info("Saving generated features to cache...")

        # Save Lexical
        save_to_cache(
            {
                "train": output["X_train_lexical"],
                "val": output["X_val_lexical"],
                "test": output["X_test_lexical"],
            },
            self.cache_files["lexical"],
            format="npz",
        )

        # Save Community
        save_to_cache(
            {
                "train": output["X_train_community"],
                "val": output["X_val_community"],
                "test": output["X_test_community"],
            },
            self.cache_files["community"],
            format="npz",
        )

        # Save Semantic
        save_to_cache(
            {
                "train": output["X_train_semantic"],
                "val": output["X_val_semantic"],
                "test": output["X_test_semantic"],
            },
            self.cache_files["semantic"],
            format="npz",
        )

        # Save Metadata
        save_to_cache(
            {
                "train": output["X_train_metadata"],
                "val": output["X_val_metadata"],
                "test": output["X_test_metadata"],
            },
            self.cache_files["metadata"],
            format="npz",
        )

        # Save Targets
        save_to_cache(
            {"train": output["y_train"], "val": output["y_val"]},
            self.cache_files["targets"],
            format="npz",
        )

        # Save IDs
        save_to_cache(
            {
                "train": output["id_train"],
                "val": output["id_val"],
                "test": output["id_test"],
            },
            self.cache_files["ids"],
            format="npz",
        )

        return output
