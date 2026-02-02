import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import get_logger, timer


class FeaturePipeline:
    """
    Implements the feature engineering pipeline for the Restored-History Hex-View Stacking Ensemble.
    Generates Lexical, Behavioral, Semantic, and Contextual feature views.
    """

    def __init__(self):
        self.logger = get_logger("FeaturePipeline")
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_paths(self):
        """Define paths for all cached artifacts."""
        paths = {
            # Train Features
            "train_lexical": os.path.join(self.cache_dir, "X_train_lexical.npz"),
            "train_behavioral": os.path.join(self.cache_dir, "X_train_behavioral.npz"),
            "train_semantic": os.path.join(self.cache_dir, "X_train_semantic.npy"),
            "train_contextual": os.path.join(self.cache_dir, "X_train_contextual.npy"),
            "y_train": os.path.join(self.cache_dir, "y_train.npy"),
            # Test Features
            "test_lexical": os.path.join(self.cache_dir, "X_test_lexical.npz"),
            "test_behavioral": os.path.join(self.cache_dir, "X_test_behavioral.npz"),
            "test_semantic": os.path.join(self.cache_dir, "X_test_semantic.npy"),
            "test_contextual": os.path.join(self.cache_dir, "X_test_contextual.npy"),
            "test_ids": os.path.join(self.cache_dir, "test_ids.npy"),
        }
        return paths

    def _check_cache_exists(self, paths):
        """Check if all cache files exist."""
        return all(os.path.exists(p) for p in paths.values())

    def _load_cache(self, paths):
        """Load all data from cache."""
        self.logger.info("Loading features from cache...")

        data = {}
        # Load Sparse
        data["X_train_lexical"] = sp.load_npz(paths["train_lexical"])
        data["X_train_behavioral"] = sp.load_npz(paths["train_behavioral"])
        data["X_test_lexical"] = sp.load_npz(paths["test_lexical"])
        data["X_test_behavioral"] = sp.load_npz(paths["test_behavioral"])

        # Load Dense
        data["X_train_semantic"] = np.load(paths["train_semantic"])
        data["X_train_contextual"] = np.load(paths["train_contextual"])
        data["y_train"] = np.load(paths["y_train"])

        data["X_test_semantic"] = np.load(paths["test_semantic"])
        data["X_test_contextual"] = np.load(paths["test_contextual"])
        data["test_ids"] = np.load(paths["test_ids"], allow_pickle=True)

        return data

    def _save_cache(self, data, paths):
        """Save all data to cache."""
        self.logger.info("Saving features to cache...")

        # Save Sparse
        sp.save_npz(paths["train_lexical"], data["X_train_lexical"])
        sp.save_npz(paths["train_behavioral"], data["X_train_behavioral"])
        sp.save_npz(paths["test_lexical"], data["X_test_lexical"])
        sp.save_npz(paths["test_behavioral"], data["X_test_behavioral"])

        # Save Dense
        np.save(paths["train_semantic"], data["X_train_semantic"])
        np.save(paths["train_contextual"], data["X_train_contextual"])
        np.save(paths["y_train"], data["y_train"])

        np.save(paths["test_semantic"], data["X_test_semantic"])
        np.save(paths["test_contextual"], data["X_test_contextual"])
        np.save(paths["test_ids"], data["test_ids"])

    def _prepare_text(self, df):
        """Concatenate title and edit-aware text."""
        # Fill NaNs with empty string
        title = df["request_title"].fillna("").astype(str)
        body = df["request_text_edit_aware"].fillna("").astype(str)
        return title + " " + body

    def _prepare_subreddits(self, df):
        """Convert subreddit list column to space-separated string."""

        def join_subs(x):
            if isinstance(x, (list, np.ndarray)):
                return " ".join([str(s) for s in x])
            return ""

        return df["requester_subreddits_at_request"].apply(join_subs)

    def get_data(self, load_cached_data=True):
        """
        Main entry point to get processed feature matrices.

        Args:
            load_cached_data (bool): If True, attempt to load from disk.

        Returns:
            tuple: (X_train_dict, y_train, X_test_dict, test_ids)
        """
        paths = self._get_cache_paths()

        if load_cached_data and self._check_cache_exists(paths):
            data = self._load_cache(paths)
        else:
            with timer("Full Feature Engineering Pipeline"):
                # 1. Load Raw Data
                self.logger.info("Loading raw metadata...")
                df_train_part = pd.read_parquet(Config.TRAIN_METADATA_PATH)
                df_val_part = pd.read_parquet(Config.VAL_METADATA_PATH)
                df_test = pd.read_parquet(Config.TEST_METADATA_PATH)

                # Merge Train and Val for CV-Bagging
                df_train = pd.concat(
                    [df_train_part, df_val_part], axis=0, ignore_index=True
                )

                y_train = df_train[Config.TARGET_COL].values
                test_ids = df_test[Config.ID_COL].values

                # 2. Text Preprocessing
                self.logger.info("Preprocessing text...")
                train_text = self._prepare_text(df_train)
                test_text = self._prepare_text(df_test)

                # 3. Subreddit Preprocessing
                self.logger.info("Preprocessing subreddits...")
                train_subs = self._prepare_subreddits(df_train)
                test_subs = self._prepare_subreddits(df_test)

                # 4. Build Lexical View (Sparse)
                self.logger.info("Building Lexical View (TF-IDF)...")
                # Unigrams + Bigrams, sublinear_tf=True, min_df=5
                lexical_vec = TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=5,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w\w+\b",
                )
                X_train_lexical = lexical_vec.fit_transform(train_text)
                X_test_lexical = lexical_vec.transform(test_text)

                # 5. Build Behavioral View (Sparse)
                self.logger.info("Building Behavioral View (Community TF-IDF)...")
                # Bag of Concepts, max 1000 features
                behavioral_vec = TfidfVectorizer(
                    max_features=Config.COMMUNITY_MAX_FEATURES,
                    token_pattern=r"(?u)\b\w+\b",  # Simple tokenization for subreddit names
                )
                X_train_behavioral = behavioral_vec.fit_transform(train_subs)
                X_test_behavioral = behavioral_vec.transform(test_subs)

                # 6. Build Semantic View (Dense)
                self.logger.info("Building Semantic View (Embeddings)...")
                # Frozen all-MiniLM-L6-v2
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                # Encode in batches
                X_train_semantic = embedding_model.encode(
                    train_text.tolist(),
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                X_test_semantic = embedding_model.encode(
                    test_text.tolist(),
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )

                # 7. Build Contextual View (Dense Metadata)
                self.logger.info("Building Contextual View (Dense Metadata)...")
                # Select features
                dense_cols = Config.DENSE_FEATURES

                # Extract raw values
                X_train_ctx_raw = df_train[dense_cols].values
                X_test_ctx_raw = df_test[dense_cols].values

                # Impute (Median)
                imputer = SimpleImputer(strategy="median")
                X_train_ctx_imp = imputer.fit_transform(X_train_ctx_raw)
                X_test_ctx_imp = imputer.transform(X_test_ctx_raw)

                # Scale (StandardScaler)
                scaler = StandardScaler()
                X_train_contextual = scaler.fit_transform(X_train_ctx_imp)
                X_test_contextual = scaler.transform(X_test_ctx_imp)

                # Pack data
                data = {
                    "X_train_lexical": X_train_lexical,
                    "X_train_behavioral": X_train_behavioral,
                    "X_train_semantic": X_train_semantic,
                    "X_train_contextual": X_train_contextual,
                    "y_train": y_train,
                    "X_test_lexical": X_test_lexical,
                    "X_test_behavioral": X_test_behavioral,
                    "X_test_semantic": X_test_semantic,
                    "X_test_contextual": X_test_contextual,
                    "test_ids": test_ids,
                }

                # Save to cache
                self._save_cache(data, paths)

        # Structure output as nested dicts for the model pipeline
        X_train_dict = {
            "lexical": data["X_train_lexical"],
            "behavioral": data["X_train_behavioral"],
            "semantic": data["X_train_semantic"],
            "contextual": data["X_train_contextual"],
        }

        X_test_dict = {
            "lexical": data["X_test_lexical"],
            "behavioral": data["X_test_behavioral"],
            "semantic": data["X_test_semantic"],
            "contextual": data["X_test_contextual"],
        }

        return X_train_dict, data["y_train"], X_test_dict, data["test_ids"]
