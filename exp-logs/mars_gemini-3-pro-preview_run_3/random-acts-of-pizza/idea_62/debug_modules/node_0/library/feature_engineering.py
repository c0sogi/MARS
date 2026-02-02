import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import Config
from library.utils import setup_logger

logger = setup_logger("feature_engineering")


class FeatureEngineeringPipeline:
    """
    Implements the feature engineering pipeline for the Conservative Granular Hept-View Stacking Ensemble.
    Generates four distinct views of the data:
    1. Lexical (Sparse TF-IDF of Text)
    2. Community (Sparse TF-IDF of Subreddit History)
    3. Semantic (Dense Embeddings of Text)
    4. Metadata (Scaled Numerical Features)
    """

    def __init__(self):
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_paths(self):
        """Returns a dictionary of file paths for caching."""
        return {
            "train_lexical": os.path.join(self.cache_dir, "X_train_lexical.npz"),
            "test_lexical": os.path.join(self.cache_dir, "X_test_lexical.npz"),
            "train_community": os.path.join(self.cache_dir, "X_train_community.npz"),
            "test_community": os.path.join(self.cache_dir, "X_test_community.npz"),
            "train_semantic": os.path.join(self.cache_dir, "X_train_semantic.npy"),
            "test_semantic": os.path.join(self.cache_dir, "X_test_semantic.npy"),
            "train_meta": os.path.join(self.cache_dir, "X_train_meta.npy"),
            "test_meta": os.path.join(self.cache_dir, "X_test_meta.npy"),
            "y_train": os.path.join(self.cache_dir, "y_train.npy"),
        }

    def _save_data(self, data_dict):
        """Saves generated data to cache."""
        paths = self._get_cache_paths()

        # Save Sparse Matrices
        sp.save_npz(paths["train_lexical"], data_dict["train_lexical"])
        sp.save_npz(paths["test_lexical"], data_dict["test_lexical"])
        sp.save_npz(paths["train_community"], data_dict["train_community"])
        sp.save_npz(paths["test_community"], data_dict["test_community"])

        # Save Dense Arrays
        np.save(paths["train_semantic"], data_dict["train_semantic"])
        np.save(paths["test_semantic"], data_dict["test_semantic"])
        np.save(paths["train_meta"], data_dict["train_meta"])
        np.save(paths["test_meta"], data_dict["test_meta"])
        np.save(paths["y_train"], data_dict["y_train"])

        logger.info(f"Feature sets saved to {self.cache_dir}")

    def _load_data(self):
        """Attempts to load data from cache."""
        paths = self._get_cache_paths()
        data_dict = {}

        try:
            # Check existence
            for p in paths.values():
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Cache file missing: {p}")

            # Load
            data_dict["train_lexical"] = sp.load_npz(paths["train_lexical"])
            data_dict["test_lexical"] = sp.load_npz(paths["test_lexical"])
            data_dict["train_community"] = sp.load_npz(paths["train_community"])
            data_dict["test_community"] = sp.load_npz(paths["test_community"])

            data_dict["train_semantic"] = np.load(paths["train_semantic"])
            data_dict["test_semantic"] = np.load(paths["test_semantic"])
            data_dict["train_meta"] = np.load(paths["train_meta"])
            data_dict["test_meta"] = np.load(paths["test_meta"])
            data_dict["y_train"] = np.load(paths["y_train"])

            logger.info("Successfully loaded features from cache.")
            return data_dict

        except Exception as e:
            logger.warning(f"Cache loading failed: {e}. Proceeding to recompute.")
            return None

    def _prepare_text(self, df):
        """Concatenates title and edit-aware body text."""
        # Fill NAs with empty string
        title = df["request_title"].fillna("").astype(str)
        body = df["request_text_edit_aware"].fillna("").astype(str)
        return (title + " " + body).tolist()

    def _prepare_community(self, df):
        """Converts list of subreddits to space-separated string."""

        def join_subs(sub_list):
            if isinstance(sub_list, (list, np.ndarray)):
                return " ".join([str(s) for s in sub_list])
            return ""

        return df[Config.COMMUNITY_COL].apply(join_subs).tolist()

    def _process_lexical(self, train_text, test_text):
        """Generates Granular Lexical Features (Sparse TF-IDF)."""
        logger.info("Generating Lexical Features...")
        vectorizer = TfidfVectorizer(
            token_pattern=Config.TOKEN_PATTERN,  # Granular pattern (e.g., \w{1,})
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=5,
            stop_words="english",
        )
        X_train = vectorizer.fit_transform(train_text)
        X_test = vectorizer.transform(test_text)
        return X_train, X_test

    def _process_community(self, train_subs, test_subs):
        """Generates Community Features (Bag-of-Concepts TF-IDF)."""
        logger.info("Generating Community Features...")
        vectorizer = TfidfVectorizer(
            max_features=Config.VOCAB_SIZE_COMMUNITY,
            token_pattern=r"(?u)\b\w+\b",  # Standard word boundary for subreddits
            stop_words=None,
        )
        X_train = vectorizer.fit_transform(train_subs)
        X_test = vectorizer.transform(test_subs)
        return X_train, X_test

    def _process_semantic(self, train_text, test_text):
        """Generates Semantic Features (Dense Embeddings)."""
        logger.info(f"Generating Semantic Features using {Config.EMBEDDING_MODEL}...")
        model = SentenceTransformer(Config.EMBEDDING_MODEL)

        # Encode in batches
        X_train = model.encode(
            train_text, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        X_test = model.encode(
            test_text, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        return X_train, X_test

    def _process_metadata(self, train_df, test_df):
        """Generates Augmented Global Metadata (Scaled)."""
        logger.info("Generating Metadata Features...")

        # Select columns
        cols = Config.METADATA_COLS
        X_train_raw = train_df[cols].copy()
        X_test_raw = test_df[cols].copy()

        # Impute
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_test_imp = imputer.transform(X_test_raw)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        return X_train_scaled, X_test_scaled

    def run(self, train_df, test_df, load_cached_data=True):
        """
        Main execution method.

        Args:
            train_df (pd.DataFrame): Union training dataset.
            test_df (pd.DataFrame): Test dataset.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: Dictionary containing all feature matrices (train/test for lexical, community, semantic, meta) and targets.
        """
        # 1. Try Load Cache
        if load_cached_data:
            data = self._load_data()
            if data is not None:
                return data

        logger.info("Computing features from scratch...")

        # 2. Prepare Raw Inputs
        train_text = self._prepare_text(train_df)
        test_text = self._prepare_text(test_df)

        train_subs = self._prepare_community(train_df)
        test_subs = self._prepare_community(test_df)

        # 3. Generate Features
        # Lexical
        X_train_lex, X_test_lex = self._process_lexical(train_text, test_text)

        # Community
        X_train_comm, X_test_comm = self._process_community(train_subs, test_subs)

        # Semantic
        X_train_sem, X_test_sem = self._process_semantic(train_text, test_text)

        # Metadata
        X_train_meta, X_test_meta = self._process_metadata(train_df, test_df)

        # Target
        y_train = train_df[Config.TARGET_COL].values.astype(int)

        # 4. Aggregate
        data_dict = {
            "train_lexical": X_train_lex,
            "test_lexical": X_test_lex,
            "train_community": X_train_comm,
            "test_community": X_test_comm,
            "train_semantic": X_train_sem,
            "test_semantic": X_test_sem,
            "train_meta": X_train_meta,
            "test_meta": X_test_meta,
            "y_train": y_train,
        }

        # 5. Save to Cache
        self._save_data(data_dict)

        return data_dict
