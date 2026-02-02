import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    RAW_NUMERICAL_COLS,
    TEXT_COL,
    SUBREDDIT_COL,
    TARGET_COL,
    ID_COL,
    LEXICAL_PARAMS,
    BEHAVIORAL_PARAMS,
    SBERT_MODEL_NAME,
)
from library.utils import timer, print_header


class DataPipeline:
    def __init__(self):
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _clean_text(self, series):
        """Ensures text data is string and handles NaNs."""
        return series.fillna("").astype(str)

    def _serialize_subreddits(self, series):
        """Converts list of subreddits to space-separated string."""
        # Handle cases where the entry might be None or empty list
        return series.apply(
            lambda x: (
                " ".join(x) if isinstance(x, list) or isinstance(x, np.ndarray) else ""
            )
        )

    def _load_raw_data(self):
        """Loads raw parquet files."""
        with timer("Loading Raw Data"):
            train_df = pd.read_parquet(TRAIN_PATH)
            val_df = pd.read_parquet(VAL_PATH)
            test_df = pd.read_parquet(TEST_PATH)
        return train_df, val_df, test_df

    def _extract_metadata(self, train_df, val_df, test_df):
        """
        Extracts, imputes, and scales numerical metadata.
        Returns dense numpy arrays.
        """
        with timer("Extracting Global Metadata"):
            # Select columns
            X_train = train_df[RAW_NUMERICAL_COLS].copy()
            X_val = val_df[RAW_NUMERICAL_COLS].copy()
            X_test = test_df[RAW_NUMERICAL_COLS].copy()

            # Imputation
            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(X_train)
            X_val = imputer.transform(X_val)
            X_test = imputer.transform(X_test)

            # Scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            return (
                X_train.astype(np.float32),
                X_val.astype(np.float32),
                X_test.astype(np.float32),
            )

    def _extract_lexical(self, train_df, val_df, test_df):
        """
        Generates TF-IDF features from request text.
        Returns sparse matrices.
        """
        with timer("Extracting Lexical View (Text TF-IDF)"):
            train_text = self._clean_text(train_df[TEXT_COL])
            val_text = self._clean_text(val_df[TEXT_COL])
            test_text = self._clean_text(test_df[TEXT_COL])

            vectorizer = TfidfVectorizer(**LEXICAL_PARAMS)
            X_train = vectorizer.fit_transform(train_text)
            X_val = vectorizer.transform(val_text)
            X_test = vectorizer.transform(test_text)

            return X_train, X_val, X_test

    def _extract_behavioral(self, train_df, val_df, test_df):
        """
        Generates TF-IDF features from subreddit history.
        Returns sparse matrices.
        """
        with timer("Extracting Behavioral View (Subreddit TF-IDF)"):
            train_subs = self._serialize_subreddits(train_df[SUBREDDIT_COL])
            val_subs = self._serialize_subreddits(val_df[SUBREDDIT_COL])
            test_subs = self._serialize_subreddits(test_df[SUBREDDIT_COL])

            vectorizer = TfidfVectorizer(**BEHAVIORAL_PARAMS)
            X_train = vectorizer.fit_transform(train_subs)
            X_val = vectorizer.transform(val_subs)
            X_test = vectorizer.transform(test_subs)

            return X_train, X_val, X_test

    def _extract_semantic(self, train_df, val_df, test_df):
        """
        Generates SBERT embeddings from request text.
        Returns dense numpy arrays.
        """
        with timer("Extracting Semantic View (SBERT Embeddings)"):
            model = SentenceTransformer(SBERT_MODEL_NAME)
            # Use GPU if available
            device = (
                "cuda"
                if os.environ.get("CUDA_VISIBLE_DEVICES")
                or os.system("nvidia-smi") == 0
                else "cpu"
            )
            print(f"SBERT Inference Device: {device}")

            train_text = self._clean_text(train_df[TEXT_COL]).tolist()
            val_text = self._clean_text(val_df[TEXT_COL]).tolist()
            test_text = self._clean_text(test_df[TEXT_COL]).tolist()

            X_train = model.encode(
                train_text, batch_size=32, show_progress_bar=False, device=device
            )
            X_val = model.encode(
                val_text, batch_size=32, show_progress_bar=False, device=device
            )
            X_test = model.encode(
                test_text, batch_size=32, show_progress_bar=False, device=device
            )

            return (
                X_train.astype(np.float32),
                X_val.astype(np.float32),
                X_test.astype(np.float32),
            )

    def _save_cache(self, data):
        """Saves processed data to cache directory."""
        with timer("Saving Cache"):
            # Helper to save sparse or dense
            def save_array(name, arr):
                path = os.path.join(self.cache_dir, f"{name}")
                if sparse.issparse(arr):
                    sparse.save_npz(path + ".npz", arr)
                else:
                    np.save(path + ".npy", arr)

            # Train
            save_array("X_train_metadata", data["train"]["metadata"])
            save_array("X_train_lexical", data["train"]["lexical"])
            save_array("X_train_behavioral", data["train"]["behavioral"])
            save_array("X_train_semantic", data["train"]["semantic"])
            save_array("y_train", data["train"]["y"])

            # Val
            save_array("X_val_metadata", data["val"]["metadata"])
            save_array("X_val_lexical", data["val"]["lexical"])
            save_array("X_val_behavioral", data["val"]["behavioral"])
            save_array("X_val_semantic", data["val"]["semantic"])
            save_array("y_val", data["val"]["y"])

            # Test
            save_array("X_test_metadata", data["test"]["metadata"])
            save_array("X_test_lexical", data["test"]["lexical"])
            save_array("X_test_behavioral", data["test"]["behavioral"])
            save_array("X_test_semantic", data["test"]["semantic"])
            save_array("test_ids", data["test"]["ids"])

    def _load_cache(self):
        """Attempts to load data from cache."""
        try:
            with timer("Loading Cache"):
                data = {"train": {}, "val": {}, "test": {}}

                # Helper to load
                def load_array(name, sparse_fmt=False):
                    path = os.path.join(self.cache_dir, name)
                    if sparse_fmt:
                        return sparse.load_npz(path + ".npz")
                    else:
                        return np.load(path + ".npy")

                # Train
                data["train"]["metadata"] = load_array("X_train_metadata")
                data["train"]["lexical"] = load_array(
                    "X_train_lexical", sparse_fmt=True
                )
                data["train"]["behavioral"] = load_array(
                    "X_train_behavioral", sparse_fmt=True
                )
                data["train"]["semantic"] = load_array("X_train_semantic")
                data["train"]["y"] = load_array("y_train")

                # Val
                data["val"]["metadata"] = load_array("X_val_metadata")
                data["val"]["lexical"] = load_array("X_val_lexical", sparse_fmt=True)
                data["val"]["behavioral"] = load_array(
                    "X_val_behavioral", sparse_fmt=True
                )
                data["val"]["semantic"] = load_array("X_val_semantic")
                data["val"]["y"] = load_array("y_val")

                # Test
                data["test"]["metadata"] = load_array("X_test_metadata")
                data["test"]["lexical"] = load_array("X_test_lexical", sparse_fmt=True)
                data["test"]["behavioral"] = load_array(
                    "X_test_behavioral", sparse_fmt=True
                )
                data["test"]["semantic"] = load_array("X_test_semantic")
                data["test"]["ids"] = load_array("test_ids")

                print("Cache loaded successfully.")
                return data
        except FileNotFoundError:
            print("Cache not found or incomplete.")
            return None
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None

    def process_data(self, load_cached_data=True):
        """
        Main execution method.
        Orchestrates loading, processing, and caching.
        """
        print_header("Data Processing Pipeline")

        # 1. Try Cache
        if load_cached_data:
            data = self._load_cache()
            if data is not None:
                return data

        # 2. Load Raw
        train_df, val_df, test_df = self._load_raw_data()

        # 3. Extract Targets and IDs
        y_train = train_df[TARGET_COL].values.astype(int)
        y_val = val_df[TARGET_COL].values.astype(int)
        test_ids = test_df[ID_COL].values

        # 4. Generate Views
        # Metadata
        X_train_meta, X_val_meta, X_test_meta = self._extract_metadata(
            train_df, val_df, test_df
        )

        # Lexical
        X_train_lex, X_val_lex, X_test_lex = self._extract_lexical(
            train_df, val_df, test_df
        )

        # Behavioral
        X_train_beh, X_val_beh, X_test_beh = self._extract_behavioral(
            train_df, val_df, test_df
        )

        # Semantic
        X_train_sem, X_val_sem, X_test_sem = self._extract_semantic(
            train_df, val_df, test_df
        )

        # 5. Construct Data Dictionary
        data = {
            "train": {
                "metadata": X_train_meta,
                "lexical": X_train_lex,
                "behavioral": X_train_beh,
                "semantic": X_train_sem,
                "y": y_train,
            },
            "val": {
                "metadata": X_val_meta,
                "lexical": X_val_lex,
                "behavioral": X_val_beh,
                "semantic": X_val_sem,
                "y": y_val,
            },
            "test": {
                "metadata": X_test_meta,
                "lexical": X_test_lex,
                "behavioral": X_test_beh,
                "semantic": X_test_sem,
                "ids": test_ids,
            },
        }

        # 6. Save to Cache
        self._save_cache(data)

        return data
