import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import (
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    METADATA_FEATURES,
    TEXT_VOCAB_SIZE,
    HISTORY_VOCAB_SIZE,
    TEXT_MIN_DF,
    NGRAM_RANGE,
    EMBEDDING_MODEL_NAME,
    SEED,
)
from library.utils import (
    print_info,
    print_header,
    save_numpy,
    load_numpy,
    Timer,
    get_device,
    set_seed,
)


class FeatureFactory:
    """
    Factory class to generate specific feature views (Metadata, Lexical, Behavioral, Semantic)
    with caching and strict leakage prevention.
    """

    def __init__(self):
        self.cache_dir = WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.device = get_device()

    def load_raw_data(self):
        """
        Loads raw metadata parquet files and performs initial text normalization.
        Returns train_df, val_df, test_df.
        """
        print_info("Loading raw data from metadata parquet files...")

        train_df = pd.read_parquet(TRAIN_METADATA_PATH)
        val_df = pd.read_parquet(VAL_METADATA_PATH)
        test_df = pd.read_parquet(TEST_METADATA_PATH)

        # Normalize Text Columns
        # Train/Val use 'request_text', Test uses 'request_text_edit_aware'
        print_info("Normalizing text columns...")

        # Helper to combine title and text
        def combine_text(df, text_col):
            return df["request_title"].fillna("") + " " + df[text_col].fillna("")

        train_df["combined_text"] = combine_text(train_df, "request_text")
        val_df["combined_text"] = combine_text(val_df, "request_text")
        test_df["combined_text"] = combine_text(test_df, "request_text_edit_aware")

        # Normalize Behavioral Columns (Subreddits list -> string)
        print_info("Normalizing behavioral columns...")

        def process_subreddits(df):
            # Handle cases where column might be missing or null
            if "requester_subreddits_at_request" not in df.columns:
                return pd.Series([""] * len(df))

            return df["requester_subreddits_at_request"].apply(
                lambda x: " ".join(x) if isinstance(x, (list, np.ndarray)) else ""
            )

        train_df["subreddit_string"] = process_subreddits(train_df)
        val_df["subreddit_string"] = process_subreddits(val_df)
        test_df["subreddit_string"] = process_subreddits(test_df)

        return train_df, val_df, test_df

    def _get_cache_paths(self, view_name):
        return (
            os.path.join(self.cache_dir, f"X_train_{view_name}.npy"),
            os.path.join(self.cache_dir, f"X_val_{view_name}.npy"),
            os.path.join(self.cache_dir, f"X_test_{view_name}.npy"),
        )

    def create_metadata_view(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Generates the Contextual Branch (Metadata) features.
        Includes imputation and scaling.
        """
        print_header("Generating Metadata View")
        paths = self._get_cache_paths("metadata")

        if load_cached_data:
            loaded = [load_numpy(p) for p in paths]
            if all(x is not None for x in loaded):
                print_info("Loaded metadata view from cache.")
                return loaded

        with Timer("Processing Metadata Features"):
            # Select features
            missing_cols = [c for c in METADATA_FEATURES if c not in train_df.columns]
            if missing_cols:
                raise ValueError(f"Missing metadata columns in train: {missing_cols}")

            X_train = train_df[METADATA_FEATURES].values.astype(np.float32)
            X_val = val_df[METADATA_FEATURES].values.astype(np.float32)
            X_test = test_df[METADATA_FEATURES].values.astype(np.float32)

            # Imputation (Median)
            print_info("Imputing missing values (Median)...")
            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(X_train)
            X_val = imputer.transform(X_val)
            X_test = imputer.transform(X_test)

            # Scaling (StandardScaler)
            print_info("Scaling features (StandardScaler)...")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            # Cache
            save_numpy(X_train, paths[0])
            save_numpy(X_val, paths[1])
            save_numpy(X_test, paths[2])

        return X_train, X_val, X_test

    def create_lexical_view(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Generates the Sparse Lexical Branch features using TF-IDF on text.
        """
        print_header("Generating Lexical View")
        paths = self._get_cache_paths("lexical")

        if load_cached_data:
            loaded = [load_numpy(p) for p in paths]
            if all(x is not None for x in loaded):
                print_info("Loaded lexical view from cache.")
                return loaded

        with Timer("Processing Lexical Features (TF-IDF)"):
            vectorizer = TfidfVectorizer(
                max_features=TEXT_VOCAB_SIZE,
                ngram_range=NGRAM_RANGE,
                min_df=TEXT_MIN_DF,
                sublinear_tf=True,
                stop_words="english",
                token_pattern=r"(?u)\b\w\w+\b",
            )

            print_info("Fitting TfidfVectorizer on training text...")
            X_train_sparse = vectorizer.fit_transform(train_df["combined_text"])
            X_val_sparse = vectorizer.transform(val_df["combined_text"])
            X_test_sparse = vectorizer.transform(test_df["combined_text"])

            # Convert to dense numpy arrays for storage (dataset is small enough)
            print_info("Densifying matrices...")
            X_train = X_train_sparse.toarray().astype(np.float32)
            X_val = X_val_sparse.toarray().astype(np.float32)
            X_test = X_test_sparse.toarray().astype(np.float32)

            # Cache
            save_numpy(X_train, paths[0])
            save_numpy(X_val, paths[1])
            save_numpy(X_test, paths[2])

        return X_train, X_val, X_test

    def create_behavioral_view(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Generates the Sparse Behavioral Branch features using TF-IDF on subreddits.
        """
        print_header("Generating Behavioral View")
        paths = self._get_cache_paths("behavioral")

        if load_cached_data:
            loaded = [load_numpy(p) for p in paths]
            if all(x is not None for x in loaded):
                print_info("Loaded behavioral view from cache.")
                return loaded

        with Timer("Processing Behavioral Features (TF-IDF)"):
            # Use simple whitespace tokenizer for pre-joined subreddit strings
            vectorizer = TfidfVectorizer(
                max_features=HISTORY_VOCAB_SIZE,
                ngram_range=(1, 1),
                min_df=2,
                sublinear_tf=True,
                token_pattern=r"(?u)\b\w+\b",  # Simple word tokenization
            )

            print_info("Fitting TfidfVectorizer on subreddit history...")
            X_train_sparse = vectorizer.fit_transform(train_df["subreddit_string"])
            X_val_sparse = vectorizer.transform(val_df["subreddit_string"])
            X_test_sparse = vectorizer.transform(test_df["subreddit_string"])

            # Densify
            X_train = X_train_sparse.toarray().astype(np.float32)
            X_val = X_val_sparse.toarray().astype(np.float32)
            X_test = X_test_sparse.toarray().astype(np.float32)

            # Cache
            save_numpy(X_train, paths[0])
            save_numpy(X_val, paths[1])
            save_numpy(X_test, paths[2])

        return X_train, X_val, X_test

    def create_semantic_view(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Generates the Dense Semantic Branch features using Sentence Transformers.
        """
        print_header("Generating Semantic View")
        paths = self._get_cache_paths("semantic")

        if load_cached_data:
            loaded = [load_numpy(p) for p in paths]
            if all(x is not None for x in loaded):
                print_info("Loaded semantic view from cache.")
                return loaded

        with Timer("Processing Semantic Features (Embeddings)"):
            print_info(f"Loading Sentence Transformer: {EMBEDDING_MODEL_NAME}")
            # Ensure CPU/GPU usage
            model = SentenceTransformer(
                EMBEDDING_MODEL_NAME, device=str(self.device).split(":")[0]
            )

            # Encode
            print_info("Encoding training data...")
            X_train = model.encode(
                train_df["combined_text"].tolist(),
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            print_info("Encoding validation data...")
            X_val = model.encode(
                val_df["combined_text"].tolist(),
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            print_info("Encoding test data...")
            X_test = model.encode(
                test_df["combined_text"].tolist(),
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Cache
            save_numpy(X_train, paths[0])
            save_numpy(X_val, paths[1])
            save_numpy(X_test, paths[2])

        return X_train, X_val, X_test
