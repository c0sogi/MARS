import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from library.config import Config
from library.utils import (
    load_dataset,
    save_cache_npy,
    load_cache_npy,
    save_cache_npz,
    load_cache_npz,
    ensure_cache_dir,
)
from library.text_processing import clean_text, generate_embeddings


class MetadataExtractor:
    """
    Handles extraction, imputation, and scaling of numerical and temporal metadata.
    Represents the 'Unified Context' vector.
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_cols = Config.RAW_DENSE_FEATURES

    def _extract_temporal(self, df: pd.DataFrame) -> np.ndarray:
        """Derives temporal features from unix timestamp."""
        # Handle potential NaNs in timestamp if any (though unlikely in this dataset)
        ts = df[Config.TIMESTAMP_COL].fillna(df[Config.TIMESTAMP_COL].median())
        dt = pd.to_datetime(ts, unit="s")

        # Features: Hour of day, Day of week
        hour = dt.dt.hour.values.reshape(-1, 1)
        day_of_week = dt.dt.dayofweek.values.reshape(-1, 1)

        return np.hstack([hour, day_of_week])

    def fit(self, df: pd.DataFrame):
        # Select raw numerical features
        # Ensure columns exist
        X_raw = df[self.feature_cols].values

        # Fit imputer
        self.imputer.fit(X_raw)

        # Transform to get imputed values for scaling fit
        X_imputed = self.imputer.transform(X_raw)

        # Extract temporal features
        X_temp = self._extract_temporal(df)

        # Combine for scaling fit
        X_full = np.hstack([X_imputed, X_temp])

        # Fit scaler
        self.scaler.fit(X_full)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X_raw = df[self.feature_cols].values

        # Impute
        X_imputed = self.imputer.transform(X_raw)

        # Temporal
        X_temp = self._extract_temporal(df)

        # Combine
        X_full = np.hstack([X_imputed, X_temp])

        # Scale
        X_scaled = self.scaler.transform(X_full)

        return X_scaled


class LexicalVectorizer:
    """
    Handles TF-IDF vectorization of the request text.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=Config.TFIDF_TEXT_MAX_FEATURES,
            ngram_range=Config.TFIDF_TEXT_NGRAM_RANGE,
            stop_words="english",
            preprocessor=clean_text,
        )

    def fit(self, df: pd.DataFrame):
        text_data = df[Config.TEXT_COL].fillna("").astype(str)
        self.vectorizer.fit(text_data)
        return self

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        text_data = df[Config.TEXT_COL].fillna("").astype(str)
        return self.vectorizer.transform(text_data)


class BehavioralVectorizer:
    """
    Handles TF-IDF vectorization of the subreddit history.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=Config.TFIDF_SUBREDDIT_MAX_FEATURES,
            ngram_range=Config.TFIDF_SUBREDDIT_NGRAM_RANGE,
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",  # Simple word token pattern
        )

    def _process_subreddits(self, df: pd.DataFrame) -> pd.Series:
        """Converts list of subreddits to a space-separated string."""

        def join_subs(x):
            if isinstance(x, (list, np.ndarray)):
                return " ".join([str(s) for s in x])
            return str(x) if pd.notnull(x) else ""

        return df[Config.SUBREDDIT_COL].apply(join_subs)

    def fit(self, df: pd.DataFrame):
        processed_subs = self._process_subreddits(df)
        self.vectorizer.fit(processed_subs)
        return self

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        processed_subs = self._process_subreddits(df)
        return self.vectorizer.transform(processed_subs)


class DataPreparer:
    """
    Orchestrates data loading, feature extraction, concatenation, and caching.
    """

    def __init__(self):
        self.meta_extractor = MetadataExtractor()
        self.lexical_vectorizer = LexicalVectorizer()
        self.behavioral_vectorizer = BehavioralVectorizer()

    def _load_sparse_from_cache(self, filename: str):
        loader = load_cache_npz(filename)
        if loader is None:
            return None
        try:
            return sparse.csr_matrix(
                (loader["data"], loader["indices"], loader["indptr"]),
                shape=loader["shape"],
            )
        except KeyError:
            return None

    def get_features(self, split: str, load_cached_data: bool = True):
        """
        Main method to get processed features for a specific split.
        Returns (X_lexical, X_behavioral, X_semantic, y, ids)
        """
        # 1. Define Cache Filenames
        cache_files = {
            "lexical": f"X_{split}_lexical.npz",
            "behavioral": f"X_{split}_behavioral.npz",
            "semantic": f"X_{split}_semantic.npy",
            "y": f"y_{split}.npy",
            "ids": f"{split}_ids.npy",
        }

        # 2. Check Cache
        if load_cached_data:
            X_lex = self._load_sparse_from_cache(cache_files["lexical"])
            X_beh = self._load_sparse_from_cache(cache_files["behavioral"])
            X_sem = load_cache_npy(cache_files["semantic"])
            y = load_cache_npy(cache_files["y"])
            ids = load_cache_npy(cache_files["ids"])

            # Verify all components exist (y can be None for test)
            has_features = all(v is not None for v in [X_lex, X_beh, X_sem, ids])

            if has_features:
                if split == "test":
                    print(f"Loaded {split} features from cache.")
                    return X_lex, X_beh, X_sem, y, ids
                elif y is not None:
                    print(f"Loaded {split} features from cache.")
                    return X_lex, X_beh, X_sem, y, ids

        print(f"Computing features for {split} (Cache miss or force reload)...")

        # 3. Load Data
        df = load_dataset(split)

        # Handle Debug
        if Config.DEBUG:
            df = df.head(Config.DEBUG_SAMPLE_SIZE)

        # 4. Fit Extractors (Always fit on Train)
        # We must fit on train data to ensure consistent scaling and vocabulary
        if split == "train":
            train_df = df
        else:
            train_df = load_dataset("train")
            if Config.DEBUG:
                train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)

        print("Fitting extractors on training data...")
        self.meta_extractor.fit(train_df)
        self.lexical_vectorizer.fit(train_df)
        self.behavioral_vectorizer.fit(train_df)

        # 5. Transform Target Split
        print(f"Transforming {split} data...")

        # A. Metadata (Dense) - The Unified Context
        X_meta = self.meta_extractor.transform(df)

        # B. Lexical (Sparse Text)
        X_lex_raw = self.lexical_vectorizer.transform(df)
        # Concatenate: [Sparse Text | Dense Meta] -> Sparse CSR
        X_lex_combined = sparse.hstack([X_lex_raw, sparse.csr_matrix(X_meta)])

        # C. Behavioral (Sparse Subreddits)
        X_beh_raw = self.behavioral_vectorizer.transform(df)
        # Concatenate: [Sparse Subs | Dense Meta] -> Sparse CSR
        X_beh_combined = sparse.hstack([X_beh_raw, sparse.csr_matrix(X_meta)])

        # D. Semantic (Dense Embeddings)
        # Check cache for raw embeddings first (handled by generate_embeddings)
        raw_embed_filename = f"raw_embeddings_{split}.npy"
        if Config.DEBUG:
            raw_embed_filename = f"debug_{raw_embed_filename}"

        X_sem_raw = generate_embeddings(
            df[Config.TEXT_COL], raw_embed_filename, load_cached_data=load_cached_data
        )
        # Concatenate: [Dense Embed | Dense Meta] -> Dense Numpy
        X_sem_combined = np.hstack([X_sem_raw, X_meta])

        # 6. Targets and IDs
        ids = df[Config.ID_COL].values.astype(str)
        if Config.TARGET_COL in df.columns:
            y = df[Config.TARGET_COL].values
        else:
            y = None

        # 7. Save to Cache
        print(f"Saving {split} features to cache...")

        # Save sparse matrices decomposed for compatibility with np.savez_compressed
        save_cache_npz(
            {
                "data": X_lex_combined.data,
                "indices": X_lex_combined.indices,
                "indptr": X_lex_combined.indptr,
                "shape": X_lex_combined.shape,
            },
            cache_files["lexical"],
        )

        save_cache_npz(
            {
                "data": X_beh_combined.data,
                "indices": X_beh_combined.indices,
                "indptr": X_beh_combined.indptr,
                "shape": X_beh_combined.shape,
            },
            cache_files["behavioral"],
        )

        save_cache_npy(X_sem_combined, cache_files["semantic"])
        save_cache_npy(ids, cache_files["ids"])

        if y is not None:
            save_cache_npy(y, cache_files["y"])

        return X_lex_combined, X_beh_combined, X_sem_combined, y, ids
