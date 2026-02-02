import os
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import (
    CACHE_DIR,
    SEED,
    TEXT_COLS,
    COMMUNITY_COL,
    ALLOW_LIST_METADATA,
    TFIDF_PARAMS,
    COMMUNITY_VOCAB_SIZE,
)
from library.data_loader import clean_dataset


class FeatureFactory:
    """
    Manages feature engineering for the Oct-View architecture.
    Handles Lexical, Behavioral, Semantic, and Metadata feature generation
    with caching and leakage prevention.
    """

    def __init__(self):
        self.transformers_path = os.path.join(CACHE_DIR, "transformers.joblib")

        # 1. Lexical Branch (Sparse)
        # Used for Lexical Bagger and Lexical Anchor
        self.lexical_vectorizer = TfidfVectorizer(**TFIDF_PARAMS)

        # 2. Behavioral Branch (Sparse)
        # We treat subreddits as tokens in a "Bag of Communities"
        self.community_vectorizer = TfidfVectorizer(
            max_features=COMMUNITY_VOCAB_SIZE,
            ngram_range=(1, 1),  # Unigrams of subreddits
            token_pattern=r"(?u)\b\w+\b",  # Simple alphanumeric tokenization
            stop_words=None,  # Subreddit names are significant, don't remove "us", "it", etc.
            sublinear_tf=True,
        )

        # 3. Metadata Branch (Dense)
        # Used for Metadata Anchor and Temporal Booster
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        # 4. Semantic Branch (Dense)
        # Loaded lazily to manage resource usage
        self.semantic_model = None

    def fit(self, train_df: pd.DataFrame):
        """
        Fits all stateful transformers (TF-IDF, Scalers) on the training data.
        """
        print("Fitting FeatureFactory transformers...")

        # Fit Lexical
        print("  - Fitting Lexical Vectorizer...")
        text_data = self._prepare_text(train_df)
        self.lexical_vectorizer.fit(text_data)

        # Fit Behavioral
        print("  - Fitting Community Vectorizer...")
        community_data = self._prepare_community(train_df)
        self.community_vectorizer.fit(community_data)

        # Fit Metadata
        print("  - Fitting Metadata Scaler/Imputer...")
        meta_data = train_df[ALLOW_LIST_METADATA]
        self.imputer.fit(meta_data)
        meta_imputed = self.imputer.transform(meta_data)
        self.scaler.fit(meta_imputed)

        # Save fitted state
        self._save_transformers()
        print("FeatureFactory successfully fitted.")
        return self

    def transform(self, df: pd.DataFrame, split_name: str, load_cache: bool = True):
        """
        Transforms data into feature matrices. Uses caching.

        Args:
            df: Dataframe to transform.
            split_name: 'train', 'val', or 'test' (used for cache naming).
            load_cache: Whether to try loading from cache.

        Returns:
            dict: Dictionary containing 'lexical', 'behavioral', 'semantic', 'metadata' matrices.
        """
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)

        outputs = {}

        # --- 1. Lexical Features ---
        path_lex = os.path.join(CACHE_DIR, f"X_{split_name}_lexical.npz")
        if load_cache and os.path.exists(path_lex):
            outputs["lexical"] = sp.load_npz(path_lex)
        else:
            print(f"  - Generating Lexical features for {split_name}...")
            text_data = self._prepare_text(df)
            X_lex = self.lexical_vectorizer.transform(text_data)
            sp.save_npz(path_lex, X_lex)
            outputs["lexical"] = X_lex

        # --- 2. Behavioral Features ---
        path_beh = os.path.join(CACHE_DIR, f"X_{split_name}_behavioral.npz")
        if load_cache and os.path.exists(path_beh):
            outputs["behavioral"] = sp.load_npz(path_beh)
        else:
            print(f"  - Generating Behavioral features for {split_name}...")
            comm_data = self._prepare_community(df)
            X_beh = self.community_vectorizer.transform(comm_data)
            sp.save_npz(path_beh, X_beh)
            outputs["behavioral"] = X_beh

        # --- 3. Semantic Features ---
        path_sem = os.path.join(CACHE_DIR, f"X_{split_name}_semantic.npy")
        if load_cache and os.path.exists(path_sem):
            outputs["semantic"] = np.load(path_sem)
        else:
            print(f"  - Generating Semantic features for {split_name}...")
            if self.semantic_model is None:
                # Initialize model only when needed
                self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

            text_data = self._prepare_text(df)
            # Encode returns numpy array by default with convert_to_numpy=True
            X_sem = self.semantic_model.encode(
                text_data.tolist(),
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            np.save(path_sem, X_sem)
            outputs["semantic"] = X_sem

        # --- 4. Metadata Features ---
        path_meta = os.path.join(CACHE_DIR, f"X_{split_name}_meta.npy")
        if load_cache and os.path.exists(path_meta):
            outputs["metadata"] = np.load(path_meta)
        else:
            print(f"  - Generating Metadata features for {split_name}...")
            meta_data = df[ALLOW_LIST_METADATA]
            X_meta = self.imputer.transform(meta_data)
            X_meta = self.scaler.transform(X_meta)
            np.save(path_meta, X_meta)
            outputs["metadata"] = X_meta

        return outputs

    def _prepare_text(self, df: pd.DataFrame) -> pd.Series:
        """Concatenates title and body."""
        # Use TEXT_COLS from config: ["request_title", "request_text_edit_aware"]
        t = df[TEXT_COLS[0]].fillna("").astype(str)
        b = df[TEXT_COLS[1]].fillna("").astype(str)
        return t + " " + b

    def _prepare_community(self, df: pd.DataFrame) -> pd.Series:
        """Converts subreddit lists to space-separated strings."""

        def join_subs(x):
            if isinstance(x, list):
                return " ".join(x)
            elif isinstance(x, np.ndarray):
                return " ".join(x)
            return ""

        return df[COMMUNITY_COL].apply(join_subs)

    def _save_transformers(self):
        """Saves fitted transformers to disk."""
        payload = {
            "lexical": self.lexical_vectorizer,
            "community": self.community_vectorizer,
            "imputer": self.imputer,
            "scaler": self.scaler,
        }
        joblib.dump(payload, self.transformers_path)

    def load_transformers(self):
        """Loads fitted transformers from disk if they exist."""
        if os.path.exists(self.transformers_path):
            payload = joblib.load(self.transformers_path)
            self.lexical_vectorizer = payload["lexical"]
            self.community_vectorizer = payload["community"]
            self.imputer = payload["imputer"]
            self.scaler = payload["scaler"]
            return True
        return False

    @staticmethod
    def combine_features(feature_dict: dict, keys: list):
        """
        Horizontally stacks selected feature matrices.

        Args:
            feature_dict: Dictionary returned by transform().
            keys: List of keys to stack (e.g., ['lexical', 'metadata']).

        Returns:
            CSR Matrix representing the combined features.
        """
        to_stack = []
        for k in keys:
            if k not in feature_dict:
                raise ValueError(f"Feature key '{k}' not found in dictionary.")

            matrix = feature_dict[k]

            # Ensure 2D shape for dense arrays before stacking
            if len(matrix.shape) == 1:
                matrix = matrix.reshape(-1, 1)

            # Explicitly convert dense arrays to sparse to avoid hstack broadcasting errors
            if not sp.issparse(matrix):
                matrix = sp.csr_matrix(matrix)

            to_stack.append(matrix)

        return sp.hstack(to_stack, format="csr")
