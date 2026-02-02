import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import (
    WORKING_DIR,
    DENSE_FEATURE_COLS,
    LEXICAL_VOCAB_SIZE,
    LEXICAL_NGRAM_RANGE,
    LEXICAL_MIN_DF,
    LEXICAL_SUBLINEAR_TF,
    BEHAVIORAL_VOCAB_SIZE,
    BEHAVIORAL_NGRAM_RANGE,
    SBERT_MODEL_NAME,
    SBERT_BATCH_SIZE,
    SEED,
)
from library.utils import set_seed


class FeatureExtractor:
    def __init__(self):
        """
        Initializes the FeatureExtractor with necessary transformers and models.
        """
        set_seed(SEED)

        # Metadata processors
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        # Lexical processor (Request Text)
        self.lexical_vectorizer = TfidfVectorizer(
            max_features=LEXICAL_VOCAB_SIZE,
            ngram_range=LEXICAL_NGRAM_RANGE,
            min_df=LEXICAL_MIN_DF,
            sublinear_tf=LEXICAL_SUBLINEAR_TF,
            stop_words="english",
        )

        # Behavioral processor (Subreddits)
        self.behavioral_vectorizer = TfidfVectorizer(
            max_features=BEHAVIORAL_VOCAB_SIZE,
            ngram_range=BEHAVIORAL_NGRAM_RANGE,
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",  # Capture alphanumeric words
        )

        # Semantic processor (SBERT)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing SBERT model on {device}...")
        self.sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=device)

    def _compute_derived_features(self, df):
        """
        Computes derived statistics like text length.
        Returns a new DataFrame with added columns.
        """
        df = df.copy()

        # Ensure text columns exist
        if "text" not in df.columns:
            df["text"] = ""
        if "title" not in df.columns:
            df["title"] = ""

        # Calculate lengths
        df["request_text_len_char"] = df["text"].astype(str).str.len()
        df["request_text_len_word"] = (
            df["text"].astype(str).apply(lambda x: len(x.split()))
        )
        df["request_title_len_char"] = df["title"].astype(str).str.len()
        df["request_title_len_word"] = (
            df["title"].astype(str).apply(lambda x: len(x.split()))
        )

        return df

    def _process_subreddits_col(self, series):
        """
        Converts a series of subreddit lists into a series of space-separated strings.
        """

        def join_subs(x):
            if isinstance(x, list):
                return " ".join(x)
            elif isinstance(x, np.ndarray):
                return " ".join(x.tolist())
            return str(x) if x is not None else ""

        return series.apply(join_subs)

    def fit(self, train_df):
        """
        Fits the internal transformers (Imputer, Scaler, Vectorizers) on the training data.
        """
        print("Fitting FeatureExtractor...")

        # 1. Metadata Fitting
        train_df_derived = self._compute_derived_features(train_df)

        # Extract dense features for fitting
        # We assume columns in DENSE_FEATURE_COLS exist after derivation
        X_dense = train_df_derived[DENSE_FEATURE_COLS].values

        self.imputer.fit(X_dense)
        X_dense_imputed = self.imputer.transform(X_dense)
        self.scaler.fit(X_dense_imputed)

        # 2. Lexical Fitting
        text_corpus = train_df_derived["text"].astype(str).fillna("").tolist()
        self.lexical_vectorizer.fit(text_corpus)

        # 3. Behavioral Fitting
        subreddit_corpus = self._process_subreddits_col(
            train_df["requester_subreddits_at_request"]
        ).tolist()
        self.behavioral_vectorizer.fit(subreddit_corpus)

        print("FeatureExtractor fitting complete.")
        return self

    def transform(self, df, split_name, load_cached_data=True):
        """
        Transforms the dataframe into feature views.
        Checks for cached files in WORKING_DIR/idea_16/ before computing.

        Args:
            df (pd.DataFrame): Data to transform.
            split_name (str): 'train', 'val', or 'test' (used for cache naming).
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: Dictionary containing 'metadata', 'lexical', 'behavioral', 'semantic' features.
        """
        # Define cache paths
        cache_dir = WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        path_meta = os.path.join(cache_dir, f"X_{split_name}_metadata.npy")
        path_lex = os.path.join(cache_dir, f"X_{split_name}_lexical.npz")
        path_beh = os.path.join(cache_dir, f"X_{split_name}_behavioral.npz")
        path_sem = os.path.join(cache_dir, f"X_{split_name}_semantic.npy")

        # 1. Try Loading Cache
        if load_cached_data:
            if (
                os.path.exists(path_meta)
                and os.path.exists(path_lex)
                and os.path.exists(path_beh)
                and os.path.exists(path_sem)
            ):
                print(f"Loading {split_name} features from cache...")
                return {
                    "metadata": np.load(path_meta),
                    "lexical": sp.load_npz(path_lex),
                    "behavioral": sp.load_npz(path_beh),
                    "semantic": np.load(path_sem),
                }
            else:
                print(f"Cache miss for {split_name}. Computing features...")
        else:
            print(f"Ignoring cache for {split_name}. Computing features...")

        # 2. Compute Features
        df_derived = self._compute_derived_features(df)

        # A. Metadata (Dense)
        X_dense = df_derived[DENSE_FEATURE_COLS].values
        X_dense = self.imputer.transform(X_dense)
        X_metadata = self.scaler.transform(X_dense)

        # B. Lexical (Sparse)
        text_corpus = df_derived["text"].astype(str).fillna("").tolist()
        X_lexical = self.lexical_vectorizer.transform(text_corpus)

        # C. Behavioral (Sparse)
        subreddit_corpus = self._process_subreddits_col(
            df["requester_subreddits_at_request"]
        ).tolist()
        X_behavioral = self.behavioral_vectorizer.transform(subreddit_corpus)

        # D. Semantic (Dense Embeddings)
        # Combine Title and Text for context
        semantic_inputs = (
            "Title: "
            + df_derived["title"].astype(str)
            + " \n Request: "
            + df_derived["text"].astype(str)
        ).tolist()

        X_semantic = self.sbert_model.encode(
            semantic_inputs,
            batch_size=SBERT_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # 3. Save to Cache
        print(f"Saving {split_name} features to cache...")
        np.save(path_meta, X_metadata)
        sp.save_npz(path_lex, X_lexical)
        sp.save_npz(path_beh, X_behavioral)
        np.save(path_sem, X_semantic)

        return {
            "metadata": X_metadata,
            "lexical": X_lexical,
            "behavioral": X_behavioral,
            "semantic": X_semantic,
        }
