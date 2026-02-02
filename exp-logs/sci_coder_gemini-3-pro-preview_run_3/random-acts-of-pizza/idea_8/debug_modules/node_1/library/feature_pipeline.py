import os
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library import config, utils


class FeaturePipeline:
    def __init__(self):
        """
        Initializes the FeaturePipeline with necessary vectorizers and scalers.
        SBERT model is initialized lazily to save resources if not needed immediately.
        """
        self.lexical_vectorizer = TfidfVectorizer(**config.LEXICAL_VECTORIZER_PARAMS)
        self.behavioral_vectorizer = TfidfVectorizer(
            **config.BEHAVIORAL_VECTORIZER_PARAMS
        )
        self.meta_imputer = SimpleImputer(strategy="median")
        self.meta_scaler = StandardScaler()
        self.sbert_model = None

    def _load_sbert(self):
        """Lazy loader for SBERT model."""
        if self.sbert_model is None:
            self.sbert_model = SentenceTransformer(config.SBERT_MODEL_NAME)

    def _clean_data(self, df):
        """
        Removes leakage columns (those ending with specific suffixes like '_at_retrieval').
        """
        cols_to_drop = [
            c
            for c in df.columns
            if any(c.endswith(suffix) for suffix in config.DROP_SUFFIXES)
        ]
        return df.drop(columns=cols_to_drop, errors="ignore")

    def _augment_metadata(self, df):
        """
        Generates augmented metadata features (text complexity) and selects numerical columns.
        """
        df = df.copy()

        # Ensure text column exists and is string
        texts = df[config.TEXT_COL].fillna("").astype(str)

        # 1. Word Count
        df["text_word_count"] = texts.apply(lambda x: len(x.split()))

        # 2. Sentence Count (Heuristic based on punctuation to avoid NLTK dependency issues)
        df["text_sentence_count"] = texts.apply(
            lambda x: x.count(".") + x.count("!") + x.count("?")
        )

        # Select numerical columns for Meta view
        # Exclude ID, Target, Text, List columns, and internal paths
        exclude_cols = {
            config.ID_COL,
            config.TARGET_COL,
            config.TEXT_COL,
            config.TITLE_COL,
            config.SUBREDDIT_COL,
            "source_file",
            "request_text",
            "request_text_edit_aware",
        }

        numerical_cols = df.select_dtypes(include=["number"]).columns
        final_cols = [c for c in numerical_cols if c not in exclude_cols]

        return df[final_cols]

    def _get_behavioral_string(self, df):
        """
        Converts the list of subreddits into a space-separated string for TF-IDF vectorization.
        Uses robust list conversion from utils.
        """

        def process_subreddits(x):
            lst = utils.safe_convert_list(x)
            # Normalize to lowercase and join
            return " ".join([str(s).lower() for s in lst])

        return df[config.SUBREDDIT_COL].apply(process_subreddits)

    def _get_cache_paths(self, split_name):
        """Returns a dictionary of file paths for caching."""
        return {
            "lexical": os.path.join(config.CACHE_DIR, f"X_lexical_{split_name}.npz"),
            "semantic": os.path.join(config.CACHE_DIR, f"X_semantic_{split_name}.npy"),
            "behavioral": os.path.join(
                config.CACHE_DIR, f"X_behavioral_{split_name}.npz"
            ),
            "meta": os.path.join(config.CACHE_DIR, f"X_meta_{split_name}.npy"),
        }

    def fit_transform(self, df, split_name="train", load_cached_data=True):
        """
        Fits the pipeline on the provided dataframe and transforms it.

        NOTE: Even if cached matrices are loaded, we MUST fit the vectorizers/scalers
        on the raw data to ensure the pipeline state is ready for subsequent 'transform'
        calls on validation/test sets. SBERT embeddings are the exception, as they
        are computationally expensive and the model is pre-trained.
        """
        print(f"[FeaturePipeline] fit_transform called for split: {split_name}")

        # 1. Clean Data
        df_clean = self._clean_data(df)
        paths = self._get_cache_paths(split_name)
        features = {}

        # --- Semantic View (SBERT) ---
        # This is the bottleneck, so we prioritize caching here.
        if load_cached_data and os.path.exists(paths["semantic"]):
            print(f"  - Loading Semantic features from cache")
            features["semantic"] = np.load(paths["semantic"])
        else:
            print(f"  - Computing Semantic features (SBERT)")
            self._load_sbert()
            texts = df_clean[config.TEXT_COL].fillna("").astype(str).tolist()
            # Encode
            embeddings = self.sbert_model.encode(
                texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
            )
            features["semantic"] = embeddings
            np.save(paths["semantic"], embeddings)

        # --- Lexical View (TF-IDF) ---
        # Must fit vectorizer
        print(f"  - Fitting and Transforming Lexical features")
        texts = df_clean[config.TEXT_COL].fillna("").astype(str)
        features["lexical"] = self.lexical_vectorizer.fit_transform(texts)
        # Save for consistency
        if load_cached_data:
            sp.save_npz(paths["lexical"], features["lexical"])

        # --- Behavioral View (TF-IDF) ---
        # Must fit vectorizer
        print(f"  - Fitting and Transforming Behavioral features")
        sub_strings = self._get_behavioral_string(df_clean)
        features["behavioral"] = self.behavioral_vectorizer.fit_transform(sub_strings)
        if load_cached_data:
            sp.save_npz(paths["behavioral"], features["behavioral"])

        # --- Meta View (Tabular) ---
        # Must fit Imputer and Scaler
        print(f"  - Fitting and Transforming Meta features")
        df_meta = self._augment_metadata(df_clean)
        X_meta = self.meta_imputer.fit_transform(df_meta)
        X_meta = self.meta_scaler.fit_transform(X_meta)
        features["meta"] = X_meta
        if load_cached_data:
            np.save(paths["meta"], X_meta)

        return features

    def transform(self, df, split_name="test", load_cached_data=True):
        """
        Transforms the dataframe using the fitted pipeline.
        Uses cached features if available and requested.
        """
        print(f"[FeaturePipeline] transform called for split: {split_name}")

        # 1. Clean Data
        df_clean = self._clean_data(df)
        paths = self._get_cache_paths(split_name)
        features = {}

        # Check if all requested cache files exist
        all_cached = all(os.path.exists(p) for p in paths.values())

        if load_cached_data and all_cached:
            print(f"  - Loading ALL features from cache")
            features["lexical"] = sp.load_npz(paths["lexical"])
            features["semantic"] = np.load(paths["semantic"])
            features["behavioral"] = sp.load_npz(paths["behavioral"])
            features["meta"] = np.load(paths["meta"])
            return features

        # If cache miss or forced recompute, process each view

        # --- Semantic ---
        if load_cached_data and os.path.exists(paths["semantic"]):
            features["semantic"] = np.load(paths["semantic"])
        else:
            print(f"  - Computing Semantic features")
            self._load_sbert()
            texts = df_clean[config.TEXT_COL].fillna("").astype(str).tolist()
            embeddings = self.sbert_model.encode(
                texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
            )
            features["semantic"] = embeddings
            np.save(paths["semantic"], embeddings)

        # --- Lexical ---
        if load_cached_data and os.path.exists(paths["lexical"]):
            features["lexical"] = sp.load_npz(paths["lexical"])
        else:
            print(f"  - Transforming Lexical features")
            texts = df_clean[config.TEXT_COL].fillna("").astype(str)
            features["lexical"] = self.lexical_vectorizer.transform(texts)
            sp.save_npz(paths["lexical"], features["lexical"])

        # --- Behavioral ---
        if load_cached_data and os.path.exists(paths["behavioral"]):
            features["behavioral"] = sp.load_npz(paths["behavioral"])
        else:
            print(f"  - Transforming Behavioral features")
            sub_strings = self._get_behavioral_string(df_clean)
            features["behavioral"] = self.behavioral_vectorizer.transform(sub_strings)
            sp.save_npz(paths["behavioral"], features["behavioral"])

        # --- Meta ---
        if load_cached_data and os.path.exists(paths["meta"]):
            features["meta"] = np.load(paths["meta"])
        else:
            print(f"  - Transforming Meta features")
            df_meta = self._augment_metadata(df_clean)
            X_meta = self.meta_imputer.transform(df_meta)
            X_meta = self.meta_scaler.transform(X_meta)
            features["meta"] = X_meta
            np.save(paths["meta"], X_meta)

        return features
