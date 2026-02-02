import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import setup_logger


class FeatureEngineer:
    """
    Implements the feature engineering logic for the Pent-View Stacking Ensemble.
    Generates Lexical, Behavioral, Semantic, and Metadata views.
    """

    def __init__(self):
        self.logger = setup_logger("feature_engineer")
        self.cache_dir = Config.CACHE_DIR
        self.embedding_model = None  # Lazy load

        # State for transformers
        self.tfidf_lexical = None
        self.tfidf_behavioral = None
        self.scaler = None
        self.feature_names_lexical = None
        self.feature_names_behavioral = None

    def _get_embedding_model(self):
        if self.embedding_model is None:
            self.logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL_NAME}")
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        return self.embedding_model

    def _compute_embeddings(self, texts, cache_name, load_cache=True):
        """
        Computes or loads embeddings for a list of texts.
        """
        cache_path = os.path.join(self.cache_dir, cache_name)

        if load_cache and os.path.exists(cache_path):
            self.logger.info(f"Loading embeddings from {cache_path}")
            return np.load(cache_path)

        self.logger.info(f"Computing embeddings for {cache_name}...")
        model = self._get_embedding_model()
        # Ensure texts are strings
        texts = [str(t) for t in texts]
        embeddings = model.encode(
            texts,
            batch_size=Config.BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Save to cache
        np.save(cache_path, embeddings)
        return embeddings

    def _process_subreddit_list(self, df):
        """Converts list of subreddits to space-separated string."""
        if Config.SUBREDDIT_LIST_COL not in df.columns:
            return [""] * len(df)

        def join_subs(x):
            if isinstance(x, (list, np.ndarray)):
                return " ".join(x)
            return str(x) if pd.notnull(x) else ""

        return df[Config.SUBREDDIT_LIST_COL].apply(join_subs).tolist()

    def fit(self, train_df):
        """
        Fits the vectorizers and scalers on the training data.
        """
        self.logger.info("Fitting feature transformers on training data...")

        # 1. Lexical TF-IDF (Request Text)
        self.logger.info("Fitting Lexical TF-IDF...")
        self.tfidf_lexical = TfidfVectorizer(**Config.TFIDF_PARAMS)
        self.tfidf_lexical.fit(train_df[Config.TEXT_COL].astype(str).fillna(""))

        # 2. Behavioral TF-IDF (Subreddit History)
        self.logger.info("Fitting Behavioral TF-IDF...")
        train_subs = self._process_subreddit_list(train_df)
        self.tfidf_behavioral = TfidfVectorizer(**Config.TFIDF_PARAMS)
        self.tfidf_behavioral.fit(train_subs)

        # 3. Metadata Scaler
        self.logger.info("Fitting Metadata Scaler...")
        # We need to pre-calculate the consistency feature for the train set to fit the scaler correctly
        # However, to avoid circular dependency, we calculate raw metadata + consistency for train here temporarily

        # Extract raw numeric
        meta_df = train_df[Config.NUMERICAL_FEATURES].copy()

        # Calculate Consistency for fit (requires embeddings)
        # We use the internal helper but ensure we don't trigger full transform logic yet
        train_text_emb = self._compute_embeddings(
            train_df[Config.TEXT_COL].astype(str).fillna("").tolist(),
            "train_text_emb_temp.npy",
        )
        train_hist_emb = self._compute_embeddings(train_subs, "train_hist_emb_temp.npy")

        # Cosine Similarity (Row-wise)
        # Normalize manually to be safe, though SentenceTransformer usually gives normalized vectors
        norm_text = np.linalg.norm(train_text_emb, axis=1, keepdims=True)
        norm_hist = np.linalg.norm(train_hist_emb, axis=1, keepdims=True)
        # Avoid division by zero
        norm_text[norm_text == 0] = 1e-10
        norm_hist[norm_hist == 0] = 1e-10

        dot_product = np.sum(train_text_emb * train_hist_emb, axis=1, keepdims=True)
        consistency = dot_product / (norm_text * norm_hist)

        meta_df[Config.CROSS_MODAL_SIM_COL] = consistency.flatten()

        # Impute
        self.imputer_values = meta_df.median()
        meta_df = meta_df.fillna(self.imputer_values)

        # Fit Scaler
        self.scaler = StandardScaler()
        self.scaler.fit(meta_df)

        self.logger.info("Fitting complete.")

    def transform(self, df, split_name, load_cache=True):
        """
        Transforms the dataframe into the 4 views.
        """
        self.logger.info(f"Transforming features for split: {split_name}")

        # --- 1. Embeddings (Semantic & Consistency) ---
        # Text Embeddings
        text_emb = self._compute_embeddings(
            df[Config.TEXT_COL].astype(str).fillna("").tolist(),
            f"{split_name}_text_embeddings.npy",
            load_cache=load_cache,
        )

        # History Embeddings
        subs_list = self._process_subreddit_list(df)
        hist_emb = self._compute_embeddings(
            subs_list, f"{split_name}_hist_embeddings.npy", load_cache=load_cache
        )

        # --- 2. Metadata & Consistency ---
        meta_df = df[Config.NUMERICAL_FEATURES].copy()

        # Compute Consistency
        norm_text = np.linalg.norm(text_emb, axis=1, keepdims=True)
        norm_hist = np.linalg.norm(hist_emb, axis=1, keepdims=True)
        norm_text[norm_text == 0] = 1e-10
        norm_hist[norm_hist == 0] = 1e-10
        dot_product = np.sum(text_emb * hist_emb, axis=1, keepdims=True)
        consistency = dot_product / (norm_text * norm_hist)

        meta_df[Config.CROSS_MODAL_SIM_COL] = consistency.flatten()

        # Impute and Scale
        meta_df = meta_df.fillna(self.imputer_values)
        meta_scaled = self.scaler.transform(meta_df)

        # --- 3. Sparse Views (Lexical & Behavioral) ---
        # Lexical TF-IDF
        tfidf_lex = self.tfidf_lexical.transform(
            df[Config.TEXT_COL].astype(str).fillna("")
        )

        # Behavioral TF-IDF
        tfidf_beh = self.tfidf_behavioral.transform(subs_list)

        # --- 4. Construct Views (Concatenation) ---
        # Architecture requires Global Metadata Vector concatenated to all views

        # View 1: Lexical (Sparse TFIDF + Dense Meta)
        # We need to stack sparse and dense.
        X_lexical = sp.hstack([tfidf_lex, meta_scaled], format="csr")

        # View 2: Behavioral (Sparse TFIDF + Dense Meta)
        X_behavioral = sp.hstack([tfidf_beh, meta_scaled], format="csr")

        # View 3: Semantic (Dense Embedding + Dense Meta)
        X_semantic = np.hstack([text_emb, meta_scaled])

        # View 4: Metadata (Dense Meta)
        X_metadata = meta_scaled

        return {
            "lexical": X_lexical,
            "behavioral": X_behavioral,
            "semantic": X_semantic,
            "metadata": X_metadata,
        }

    def _save_views(self, views, split_name):
        """Saves the generated views to cache."""
        for view_name, data in views.items():
            filename = f"X_{split_name}_{view_name}"
            path = os.path.join(self.cache_dir, filename)

            if sp.issparse(data):
                sp.save_npz(path + ".npz", data)
            else:
                np.save(path + ".npy", data)

    def _load_views(self, split_name):
        """Attempts to load views from cache."""
        views = {}
        required_views = ["lexical", "behavioral", "semantic", "metadata"]

        for view_name in required_views:
            filename = f"X_{split_name}_{view_name}"
            path_base = os.path.join(self.cache_dir, filename)

            if os.path.exists(path_base + ".npz"):
                views[view_name] = sp.load_npz(path_base + ".npz")
            elif os.path.exists(path_base + ".npy"):
                views[view_name] = np.load(path_base + ".npy")
            else:
                return None  # Cache incomplete
        return views

    def create_features(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Main driver to create or load features for all splits.
        """
        self.logger.info(f"Creating features (load_cache={load_cached_data})...")

        # 1. Try Loading from Cache
        if load_cached_data:
            train_views = self._load_views("train")
            val_views = self._load_views("val")
            test_views = self._load_views("test")

            if train_views and val_views and test_views:
                self.logger.info("All feature views loaded from cache successfully.")
                return train_views, val_views, test_views
            else:
                self.logger.info("Cache incomplete or missing. Recomputing features...")

        # 2. Compute from Scratch
        # Fit on Train
        self.fit(train_df)

        # Transform all
        train_views = self.transform(train_df, "train", load_cache=load_cached_data)
        val_views = self.transform(val_df, "val", load_cache=load_cached_data)
        test_views = self.transform(test_df, "test", load_cache=load_cached_data)

        # 3. Save to Cache
        self.logger.info("Saving feature views to cache...")
        self._save_views(train_views, "train")
        self._save_views(val_views, "val")
        self._save_views(test_views, "test")

        return train_views, val_views, test_views
