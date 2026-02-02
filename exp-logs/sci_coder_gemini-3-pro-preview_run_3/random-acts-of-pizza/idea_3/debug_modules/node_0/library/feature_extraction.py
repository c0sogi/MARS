import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from library.config import Config


class FeatureGenerator:
    """
    Generates specific feature views (Lexical and Semantic) for the
    Multi-Paradigm Stacking Ensemble. Handles vectorization, embedding,
    metadata concatenation, and caching.
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_lexical_view(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Generates the Lexical View: Sparse TF-IDF vectors concatenated with
        dense metadata features.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
            test_df (pd.DataFrame): Test data.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            tuple: (X_train_sparse, X_val_sparse, X_test_sparse) as scipy.sparse.csr_matrix
        """
        # Define cache paths
        train_path = os.path.join(self.cache_dir, "X_train_lexical.npz")
        val_path = os.path.join(self.cache_dir, "X_val_lexical.npz")
        test_path = os.path.join(self.cache_dir, "X_test_lexical.npz")

        # 1. Try Loading from Cache
        if load_cached_data:
            if (
                os.path.exists(train_path)
                and os.path.exists(val_path)
                and os.path.exists(test_path)
            ):
                print("Loading Lexical View from cache...")
                X_train = sparse.load_npz(train_path)
                X_val = sparse.load_npz(val_path)
                X_test = sparse.load_npz(test_path)
                return X_train, X_val, X_test

        print("Generating Lexical View (TF-IDF + Metadata)...")

        # 2. Extract Metadata (Dense)
        meta_train, meta_val, meta_test = self._extract_metadata(
            train_df, val_df, test_df
        )

        # 3. Generate TF-IDF Features (Sparse)
        print("Fitting TF-IDF Vectorizer...")
        vectorizer = TfidfVectorizer(**Config.TFIDF_PARAMS)

        # Fit on train, transform all
        txt_train = train_df[Config.TEXT_COL].fillna("").astype(str)
        txt_val = val_df[Config.TEXT_COL].fillna("").astype(str)
        txt_test = test_df[Config.TEXT_COL].fillna("").astype(str)

        tfidf_train = vectorizer.fit_transform(txt_train)
        tfidf_val = vectorizer.transform(txt_val)
        tfidf_test = vectorizer.transform(txt_test)

        # 4. Concatenate Sparse + Dense
        # Convert metadata to sparse for efficient stacking
        X_train = sparse.hstack([tfidf_train, sparse.csr_matrix(meta_train)])
        X_val = sparse.hstack([tfidf_val, sparse.csr_matrix(meta_val)])
        X_test = sparse.hstack([tfidf_test, sparse.csr_matrix(meta_test)])

        # Ensure CSR format
        X_train = X_train.tocsr()
        X_val = X_val.tocsr()
        X_test = X_test.tocsr()

        # 5. Save to Cache
        print("Saving Lexical View to cache...")
        sparse.save_npz(train_path, X_train)
        sparse.save_npz(val_path, X_val)
        sparse.save_npz(test_path, X_test)

        return X_train, X_val, X_test

    def get_semantic_view(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Generates the Semantic View: Dense SBERT embeddings concatenated with
        dense metadata features.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
            test_df (pd.DataFrame): Test data.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            tuple: (X_train_dense, X_val_dense, X_test_dense) as numpy.ndarray
        """
        # Define cache paths
        train_path = os.path.join(self.cache_dir, "X_train_semantic.npy")
        val_path = os.path.join(self.cache_dir, "X_val_semantic.npy")
        test_path = os.path.join(self.cache_dir, "X_test_semantic.npy")

        # 1. Try Loading from Cache
        if load_cached_data:
            if (
                os.path.exists(train_path)
                and os.path.exists(val_path)
                and os.path.exists(test_path)
            ):
                print("Loading Semantic View from cache...")
                X_train = np.load(train_path)
                X_val = np.load(val_path)
                X_test = np.load(test_path)
                return X_train, X_val, X_test

        print("Generating Semantic View (Embeddings + Metadata)...")

        # 2. Extract Metadata (Dense)
        meta_train, meta_val, meta_test = self._extract_metadata(
            train_df, val_df, test_df
        )

        # 3. Generate Embeddings (Dense)
        print(f"Loading SBERT model: {Config.SBERT_MODEL_NAME}...")
        model = SentenceTransformer(Config.SBERT_MODEL_NAME)

        # Encode texts
        # Note: show_progress_bar=False to reduce log clutter
        print("Encoding training text...")
        emb_train = model.encode(
            train_df[Config.TEXT_COL].fillna("").astype(str).tolist(),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        print("Encoding validation text...")
        emb_val = model.encode(
            val_df[Config.TEXT_COL].fillna("").astype(str).tolist(),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        print("Encoding test text...")
        emb_test = model.encode(
            test_df[Config.TEXT_COL].fillna("").astype(str).tolist(),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # 4. Concatenate Dense + Dense
        X_train = np.hstack([emb_train, meta_train])
        X_val = np.hstack([emb_val, meta_val])
        X_test = np.hstack([emb_test, meta_test])

        # 5. Save to Cache
        print("Saving Semantic View to cache...")
        np.save(train_path, X_train)
        np.save(val_path, X_val)
        np.save(test_path, X_test)

        return X_train, X_val, X_test

    def _extract_metadata(self, train_df, val_df, test_df):
        """
        Helper to extract consistent numerical metadata columns from DataFrames.
        Excludes ID, Target, and Text columns.
        """
        # Identify feature columns based on training data
        exclude_cols = {Config.ID_COL, Config.TARGET_COL, Config.TEXT_COL}
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]

        # Sort for consistency
        feature_cols.sort()

        # Verify columns exist in all sets
        missing_val = set(feature_cols) - set(val_df.columns)
        missing_test = set(feature_cols) - set(test_df.columns)

        if missing_val:
            raise ValueError(f"Validation set missing columns: {missing_val}")
        if missing_test:
            raise ValueError(f"Test set missing columns: {missing_test}")

        # Extract values
        meta_train = train_df[feature_cols].values.astype(np.float32)
        meta_val = val_df[feature_cols].values.astype(np.float32)
        meta_test = test_df[feature_cols].values.astype(np.float32)

        return meta_train, meta_val, meta_test
