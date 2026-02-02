import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
import torch

from library.config import (
    CACHE_DIR,
    TEXT_COLUMN,
    BOW_NGRAM_RANGE,
    BOW_MAX_FEATURES,
    SENTENCE_TRANSFORMER_MODEL,
    RANDOM_SEED,
)


class SparseStreamTransformer:
    """
    Handles the Sparse Stream (Stream A) for the Random Forest model.
    Combines Bag-of-Words text features with raw numerical features.
    """

    def __init__(self):
        self.vectorizer = CountVectorizer(
            ngram_range=BOW_NGRAM_RANGE,
            max_features=BOW_MAX_FEATURES,
            stop_words="english",
        )
        self.imputer = SimpleImputer(strategy="constant", fill_value=0)

    def fit(self, df, numeric_cols):
        """
        Fits the vectorizer and imputer on the training data.
        """
        # Fit text vectorizer
        text_data = df[TEXT_COLUMN].fillna("").astype(str)
        self.vectorizer.fit(text_data)

        # Fit numerical imputer
        if numeric_cols:
            self.imputer.fit(df[numeric_cols])

        return self

    def transform(self, df, numeric_cols):
        """
        Transforms data into a sparse feature matrix.
        """
        # Transform text
        text_data = df[TEXT_COLUMN].fillna("").astype(str)
        text_features = self.vectorizer.transform(text_data)

        # Transform numerical features
        if numeric_cols:
            numeric_data = self.imputer.transform(df[numeric_cols])
            # Convert to sparse matrix to allow hstack
            numeric_features = sparse.csr_matrix(numeric_data)

            # Combine
            X = sparse.hstack([text_features, numeric_features])
        else:
            X = text_features

        return X.tocsr()


class DenseStreamTransformer:
    """
    Handles the Dense Stream (Stream B) for the Logistic Regression model.
    Combines Sentence Embeddings with standardized numerical features.
    """

    def __init__(self):
        self.model_name = SENTENCE_TRANSFORMER_MODEL
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="constant", fill_value=0)
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, df, numeric_cols):
        """
        Fits the scaler and imputer. The sentence transformer is pre-trained.
        """
        # Fit numerical scaler (pipeline: impute -> scale)
        if numeric_cols:
            imputed = self.imputer.fit_transform(df[numeric_cols])
            self.scaler.fit(imputed)
        return self

    def transform(self, df, numeric_cols):
        """
        Transforms data into a dense feature matrix.
        """
        # Encode text
        # Load model inside transform/fit to avoid holding it in memory unnecessarily if not used immediately
        # or instantiate in __init__. Here we instantiate locally to ensure clean state or use cached instance.
        # For efficiency, we assume the model is loaded once.
        model = SentenceTransformer(self.model_name, device=self.device)

        text_data = df[TEXT_COLUMN].fillna("").astype(str).tolist()
        # Encode returns a numpy array
        text_embeddings = model.encode(
            text_data, show_progress_bar=False, convert_to_numpy=True
        )

        # Transform numerical features
        if numeric_cols:
            imputed = self.imputer.transform(df[numeric_cols])
            numeric_features = self.scaler.transform(imputed)

            # Combine
            X = np.hstack([text_embeddings, numeric_features])
        else:
            X = text_embeddings

        return X


def generate_streams(train_df, val_df, test_df, numeric_cols, load_cached_data=True):
    """
    Orchestrates the generation of Sparse and Dense feature streams.
    Handles caching of the resulting matrices.

    Args:
        train_df, val_df, test_df: DataFrames containing raw data.
        numeric_cols: List of numerical column names to include.
        load_cached_data: Boolean to enable loading from disk.

    Returns:
        sparse_data: Dict containing 'train', 'val', 'test' sparse matrices.
        dense_data: Dict containing 'train', 'val', 'test' dense arrays.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define Cache Paths
    # Sparse (npz for CSR)
    sparse_paths = {
        "train": os.path.join(CACHE_DIR, "sparse_train.npz"),
        "val": os.path.join(CACHE_DIR, "sparse_val.npz"),
        "test": os.path.join(CACHE_DIR, "sparse_test.npz"),
    }
    # Dense (npy for numpy array)
    dense_paths = {
        "train": os.path.join(CACHE_DIR, "dense_train.npy"),
        "val": os.path.join(CACHE_DIR, "dense_val.npy"),
        "test": os.path.join(CACHE_DIR, "dense_test.npy"),
    }

    # Check if all files exist
    all_sparse_exist = all(os.path.exists(p) for p in sparse_paths.values())
    all_dense_exist = all(os.path.exists(p) for p in dense_paths.values())

    sparse_data = {}
    dense_data = {}

    # --- Process Sparse Stream ---
    if load_cached_data and all_sparse_exist:
        print("Loading cached Sparse Stream data...")
        sparse_data["train"] = sparse.load_npz(sparse_paths["train"])
        sparse_data["val"] = sparse.load_npz(sparse_paths["val"])
        sparse_data["test"] = sparse.load_npz(sparse_paths["test"])
    else:
        print("Generating Sparse Stream data...")
        transformer = SparseStreamTransformer()
        transformer.fit(train_df, numeric_cols)

        sparse_data["train"] = transformer.transform(train_df, numeric_cols)
        sparse_data["val"] = transformer.transform(val_df, numeric_cols)
        sparse_data["test"] = transformer.transform(test_df, numeric_cols)

        # Save to cache
        print("Caching Sparse Stream data...")
        sparse.save_npz(sparse_paths["train"], sparse_data["train"])
        sparse.save_npz(sparse_paths["val"], sparse_data["val"])
        sparse.save_npz(sparse_paths["test"], sparse_data["test"])

    # --- Process Dense Stream ---
    if load_cached_data and all_dense_exist:
        print("Loading cached Dense Stream data...")
        dense_data["train"] = np.load(dense_paths["train"])
        dense_data["val"] = np.load(dense_paths["val"])
        dense_data["test"] = np.load(dense_paths["test"])
    else:
        print("Generating Dense Stream data (this may take a while)...")
        transformer = DenseStreamTransformer()
        transformer.fit(train_df, numeric_cols)

        dense_data["train"] = transformer.transform(train_df, numeric_cols)
        dense_data["val"] = transformer.transform(val_df, numeric_cols)
        dense_data["test"] = transformer.transform(test_df, numeric_cols)

        # Save to cache
        print("Caching Dense Stream data...")
        np.save(dense_paths["train"], dense_data["train"])
        np.save(dense_paths["val"], dense_data["val"])
        np.save(dense_paths["test"], dense_data["test"])

    return sparse_data, dense_data
