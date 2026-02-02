import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from library.config import (
    MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    TEXT_COLS,
    NUMERICAL_FEATURES,
    WORKING_DIR,
    SEED,
)
from library.utils import set_seed


class HybridFeaturePipeline:
    """
    A pipeline to extract semantic text features using Sentence Transformers
    and scale numerical metadata for a hybrid linear model.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model_name = MODEL_NAME
        self.batch_size = EMBEDDING_BATCH_SIZE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        set_seed(SEED)

    def fit_transform(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        load_cached_data: bool = True,
    ):
        """
        Generates feature matrices for train, validation, and test sets.

        Logic:
        1. Check if cached .npy files exist. If yes and load_cached_data=True, return them.
        2. If not, generate embeddings for text columns.
        3. Scale numerical columns.
        4. Concatenate text and numerical features.
        5. Save to cache and return.

        Args:
            df_train (pd.DataFrame): Training data.
            df_val (pd.DataFrame): Validation data.
            df_test (pd.DataFrame): Test data.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            X_train, y_train, X_val, y_val, X_test
        """
        # Define cache file paths
        cache_files = {
            "X_train": os.path.join(WORKING_DIR, "X_train.npy"),
            "y_train": os.path.join(WORKING_DIR, "y_train.npy"),
            "X_val": os.path.join(WORKING_DIR, "X_val.npy"),
            "y_val": os.path.join(WORKING_DIR, "y_val.npy"),
            "X_test": os.path.join(WORKING_DIR, "X_test.npy"),
        }

        # 1. Attempt to load from cache
        if load_cached_data:
            all_exist = all(os.path.exists(p) for p in cache_files.values())
            if all_exist:
                print("Loading features from cache...")
                X_train = np.load(cache_files["X_train"])
                y_train = np.load(cache_files["y_train"])
                X_val = np.load(cache_files["X_val"])
                y_val = np.load(cache_files["y_val"])
                X_test = np.load(cache_files["X_test"])
                return X_train, y_train, X_val, y_val, X_test

        print("Cache miss or force reload. Generating features from scratch...")
        os.makedirs(WORKING_DIR, exist_ok=True)

        # 2. Text Processing (Semantic Embeddings)
        print(
            f"Initializing SentenceTransformer: {self.model_name} on {self.device}..."
        )
        model = SentenceTransformer(self.model_name, device=self.device)

        def prepare_text_and_encode(df):
            # Concatenate title and text body
            # Ensure we handle potential NaNs by filling with empty string
            t1 = df[TEXT_COLS[0]].fillna("").astype(str)
            t2 = df[TEXT_COLS[1]].fillna("").astype(str)
            combined_text = (t1 + " " + t2).tolist()

            # Encode
            embeddings = model.encode(
                combined_text,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings

        print("Encoding training text...")
        X_train_text = prepare_text_and_encode(df_train)

        print("Encoding validation text...")
        X_val_text = prepare_text_and_encode(df_val)

        print("Encoding test text...")
        X_test_text = prepare_text_and_encode(df_test)

        # 3. Tabular Processing (Scaling)
        print("Processing numerical features...")
        # Extract raw numerical data
        X_train_num = df_train[NUMERICAL_FEATURES].values.astype(np.float32)
        X_val_num = df_val[NUMERICAL_FEATURES].values.astype(np.float32)
        X_test_num = df_test[NUMERICAL_FEATURES].values.astype(np.float32)

        # Fit Scaler on Train, Transform all
        X_train_num = self.scaler.fit_transform(X_train_num)
        X_val_num = self.scaler.transform(X_val_num)
        X_test_num = self.scaler.transform(X_test_num)

        # 4. Concatenation
        print("Concatenating features...")
        X_train = np.hstack([X_train_text, X_train_num])
        X_val = np.hstack([X_val_text, X_val_num])
        X_test = np.hstack([X_test_text, X_test_num])

        # 5. Extract Labels
        y_train = df_train["requester_received_pizza"].values.astype(int)
        y_val = df_val["requester_received_pizza"].values.astype(int)

        # 6. Save to Cache
        print("Saving features to cache...")
        np.save(cache_files["X_train"], X_train)
        np.save(cache_files["y_train"], y_train)
        np.save(cache_files["X_val"], X_val)
        np.save(cache_files["y_val"], y_val)
        np.save(cache_files["X_test"], X_test)

        return X_train, y_train, X_val, y_val, X_test
