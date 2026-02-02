import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import DataLoader
from library.feature_engine import EmbeddingGenerator

logger = setup_logger("dataset_builder")


class DatasetBuilder:
    """
    Orchestrates the data loading and feature engineering processes to construct
    the final feature matrices and target vectors for the machine learning pipeline.
    Merges metadata, high-resolution embeddings, and low-resolution embeddings.
    """

    def __init__(self):
        self.config = Config
        # Define cache paths
        self.cache_dir = self.config.WORKING_DIR
        self.paths = {
            "X_train": os.path.join(self.cache_dir, "X_train.parquet"),
            "y_train": os.path.join(self.cache_dir, "y_train.npy"),
            "X_val": os.path.join(self.cache_dir, "X_val.parquet"),
            "y_val": os.path.join(self.cache_dir, "y_val.npy"),
            "X_test": os.path.join(self.cache_dir, "X_test.parquet"),
            "test_ids": os.path.join(self.cache_dir, "test_ids.parquet"),
        }

    def _build_feature_matrix(self, df, emb_high, emb_low):
        """
        Constructs a unified feature matrix from metadata and embeddings.

        Args:
            df (pd.DataFrame): DataFrame containing metadata columns.
            emb_high (np.ndarray): High-resolution embeddings (MiniLM).
            emb_low (np.ndarray): Low-resolution embeddings (MPNet).

        Returns:
            pd.DataFrame: Combined feature matrix with named columns.
        """
        # 1. Extract Metadata
        # Ensure we copy to avoid SettingWithCopy warnings and reset index for concatenation
        meta_df = df[self.config.METADATA_COLS].copy().reset_index(drop=True)

        # 2. Process High-Res Embeddings (MiniLM)
        # Naming convention: minilm_0, minilm_1, ...
        high_cols = [f"minilm_{i}" for i in range(emb_high.shape[1])]
        high_df = pd.DataFrame(emb_high, columns=high_cols)

        # 3. Process Low-Res Embeddings (MPNet)
        # Naming convention: mpnet_0, mpnet_1, ...
        low_cols = [f"mpnet_{i}" for i in range(emb_low.shape[1])]
        low_df = pd.DataFrame(emb_low, columns=low_cols)

        # 4. Concatenate all views
        # axis=1 merges columns. Indexes are reset to 0..N so they align.
        X = pd.concat([meta_df, high_df, low_df], axis=1)

        return X

    def build_datasets(self, load_cached_data=True):
        """
        Main method to build training, validation, and testing datasets.

        Args:
            load_cached_data (bool): If True, attempts to load from cache.

        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
        """
        set_seed()
        os.makedirs(self.cache_dir, exist_ok=True)

        # 1. Attempt to load from cache
        if load_cached_data:
            all_exist = all(os.path.exists(p) for p in self.paths.values())
            if all_exist:
                logger.info("Loading constructed datasets from cache...")
                try:
                    X_train = pd.read_parquet(self.paths["X_train"])
                    y_train = np.load(self.paths["y_train"])
                    X_val = pd.read_parquet(self.paths["X_val"])
                    y_val = np.load(self.paths["y_val"])
                    X_test = pd.read_parquet(self.paths["X_test"])
                    # Load test_ids from parquet and convert to numpy array of strings
                    test_ids_df = pd.read_parquet(self.paths["test_ids"])
                    test_ids = test_ids_df["request_id"].values

                    logger.info(f"Loaded datasets. X_train shape: {X_train.shape}")
                    return X_train, y_train, X_val, y_val, X_test, test_ids
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}. Rebuilding...")

        # 2. Load Dependencies
        logger.info("Building datasets from scratch...")
        loader = DataLoader()
        train_df, val_df, test_df = loader.load_data(load_cached_data=load_cached_data)

        embedder = EmbeddingGenerator()
        # Returns: (train_high, val_high, test_high, train_low, val_low, test_low)
        embs = embedder.generate_embeddings(
            train_df, val_df, test_df, load_cached_data=load_cached_data
        )
        train_high, val_high, test_high, train_low, val_low, test_low = embs

        # 3. Build Feature Matrices
        logger.info("Constructing X_train...")
        X_train = self._build_feature_matrix(train_df, train_high, train_low)
        y_train = train_df["requester_received_pizza"].values.astype(int)

        logger.info("Constructing X_val...")
        X_val = self._build_feature_matrix(val_df, val_high, val_low)
        y_val = val_df["requester_received_pizza"].values.astype(int)

        logger.info("Constructing X_test...")
        X_test = self._build_feature_matrix(test_df, test_high, test_low)
        test_ids = test_df["request_id"].values

        # 4. Save to Cache
        logger.info("Saving datasets to cache...")
        try:
            X_train.to_parquet(self.paths["X_train"], index=False)
            np.save(self.paths["y_train"], y_train)

            X_val.to_parquet(self.paths["X_val"], index=False)
            np.save(self.paths["y_val"], y_val)

            X_test.to_parquet(self.paths["X_test"], index=False)

            # Save test_ids as DataFrame to avoid pickle issues with object arrays
            pd.DataFrame({"request_id": test_ids}).to_parquet(
                self.paths["test_ids"], index=False
            )

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

        logger.info(f"Dataset construction complete. X_train shape: {X_train.shape}")
        return X_train, y_train, X_val, y_val, X_test, test_ids
