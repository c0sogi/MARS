import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from library.config import Config
from library.utils import save_cache, load_cache


class FeaturePreprocessor:
    """
    Handles dimensionality reduction (PCA) and interaction feature engineering.
    Implements Independent Component Compression and Metadata-Visual Interaction.
    """

    def __init__(self):
        self.pca_variance = Config.PCA_VARIANCE
        self.interaction_k = Config.INTERACTION_TOP_K
        self.meta_scaling = Config.METADATA_SCALING
        self.metadata_cols = Config.METADATA_COLS
        self.target_col = Config.TARGET_COL
        self.seed = Config.SEED

    def _make_interactions(
        self, top_k_features: np.ndarray, meta_features: np.ndarray
    ) -> np.ndarray:
        """
        Computes interaction terms between Top-K PCA features and Metadata features.
        Performs an element-wise product (outer product per sample) and flattens.

        Args:
            top_k_features: (N, K) array of top K principal components.
            meta_features: (N, M) array of binary metadata features.

        Returns:
            interactions: (N, K*M) array of interaction terms.
        """
        # Expand dimensions for broadcasting
        # (N, K, 1)
        tk = top_k_features[:, :, np.newaxis]
        # (N, 1, M)
        mt = meta_features[:, np.newaxis, :]

        # Element-wise multiplication with broadcasting -> (N, K, M)
        interactions = tk * mt

        # Flatten the last two dimensions -> (N, K*M)
        N = interactions.shape[0]
        return interactions.reshape(N, -1)

    def preprocess(self, raw_features_dict: dict, load_cached_data: bool = True):
        """
        Main pipeline: PCA -> Interaction -> Concatenation.

        Args:
            raw_features_dict: Dictionary {backbone_name: {'train': np, 'val': np, 'test': np}}
            load_cached_data: Whether to try loading from disk first.

        Returns:
            X_train, y_train, X_val, y_val, X_test, test_ids
        """
        # Define cache filenames
        cache_files = {
            "X_train": "final_X_train.npy",
            "y_train": "final_y_train.npy",
            "X_val": "final_X_val.npy",
            "y_val": "final_y_val.npy",
            "X_test": "final_X_test.npy",
            "test_ids": "final_test_ids.npy",
        }

        # 1. Attempt to load from cache
        if load_cached_data:
            loaded_data = {}
            all_found = True
            for key, filename in cache_files.items():
                data = load_cache(filename)
                if data is None:
                    all_found = False
                    break
                loaded_data[key] = data

            if all_found:
                print("Loaded preprocessed features from cache.")
                return (
                    loaded_data["X_train"],
                    loaded_data["y_train"],
                    loaded_data["X_val"],
                    loaded_data["y_val"],
                    loaded_data["X_test"],
                    loaded_data["test_ids"],
                )

        print("Starting feature preprocessing from scratch...")

        # 2. Load Metadata
        # We need metadata for interaction terms and targets
        if not os.path.exists(Config.TRAIN_META_PATH):
            raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_META_PATH}")

        df_train = pd.read_csv(Config.TRAIN_META_PATH)
        df_val = pd.read_csv(Config.VAL_META_PATH)
        df_test = pd.read_csv(Config.TEST_META_PATH)

        # Extract Targets
        y_train = df_train[self.target_col].values.astype(np.float32)
        y_val = df_val[self.target_col].values.astype(np.float32)

        # Extract IDs for test set
        test_ids = df_test["Id"].values

        # Extract Metadata Features (Binary)
        meta_train = df_train[self.metadata_cols].values.astype(np.float32)
        meta_val = df_val[self.metadata_cols].values.astype(np.float32)
        meta_test = df_test[self.metadata_cols].values.astype(np.float32)

        # Initialize lists to hold feature blocks
        train_blocks = []
        val_blocks = []
        test_blocks = []

        # 3. Process each backbone
        # Sort keys to ensure deterministic feature order
        backbone_names = sorted(raw_features_dict.keys())

        for name in backbone_names:
            print(f"Preprocessing features for backbone: {name}")

            # Get raw features
            # These are (N, D) arrays (Dual-Pooled)
            f_train = raw_features_dict[name]["train"]
            f_val = raw_features_dict[name]["val"]
            f_test = raw_features_dict[name]["test"]

            # --- A. PCA Compression ---
            # Fit PCA on training data only
            pca = PCA(n_components=self.pca_variance, random_state=self.seed)
            pca.fit(f_train)

            # Transform all sets
            pca_train = pca.transform(f_train)
            pca_val = pca.transform(f_val)
            pca_test = pca.transform(f_test)

            # Append compressed features
            train_blocks.append(pca_train)
            val_blocks.append(pca_val)
            test_blocks.append(pca_test)

            # --- B. Interaction Engineering ---
            # Identify Top-K components
            # Handle case where PCA selected fewer components than K
            n_comps = pca_train.shape[1]
            k = min(self.interaction_k, n_comps)

            # Slice the top K components
            top_k_train = pca_train[:, :k]
            top_k_val = pca_val[:, :k]
            top_k_test = pca_test[:, :k]

            # Compute interactions
            inter_train = self._make_interactions(top_k_train, meta_train)
            inter_val = self._make_interactions(top_k_val, meta_val)
            inter_test = self._make_interactions(top_k_test, meta_test)

            # Append interaction features
            train_blocks.append(inter_train)
            val_blocks.append(inter_val)
            test_blocks.append(inter_test)

        # 4. Append Scaled Metadata
        # Scale metadata by factor
        scaled_meta_train = meta_train * self.meta_scaling
        scaled_meta_val = meta_val * self.meta_scaling
        scaled_meta_test = meta_test * self.meta_scaling

        train_blocks.append(scaled_meta_train)
        val_blocks.append(scaled_meta_val)
        test_blocks.append(scaled_meta_test)

        # 5. Concatenate All Features
        X_train = np.hstack(train_blocks)
        X_val = np.hstack(val_blocks)
        X_test = np.hstack(test_blocks)

        print(f"Preprocessing complete.")
        print(f"Final Train Shape: {X_train.shape}")
        print(f"Final Val Shape:   {X_val.shape}")
        print(f"Final Test Shape:  {X_test.shape}")

        # 6. Save to Cache
        save_cache(X_train, cache_files["X_train"])
        save_cache(y_train, cache_files["y_train"])
        save_cache(X_val, cache_files["X_val"])
        save_cache(y_val, cache_files["y_val"])
        save_cache(X_test, cache_files["X_test"])
        save_cache(test_ids, cache_files["test_ids"])

        return X_train, y_train, X_val, y_val, X_test, test_ids
