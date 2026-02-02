import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import save_array, load_array
from library.feature_extraction import DualStreamExtractor


class DataManager:
    """
    Manages data structuring and manifold densification.
    Aggregates 12 raw views into 3 orthogonal centroids and prepares
    densified training data and structured test data.
    """

    def __init__(self):
        self.feature_extractor = DualStreamExtractor()

    def get_data(self, load_cached_data=True):
        """
        Main method to get processed data.
        Returns a dictionary containing densified train/val data and structured test data.

        Returns:
            dict: {
                'train_X': np.ndarray (3N, D),
                'train_y': np.ndarray (3N,),
                'train_ids': np.ndarray (3N,),
                'val_X': np.ndarray (3M, D),
                'val_y': np.ndarray (3M,),
                'val_ids': np.ndarray (3M,),
                'test_X': np.ndarray (K, 3, D),
                'test_ids': np.ndarray (K,)
            }
        """
        # Define cache paths
        cache_paths = {
            "train_X": os.path.join(Config.WORKING_DIR, "train_densified_X.npy"),
            "train_y": os.path.join(Config.WORKING_DIR, "train_densified_y.npy"),
            "train_ids": os.path.join(Config.WORKING_DIR, "train_densified_ids.npy"),
            "val_X": os.path.join(Config.WORKING_DIR, "val_densified_X.npy"),
            "val_y": os.path.join(Config.WORKING_DIR, "val_densified_y.npy"),
            "val_ids": os.path.join(Config.WORKING_DIR, "val_densified_ids.npy"),
            "test_X": os.path.join(Config.WORKING_DIR, "test_structured_X.npy"),
            "test_ids": os.path.join(Config.WORKING_DIR, "test_ids_final.npy"),
        }

        # Try loading from cache
        if load_cached_data:
            cached_data = {}
            all_exist = True
            for key, path in cache_paths.items():
                arr = load_array(path)
                if arr is None:
                    all_exist = False
                    break
                cached_data[key] = arr

            if all_exist:
                print("DataManager: Loaded densified data from cache.")
                return cached_data

        print("DataManager: Processing data from scratch...")

        # 1. Get Raw Features (Visual)
        # This handles its own caching internally
        raw_features = self.feature_extractor.extract_features(
            load_cached_data=load_cached_data
        )

        # 2. Load Metadata (Tabular + Labels)
        train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
        val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
        test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

        # 3. Process Splits
        train_data = self._process_split(
            raw_features["train_dino"],
            raw_features["train_conv"],
            raw_features["train_ids"],
            train_df,
            is_train=True,
        )

        val_data = self._process_split(
            raw_features["val_dino"],
            raw_features["val_conv"],
            raw_features["val_ids"],
            val_df,
            is_train=True,
        )

        test_data = self._process_split(
            raw_features["test_dino"],
            raw_features["test_conv"],
            raw_features["test_ids"],
            test_df,
            is_train=False,
        )

        # 4. Save to Cache
        data_map = {
            "train_X": train_data["X"],
            "train_y": train_data["y"],
            "train_ids": train_data["ids"],
            "val_X": val_data["X"],
            "val_y": val_data["y"],
            "val_ids": val_data["ids"],
            "test_X": test_data["X"],
            "test_ids": test_data["ids"],
        }

        for key, arr in data_map.items():
            if arr is not None:
                save_array(arr, cache_paths[key])

        print("DataManager: Data processing complete and cached.")
        return data_map

    def _process_split(self, dino_feats, conv_feats, ids, metadata_df, is_train=True):
        """
        Processes a single data split:
        1. Aligns metadata with feature IDs.
        2. Computes centroids.
        3. Concatenates tabular data.
        4. Densifies (if train/val) or structures (if test).
        """
        # Align metadata
        # Filter metadata to match IDs and preserve order
        metadata_df = metadata_df.set_index("id")
        aligned_df = metadata_df.loc[ids].reset_index()

        # Extract Tabular Features
        # Columns: margin_1..64, shape_1..64, texture_1..64
        margin_cols = [f"margin_{i+1}" for i in range(64)]
        shape_cols = [f"shape_{i+1}" for i in range(64)]
        texture_cols = [f"texture_{i+1}" for i in range(64)]
        tabular_cols = margin_cols + shape_cols + texture_cols

        tabular_feats = aligned_df[tabular_cols].values.astype(np.float32)  # (N, 192)

        # Compute Centroids
        # dino_feats: (N, 12, 1024) -> (N, 3, 1024)
        dino_centroids = self._compute_centroids(dino_feats)
        # conv_feats: (N, 12, 1536) -> (N, 3, 1536)
        conv_centroids = self._compute_centroids(conv_feats)

        # Expand Tabular Features to (N, 3, 192)
        # Repeat the vector 3 times for each sample
        tabular_expanded = np.repeat(tabular_feats[:, np.newaxis, :], 3, axis=1)

        # Concatenate all streams: DINO | ConvNeXt | Tabular
        # Shape: (N, 3, 1024 + 1536 + 192) = (N, 3, 2752)
        X_structured = np.concatenate(
            [dino_centroids, conv_centroids, tabular_expanded], axis=2
        )

        if is_train:
            # Densification: Flatten (N, 3, D) -> (3N, D)
            N, C, D = X_structured.shape
            X_densified = X_structured.reshape(N * C, D)

            # Replicate Labels
            labels = aligned_df["species"].values
            y_densified = np.repeat(labels, C)  # (3N,)

            # Replicate IDs
            ids_densified = np.repeat(ids, C)  # (3N,)

            return {"X": X_densified, "y": y_densified, "ids": ids_densified}
        else:
            # For Test, keep structure (N, 3, D) for aggregation
            return {"X": X_structured, "y": None, "ids": ids}

    def _compute_centroids(self, features):
        """
        Aggregates 12 views into 3 orthogonal centroids.
        Centroid A: Avg(0, 90, 180, 270) -> Indices 0, 3, 6, 9
        Centroid B: Avg(30, 120, 210, 300) -> Indices 1, 4, 7, 10
        Centroid C: Avg(60, 150, 240, 330) -> Indices 2, 5, 8, 11

        Args:
            features: (N, 12, D)
        Returns:
            centroids: (N, 3, D)
        """
        indices_A = [0, 3, 6, 9]
        indices_B = [1, 4, 7, 10]
        indices_C = [2, 5, 8, 11]

        # Compute means along the rotation axis (axis 1)
        # We select specific indices for each centroid

        # Shape: (N, 4, D) -> mean -> (N, D)
        centroid_A = np.mean(features[:, indices_A, :], axis=1)
        centroid_B = np.mean(features[:, indices_B, :], axis=1)
        centroid_C = np.mean(features[:, indices_C, :], axis=1)

        # Stack to (N, 3, D)
        return np.stack([centroid_A, centroid_B, centroid_C], axis=1)
