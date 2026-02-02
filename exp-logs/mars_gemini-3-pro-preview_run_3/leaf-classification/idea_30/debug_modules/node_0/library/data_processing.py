import os
import numpy as np
import pandas as pd
import torch
import random
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.feature_extraction import FeatureExtractor


class DatasetManager:
    """
    Manages data loading, manifold densification, and fold splitting for the
    Selective-Topology Orthogonal Manifold-Densified LDA strategy.
    """

    def __init__(self):
        self._set_seed(Config.SEED)
        self.feature_extractor = FeatureExtractor()

        # Feature Dimensions
        # DINOv2 ViT-Large: 1024
        # ConvNeXt Large: 1536
        # Tabular: 192 (64 margin + 64 shape + 64 texture)
        self.dim_dino = 1024
        self.dim_conv = 1536
        self.dim_tab = 192

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_data(self, load_cached_data=True):
        """
        Loads metadata and features for training (combined train+val) and test sets.

        Args:
            load_cached_data (bool): Whether to use cached features from FeatureExtractor.

        Returns:
            dict: A dictionary containing 'train' and 'test' data sub-dictionaries.
                  Each sub-dictionary contains 'ids', 'dino', 'conv', 'tabular', and 'labels' (train only).
        """
        # 1. Load Metadata
        # We combine original train and val for Cross-Validation
        df_train_orig = pd.read_csv(Config.TRAIN_CSV)
        df_val_orig = pd.read_csv(Config.VAL_CSV)
        df_train_combined = pd.concat(
            [df_train_orig, df_val_orig], axis=0, ignore_index=True
        )

        df_test = pd.read_csv(Config.TEST_CSV)

        # 2. Extract Deep Features (DINOv2 + ConvNeXt)
        # FeatureExtractor handles caching internally based on dataset name
        print("Extracting/Loading features for Combined Train+Val...")
        train_feats = self.feature_extractor.extract_features(
            df_train_combined,
            dataset_name="combined_train_val",
            load_cached_data=load_cached_data,
        )

        print("Extracting/Loading features for Test...")
        test_feats = self.feature_extractor.extract_features(
            df_test, dataset_name="test", load_cached_data=load_cached_data
        )

        # 3. Extract Tabular Features
        train_tab = self._extract_tabular(df_train_combined)
        test_tab = self._extract_tabular(df_test)

        # 4. Extract Labels (Train only)
        train_labels = df_train_combined["species"].values

        # 5. Structure Data
        data = {
            "train": {
                "ids": train_feats["ids"],
                "dino": train_feats["dino_features"],  # Shape: (N, 12, 1024)
                "conv": train_feats["conv_features"],  # Shape: (N, 12, 1536)
                "tabular": train_tab,  # Shape: (N, 192)
                "labels": train_labels,  # Shape: (N,)
            },
            "test": {
                "ids": test_feats["ids"],
                "dino": test_feats["dino_features"],
                "conv": test_feats["conv_features"],
                "tabular": test_tab,
            },
        }

        return data

    def _extract_tabular(self, df):
        """Helper to extract the 192 tabular features from dataframe."""
        # Columns are margin_1..64, shape_1..64, texture_1..64
        # We rely on column naming convention
        cols = [c for c in df.columns if c.startswith(("margin", "shape", "texture"))]
        if len(cols) != self.dim_tab:
            raise ValueError(
                f"Expected {self.dim_tab} tabular features, found {len(cols)}"
            )
        return df[cols].values.astype(np.float32)

    def generate_orthogonal_centroids(self, dino_features, conv_features):
        """
        Aggregates 12 views into 3 orthogonal centroids.

        Args:
            dino_features: (N, 12, D_dino)
            conv_features: (N, 12, D_conv)

        Returns:
            dino_centroids: (N, 3, D_dino)
            conv_centroids: (N, 3, D_conv)
        """
        # Centroid A: Views 0, 3, 6, 9 (0, 90, 180, 270 degrees)
        idx_a = [0, 3, 6, 9]
        # Centroid B: Views 1, 4, 7, 10 (30, 120, 210, 300 degrees)
        idx_b = [1, 4, 7, 10]
        # Centroid C: Views 2, 5, 8, 11 (60, 150, 240, 330 degrees)
        idx_c = [2, 5, 8, 11]

        # Compute means along the view dimension (axis 1)
        d_a = np.mean(dino_features[:, idx_a, :], axis=1)
        d_b = np.mean(dino_features[:, idx_b, :], axis=1)
        d_c = np.mean(dino_features[:, idx_c, :], axis=1)

        c_a = np.mean(conv_features[:, idx_a, :], axis=1)
        c_b = np.mean(conv_features[:, idx_b, :], axis=1)
        c_c = np.mean(conv_features[:, idx_c, :], axis=1)

        # Stack back to (N, 3, D)
        dino_centroids = np.stack([d_a, d_b, d_c], axis=1)
        conv_centroids = np.stack([c_a, c_b, c_c], axis=1)

        return dino_centroids, conv_centroids

    def prepare_training_set(self, data_dict, indices=None):
        """
        Prepares the densified dataset for training or validation.
        Converts (N) samples into (3N) samples by flattening centroids.

        Args:
            data_dict (dict): Dictionary with keys 'ids', 'dino', 'conv', 'tabular', 'labels'.
            indices (array-like, optional): Indices to subset the data (e.g., for a specific fold).

        Returns:
            X (np.ndarray): Concatenated features (3N, Total_Dim).
            y (np.ndarray): Labels (3N,).
            ids (np.ndarray): Original Image IDs (3N,).
        """
        # 1. Subset data if indices provided
        if indices is not None:
            ids = data_dict["ids"][indices]
            dino = data_dict["dino"][indices]
            conv = data_dict["conv"][indices]
            tab = data_dict["tabular"][indices]
            labels = data_dict["labels"][indices] if "labels" in data_dict else None
        else:
            ids = data_dict["ids"]
            dino = data_dict["dino"]
            conv = data_dict["conv"]
            tab = data_dict["tabular"]
            labels = data_dict["labels"] if "labels" in data_dict else None

        # 2. Generate Centroids (N, 3, D)
        dino_cent, conv_cent = self.generate_orthogonal_centroids(dino, conv)

        # 3. Flatten Centroids (Densification) -> (3N, D)
        # We reshape such that the 3 centroids for sample i are consecutive
        dino_flat = dino_cent.reshape(-1, self.dim_dino)
        conv_flat = conv_cent.reshape(-1, self.dim_conv)

        # 4. Replicate Tabular Features -> (3N, D_tab)
        # (N, D) -> (N, 3, D) -> (3N, D)
        tab_flat = np.repeat(tab[:, np.newaxis, :], 3, axis=1).reshape(-1, self.dim_tab)

        # 5. Concatenate All Features
        # Order: [DINO | CONV | TABULAR]
        X = np.hstack([dino_flat, conv_flat, tab_flat])

        # 6. Replicate IDs and Labels
        ids_flat = np.repeat(ids, 3)

        if labels is not None:
            y_flat = np.repeat(labels, 3)
            return X, y_flat, ids_flat

        return X, ids_flat

    def get_feature_indices(self):
        """
        Returns the column indices for each feature type in the concatenated X matrix.
        Useful for applying selective transformations (PCA vs QT).

        Returns:
            dict: { 'dino': (start, end), 'conv': (start, end), 'tabular': (start, end) }
        """
        start_dino = 0
        end_dino = start_dino + self.dim_dino

        start_conv = end_dino
        end_conv = start_conv + self.dim_conv

        start_tab = end_conv
        end_tab = start_tab + self.dim_tab

        return {
            "dino": (start_dino, end_dino),
            "conv": (start_conv, end_conv),
            "tabular": (start_tab, end_tab),
        }

    def get_stratified_kfold(self, n_folds=None):
        """
        Returns a StratifiedKFold object configured with the global seed.
        """
        k = n_folds if n_folds else Config.N_FOLDS
        return StratifiedKFold(n_splits=k, shuffle=True, random_state=Config.SEED)
