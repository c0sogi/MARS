import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer
from library.config import Config
from library.image_processing import LeafImageProcessor
from library.feature_extractors import DualStreamExtractor


class DataManager:
    """
    Orchestrates data preparation, feature caching, and the core Manifold Densification logic.
    """

    def __init__(self):
        """
        Initializes the data manager with image processors, feature extractors,
        and the tabular transformer.
        """
        self.img_processor = LeafImageProcessor()
        self.feature_extractor = DualStreamExtractor()
        # Initialize QuantileTransformer; fit will be called on training data later
        self.tabular_transformer = QuantileTransformer(
            output_distribution="normal", random_state=Config.SEED
        )

    def extract_all_views(
        self, df: pd.DataFrame, cache_name: str, load_cached_data: bool = True
    ) -> np.ndarray:
        """
        Iterates through the dataset to generate and cache a (N x 12 x D) feature tensor.
        Uses the DualStreamExtractor to compute features for 12 rotated views per image.

        Args:
            df (pd.DataFrame): Metadata dataframe containing 'file_path'.
            cache_name (str): Identifier for the cache file (e.g., 'train', 'test').
            load_cached_data (bool): Whether to attempt loading from disk.

        Returns:
            np.ndarray: Extracted features with shape (N, 12, Feature_Dim).
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}_features.npy")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}...")
            try:
                data = np.load(cache_path)
                print(f"Successfully loaded features: {data.shape}")
                return data
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Extracting features for {cache_name} (this may take a while)...")
        all_features = []
        file_paths = df["file_path"].tolist()

        for rel_path in file_paths:
            try:
                # Load image
                img = self.img_processor.load_image(rel_path)

                # Generate 12 rotated views (List of tensors)
                views = self.img_processor.generate_rotated_views(img)

                # Stack views into a batch: (12, 3, H, W)
                batch_tensor = torch.stack(views)

                # Extract features: (12, D)
                features = self.feature_extractor.extract_batch(batch_tensor)
                all_features.append(features)

            except Exception as e:
                print(f"Error processing {rel_path}: {e}")
                raise e

        # Stack all samples: (N, 12, D)
        all_features_array = np.stack(all_features)

        # 3. Save to cache
        np.save(cache_path, all_features_array)
        print(f"Saved extracted features to {cache_path}")

        return all_features_array

    def process_tabular_features(
        self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
    ):
        """
        Loads margin/shape/texture features and applies QuantileTransformer.
        Fits the transformer ONLY on training data, then transforms all sets.

        Args:
            df_train (pd.DataFrame): Training metadata.
            df_val (pd.DataFrame): Validation metadata.
            df_test (pd.DataFrame): Test metadata.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Transformed tabular features for train, val, test.
        """
        # Identify feature columns (margin_*, shape_*, texture_*)
        # We assume columns are consistent across dataframes
        cols = df_train.columns
        feature_cols = [
            c
            for c in cols
            if c.startswith("margin")
            or c.startswith("shape")
            or c.startswith("texture")
        ]
        # Sort to ensure deterministic order
        feature_cols.sort()

        print(f"Processing {len(feature_cols)} tabular features...")

        # Extract raw values
        X_train = df_train[feature_cols].values.astype(np.float32)
        X_val = df_val[feature_cols].values.astype(np.float32)
        X_test = df_test[feature_cols].values.astype(np.float32)

        # Fit transformer on training data
        self.tabular_transformer.fit(X_train)

        # Transform all sets
        X_train_trans = self.tabular_transformer.transform(X_train)
        X_val_trans = self.tabular_transformer.transform(X_val)
        X_test_trans = self.tabular_transformer.transform(X_test)

        return X_train_trans, X_val_trans, X_test_trans

    def densify_training_data(
        self, img_features: np.ndarray, tab_features: np.ndarray, labels: np.ndarray
    ):
        """
        Implements Manifold Densification logic.
        Generates 3 distinct centroids per training sample by averaging mutually exclusive
        sets of 4 orthogonal views.

        Args:
            img_features (np.ndarray): Shape (N, 12, D).
            tab_features (np.ndarray): Shape (N, T).
            labels (np.ndarray): Shape (N,).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Densified Image Features (3N, D)
                - Densified Tabular Features (3N, T)
                - Densified Labels (3N,)
        """
        print("Applying Manifold Densification to training data...")

        densified_img_list = []
        densified_tab_list = []
        densified_labels_list = []

        # Iterate through the 3 view groups defined in Config
        # Group A: [0, 90, 180, 270], Group B: [30, 120...], Group C: [60, 150...]
        for group_angles in Config.VIEW_GROUPS:
            # Convert angles to indices (0-11) based on ROTATION_ANGLES
            indices = [Config.ROTATION_ANGLES.index(a) for a in group_angles]

            # Select the specific views for this group: (N, 4, D)
            group_views = img_features[:, indices, :]

            # Compute Centroid (Mean across the 4 views): (N, D)
            centroids = np.mean(group_views, axis=1)

            densified_img_list.append(centroids)
            densified_tab_list.append(tab_features)  # Duplicate tabular features
            densified_labels_list.append(labels)  # Duplicate labels

        # Concatenate all groups vertically to triple the dataset size
        X_img_dense = np.concatenate(densified_img_list, axis=0)
        X_tab_dense = np.concatenate(densified_tab_list, axis=0)
        y_dense = np.concatenate(densified_labels_list, axis=0)

        print(f"Densified data shape: {X_img_dense.shape}")
        return X_img_dense, X_tab_dense, y_dense

    def prepare_inference_data(
        self, img_features: np.ndarray, tab_features: np.ndarray
    ):
        """
        Prepares validation or test data using the standard orthogonal view set (Group A).
        This ensures the inference distribution matches the training distribution components.

        Args:
            img_features (np.ndarray): Shape (N, 12, D).
            tab_features (np.ndarray): Shape (N, T).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Centroid Image Features (N, D)
                - Tabular Features (N, T)
        """
        # Get indices for the standard inference angles (0, 90, 180, 270)
        indices = [Config.ROTATION_ANGLES.index(a) for a in Config.INFERENCE_ANGLES]

        # Select views: (N, 4, D)
        selected_views = img_features[:, indices, :]

        # Compute Mean Centroid: (N, D)
        X_img_centroid = np.mean(selected_views, axis=1)

        return X_img_centroid, tab_features

    def fuse_features(
        self, img_features: np.ndarray, tab_features: np.ndarray
    ) -> np.ndarray:
        """
        Concatenates the computed image centroids with the transformed tabular features.

        Args:
            img_features (np.ndarray): Shape (N, D).
            tab_features (np.ndarray): Shape (N, T).

        Returns:
            np.ndarray: Fused feature matrix (N, D+T).
        """
        return np.concatenate([img_features, tab_features], axis=1)
