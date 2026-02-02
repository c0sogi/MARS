import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import setup_logger, seed_everything

# Initialize logger
logger = setup_logger("data_processor")


class LeafDataset:
    """
    A container for the densified dataset arrays.
    """

    def __init__(self, dino_features, conv_features, tab_features, ids, labels=None):
        self.dino_features = dino_features
        self.conv_features = conv_features
        self.tab_features = tab_features
        self.ids = ids
        self.labels = labels

    def __len__(self):
        return len(self.ids)


class DataProcessor:
    """
    Handles the transformation of raw 12-view features into densified orthogonal centroids.
    """

    def __init__(self):
        self.centroid_map = Config.CENTROID_INDICES

    def compute_orthogonal_centroids(self, features_12view):
        """
        Aggregates 12 views into 3 orthogonal centroids based on Config indices.

        Args:
            features_12view (np.ndarray): Shape (N, 12, D)

        Returns:
            np.ndarray: Shape (N, 3, D)
        """
        centroids = []
        # Ensure consistent order: A, B, C
        for key in ["A", "B", "C"]:
            indices = self.centroid_map[key]
            # Select the specific views for this centroid
            views = features_12view[:, indices, :]  # (N, 4, D)
            # Compute mean across the selected views
            centroid = np.mean(views, axis=1)  # (N, D)
            centroids.append(centroid)

        # Stack centroids along the second dimension
        return np.stack(centroids, axis=1)  # (N, 3, D)

    def densify_data(
        self, dino_centroids, conv_centroids, tab_features, ids, labels=None
    ):
        """
        Flattens the centroid dimension to create a densified dataset where each
        centroid becomes an independent sample.

        Args:
            dino_centroids: (N, 3, D_dino)
            conv_centroids: (N, 3, D_conv)
            tab_features: (N, T)
            ids: (N,)
            labels: (N,) or None

        Returns:
            LeafDataset: Contains arrays of shape (N*3, ...)
        """
        N = len(ids)

        # Flatten Visual Features: (N, 3, D) -> (N*3, D)
        # Reshape with order='C' (default) ensures:
        # [Img0_A, Img0_B, Img0_C, Img1_A, ...]
        X_dino = dino_centroids.reshape(N * 3, -1)
        X_conv = conv_centroids.reshape(N * 3, -1)

        # Replicate Tabular Features: (N, T) -> (N*3, T)
        # np.repeat with axis=0 produces: [Row0, Row0, Row0, Row1, ...]
        X_tab = np.repeat(tab_features, 3, axis=0)

        # Replicate IDs
        X_ids = np.repeat(ids, 3, axis=0)

        # Replicate Labels if provided
        X_labels = None
        if labels is not None:
            X_labels = np.repeat(labels, 3, axis=0)

        return LeafDataset(X_dino, X_conv, X_tab, X_ids, X_labels)

    def get_stratified_folds(self, unique_ids, unique_labels, n_folds=Config.N_FOLDS):
        """
        Generates stratified folds based on unique images, then maps them to densified indices.

        Args:
            unique_ids (np.ndarray): Original unique image IDs (N,)
            unique_labels (np.ndarray): Original labels (N,)
            n_folds (int): Number of folds

        Yields:
            tuple: (fold_index, train_indices_densified, val_indices_densified)
        """
        seed_everything(Config.SEED)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

        for fold, (train_idx_unique, val_idx_unique) in enumerate(
            skf.split(unique_ids, unique_labels)
        ):
            # Helper to expand unique indices to densified indices
            # If index i is in the set, then 3*i, 3*i+1, 3*i+2 are in the densified set
            def expand_indices(indices):
                base = indices * 3
                # Create [3*i, 3*i+1, 3*i+2] for each i and flatten
                expanded = np.vstack([base, base + 1, base + 2]).T.flatten()
                return expanded

            train_idx_dense = expand_indices(train_idx_unique)
            val_idx_dense = expand_indices(val_idx_unique)

            yield fold, train_idx_dense, val_idx_dense

    def process_train_data(self, raw_data, load_cached_data=True):
        """
        Orchestrates the processing of training data: Centroid Computation -> Densification -> Caching.

        Args:
            raw_data (dict): Dictionary from feature_extraction.py containing 'train_dino', etc.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            LeafDataset: The processed training dataset.
        """
        cache_files = {
            "dino": os.path.join(Config.WORKING_DIR, "densified_train_dino.npy"),
            "conv": os.path.join(Config.WORKING_DIR, "densified_train_conv.npy"),
            "tab": os.path.join(Config.WORKING_DIR, "densified_train_tab.npy"),
            "ids": os.path.join(Config.WORKING_DIR, "densified_train_ids.npy"),
            "y": os.path.join(Config.WORKING_DIR, "densified_train_y.npy"),
        }

        # Check cache
        if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
            logger.info("Loading densified training data from cache...")
            return LeafDataset(
                np.load(cache_files["dino"]),
                np.load(cache_files["conv"]),
                np.load(cache_files["tab"]),
                np.load(cache_files["ids"]),
                np.load(cache_files["y"]),
            )

        logger.info("Processing training data from scratch...")
        if raw_data is None:
            raise ValueError("Raw data is required when cache is missing.")

        # 1. Compute Orthogonal Centroids
        dino_3 = self.compute_orthogonal_centroids(raw_data["train_dino"])
        conv_3 = self.compute_orthogonal_centroids(raw_data["train_convnext"])

        # 2. Densify Data
        dataset = self.densify_data(
            dino_3,
            conv_3,
            raw_data["train_tab"],
            raw_data["train_ids"],
            raw_data["train_labels"],
        )

        # 3. Save to Cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        np.save(cache_files["dino"], dataset.dino_features)
        np.save(cache_files["conv"], dataset.conv_features)
        np.save(cache_files["tab"], dataset.tab_features)
        np.save(cache_files["ids"], dataset.ids)
        np.save(cache_files["y"], dataset.labels)

        logger.info(
            f"Densified training data saved. Shape: {dataset.dino_features.shape}"
        )
        return dataset

    def process_test_data(self, raw_data, load_cached_data=True):
        """
        Orchestrates the processing of test data: Centroid Computation -> Densification -> Caching.
        Note: Test data is also densified to allow for Full-Manifold Test-Time Aggregation.

        Args:
            raw_data (dict): Dictionary from feature_extraction.py containing 'test_dino', etc.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            LeafDataset: The processed test dataset.
        """
        cache_files = {
            "dino": os.path.join(Config.WORKING_DIR, "densified_test_dino.npy"),
            "conv": os.path.join(Config.WORKING_DIR, "densified_test_conv.npy"),
            "tab": os.path.join(Config.WORKING_DIR, "densified_test_tab.npy"),
            "ids": os.path.join(Config.WORKING_DIR, "densified_test_ids.npy"),
        }

        # Check cache
        if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
            logger.info("Loading densified test data from cache...")
            return LeafDataset(
                np.load(cache_files["dino"]),
                np.load(cache_files["conv"]),
                np.load(cache_files["tab"]),
                np.load(cache_files["ids"]),
                None,
            )

        logger.info("Processing test data from scratch...")
        if raw_data is None:
            raise ValueError("Raw data is required when cache is missing.")

        # 1. Compute Orthogonal Centroids
        dino_3 = self.compute_orthogonal_centroids(raw_data["test_dino"])
        conv_3 = self.compute_orthogonal_centroids(raw_data["test_convnext"])

        # 2. Densify Data
        dataset = self.densify_data(
            dino_3,
            conv_3,
            raw_data["test_tab"],
            raw_data["test_ids"],
            None,
        )

        # 3. Save to Cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        np.save(cache_files["dino"], dataset.dino_features)
        np.save(cache_files["conv"], dataset.conv_features)
        np.save(cache_files["tab"], dataset.tab_features)
        np.save(cache_files["ids"], dataset.ids)

        logger.info(f"Densified test data saved. Shape: {dataset.dino_features.shape}")
        return dataset
