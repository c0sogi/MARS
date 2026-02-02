import os
import numpy as np
from library.config import Config
from library.utils import get_logger


class ManifoldDensifier:
    """
    Implements the Orthogonal View-Set Averaging strategy (Manifold Densification).
    Transforms N samples with 12 views each into 3N samples (3 centroids per image).
    """

    def __init__(self):
        self.logger = get_logger("manifold_densifier")

    def compute_orthogonal_centroids(self, features):
        """
        Aggregates 12 views into 3 orthogonal centroids based on Config.CENTROID_INDICES.

        Args:
            features (np.ndarray): Input features of shape (N, 12, D).

        Returns:
            np.ndarray: Centroid features of shape (N, 3, D).
        """
        centroids = []
        # Config.CENTROID_INDICES is a list of lists, e.g., [[0,3,6,9], [1,4,7,10], [2,5,8,11]]
        for indices in Config.CENTROID_INDICES:
            # Select the specific views for this centroid: Shape (N, 4, D)
            subset = features[:, indices, :]
            # Compute the mean across the view dimension (axis 1): Shape (N, D)
            centroid = np.mean(subset, axis=1)
            centroids.append(centroid)

        # Stack the 3 centroids along axis 1: Shape (N, 3, D)
        return np.stack(centroids, axis=1)

    def prepare_densified_dataset(self, raw_data, split, load_cached_data=True):
        """
        Processes raw multi-view data into a densified dataset where each image
        yields 3 independent samples (one for each orthogonal centroid).

        Handles caching of the densified numpy arrays.

        Args:
            raw_data (dict): Dictionary containing 'dino_features', 'conv_features',
                             'tabular_features', 'ids', and optionally 'labels'.
            split (str): The dataset split ('train', 'val', or 'test').
            load_cached_data (bool): If True, attempts to load from cache first.

        Returns:
            dict: Dictionary containing flattened densified arrays:
                  'X_dino', 'X_conv', 'X_tab', 'ids', and optionally 'y'.
        """
        # Construct cache file paths
        suffix = "_debug" if Config.DEBUG else ""
        cache_base = os.path.join(Config.CACHE_DIR, f"densified_{split}{suffix}")

        paths = {
            "X_dino": f"{cache_base}_X_dino.npy",
            "X_conv": f"{cache_base}_X_conv.npy",
            "X_tab": f"{cache_base}_X_tab.npy",
            "ids": f"{cache_base}_ids.npy",
            "y": f"{cache_base}_y.npy",
        }

        has_labels = "labels" in raw_data

        # 1. Try to load from cache
        cache_valid = True
        keys_to_check = ["X_dino", "X_conv", "X_tab", "ids"]
        if has_labels:
            keys_to_check.append("y")

        if load_cached_data:
            for k in keys_to_check:
                if not os.path.exists(paths[k]):
                    cache_valid = False
                    break

            if cache_valid:
                self.logger.info(f"Loading densified data for '{split}' from cache...")
                result = {
                    "X_dino": np.load(paths["X_dino"]),
                    "X_conv": np.load(paths["X_conv"]),
                    "X_tab": np.load(paths["X_tab"]),
                    "ids": np.load(paths["ids"]),
                }
                if has_labels:
                    result["y"] = np.load(paths["y"])
                return result

        # 2. Compute if cache miss
        self.logger.info(f"Computing densified data for '{split}'...")

        # Extract raw components
        # Shapes: (N, 12, D_dino), (N, 12, D_conv), (N, 192), (N,)
        dino_in = raw_data["dino_features"]
        conv_in = raw_data["conv_features"]
        tab_in = raw_data["tabular_features"]
        ids_in = raw_data["ids"]

        # Compute Centroids -> Shapes: (N, 3, D)
        dino_centroids = self.compute_orthogonal_centroids(dino_in)
        conv_centroids = self.compute_orthogonal_centroids(conv_in)

        # Flatten/Expand to (3N, ...)
        N = len(ids_in)

        # Reshape visual features: (N, 3, D) -> (3N, D)
        # Order: [Img1_C1, Img1_C2, Img1_C3, Img2_C1, ...]
        X_dino = dino_centroids.reshape(N * 3, -1)
        X_conv = conv_centroids.reshape(N * 3, -1)

        # Replicate tabular features: (N, 192) -> (3N, 192)
        # np.repeat(..., 3, axis=0) produces [Row1, Row1, Row1, Row2, ...]
        # This aligns with the reshape order of visual features.
        X_tab = np.repeat(tab_in, 3, axis=0)

        # Replicate IDs
        expanded_ids = np.repeat(ids_in, 3)

        result = {
            "X_dino": X_dino,
            "X_conv": X_conv,
            "X_tab": X_tab,
            "ids": expanded_ids,
        }

        # Handle labels if present
        if has_labels:
            y_in = raw_data["labels"]
            y_expanded = np.repeat(y_in, 3)
            result["y"] = y_expanded

        # 3. Save to cache
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        np.save(paths["X_dino"], X_dino)
        np.save(paths["X_conv"], X_conv)
        np.save(paths["X_tab"], X_tab)
        np.save(paths["ids"], expanded_ids)

        if has_labels:
            np.save(paths["y"], result["y"])

        self.logger.info(f"Densified data for '{split}' saved to {Config.CACHE_DIR}")
        return result
