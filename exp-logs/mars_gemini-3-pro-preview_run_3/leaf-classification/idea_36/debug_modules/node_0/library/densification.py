import os
import numpy as np
import logging
from library.config import Config
from library.utils import seed_everything


class ManifoldDensifier:
    """
    Implements Convex-Interpolated Manifold Densification.
    Transforms 12-view feature representations into a dense convex hull of 6 centroids
    (3 Primary Orthogonal + 3 Secondary Interpolated) per image.
    """

    def __init__(self, config: Config):
        self.config = config
        self.working_dir = self.config.WORKING_DIR
        seed_everything(self.config.SEED)

    def compute_centroids(self, features):
        """
        Computes 3 Primary and 3 Secondary centroids from 12-view features.

        Args:
            features (np.ndarray): Array of shape (N_samples, 12, D_dim).

        Returns:
            np.ndarray: Array of shape (N_samples, 6, D_dim).
        """
        N, V, D = features.shape
        if V != self.config.N_ROTATIONS:
            raise ValueError(f"Expected {self.config.N_ROTATIONS} views, got {V}")

        # 1. Compute Primary Centroids (Orthogonal Averages)
        # indices shape: (3, 4) -> 3 centroids, each averaging 4 views
        indices = self.config.PRIMARY_CENTROIDS

        # c1, c2, c3 shape: (N, D) each
        c1 = np.mean(features[:, indices[0], :], axis=1)
        c2 = np.mean(features[:, indices[1], :], axis=1)
        c3 = np.mean(features[:, indices[2], :], axis=1)

        # 2. Compute Secondary Centroids (Convex Interpolation)
        alpha = self.config.INTERPOLATION_ALPHA

        # Linear interpolation (MixUp in feature space)
        c12 = alpha * c1 + (1 - alpha) * c2
        c23 = alpha * c2 + (1 - alpha) * c3
        c31 = alpha * c3 + (1 - alpha) * c1

        # 3. Stack all centroids
        # Result shape: (N, 6, D)
        # Order: C1, C2, C3, C12, C23, C31
        centroids = np.stack([c1, c2, c3, c12, c23, c31], axis=1)

        return centroids

    def densify_dataset(self, data_dict, dataset_name, load_cached_data=True):
        """
        Expands the dataset by generating 6 centroids per image.
        Flattens the structure so each centroid becomes an independent sample (6N samples).

        Args:
            data_dict (dict): Dictionary containing 'ids', 'dino', 'conv', 'tab', and optionally 'labels'.
                              'dino'/'conv' shape: (N, 12, D)
            dataset_name (str): Name for caching (e.g., 'train', 'test').
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: Densified dataset with keys 'ids', 'dino', 'conv', 'tab', 'labels' (if present).
                  All arrays will have length 6 * N.
        """
        cache_prefix = os.path.join(self.working_dir, f"densified_{dataset_name}")
        paths = {
            "ids": f"{cache_prefix}_ids.npy",
            "dino": f"{cache_prefix}_dino.npy",
            "conv": f"{cache_prefix}_conv.npy",
            "tab": f"{cache_prefix}_tab.npy",
            "labels": f"{cache_prefix}_labels.npy",
        }

        # Check cache
        if load_cached_data:
            if (
                os.path.exists(paths["ids"])
                and os.path.exists(paths["dino"])
                and os.path.exists(paths["conv"])
                and os.path.exists(paths["tab"])
            ):

                logging.info(
                    f"Loading densified features for '{dataset_name}' from {self.working_dir}..."
                )
                data = {
                    "ids": np.load(paths["ids"]),
                    "dino": np.load(paths["dino"]),
                    "conv": np.load(paths["conv"]),
                    "tab": np.load(paths["tab"]),
                }
                if os.path.exists(paths["labels"]):
                    data["labels"] = np.load(paths["labels"])
                return data

        logging.info(f"Densifying dataset '{dataset_name}' (Computing Convex Hull)...")

        # Extract inputs
        ids = data_dict["ids"]
        dino_feats = data_dict["dino"]
        conv_feats = data_dict["conv"]
        tab_feats = data_dict["tab"]

        has_labels = "labels" in data_dict
        labels = data_dict["labels"] if has_labels else None

        N = len(ids)

        # 1. Compute Centroids for Visual Streams
        # Shape: (N, 6, D)
        dino_centroids = self.compute_centroids(dino_feats)
        conv_centroids = self.compute_centroids(conv_feats)

        # 2. Flatten Visual Features
        # Shape: (6*N, D)
        dino_dense = dino_centroids.reshape(-1, dino_centroids.shape[-1])
        conv_dense = conv_centroids.reshape(-1, conv_centroids.shape[-1])

        # 3. Repeat Metadata and Tabular Features
        # We repeat each row 6 times to match the 6 centroids per image
        # np.repeat repeats elements: [1, 2] -> [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        # This aligns with reshape(-1, D) which unrolls the axis 1 (centroids)

        ids_dense = np.repeat(ids, 6)
        tab_dense = np.repeat(tab_feats, 6, axis=0)

        labels_dense = None
        if has_labels:
            labels_dense = np.repeat(labels, 6)

        # 4. Save to Cache
        logging.info(f"Saving densified features to {self.working_dir}...")
        np.save(paths["ids"], ids_dense)
        np.save(paths["dino"], dino_dense)
        np.save(paths["conv"], conv_dense)
        np.save(paths["tab"], tab_dense)

        densified_data = {
            "ids": ids_dense,
            "dino": dino_dense,
            "conv": conv_dense,
            "tab": tab_dense,
        }

        if has_labels:
            np.save(paths["labels"], labels_dense)
            densified_data["labels"] = labels_dense

        return densified_data
