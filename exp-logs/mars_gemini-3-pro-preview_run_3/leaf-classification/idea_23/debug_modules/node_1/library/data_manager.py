import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.feature_extraction import extract_and_cache_features


class DataManager:
    """
    Manages data loading, stratification, and the core 'Manifold Densification' logic.
    Prepares distinct data streams (DINO, ConvNeXt, Tabular) for the model.
    """

    def __init__(self):
        # DINOv2 Large output dimension is 1024
        self.dino_dim = 1024

    def load_raw_data(self, load_cached_data=True):
        """
        Loads the raw 12-view image features and tabular features.
        Uses the feature_extraction library to handle caching or computation.

        Args:
            load_cached_data (bool): If True, attempts to load from disk cache.

        Returns:
            tuple: (train_data, test_data)
                train_data: (img_feats, tab_feats, labels, ids)
                test_data: (img_feats, tab_feats, ids)
        """
        return extract_and_cache_features(load_cached_data=load_cached_data)

    def create_orthogonal_centroids(
        self, img_features, tab_features, labels=None, ids=None
    ):
        """
        Implements Manifold Densification.

        Takes raw 12-view features and aggregates them into 3 orthogonal centroids per image
        based on the configuration (e.g., Centroid A = Avg(0, 90, 180, 270)).

        Also splits the combined image features into 'DINO' (Global Geometry) and
        'ConvNeXt' (Local Texture) streams.

        Args:
            img_features (np.ndarray): Shape (N, 12, 2560). Combined DINO+ConvNeXt features.
            tab_features (np.ndarray): Shape (N, 192). Tabular features.
            labels (np.ndarray, optional): Shape (N,). Target labels.
            ids (np.ndarray, optional): Shape (N,). Image IDs.

        Returns:
            dict: A dictionary containing the densified data streams:
                - 'dino': (3*N, 1024)
                - 'convnext': (3*N, 1536)
                - 'tabular': (3*N, 192)
                - 'labels': (3*N,) or None
                - 'ids': (3*N,) or None
        """
        # 1. Split the combined image features into DINO and ConvNeXt streams
        # The first 1024 dimensions correspond to DINOv2 Large
        dino_raw = img_features[:, :, : self.dino_dim]
        # The remaining dimensions (1536) correspond to ConvNeXt Large
        conv_raw = img_features[:, :, self.dino_dim :]

        # 2. Compute Orthogonal Centroids
        # Config.CENTROID_INDICES defines the groups of views to average
        # e.g., [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]
        dino_centroids_list = []
        conv_centroids_list = []

        for indices in Config.CENTROID_INDICES:
            # Extract specific views for this centroid group: (N, 4, Dim)
            d_subset = dino_raw[:, indices, :]
            c_subset = conv_raw[:, indices, :]

            # Compute mean across the view dimension (axis 1) -> (N, Dim)
            dino_centroids_list.append(np.mean(d_subset, axis=1))
            conv_centroids_list.append(np.mean(c_subset, axis=1))

        # 3. Densify Data (Concatenate Vertically)
        # This triples the sample size (N -> 3N).
        # Order: [Centroid_A_Samples, Centroid_B_Samples, Centroid_C_Samples]
        dino_densified = np.concatenate(dino_centroids_list, axis=0)
        conv_densified = np.concatenate(conv_centroids_list, axis=0)

        # 4. Replicate Invariant Data
        # Tabular features, labels, and IDs are invariant to rotation/view.
        # We replicate them to match the 3x increase in sample size.
        num_groups = len(Config.CENTROID_INDICES)

        # Tile tabular features: (N, 192) -> (3*N, 192)
        # np.tile with shape (3, 1) stacks the array vertically 3 times, matching the centroid concatenation order.
        tab_densified = np.tile(tab_features, (num_groups, 1))

        labels_densified = None
        if labels is not None:
            labels_densified = np.tile(labels, num_groups)

        ids_densified = None
        if ids is not None:
            ids_densified = np.tile(ids, num_groups)

        return {
            "dino": dino_densified,
            "convnext": conv_densified,
            "tabular": tab_densified,
            "labels": labels_densified,
            "ids": ids_densified,
        }

    def get_stratified_kfold(self, n_splits=Config.N_FOLDS, seed=Config.SEED):
        """
        Returns a StratifiedKFold splitter.

        This should be applied to the ORIGINAL (undensified) labels/IDs to ensure
        that all centroids of a specific image remain in the same fold (preventing leakage).

        Args:
            n_splits (int): Number of folds.
            seed (int): Random seed for reproducibility.

        Returns:
            StratifiedKFold: The splitter object.
        """
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
