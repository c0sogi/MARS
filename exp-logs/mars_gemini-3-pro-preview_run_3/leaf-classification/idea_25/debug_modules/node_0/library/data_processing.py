import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import setup_logger
from library.feature_extraction import FeatureExtractor


class DensificationManager:
    """
    Manages data transformation and structuring.
    Implements Manifold Densification to convert raw 12-view features into
    3 'Orthogonal Centroids' and prepares densified datasets.
    """

    def __init__(self):
        self.logger = setup_logger(
            os.path.join(Config.WORKING_DIR, "data_processing.log")
        )
        self.feature_extractor = FeatureExtractor()

        # Define tabular feature columns
        self.margin_cols = [f"margin_{i}" for i in range(1, 65)]
        self.shape_cols = [f"shape_{i}" for i in range(1, 65)]
        self.texture_cols = [f"texture_{i}" for i in range(1, 65)]
        self.tabular_cols = self.margin_cols + self.shape_cols + self.texture_cols

    def _compute_centroids(self, features):
        """
        Computes 3 orthogonal centroids from 12 views.
        Args:
            features: (N, 12, D) numpy array
        Returns:
            centroids: (N, 3, D) numpy array
        """
        # Config.CENTROID_INDICES is [[0,3,6,9], [1,4,7,10], [2,5,8,11]]
        centroids_list = []
        for indices in Config.CENTROID_INDICES:
            # Extract the 4 orthogonal views for this centroid
            # Shape: (N, 4, D)
            view_subset = features[:, indices, :]
            # Average them
            # Shape: (N, D)
            centroid = np.mean(view_subset, axis=1)
            centroids_list.append(centroid)

        # Stack along the 2nd dimension (views dimension)
        # Result shape: (N, 3, D)
        return np.stack(centroids_list, axis=1)

    def _densify_and_flatten(
        self, ids, dino_feats, conv_feats, tabular_df, labels=None
    ):
        """
        Expands the dataset by a factor of 3 (3 centroids per image).

        Args:
            ids: (N,) array of image IDs
            dino_feats: (N, 12, D1) array
            conv_feats: (N, 12, D2) array
            tabular_df: DataFrame containing tabular features aligned with ids
            labels: (N,) array of labels (optional)

        Returns:
            Tuple of densified arrays:
            (ids_flat, dino_flat, conv_flat, tab_flat, labels_flat)
            Shapes: (3N,), (3N, D1), (3N, D2), (3N, 192), (3N,)
        """
        N = len(ids)

        # 1. Compute Centroids: (N, 12, D) -> (N, 3, D)
        dino_centroids = self._compute_centroids(dino_feats)
        conv_centroids = self._compute_centroids(conv_feats)

        # 2. Flatten Visual Features: (N, 3, D) -> (3N, D)
        # We use reshape to flatten the first two dimensions
        # Order='C' ensures (Sample1_C1, Sample1_C2, Sample1_C3, Sample2_C1...)
        dino_flat = dino_centroids.reshape(N * Config.NUM_CENTROIDS, -1)
        conv_flat = conv_centroids.reshape(N * Config.NUM_CENTROIDS, -1)

        # 3. Expand Tabular Features: (N, F) -> (3N, F)
        # Extract numpy array
        tab_features = tabular_df[self.tabular_cols].values.astype(np.float32)
        # Repeat each row 3 times to match the visual centroids order
        # np.repeat([a, b], 3) -> [a, a, a, b, b, b]
        tab_flat = np.repeat(tab_features, Config.NUM_CENTROIDS, axis=0)

        # 4. Expand IDs
        ids_flat = np.repeat(ids, Config.NUM_CENTROIDS, axis=0)

        # 5. Expand Labels if present
        labels_flat = None
        if labels is not None:
            labels_flat = np.repeat(labels, Config.NUM_CENTROIDS, axis=0)

        return ids_flat, dino_flat, conv_flat, tab_flat, labels_flat

    def _load_metadata(self, subset_name):
        """Loads the appropriate metadata CSV."""
        if subset_name == "train":
            path = Config.TRAIN_METADATA_PATH
        elif subset_name == "val":
            path = Config.VAL_METADATA_PATH
        elif subset_name == "test":
            path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown subset: {subset_name}")

        return pd.read_csv(path)

    def _process_subset(self, subset_name, load_cached_data=True):
        """
        Generic pipeline to load raw features, densify, and cache results.
        """
        # Define Cache Paths
        cache_prefix = os.path.join(Config.WORKING_DIR, f"{subset_name}_densified")
        path_ids = f"{cache_prefix}_ids.npy"
        path_dino = f"{cache_prefix}_dino.npy"
        path_conv = f"{cache_prefix}_conv.npy"
        path_tab = f"{cache_prefix}_tab.npy"
        path_labels = f"{cache_prefix}_labels.npy"

        has_labels = subset_name in ["train", "val"]

        # Check Cache
        if load_cached_data:
            files_exist = (
                os.path.exists(path_ids)
                and os.path.exists(path_dino)
                and os.path.exists(path_conv)
                and os.path.exists(path_tab)
            )
            if has_labels:
                files_exist = files_exist and os.path.exists(path_labels)

            if files_exist:
                self.logger.info(f"Loading densified {subset_name} data from cache...")
                ids = np.load(path_ids)
                dino = np.load(path_dino)
                conv = np.load(path_conv)
                tab = np.load(path_tab)
                if has_labels:
                    labels = np.load(path_labels)
                    return ids, dino, conv, tab, labels
                return ids, dino, conv, tab, None

        # Process from scratch
        self.logger.info(f"Processing {subset_name} data (Densification)...")

        # 1. Get Metadata
        path_meta = (
            Config.TRAIN_METADATA_PATH
            if subset_name == "train"
            else (
                Config.VAL_METADATA_PATH
                if subset_name == "val"
                else Config.TEST_METADATA_PATH
            )
        )
        df_meta = self._load_metadata(subset_name)

        # 2. Get Raw Features (N, 12, D)
        # FeatureExtractor handles its own caching of raw features
        raw_ids, raw_dino, raw_conv = self.feature_extractor.extract_features(
            path_meta, subset_name, load_cached_data=load_cached_data
        )

        # 3. Alignment Check
        # Ensure the features correspond to the metadata rows
        # LeafDataset reads CSV sequentially, so they should match.
        # We perform a strict check.
        if not np.array_equal(raw_ids, df_meta["id"].values):
            self.logger.warning(
                f"ID mismatch detected in {subset_name}. Re-aligning..."
            )
            # Create a map from ID to index in raw features
            id_to_idx = {id_: i for i, id_ in enumerate(raw_ids)}
            # Reorder indices based on dataframe order
            indices = [id_to_idx[id_] for id_ in df_meta["id"].values]
            raw_ids = raw_ids[indices]
            raw_dino = raw_dino[indices]
            raw_conv = raw_conv[indices]

        # 4. Extract Labels
        labels = None
        if has_labels:
            labels = df_meta["species"].values

        # 5. Densify
        ids_out, dino_out, conv_out, tab_out, labels_out = self._densify_and_flatten(
            raw_ids, raw_dino, raw_conv, df_meta, labels
        )

        # 6. Save to Cache
        np.save(path_ids, ids_out)
        np.save(path_dino, dino_out)
        np.save(path_conv, conv_out)
        np.save(path_tab, tab_out)
        if has_labels:
            np.save(path_labels, labels_out)

        self.logger.info(
            f"Densified {subset_name} data cached. Shape: {dino_out.shape}"
        )

        return ids_out, dino_out, conv_out, tab_out, labels_out

    def prepare_training_data(self, load_cached_data=True):
        """
        Prepares the training dataset.
        Returns:
            ids, X_dino, X_conv, X_tab, y
        """
        return self._process_subset("train", load_cached_data)

    def prepare_validation_data(self, load_cached_data=True):
        """
        Prepares the validation dataset.
        Returns:
            ids, X_dino, X_conv, X_tab, y
        """
        return self._process_subset("val", load_cached_data)

    def prepare_inference_data(self, load_cached_data=True):
        """
        Prepares the test dataset.
        Returns:
            ids, X_dino, X_conv, X_tab
        """
        ids, dino, conv, tab, _ = self._process_subset("test", load_cached_data)
        return ids, dino, conv, tab
