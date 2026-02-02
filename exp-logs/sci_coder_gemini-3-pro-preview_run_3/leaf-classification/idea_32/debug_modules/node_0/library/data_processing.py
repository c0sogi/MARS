import os
import numpy as np
from library.config import Config


class DataProcessor:
    """
    Handles the 'Manifold Densification' strategy and data structuring.

    Responsibilities:
    1. Load raw extracted features (Image: [N, 12, D], Tabular: [N, 192]).
    2. Compute 3 Orthogonal Centroids per image from the 12 views.
    3. Densify the dataset by flattening centroids and replicating tabular features/labels.
    4. Cache the processed datasets to disk.
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        self.working_dir = Config.WORKING_DIR
        self.centroid_indices = Config.CENTROID_INDICES

        # Mapping for raw cache filenames based on Config
        self.raw_cache_map = {
            "train": {
                "img": Config.CACHE_TRAIN_IMG_FEATURES,
                "tab": Config.CACHE_TRAIN_TAB_FEATURES,
                "ids": Config.CACHE_TRAIN_IDS,
                "lbl": Config.CACHE_TRAIN_LABELS,
            },
            "test": {
                "img": Config.CACHE_TEST_IMG_FEATURES,
                "tab": Config.CACHE_TEST_TAB_FEATURES,
                "ids": Config.CACHE_TEST_IDS,
                "lbl": None,
            },
            "val": {
                "img": "val_img_features.npy",
                "tab": "val_tab_features.npy",
                "ids": "val_ids.npy",
                "lbl": "val_labels.npy",
            },
        }

    def _compute_centroids(self, img_features):
        """
        Computes 3 orthogonal centroids from 12 views.

        Args:
            img_features (np.ndarray): Shape (N, 12, D)

        Returns:
            np.ndarray: Shape (N, 3, D)
        """
        centroids_list = []
        for indices in self.centroid_indices:
            # Select the specific 4 orthogonal views for this centroid
            # Shape: (N, 4, D)
            views = img_features[:, indices, :]
            # Compute mean: Shape (N, D)
            centroid = np.mean(views, axis=1)
            centroids_list.append(centroid)

        # Stack along axis 1 to get (N, 3, D)
        return np.stack(centroids_list, axis=1)

    def _densify(self, centroids, tab_features, ids, labels=None):
        """
        Flattens the centroids and replicates tabular data/ids/labels to match.

        Args:
            centroids (np.ndarray): Shape (N, 3, D)
            tab_features (np.ndarray): Shape (N, F)
            ids (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,)

        Returns:
            tuple: (densified_img, densified_tab, densified_ids, densified_labels)
                   densified_img shape: (N*3, D)
                   densified_tab shape: (N*3, F)
        """
        N, n_centroids, D = centroids.shape

        # Flatten centroids: (N*3, D)
        # Order: [Img1_C1, Img1_C2, Img1_C3, Img2_C1, ...]
        densified_img = centroids.reshape(N * n_centroids, D)

        # Replicate tabular features: (N*3, F)
        # We repeat each row 3 times to align with the flattened centroids
        densified_tab = np.repeat(tab_features, n_centroids, axis=0)

        # Replicate IDs: (N*3,)
        densified_ids = np.repeat(ids, n_centroids, axis=0)

        densified_labels = None
        if labels is not None:
            densified_labels = np.repeat(labels, n_centroids, axis=0)

        return densified_img, densified_tab, densified_ids, densified_labels

    def _process_dataset(self, split_name, metadata_path, load_cached=True):
        """
        Generic driver to process a specific dataset split (train/val/test).
        """
        # Define paths for the processed (densified) data
        # Stored in WORKING_DIR
        prefix = f"densified_{split_name}"
        d_img_path = os.path.join(self.working_dir, f"{prefix}_img.npy")
        d_tab_path = os.path.join(self.working_dir, f"{prefix}_tab.npy")
        d_ids_path = os.path.join(self.working_dir, f"{prefix}_ids.npy")
        d_lbl_path = os.path.join(self.working_dir, f"{prefix}_labels.npy")

        has_labels = split_name != "test"

        # 1. Try to load from processed cache
        if load_cached:
            files_exist = (
                os.path.exists(d_img_path)
                and os.path.exists(d_tab_path)
                and os.path.exists(d_ids_path)
            )
            if has_labels:
                files_exist = files_exist and os.path.exists(d_lbl_path)

            if files_exist:
                print(f"Loading densified {split_name} data from {self.working_dir}...")
                d_img = np.load(d_img_path)
                d_tab = np.load(d_tab_path)
                d_ids = np.load(d_ids_path)
                d_lbl = np.load(d_lbl_path) if has_labels else None
                return d_img, d_tab, d_ids, d_lbl

        # 2. Prepare Raw Feature Paths
        raw_map = self.raw_cache_map[split_name]
        r_img_path = os.path.join(self.cache_dir, raw_map["img"])
        r_tab_path = os.path.join(self.cache_dir, raw_map["tab"])
        r_ids_path = os.path.join(self.cache_dir, raw_map["ids"])
        r_lbl_path = (
            os.path.join(self.cache_dir, raw_map["lbl"]) if has_labels else None
        )

        # 3. Check if raw features exist; if not, run extraction
        raw_exists = (
            os.path.exists(r_img_path)
            and os.path.exists(r_tab_path)
            and os.path.exists(r_ids_path)
        )
        if has_labels:
            raw_exists = raw_exists and os.path.exists(r_lbl_path)

        if not raw_exists:
            print(
                f"Raw features for {split_name} not found. Running FeatureExtractor..."
            )
            from library.feature_extraction import FeatureExtractor

            extractor = FeatureExtractor()
            extractor.extract_dataset_features(
                metadata_path,
                r_img_path,
                r_tab_path,
                r_ids_path,
                r_lbl_path,
                load_cached_data=True,
            )

        # 4. Load Raw Features
        print(f"Loading raw features for {split_name}...")
        img_features = np.load(r_img_path)
        tab_features = np.load(r_tab_path)
        ids = np.load(r_ids_path)
        labels = np.load(r_lbl_path) if has_labels else None

        # 5. Compute Centroids and Densify
        print(f"Computing orthogonal centroids and densifying {split_name} data...")
        centroids = self._compute_centroids(img_features)
        d_img, d_tab, d_ids, d_lbl = self._densify(centroids, tab_features, ids, labels)

        # 6. Save Processed Data
        print(f"Saving densified {split_name} data to {self.working_dir}...")
        np.save(d_img_path, d_img)
        np.save(d_tab_path, d_tab)
        np.save(d_ids_path, d_ids)
        if has_labels:
            np.save(d_lbl_path, d_lbl)

        return d_img, d_tab, d_ids, d_lbl

    def get_train_data(self, load_cached=True):
        """
        Returns the densified training data (3x original size).
        """
        return self._process_dataset(
            "train", Config.TRAIN_METADATA_PATH, load_cached=load_cached
        )

    def get_val_data(self, load_cached=True):
        """
        Returns the densified validation data (3x original size).
        """
        return self._process_dataset(
            "val", Config.VAL_METADATA_PATH, load_cached=load_cached
        )

    def get_test_data(self, load_cached=True):
        """
        Returns the densified test data (3x original size).
        Note: Labels will be None.
        """
        return self._process_dataset(
            "test", Config.TEST_METADATA_PATH, load_cached=load_cached
        )
