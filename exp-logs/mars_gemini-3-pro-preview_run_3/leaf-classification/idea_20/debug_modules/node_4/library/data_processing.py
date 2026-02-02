import os
import numpy as np
from library.config import Config
from library.utils import save_npy, load_npy, seed_everything
from library.feature_extraction import FeatureExtractor


class ManifoldDensifier:
    """
    Implements Manifold Densification by creating multiple orthogonal centroids
    from multi-view image features. This structurally increases the dataset size
    and stabilizes feature representation for covariance estimation.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR

        # Define filenames for the processed (densified 3x) data
        # We append '_3x' to distinguish from the raw multi-view output of FeatureExtractor
        self.processed_files = {
            "train_img": "train_densified_3x_img.npy",
            "train_tab": "train_densified_3x_tab.npy",
            "train_ids": "train_densified_3x_ids.npy",
            "train_labels": "train_densified_3x_labels.npy",
            "test_img": "test_densified_3x_img.npy",
            "test_tab": "test_densified_3x_tab.npy",
            "test_ids": "test_densified_3x_ids.npy",
        }

    def _get_path(self, key):
        """Returns the full path for a processed file key."""
        return os.path.join(self.working_dir, self.processed_files[key])

    def _compute_centroids(self, img_features):
        """
        Computes 3 orthogonal centroids from 12-view features.

        Args:
            img_features: (N, 12, D) numpy array

        Returns:
            (3N, D) numpy array where the first N rows are Centroid A,
            the next N are Centroid B, and the last N are Centroid C.
        """
        # Define orthogonal view indices
        # Centroid A: 0°, 90°, 180°, 270° -> Indices [0, 3, 6, 9]
        # Centroid B: 30°, 120°, 210°, 300° -> Indices [1, 4, 7, 10]
        # Centroid C: 60°, 150°, 240°, 330° -> Indices [2, 5, 8, 11]
        idx_a = [0, 3, 6, 9]
        idx_b = [1, 4, 7, 10]
        idx_c = [2, 5, 8, 11]

        # Compute means along the view axis (axis 1)
        # Resulting shape for each: (N, D)
        centroid_a = np.mean(img_features[:, idx_a, :], axis=1)
        centroid_b = np.mean(img_features[:, idx_b, :], axis=1)
        centroid_c = np.mean(img_features[:, idx_c, :], axis=1)

        # Stack vertically to create the densified dataset
        # Shape: (3N, D)
        return np.concatenate([centroid_a, centroid_b, centroid_c], axis=0)

    def _replicate_metadata(self, array):
        """
        Replicates metadata (tabular features, IDs, labels) 3 times
        to align with the stacked centroids.

        Args:
            array: (N, ...) numpy array

        Returns:
            (3N, ...) numpy array
        """
        return np.concatenate([array, array, array], axis=0)

    def _process_split(self, img_key, tab_key, id_key, label_key=None):
        """
        Loads raw multi-view data, applies densification, and returns processed arrays.
        """
        # Load raw data using Config paths (output of FeatureExtractor)
        raw_img = load_npy(Config.get_cache_path(img_key))
        raw_tab = load_npy(Config.get_cache_path(tab_key))
        raw_ids = load_npy(Config.get_cache_path(id_key))

        # Apply Manifold Densification
        densified_img = self._compute_centroids(raw_img)
        densified_tab = self._replicate_metadata(raw_tab)
        densified_ids = self._replicate_metadata(raw_ids)

        result = {"img": densified_img, "tab": densified_tab, "ids": densified_ids}

        # Handle labels if present (Training set)
        if label_key:
            raw_labels = load_npy(Config.get_cache_path(label_key))
            densified_labels = self._replicate_metadata(raw_labels)
            result["labels"] = densified_labels

        return result

    def run(self, load_cached_data=True):
        """
        Main execution method.
        1. Ensures raw features exist (calls FeatureExtractor).
        2. Checks if densified (3x) features exist in cache.
        3. If not, processes raw features and saves to cache.
        4. Returns dictionaries containing the densified training and test data.

        Returns:
            train_data (dict): Keys 'img', 'tab', 'ids', 'labels'
            test_data (dict): Keys 'img', 'tab', 'ids'
        """
        seed_everything()

        # 1. Ensure raw features are available
        # FeatureExtractor handles its own caching logic.
        extractor = FeatureExtractor()
        extractor.run(load_cached_data=load_cached_data)

        # 2. Check local cache for densified data
        cache_complete = True
        if load_cached_data:
            for key in self.processed_files:
                if not os.path.exists(self._get_path(key)):
                    cache_complete = False
                    break
        else:
            cache_complete = False

        if cache_complete:
            print("Loading densified (3x) data from cache...")
            train_data = {
                "img": load_npy(self._get_path("train_img")),
                "tab": load_npy(self._get_path("train_tab")),
                "ids": load_npy(self._get_path("train_ids")),
                "labels": load_npy(self._get_path("train_labels")),
            }
            test_data = {
                "img": load_npy(self._get_path("test_img")),
                "tab": load_npy(self._get_path("test_tab")),
                "ids": load_npy(self._get_path("test_ids")),
            }
            return train_data, test_data

        print("Generating densified dataset (3x expansion)...")

        # 3. Process Training Data
        print("Processing training split...")
        train_res = self._process_split(
            "train_img_features", "train_tab_features", "train_ids", "train_labels"
        )

        # Save to cache
        save_npy(train_res["img"], self._get_path("train_img"))
        save_npy(train_res["tab"], self._get_path("train_tab"))
        save_npy(train_res["ids"], self._get_path("train_ids"))
        save_npy(train_res["labels"], self._get_path("train_labels"))

        # 4. Process Test Data
        print("Processing test split...")
        test_res = self._process_split(
            "test_img_features", "test_tab_features", "test_ids"
        )

        # Save to cache
        save_npy(test_res["img"], self._get_path("test_img"))
        save_npy(test_res["tab"], self._get_path("test_tab"))
        save_npy(test_res["ids"], self._get_path("test_ids"))

        print("Manifold densification complete.")
        return train_res, test_res
