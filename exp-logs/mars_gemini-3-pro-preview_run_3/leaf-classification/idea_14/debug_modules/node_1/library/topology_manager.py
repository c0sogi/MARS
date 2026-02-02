import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List

from library.configuration import Config
from library.utilities import setup_logger
from library.vision_extractor import extract_rotational_features


class TopologyManager:
    """
    Manages the data topology for the Manifold Densification strategy.
    Handles the creation of densified training sets (3 centroids per image)
    and canonical inference sets (1 centroid per image), including
    synchronization of tabular features and labels.
    """

    def __init__(self):
        self.logger = setup_logger()

        # Define tabular feature columns based on dataset description
        self.margin_cols = [f"margin_{i+1}" for i in range(64)]
        self.shape_cols = [f"shape_{i+1}" for i in range(64)]
        self.texture_cols = [f"texture_{i+1}" for i in range(64)]
        self.feature_cols = self.margin_cols + self.shape_cols + self.texture_cols

    def _load_metadata(self, subset: str) -> pd.DataFrame:
        """Helper to load the correct metadata CSV."""
        if subset == "train":
            path = Config.TRAIN_META_PATH
        elif subset == "val":
            path = Config.VAL_META_PATH
        elif subset == "test":
            path = Config.TEST_META_PATH
        else:
            raise ValueError(f"Unknown subset: {subset}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")

        return pd.read_csv(path)

    def _get_tabular_array(self, df: pd.DataFrame) -> np.ndarray:
        """Extracts the 192 tabular features as a float32 array."""
        return df[self.feature_cols].values.astype(np.float32)

    def _get_labels(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extracts labels if they exist."""
        if "species" in df.columns:
            return df["species"].values
        return None

    def _get_ids(self, df: pd.DataFrame) -> np.ndarray:
        """Extracts image IDs."""
        return df["id"].values

    def _compute_centroid(
        self, features: np.ndarray, view_indices: List[int]
    ) -> np.ndarray:
        """
        Computes the centroid (mean) of specific views.
        Args:
            features: (N, 12, D) tensor of raw features.
            view_indices: List of indices to average (e.g., [0, 3, 6, 9]).
        Returns:
            (N, D) tensor of averaged features.
        """
        # Select specific views: (N, 4, D)
        selected_views = features[:, view_indices, :]
        # Average across the view dimension (axis 1): (N, D)
        centroid = np.mean(selected_views, axis=1)
        return centroid

    def get_densified_train_data(
        self, load_cached_data: bool = True, limit: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates the Densified Training Set.

        Logic:
            1. Extract 12 views for each training image.
            2. Generate 3 centroids per image using orthogonal subsets A, B, and C.
            3. Stack these centroids to create a dataset 3x the size of the original.
            4. Replicate tabular features, labels, and IDs 3x to match.

        Returns:
            Tuple containing:
            - img_features: (3N, D)
            - tab_features: (3N, 192)
            - labels: (3N,)
            - ids: (3N,)
        """
        cache_prefix = "train_densified"
        if limit is not None:
            cache_prefix += f"_limit{limit}"

        # Define cache paths
        path_img = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_img.npy")
        path_tab = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_tab.npy")
        path_y = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_y.npy")
        path_ids = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_ids.npy")

        # 1. Try Loading Cache
        if load_cached_data:
            if (
                os.path.exists(path_img)
                and os.path.exists(path_tab)
                and os.path.exists(path_y)
                and os.path.exists(path_ids)
            ):
                self.logger.info(
                    f"Loading cached densified training data from {Config.WORKING_DIR}"
                )
                return (
                    np.load(path_img),
                    np.load(path_tab),
                    np.load(path_y),
                    np.load(path_ids),
                )
            else:
                self.logger.info(
                    "Cached densified training data not found or incomplete. Recomputing..."
                )

        # 2. Compute from Scratch
        df = self._load_metadata("train")
        if limit is not None:
            df = df.iloc[:limit]

        # Extract Raw Vision Features (N, 12, D)
        # Note: We pass load_cached_data to the extractor as well, so it can use its own raw cache
        raw_features = extract_rotational_features(
            df, "train", load_cached_data=load_cached_data, limit=limit
        )

        # Compute Centroids
        self.logger.info("Computing orthogonal centroids (A, B, C)...")
        centroid_a = self._compute_centroid(raw_features, Config.VIEW_INDICES_A)
        centroid_b = self._compute_centroid(raw_features, Config.VIEW_INDICES_B)
        centroid_c = self._compute_centroid(raw_features, Config.VIEW_INDICES_C)

        # Stack Centroids: (3N, D)
        # Order: All A's, then all B's, then all C's
        img_features = np.concatenate([centroid_a, centroid_b, centroid_c], axis=0)

        # Process Tabular, Labels, IDs
        raw_tab = self._get_tabular_array(df)
        raw_y = self._get_labels(df)
        raw_ids = self._get_ids(df)

        # Replicate 3x
        tab_features = np.concatenate([raw_tab, raw_tab, raw_tab], axis=0)
        labels = np.concatenate([raw_y, raw_y, raw_y], axis=0)
        ids = np.concatenate([raw_ids, raw_ids, raw_ids], axis=0)

        # 3. Save to Cache
        self.logger.info(f"Saving densified training data to {Config.WORKING_DIR}")
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        np.save(path_img, img_features)
        np.save(path_tab, tab_features)
        np.save(path_y, labels)
        np.save(path_ids, ids)

        return img_features, tab_features, labels, ids

    def get_canonical_inference_data(
        self, subset: str, load_cached_data: bool = True, limit: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Generates the Canonical Inference Set (Validation or Test).

        Logic:
            1. Extract 12 views for each image.
            2. Generate 1 canonical centroid per image using Subset A (0, 90, 180, 270).
            3. Maintain 1:1 mapping with original samples.

        Args:
            subset: "val" or "test"

        Returns:
            Tuple containing:
            - img_features: (N, D)
            - tab_features: (N, 192)
            - labels: (N,) or None (for test)
            - ids: (N,)
        """
        if subset not in ["val", "test"]:
            raise ValueError("Subset must be 'val' or 'test'")

        cache_prefix = f"{subset}_canonical"
        if limit is not None:
            cache_prefix += f"_limit{limit}"

        # Define cache paths
        path_img = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_img.npy")
        path_tab = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_tab.npy")
        path_y = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_y.npy")
        path_ids = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_ids.npy")

        # 1. Try Loading Cache
        # Note: labels file might not exist for test if we didn't save None,
        # but we handle that logic below.
        cache_exists = (
            os.path.exists(path_img)
            and os.path.exists(path_tab)
            and os.path.exists(path_ids)
        )
        if subset == "val":
            cache_exists = cache_exists and os.path.exists(path_y)

        if load_cached_data and cache_exists:
            self.logger.info(
                f"Loading cached canonical {subset} data from {Config.WORKING_DIR}"
            )
            img = np.load(path_img)
            tab = np.load(path_tab)
            ids = np.load(path_ids)
            y = np.load(path_y) if os.path.exists(path_y) else None
            return img, tab, y, ids
        elif load_cached_data:
            self.logger.info(
                f"Cached canonical {subset} data not found. Recomputing..."
            )

        # 2. Compute from Scratch
        df = self._load_metadata(subset)
        if limit is not None:
            df = df.iloc[:limit]

        # Extract Raw Vision Features (N, 12, D)
        raw_features = extract_rotational_features(
            df, subset, load_cached_data=load_cached_data, limit=limit
        )

        # Compute Canonical Centroid (Set A)
        self.logger.info("Computing canonical centroid (A)...")
        img_features = self._compute_centroid(raw_features, Config.VIEW_INDICES_A)

        # Process Tabular, Labels, IDs
        tab_features = self._get_tabular_array(df)
        labels = self._get_labels(df)
        ids = self._get_ids(df)

        # 3. Save to Cache
        self.logger.info(f"Saving canonical {subset} data to {Config.WORKING_DIR}")
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        np.save(path_img, img_features)
        np.save(path_tab, tab_features)
        np.save(path_ids, ids)

        if labels is not None:
            np.save(path_y, labels)
        elif os.path.exists(path_y):
            # Clean up old label file if it exists but shouldn't (e.g. switching from val to test with same name logic error)
            os.remove(path_y)

        return img_features, tab_features, labels, ids
