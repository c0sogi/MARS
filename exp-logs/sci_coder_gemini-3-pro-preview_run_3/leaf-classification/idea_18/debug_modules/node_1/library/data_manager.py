import os
import numpy as np
from library.config import Config
from library.utils import setup_logging
from library.feature_extractor import process_dataset

# Initialize logger
logger = setup_logging()


class DataManager:
    """
    Manages data loading and topological transformations for the Hyper-Densified strategy.
    Implements Orthogonal View-Set Averaging for training densification and
    Canonical Centroid generation for inference.
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def create_densified_training_set(self, load_cached_data: bool = True):
        """
        Constructs the Hyper-Densified training set.
        Generates 9 distinct centroids per image by averaging orthogonal views.
        Replicates tabular data and labels to match.

        The resulting dataset is 9x larger than the original training set.

        Args:
            load_cached_data (bool): Whether to attempt loading from disk.

        Returns:
            dino_feats (np.ndarray): (9N, 1024)
            conv_feats (np.ndarray): (9N, 1536)
            tab_feats (np.ndarray): (9N, 192)
            ids (np.ndarray): (9N,)
            labels (np.ndarray): (9N,)
        """
        # Define cache paths
        path_dino = os.path.join(self.cache_dir, "train_densified_dino.npy")
        path_conv = os.path.join(self.cache_dir, "train_densified_conv.npy")
        path_tab = os.path.join(self.cache_dir, "train_densified_tab.npy")
        path_ids = os.path.join(self.cache_dir, "train_densified_ids.npy")
        path_labels = os.path.join(self.cache_dir, "train_densified_labels.npy")

        # Check if cache exists
        if load_cached_data and all(
            os.path.exists(p)
            for p in [path_dino, path_conv, path_tab, path_ids, path_labels]
        ):
            logger.info("Loading densified training set from cache...")
            return (
                np.load(path_dino),
                np.load(path_conv),
                np.load(path_tab),
                np.load(path_ids),
                np.load(path_labels),
            )

        logger.info("Creating densified training set (Cache miss or force reload)...")

        # 1. Load Raw Data (N, 36, D)
        # We rely on feature_extractor to handle its own caching of the raw 36 views
        raw_dino, raw_conv, raw_tab, raw_ids, raw_labels = process_dataset(
            "train", load_cached_data=load_cached_data
        )

        # 2. Generate 9 Orthogonal Centroids
        # Config.NUM_TRAIN_CENTROIDS should be 9
        # Config.VIEWS_PER_CENTROID should be 4
        # Stride is 9 (36 // 4)

        num_centroids = Config.NUM_TRAIN_CENTROIDS
        stride = Config.NUM_VIEWS // Config.VIEWS_PER_CENTROID

        dino_centroids_list = []
        conv_centroids_list = []

        # We iterate through shifts 0 to 8 to create 9 distinct centroids
        for shift in range(num_centroids):
            # Indices: shift, shift+9, shift+18, shift+27
            # e.g., shift 0 -> 0, 9, 18, 27 (0, 90, 180, 270 degrees)
            indices = [shift + i * stride for i in range(Config.VIEWS_PER_CENTROID)]

            # Compute mean over these 4 views
            # raw shape: (N, 36, D) -> slice (N, 4, D) -> mean (N, D)
            dino_mean = np.mean(raw_dino[:, indices, :], axis=1)
            conv_mean = np.mean(raw_conv[:, indices, :], axis=1)

            dino_centroids_list.append(dino_mean)
            conv_centroids_list.append(conv_mean)

        # 3. Stack and Reshape to Interleave
        # Stack shape: (N, 9, D)
        dino_stacked = np.stack(dino_centroids_list, axis=1)
        conv_stacked = np.stack(conv_centroids_list, axis=1)

        # Reshape to (N*9, D)
        # This orders as: Img1_C1, Img1_C2... Img1_C9, Img2_C1...
        # This grouping keeps all variations of a single image adjacent
        N = dino_stacked.shape[0]
        dino_densified = dino_stacked.reshape(N * num_centroids, -1)
        conv_densified = conv_stacked.reshape(N * num_centroids, -1)

        # 4. Replicate Metadata
        # np.repeat repeats elements: [1, 2] -> [1, 1, 2, 2] (if repeats=2)
        # This matches the interleaved order of the visual features
        tab_densified = np.repeat(raw_tab, num_centroids, axis=0)
        ids_densified = np.repeat(raw_ids, num_centroids, axis=0)
        labels_densified = np.repeat(raw_labels, num_centroids, axis=0)

        # 5. Save to Cache
        np.save(path_dino, dino_densified)
        np.save(path_conv, conv_densified)
        np.save(path_tab, tab_densified)
        np.save(path_ids, ids_densified)
        np.save(path_labels, labels_densified)

        logger.info(f"Densified training set created. Shape: {dino_densified.shape}")

        return (
            dino_densified,
            conv_densified,
            tab_densified,
            ids_densified,
            labels_densified,
        )

    def create_canonical_inference_set(
        self, subset: str, load_cached_data: bool = True
    ):
        """
        Constructs the Canonical inference set (Validation or Test).
        Generates 1 canonical centroid per image (Avg of 0, 90, 180, 270 degrees).

        Args:
            subset (str): 'val' or 'test'
            load_cached_data (bool): Whether to attempt loading from disk.

        Returns:
            dino_feats (np.ndarray): (N, 1024)
            conv_feats (np.ndarray): (N, 1536)
            tab_feats (np.ndarray): (N, 192)
            ids (np.ndarray): (N,)
            labels (np.ndarray or None): (N,) or None (for test)
        """
        if subset not in ["val", "test"]:
            raise ValueError("Subset must be 'val' or 'test'")

        # Define cache paths
        path_dino = os.path.join(self.cache_dir, f"{subset}_canonical_dino.npy")
        path_conv = os.path.join(self.cache_dir, f"{subset}_canonical_conv.npy")
        path_tab = os.path.join(self.cache_dir, f"{subset}_canonical_tab.npy")
        path_ids = os.path.join(self.cache_dir, f"{subset}_canonical_ids.npy")
        path_labels = os.path.join(self.cache_dir, f"{subset}_canonical_labels.npy")

        # Check cache
        # Labels might not exist for test, so we handle that conditionally in check
        required_files = [path_dino, path_conv, path_tab, path_ids]
        if subset == "val":
            required_files.append(path_labels)

        if load_cached_data and all(os.path.exists(p) for p in required_files):
            logger.info(f"Loading canonical {subset} set from cache...")
            dino = np.load(path_dino)
            conv = np.load(path_conv)
            tab = np.load(path_tab)
            ids = np.load(path_ids)
            labels = np.load(path_labels) if subset == "val" else None
            return dino, conv, tab, ids, labels

        logger.info(f"Creating canonical {subset} set (Cache miss or force reload)...")

        # 1. Load Raw Data
        raw_dino, raw_conv, raw_tab, raw_ids, raw_labels = process_dataset(
            subset, load_cached_data=load_cached_data
        )

        # 2. Generate Canonical Centroid (Shift 0)
        # Indices: 0, 9, 18, 27 (corresponding to 0, 90, 180, 270 degrees)
        stride = Config.NUM_VIEWS // Config.VIEWS_PER_CENTROID
        indices = [0 + i * stride for i in range(Config.VIEWS_PER_CENTROID)]

        dino_canonical = np.mean(raw_dino[:, indices, :], axis=1)
        conv_canonical = np.mean(raw_conv[:, indices, :], axis=1)

        # 3. Metadata (No replication needed)
        tab_canonical = raw_tab
        ids_canonical = raw_ids
        labels_canonical = raw_labels

        # 4. Save to Cache
        np.save(path_dino, dino_canonical)
        np.save(path_conv, conv_canonical)
        np.save(path_tab, tab_canonical)
        np.save(path_ids, ids_canonical)
        if labels_canonical is not None:
            np.save(path_labels, labels_canonical)

        logger.info(f"Canonical {subset} set created. Shape: {dino_canonical.shape}")

        return (
            dino_canonical,
            conv_canonical,
            tab_canonical,
            ids_canonical,
            labels_canonical,
        )
