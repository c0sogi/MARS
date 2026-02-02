import os
import numpy as np
import pandas as pd
from library.config import Config
from library.feature_extractor import FeatureExtractor


class ManifoldDensifier:
    """
    Implements the Convex-Hull Manifold Densification strategy.
    Transforms 12-view rotational features into a dense convex hull of centroids.
    """

    def __init__(self):
        # Mapping from rotation angle to centroid index (A=0, B=1, C=2)
        # A: 0, 90, 180, 270
        # B: 30, 120, 210, 300
        # C: 60, 150, 240, 330
        self.angle_to_centroid = {
            0: 0,
            90: 0,
            180: 0,
            270: 0,
            30: 1,
            120: 1,
            210: 1,
            300: 1,
            60: 2,
            150: 2,
            240: 2,
            330: 2,
        }

    def _compute_centroids(self, group_df):
        """
        Computes the 3 primary orthogonal centroids (A, B, C) for a single image.
        """
        # Extract features from the dataframe cells
        # The features are expected to be lists or arrays within the dataframe
        dino_matrix = np.stack(group_df["dino_features"].values)
        conv_matrix = np.stack(group_df["convnext_features"].values)
        angles = group_df["view_angle"].values

        # Initialize centroids (3, Feature_Dim)
        c_dino = np.zeros((3, dino_matrix.shape[1]), dtype=np.float32)
        c_conv = np.zeros((3, conv_matrix.shape[1]), dtype=np.float32)
        counts = np.zeros(3, dtype=np.float32)

        for i, angle in enumerate(angles):
            if angle in self.angle_to_centroid:
                idx = self.angle_to_centroid[angle]
                c_dino[idx] += dino_matrix[i]
                c_conv[idx] += conv_matrix[i]
                counts[idx] += 1.0

        # Compute averages
        # Avoid division by zero
        counts[counts == 0] = 1.0
        c_dino /= counts[:, None]
        c_conv /= counts[:, None]

        return c_dino, c_conv

    def _interpolate_centroids(self, c_dino, c_conv):
        """
        Generates 3 synthetic interpolated centroids (AB, BC, CA) to fill the convex hull.
        """
        # A=0, B=1, C=2
        # AB = 0.5*A + 0.5*B
        ab_dino = 0.5 * c_dino[0] + 0.5 * c_dino[1]
        bc_dino = 0.5 * c_dino[1] + 0.5 * c_dino[2]
        ca_dino = 0.5 * c_dino[2] + 0.5 * c_dino[0]

        ab_conv = 0.5 * c_conv[0] + 0.5 * c_conv[1]
        bc_conv = 0.5 * c_conv[1] + 0.5 * c_conv[2]
        ca_conv = 0.5 * c_conv[2] + 0.5 * c_conv[0]

        interp_dino = np.stack([ab_dino, bc_dino, ca_dino])
        interp_conv = np.stack([ab_conv, bc_conv, ca_conv])

        return interp_dino, interp_conv

    def densify_dataset(self, view_features_df, metadata_df, mode="train"):
        """
        Orchestrates the densification process for the entire dataset.

        Args:
            view_features_df (pd.DataFrame): DF with 12 rows per image (raw features).
            metadata_df (pd.DataFrame): DF with 1 row per image (tabular features & labels).
            mode (str): 'train' -> Generates 6 samples per image (A, B, C, AB, BC, CA).
                        'test'  -> Generates 3 samples per image (A, B, C).

        Returns:
            tuple: (ids, X_dino, X_conv, X_tab, y)
        """
        # Identify tabular columns
        tab_cols = [
            c
            for c in metadata_df.columns
            if c.startswith(("margin", "shape", "texture"))
        ]
        has_species = "species" in metadata_df.columns

        # Optimize metadata lookup
        meta_lookup = metadata_df.set_index("id")

        # Accumulators
        out_ids = []
        out_y = []
        out_dino = []
        out_conv = []
        out_tab = []

        # Group raw features by Image ID
        grouped = view_features_df.groupby("id")

        for img_id, group in grouped:
            if img_id not in meta_lookup.index:
                continue

            # Retrieve metadata (tabular + label)
            meta_row = meta_lookup.loc[img_id]

            # 1. Compute Primary Centroids
            p_dino, p_conv = self._compute_centroids(group)

            final_dino = [p_dino]
            final_conv = [p_conv]

            # 2. Compute Interpolated Centroids (Training only)
            if mode == "train":
                i_dino, i_conv = self._interpolate_centroids(p_dino, p_conv)
                final_dino.append(i_dino)
                final_conv.append(i_conv)

            # Concatenate all samples for this image
            img_dino = np.concatenate(final_dino, axis=0)
            img_conv = np.concatenate(final_conv, axis=0)
            n_samples = img_dino.shape[0]

            # 3. Replicate Tabular Features
            tab_vec = meta_row[tab_cols].values.astype(np.float32)
            img_tab = np.tile(tab_vec, (n_samples, 1))

            # 4. Replicate ID and Label
            img_ids = np.full(n_samples, img_id, dtype=np.int64)

            out_dino.append(img_dino)
            out_conv.append(img_conv)
            out_tab.append(img_tab)
            out_ids.append(img_ids)

            if has_species:
                label = meta_row["species"]
                out_y.extend([label] * n_samples)

        # Stack everything into final arrays
        if not out_ids:
            return np.array([]), np.array([]), np.array([]), np.array([]), None

        ids = np.concatenate(out_ids, axis=0)
        X_dino = np.concatenate(out_dino, axis=0)
        X_conv = np.concatenate(out_conv, axis=0)
        X_tab = np.concatenate(out_tab, axis=0)
        y = np.array(out_y) if out_y else None

        return ids, X_dino, X_conv, X_tab, y


def get_densified_data(split="train", load_cached_data=True):
    """
    Retrieves the densified dataset for a specific split.
    Manages caching for both raw feature extraction and densification.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (ids, X_dino, X_conv, X_tab, y)
    """
    # Configuration Mapping
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
        raw_cache_path = Config.TRAIN_FEATURES_CACHE
        # Train mode: 6 samples (densified)
        mode = "train"
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
        raw_cache_path = Config.VAL_FEATURES_CACHE
        # Val mode: 3 samples (evaluation)
        mode = "test"
    elif split == "test":
        meta_path = Config.TEST_METADATA_PATH
        raw_cache_path = Config.TEST_FEATURES_CACHE
        # Test mode: 3 samples (inference)
        mode = "test"
    else:
        raise ValueError(f"Invalid split: {split}")

    # Densified Cache Paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    f_dino = os.path.join(cache_dir, f"densified_{split}_dino.npy")
    f_conv = os.path.join(cache_dir, f"densified_{split}_conv.npy")
    f_tab = os.path.join(cache_dir, f"densified_{split}_tab.npy")
    f_ids = os.path.join(cache_dir, f"densified_{split}_ids.npy")
    f_y = os.path.join(cache_dir, f"densified_{split}_y.npy")

    # 1. Try to Load Densified Data from Cache
    if load_cached_data:
        # Check if all required feature files exist
        if (
            os.path.exists(f_dino)
            and os.path.exists(f_conv)
            and os.path.exists(f_tab)
            and os.path.exists(f_ids)
        ):

            # For train/val, check if labels exist. For test, labels might not exist.
            if split == "test" or os.path.exists(f_y):
                print(f"Loading densified {split} data from cache...")
                X_dino = np.load(f_dino)
                X_conv = np.load(f_conv)
                X_tab = np.load(f_tab)
                ids = np.load(f_ids)
                y = np.load(f_y) if os.path.exists(f_y) else None
                return ids, X_dino, X_conv, X_tab, y

    print(f"Generating densified {split} data...")

    # 2. Get Raw 12-View Features
    # Optimization: Check if raw cache exists to avoid instantiating FeatureExtractor (which loads models)
    if load_cached_data and os.path.exists(raw_cache_path):
        print(f"Loading raw features from {raw_cache_path}...")
        raw_df = pd.read_parquet(raw_cache_path)
    else:
        # Must compute from scratch
        print(
            "Raw features not found or reload requested. Initializing FeatureExtractor..."
        )
        extractor = FeatureExtractor()
        raw_df = extractor.extract_dataset_features(
            metadata_path=meta_path,
            cache_path=raw_cache_path,
            load_cached_data=load_cached_data,
        )

    # 3. Load Metadata
    meta_df = pd.read_csv(meta_path)

    # 4. Perform Densification
    densifier = ManifoldDensifier()
    ids, X_dino, X_conv, X_tab, y = densifier.densify_dataset(
        raw_df, meta_df, mode=mode
    )

    # 5. Save Densified Data to Cache
    print(f"Saving densified {split} data to {cache_dir}...")
    np.save(f_dino, X_dino)
    np.save(f_conv, X_conv)
    np.save(f_tab, X_tab)
    np.save(f_ids, ids)
    if y is not None:
        np.save(f_y, y)

    return ids, X_dino, X_conv, X_tab, y
