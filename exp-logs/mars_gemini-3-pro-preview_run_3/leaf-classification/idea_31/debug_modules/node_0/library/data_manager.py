import os
import numpy as np
import pandas as pd
import logging
from library import config, utils, feature_extractor

# Set up logging
logger = utils.setup_logger(
    "data_manager", os.path.join(config.WORKING_DIR, "data_manager.log")
)


class LeafDataManager:
    def __init__(self):
        self.cache_dir = config.WORKING_DIR
        self.tabular_cols = (
            [f"margin_{i}" for i in range(1, 65)]
            + [f"shape_{i}" for i in range(1, 65)]
            + [f"texture_{i}" for i in range(1, 65)]
        )

    def _generate_centroids(self, raw_features):
        """
        Averages 12-view features into 3 orthogonal centroids.

        Args:
            raw_features (np.ndarray): Shape (N, 12, D)

        Returns:
            np.ndarray: Shape (N, 3, D)
        """
        N, V, D = raw_features.shape
        if V != config.NUM_VIEWS:
            raise ValueError(f"Expected {config.NUM_VIEWS} views, got {V}")

        # Initialize output array
        centroids = np.zeros((N, 3, D), dtype=raw_features.dtype)

        # Compute averages for each group
        # Group A
        indices_a = config.CENTROID_GROUPS["A"]
        centroids[:, 0, :] = np.mean(raw_features[:, indices_a, :], axis=1)

        # Group B
        indices_b = config.CENTROID_GROUPS["B"]
        centroids[:, 1, :] = np.mean(raw_features[:, indices_b, :], axis=1)

        # Group C
        indices_c = config.CENTROID_GROUPS["C"]
        centroids[:, 2, :] = np.mean(raw_features[:, indices_c, :], axis=1)

        return centroids

    def _process_dataset(self, metadata_df, dataset_name, load_cached_features=True):
        """
        Internal method to process a specific dataset (train/val/test).
        Extracts features, densifies them, and aligns tabular data.
        """
        # 1. Extract Raw Features (N, 12, D)
        # This uses the feature_extractor's own caching mechanism for the raw extraction
        logger.info(f"Extracting/Loading raw features for {dataset_name}...")
        extracted_data = feature_extractor.extract_multi_view_features(
            metadata_df, dataset_name, load_cached_data=load_cached_features
        )

        raw_ids = extracted_data["ids"]
        raw_dino = extracted_data["dino_features"]
        raw_conv = extracted_data["conv_features"]

        # Verify alignment with metadata
        # The feature extractor returns IDs corresponding to the features.
        # We must ensure we align the tabular data correctly.
        # We'll re-index the metadata to match the order of raw_ids.

        # Create a mapping from ID to index in metadata
        metadata_df = metadata_df.set_index("id")
        # Reorder metadata to match the extracted features order
        aligned_metadata = metadata_df.loc[raw_ids].reset_index()

        # 2. Generate Centroids (N, 3, D)
        logger.info(f"Generating orthogonal centroids for {dataset_name}...")
        dino_centroids = self._generate_centroids(raw_dino)
        conv_centroids = self._generate_centroids(raw_conv)

        # 3. Densify / Flatten (3N, D)
        N = len(raw_ids)

        # Reshape: (N, 3, D) -> (N*3, D)
        # We use order='C' (default) which flattens the second dimension first.
        # So the order will be: [Img1_A, Img1_B, Img1_C, Img2_A, ...]
        dino_densified = dino_centroids.reshape(N * 3, -1)
        conv_densified = conv_centroids.reshape(N * 3, -1)

        # 4. Align Tabular Data and Labels
        # We need to repeat each row 3 times to match the centroids
        # repeat(3) on axis 0 does: [Row1, Row1, Row1, Row2, ...] which matches the reshape above

        # Tabular features
        tabular_data = aligned_metadata[self.tabular_cols].values
        tabular_densified = np.repeat(tabular_data, 3, axis=0)

        # IDs
        ids_densified = np.repeat(raw_ids, 3, axis=0)

        # Labels (if available)
        if "species" in aligned_metadata.columns:
            y_data = aligned_metadata["species"].values
            y_densified = np.repeat(y_data, 3, axis=0)
        else:
            y_densified = None

        logger.info(
            f"Densification complete. Original: {N}, Densified: {len(ids_densified)}"
        )

        return {
            "dino": dino_densified,
            "conv": conv_densified,
            "tabular": tabular_densified,
            "ids": ids_densified,
            "y": y_densified,
        }

    def get_dataset(self, dataset_name, load_cached_data=True):
        """
        Retrieves the processed, densified dataset.

        Args:
            dataset_name (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached processed files.

        Returns:
            dict: Contains 'dino', 'conv', 'tabular', 'ids', 'y'.
        """
        # Define cache paths for the PROCESSED (densified) data
        cache_prefix = os.path.join(self.cache_dir, f"densified_{dataset_name}")
        path_dino = f"{cache_prefix}_dino.npy"
        path_conv = f"{cache_prefix}_conv.npy"
        path_tab = f"{cache_prefix}_tab.npy"
        path_ids = f"{cache_prefix}_ids.npy"
        path_y = f"{cache_prefix}_y.npy"

        # Check if cache exists
        cache_exists = (
            os.path.exists(path_dino)
            and os.path.exists(path_conv)
            and os.path.exists(path_tab)
            and os.path.exists(path_ids)
        )
        # Check y only if it's not test
        if dataset_name != "test":
            cache_exists = cache_exists and os.path.exists(path_y)

        if load_cached_data and cache_exists:
            logger.info(f"Loading densified {dataset_name} dataset from cache...")
            data = {
                "dino": utils.load_numpy(path_dino),
                "conv": utils.load_numpy(path_conv),
                "tabular": utils.load_numpy(path_tab),
                "ids": utils.load_numpy(path_ids),
                "y": utils.load_numpy(path_y) if dataset_name != "test" else None,
            }
            return data

        # If not cached, compute from scratch
        logger.info(f"Processing {dataset_name} dataset from scratch...")

        # Load appropriate metadata
        if dataset_name == "train":
            meta_path = config.TRAIN_METADATA_PATH
        elif dataset_name == "val":
            meta_path = config.VAL_METADATA_PATH
        elif dataset_name == "test":
            meta_path = config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        metadata_df = pd.read_csv(meta_path)

        # Process
        data = self._process_dataset(
            metadata_df, dataset_name, load_cached_features=load_cached_data
        )

        # Save to cache
        logger.info(f"Saving densified {dataset_name} dataset to cache...")
        utils.save_numpy(data["dino"], path_dino)
        utils.save_numpy(data["conv"], path_conv)
        utils.save_numpy(data["tabular"], path_tab)
        utils.save_numpy(data["ids"], path_ids)
        if data["y"] is not None:
            utils.save_numpy(data["y"], path_y)

        return data
