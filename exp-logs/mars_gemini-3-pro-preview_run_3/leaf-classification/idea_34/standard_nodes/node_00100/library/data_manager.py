import os
import numpy as np
import pandas as pd
from library.config import Config


class DataManager:
    """
    Manages data loading, processing, and caching for the Manifold Densification strategy.
    Aggregates 12-view raw features into 3 orthogonal centroids and aligns tabular data.
    """

    def __init__(self):
        """
        Initialize the DataManager with configuration paths and settings.
        """
        self.working_dir = Config.WORKING_DIR
        self.tabular_cols = Config.TABULAR_COLS
        self.centroid_indices = Config.CENTROID_INDICES

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

    def load_metadata(self, path):
        """
        Loads metadata from a CSV file.

        Args:
            path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded metadata.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")
        return pd.read_csv(path)

    def create_densified_dataset(
        self, split_name, raw_data, metadata_path, load_cached_data=True
    ):
        """
        Creates a densified dataset (3 centroids per image) from raw 12-view features.

        This method aggregates the 12 views into 3 orthogonal centroids (A, B, C),
        concatenates them, and replicates the tabular features and labels to match.

        Args:
            split_name (str): 'train', 'val', or 'test'. Used for cache naming.
            raw_data (tuple): (ids, dino_features, conv_features) from FeatureExtractor.
            metadata_path (str): Path to the metadata CSV.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: Contains 'ids', 'dino', 'conv', 'tabular', and optionally 'y' (species).
                  All arrays will have length 3 * N (where N is number of images in raw_data).
        """
        # Define cache file paths
        cache_prefix = os.path.join(self.working_dir, f"densified_{split_name}")
        path_ids = f"{cache_prefix}_ids.npy"
        path_dino = f"{cache_prefix}_dino.npy"
        path_conv = f"{cache_prefix}_conv.npy"
        path_tab = f"{cache_prefix}_tab.npy"
        path_y = f"{cache_prefix}_y.npy"

        # 1. Attempt to load from cache
        if load_cached_data:
            # Check if basic feature files exist
            files_exist = (
                os.path.exists(path_ids)
                and os.path.exists(path_dino)
                and os.path.exists(path_conv)
                and os.path.exists(path_tab)
            )

            if files_exist:
                # Check if labels file exists (it might not for test set)
                y_exists = os.path.exists(path_y)

                print(
                    f"[{split_name}] Loading densified data from cache: {self.working_dir}"
                )
                data = {
                    "ids": np.load(path_ids),
                    "dino": np.load(path_dino),
                    "conv": np.load(path_conv),
                    "tabular": np.load(path_tab),
                }
                if y_exists:
                    # Allow pickle for loading string arrays
                    data["y"] = np.load(path_y, allow_pickle=True)
                return data
            else:
                print(
                    f"[{split_name}] Densified cache miss. Processing from raw features..."
                )
        else:
            print(f"[{split_name}] Force re-computation of densified data...")

        # 2. Process Data
        ids_raw, dino_raw, conv_raw = raw_data

        # Load metadata
        df_meta = self.load_metadata(metadata_path)

        # Filter and Align Metadata with Raw IDs
        # The raw features might be a subset (e.g., due to Config.DEBUG_SAMPLE_LIMIT)
        # We index df_meta by 'id' and select 'ids_raw' to ensure exact order alignment
        df_meta = df_meta.set_index("id")

        # Check if all IDs in raw data exist in metadata
        missing_ids = [i for i in ids_raw if i not in df_meta.index]
        if missing_ids:
            raise ValueError(
                f"IDs in raw data not found in metadata: {missing_ids[:5]}..."
            )

        # Reindex to match raw data order and reset index to keep 'id' as a column
        df_aligned = df_meta.loc[ids_raw].reset_index()

        # Extract Tabular Data
        # Ensure all required tabular columns exist
        missing_cols = [c for c in self.tabular_cols if c not in df_aligned.columns]
        if missing_cols:
            raise ValueError(
                f"Missing tabular columns in metadata: {missing_cols[:5]}..."
            )

        tab_raw = df_aligned[self.tabular_cols].values.astype(np.float32)

        # Extract Labels if available (e.g., 'species' column)
        has_labels = "species" in df_aligned.columns
        y_raw = df_aligned["species"].values if has_labels else None

        # 3. Compute Centroids (Densification)
        # We generate 3 blocks: Centroid A, Centroid B, Centroid C
        # The order of iteration ["A", "B", "C"] corresponds to the tiling order later.

        dino_blocks = []
        conv_blocks = []

        for key in ["A", "B", "C"]:
            indices = self.centroid_indices[key]  # e.g., [0, 3, 6, 9]

            # Average over the view dimension (axis 1)
            # Input shape: (N, 12, Features) -> Output shape: (N, Features)
            dino_mean = dino_raw[:, indices, :].mean(axis=1)
            conv_mean = conv_raw[:, indices, :].mean(axis=1)

            dino_blocks.append(dino_mean)
            conv_blocks.append(conv_mean)

        # Concatenate blocks along the sample axis
        # Resulting shape: (3*N, Features)
        # Order: [Block_A, Block_B, Block_C]
        dino_densified = np.concatenate(dino_blocks, axis=0)
        conv_densified = np.concatenate(conv_blocks, axis=0)

        # Replicate Tabular, IDs, and Labels 3 times to match the visual blocks
        # np.tile([1, 2], 3) -> [1, 2, 1, 2, 1, 2]
        # This aligns with [Block_A, Block_B, Block_C] since tabular/labels are invariant to rotation.

        tab_densified = np.tile(tab_raw, (3, 1))
        ids_densified = np.tile(ids_raw, 3)

        y_densified = None
        if has_labels:
            y_densified = np.tile(y_raw, 3)

        # 4. Save to Cache
        np.save(path_ids, ids_densified)
        np.save(path_dino, dino_densified)
        np.save(path_conv, conv_densified)
        np.save(path_tab, tab_densified)

        if has_labels:
            # Save labels (strings/objects)
            np.save(path_y, y_densified)

        print(f"[{split_name}] Saved densified data to {self.working_dir}")
        print(
            f"[{split_name}] Output Shapes: DINO {dino_densified.shape}, Conv {conv_densified.shape}"
        )

        # 5. Return Data
        data = {
            "ids": ids_densified,
            "dino": dino_densified,
            "conv": conv_densified,
            "tabular": tab_densified,
        }
        if has_labels:
            data["y"] = y_densified

        return data
