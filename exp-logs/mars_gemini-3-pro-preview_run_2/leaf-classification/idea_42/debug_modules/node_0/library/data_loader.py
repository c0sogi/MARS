import os
import numpy as np
import pandas as pd
from library.config import Config
from library.morphology import get_morphometric_features


class DataManager:
    """
    Handles data ingestion, feature view construction, and caching for the IGCME strategy.
    """

    @staticmethod
    def load_split(split_name: str, load_cached_data: bool = True) -> dict:
        """
        Loads the data for a specific split (train, val, test).
        Constructs 'global' and 'combined' feature views.

        Args:
            split_name (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): If True, attempts to load processed arrays from disk.

        Returns:
            dict: A dictionary containing:
                - 'X_global': np.ndarray (N, 192)
                - 'X_combined': np.ndarray (N, 192 + 11)
                - 'y': np.ndarray (N,) or None for test
                - 'ids': np.ndarray (N,)
        """
        # Define cache file paths
        cache_dir = Config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        file_prefix = f"data_{split_name}"
        path_X_global = os.path.join(cache_dir, f"{file_prefix}_X_global.npy")
        path_X_combined = os.path.join(cache_dir, f"{file_prefix}_X_combined.npy")
        path_y = os.path.join(cache_dir, f"{file_prefix}_y.npy")
        path_ids = os.path.join(cache_dir, f"{file_prefix}_ids.npy")

        # 1. Try Loading Cache
        if load_cached_data:
            files_exist = (
                os.path.exists(path_X_global)
                and os.path.exists(path_X_combined)
                and os.path.exists(path_ids)
            )
            # For train/val, y must also exist
            if split_name != "test":
                files_exist = files_exist and os.path.exists(path_y)

            if files_exist:
                print(f"Loading cached data for split '{split_name}'...")
                try:
                    data = {
                        "X_global": np.load(path_X_global),
                        "X_combined": np.load(path_X_combined),
                        "ids": np.load(path_ids),
                        "y": np.load(path_y) if split_name != "test" else None,
                    }
                    return data
                except Exception as e:
                    print(
                        f"Failed to load cache for '{split_name}': {e}. Recomputing..."
                    )
            else:
                print(f"Cache missing for '{split_name}'. Computing from scratch...")
        else:
            print(f"Force recomputing data for '{split_name}'...")

        # 2. Compute from Scratch

        # Determine metadata path
        if split_name == "train":
            metadata_path = Config.TRAIN_METADATA_PATH
        elif split_name == "val":
            metadata_path = Config.VAL_METADATA_PATH
        elif split_name == "test":
            metadata_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid split_name: {split_name}")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df = pd.read_csv(metadata_path)

        # Debugging subsample
        if Config.DEBUG_SAMPLE_SIZE is not None:
            print(
                f"DEBUG: Subsampling {Config.DEBUG_SAMPLE_SIZE} rows for {split_name}."
            )
            df = df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        # Extract IDs
        ids = df["id"].values

        # Extract Target (if available)
        y = None
        if "species" in df.columns:
            y = df["species"].values

        # Extract Global Features (192 columns)
        # Columns are margin_1..64, shape_1..64, texture_1..64
        feature_cols = [
            c
            for c in df.columns
            if c.startswith("margin")
            or c.startswith("shape")
            or c.startswith("texture")
        ]
        # Sort to ensure consistent order (though list comprehension usually preserves order)
        # The prompt implies the columns are already there. We trust the CSV order or enforce sorting if needed.
        # Given the metadata generation, columns are stable.

        X_global = df[feature_cols].values.astype(Config.FLOAT_PRECISION)

        # Extract Morphometric Features (11 columns)
        # This function handles its own caching for the raw extraction
        X_morph = get_morphometric_features(
            metadata_path=metadata_path,
            dataset_key=split_name,
            load_cached_data=load_cached_data,
        )

        # Ensure precision match
        X_morph = X_morph.astype(Config.FLOAT_PRECISION)

        # Construct Combined View
        X_combined = np.hstack([X_global, X_morph])

        # 3. Save to Cache
        print(f"Saving processed data for '{split_name}' to {cache_dir}...")
        np.save(path_X_global, X_global)
        np.save(path_X_combined, X_combined)
        np.save(path_ids, ids)
        if y is not None:
            np.save(path_y, y)

        return {"X_global": X_global, "X_combined": X_combined, "ids": ids, "y": y}
