import os
import json
import hashlib
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import ensure_float64, save_to_cache, load_from_cache
import library.morphology as morphology


class FeatureManager:
    """
    Manages the high-level extraction, merging, and caching of features.
    Orchestrates the combination of pre-extracted tabular features and
    newly computed morphological features.
    """

    def __init__(self):
        pass

    def _get_config_hash(self):
        """
        Generates a hash based on the current feature configuration.
        This ensures that if the feature definitions in Config change,
        the cached merged dataset is invalidated.
        """
        config_info = {
            "tabular_prefixes": sorted(Config.TABULAR_FEATURE_PREFIXES),
            "morphological_features": sorted(Config.MORPHOLOGICAL_FEATURES),
            "float_type": str(Config.FLOAT_TYPE),
        }
        # Serialize to JSON and hash
        config_str = json.dumps(config_info, sort_keys=True)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()

    def get_feature_cache(self, split_name):
        """
        Generates the cache filename for the merged dataset based on the configuration hash.

        Args:
            split_name (str): 'train', 'val', or 'test'.

        Returns:
            str: The filename for the cached parquet file.
        """
        config_hash = self._get_config_hash()
        return f"merged_features_{split_name}_{config_hash}"

    def get_dataset(self, split_name, load_cached_data=True):
        """
        Retrieves the full dataset (X, y, ids) for a given split.
        Handles caching of the combined feature matrix.

        Args:
            split_name (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (X, y, ids)
                - X (pd.DataFrame): Feature matrix (float64).
                - y (pd.Series or None): Target labels.
                - ids (pd.Series): Image IDs.
        """
        # 1. Determine Cache Filename
        cache_filename = self.get_feature_cache(split_name)

        # 2. Try Loading from Cache
        if load_cached_data:
            cached_df = load_from_cache(cache_filename, expected_type="dataframe")
            if cached_df is not None:
                print(
                    f"FeatureManager: Loaded merged dataset for '{split_name}' from cache."
                )
                return self._parse_dataframe(cached_df)

        print(f"FeatureManager: Constructing dataset for '{split_name}'...")

        # 3. Load Metadata
        if split_name == "train":
            meta_path = Config.TRAIN_DATA_PATH
        elif split_name == "val":
            meta_path = Config.VAL_DATA_PATH
        elif split_name == "test":
            meta_path = Config.TEST_DATA_PATH
        else:
            raise ValueError(f"Unknown split name: {split_name}")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")

        metadata_df = pd.read_csv(meta_path)

        # 4. Extract Morphological Features
        # Delegates to library.morphology which handles the expensive image processing cache
        morph_df = morphology.extract_morphological_features(
            metadata_df, split_name, load_cached_data=load_cached_data
        )

        # 5. Extract Tabular Features from Metadata
        # Identify columns matching the configured prefixes
        tabular_cols = [
            col
            for col in metadata_df.columns
            if any(col.startswith(prefix) for prefix in Config.TABULAR_FEATURE_PREFIXES)
        ]
        tabular_df = metadata_df[tabular_cols]

        # 6. Merge Features
        # Reset indices to ensure correct alignment (though they should align by default)
        tabular_df = tabular_df.reset_index(drop=True)
        morph_df = morph_df.reset_index(drop=True)

        if len(tabular_df) != len(morph_df):
            raise ValueError(
                f"Row count mismatch: Tabular ({len(tabular_df)}) vs Morphological ({len(morph_df)})"
            )

        # Concatenate features
        X = pd.concat([tabular_df, morph_df], axis=1)

        # 7. Deterministic Column Ordering
        # Sort columns alphabetically to ensure consistent memory layout
        X = X.reindex(sorted(X.columns), axis=1)

        # 8. Prepare Full DataFrame for Caching
        # Attach ID and Target to cache a single file
        full_df = X.copy()
        full_df[Config.ID_COL] = metadata_df[Config.ID_COL].values

        if Config.TARGET_COL in metadata_df.columns:
            full_df[Config.TARGET_COL] = metadata_df[Config.TARGET_COL].values

        # 9. Save to Cache
        save_to_cache(cache_filename, full_df)
        print(f"FeatureManager: Saved merged dataset for '{split_name}' to cache.")

        # 10. Return Parsed Data
        return self._parse_dataframe(full_df)

    def _parse_dataframe(self, df):
        """
        Splits the cached dataframe into X, y, and ids.
        """
        # Extract IDs
        ids = df[Config.ID_COL]

        # Extract Target if present
        y = None
        if Config.TARGET_COL in df.columns:
            y = df[Config.TARGET_COL]

        # Extract Features (X)
        # Drop ID and Target columns
        cols_to_drop = [Config.ID_COL]
        if Config.TARGET_COL in df.columns:
            cols_to_drop.append(Config.TARGET_COL)

        X = df.drop(columns=cols_to_drop, errors="ignore")

        # Enforce Float64 Precision
        X = ensure_float64(X)

        return X, y, ids
