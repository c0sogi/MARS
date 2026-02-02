import os
import numpy as np
import pandas as pd
from library.config import Config
from library.feature_extraction import process_dataset


class LeafDataManager:
    """
    Manages data ingestion, feature fusion, and caching for the Leaf Classification task.
    Implements the Parsimonious Integral-Geometric Fusion strategy by combining
    tabular features with extracted geometric descriptors.
    """

    def __init__(self):
        pass

    def load_data(self, subset, load_cached_data=True, max_samples=None):
        """
        Loads metadata, extracts geometric features, merges with tabular features,
        and returns the raw feature matrix X, labels y, and ids.

        Args:
            subset (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from parquet cache.
            max_samples (int, optional): Maximum number of samples to load (for debugging).

        Returns:
            tuple: (X, y, ids, feature_names)
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Define cache path for the combined dataset
        cache_path = os.path.join(Config.WORKING_DIR, f"{subset}_combined.parquet")

        df = None

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                # print(f"Loading cached {subset} data from {cache_path}")
                df = pd.read_parquet(cache_path)
            except Exception:
                # If cache is corrupt, proceed to process from scratch
                df = None

        # 2. Process from Scratch if needed
        if df is None:
            # Determine metadata path
            if subset == "train":
                meta_path = Config.TRAIN_META
            elif subset == "val":
                meta_path = Config.VAL_META
            elif subset == "test":
                meta_path = Config.TEST_META
            else:
                raise ValueError(f"Unknown subset: {subset}")

            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")

            df_meta = pd.read_csv(meta_path)

            # Extract Geometric Features
            # We delegate to library.feature_extraction.process_dataset.
            # We disable the internal cache of process_dataset to avoid redundant files
            # since we are caching the fully combined result here.
            geo_df = process_dataset(
                metadata_df=df_meta,
                input_dir=Config.INPUT_DIR,
                cache_path=None,
                load_cached_data=False,
            )

            # Identify Tabular Features in Metadata
            # Columns starting with margin, shape, texture
            tab_cols = [
                c
                for c in df_meta.columns
                if any(
                    c.startswith(prefix) for prefix in Config.TABULAR_FEATURES_PREFIX
                )
            ]

            # Construct final DataFrame
            # We start with ID and Species (if available) for alignment
            cols_to_keep = ["id"]
            if "species" in df_meta.columns:
                cols_to_keep.append("species")

            df_base = df_meta[cols_to_keep].reset_index(drop=True)
            df_tab = df_meta[tab_cols].reset_index(drop=True)

            # Handle geometric dataframe (remove ID if duplicated from process_dataset)
            if "id" in geo_df.columns:
                geo_df = geo_df.drop(columns=["id"])
            geo_df = geo_df.reset_index(drop=True)

            # Additive Fusion: Base + Tabular + Geometric
            df_combined = pd.concat([df_base, df_tab, geo_df], axis=1)

            # Save to cache
            df_combined.to_parquet(cache_path, index=False)
            df = df_combined

        # Apply max_samples for debugging
        if max_samples is not None and max_samples < len(df):
            df = df.iloc[:max_samples]

        # 3. Prepare Outputs
        # Extract IDs
        ids = df["id"].values

        # Extract Labels if present
        y = None
        if "species" in df.columns:
            y = df["species"].values

        # Extract Features
        # Exclude non-feature columns
        exclude_cols = ["id", "species"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Enforce Deterministic Schema: Alphanumeric sort
        feature_cols.sort()

        # High-Precision Loading: Convert to float64 numpy array
        X = df[feature_cols].values.astype(np.float64)

        return X, y, ids, feature_cols
