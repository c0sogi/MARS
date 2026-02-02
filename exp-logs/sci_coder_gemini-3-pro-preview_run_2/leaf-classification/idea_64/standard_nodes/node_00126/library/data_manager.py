import os
import pandas as pd
import numpy as np
from library.image_processing import process_image_batch
from library.utils import set_seed


class LeafDataManager:
    """
    Manages data loading, feature merging, and type enforcement for the Leaf Classification task.
    """

    def __init__(self, metadata_dir="./metadata", cache_dir="./working/idea_64"):
        """
        Initialize the data manager.

        Args:
            metadata_dir (str): Directory containing train.csv, val.csv, test.csv metadata.
            cache_dir (str): Directory to store cached processed dataframes.
        """
        self.metadata_dir = metadata_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define column groups structure
        # Cite debug_lesson_5: Use concatenated format (e.g., margin1) to match raw data schema
        self.margin_cols = [f"margin{i}" for i in range(1, 65)]
        self.shape_cols = [f"shape{i}" for i in range(1, 65)]
        self.texture_cols = [f"texture{i}" for i in range(1, 65)]
        self.physical_cols = [f"hu_{i}" for i in range(1, 8)] + [
            "aspect_ratio",
            "extent",
            "solidity",
            "eccentricity",
        ]

        # Enforce a specific column order for X
        self.feature_columns = (
            self.margin_cols + self.shape_cols + self.texture_cols + self.physical_cols
        )

    def _load_and_merge_split(self, split_name, load_cached_data):
        """
        Helper to load metadata, compute/load morphometrics, and merge them.
        """
        cache_path = os.path.join(self.cache_dir, f"{split_name}_merged.parquet")

        # 1. Try loading fully merged cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"[{split_name}] Loading merged data from cache: {cache_path}")
            try:
                df = pd.read_parquet(cache_path)
                # Cite debug_lesson_18: Validate cache dimensions to prevent loading corrupted/stale data
                if len(df.columns) < len(self.feature_columns):
                    raise ValueError(
                        f"Cache missing columns. Expected at least {len(self.feature_columns)}"
                    )
                return df
            except Exception as e:
                print(
                    f"[{split_name}] Failed to load merged cache or cache invalid ({e}). Recomputing..."
                )

        # 2. Load Metadata
        meta_path = os.path.join(self.metadata_dir, f"{split_name}.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        print(f"[{split_name}] Loading metadata from {meta_path}...")
        df_meta = pd.read_csv(meta_path)

        # 3. Get Physical Features (Morphometrics)
        # process_image_batch handles its own caching of the morphometrics part
        # We pass the relative image paths from the metadata
        image_paths = df_meta["image_path"].tolist()
        df_phys = process_image_batch(
            image_paths,
            cache_name=f"{split_name}_morphometrics",
            load_cached_data=load_cached_data,
        )

        # 4. Merge
        # Both dataframes should have 'id'. We merge on 'id'.
        # Note: process_image_batch returns a DF with 'id' and physical cols.
        print(f"[{split_name}] Merging provided features with physical features...")

        # Ensure IDs match types
        df_meta["id"] = df_meta["id"].astype(int)
        df_phys["id"] = df_phys["id"].astype(int)

        # Merge
        df_merged = pd.merge(df_meta, df_phys, on="id", how="left")

        # Fill NaNs in physical features (if any image failed processing) with 0
        df_merged[self.physical_cols] = df_merged[self.physical_cols].fillna(0.0)

        # 5. Save merged cache
        print(f"[{split_name}] Saving merged data to {cache_path}")
        df_merged.to_parquet(cache_path, index=False)

        return df_merged

    def load_data(self, load_cached_data=True):
        """
        Loads train, validation, and test datasets.

        Args:
            load_cached_data (bool): Whether to use cached files if available.

        Returns:
            dict: Dictionary containing:
                'X_train', 'y_train', 'train_ids'
                'X_val', 'y_val', 'val_ids'
                'X_test', 'test_ids'
                (X matrices are float64 numpy arrays)
        """
        set_seed(42)

        # Load splits
        df_train = self._load_and_merge_split("train", load_cached_data)
        df_val = self._load_and_merge_split("val", load_cached_data)
        df_test = self._load_and_merge_split("test", load_cached_data)

        # Prepare Output Dictionary
        data = {}

        # Helper to extract X, y
        def extract_features(df, is_test=False):
            # Ensure all feature columns exist (fill missing with 0 if any, though shouldn't happen)
            # Cite debug_lesson_19: Validate dynamic feature selection to prevent silent data loss
            missing = [c for c in self.feature_columns if c not in df.columns]
            if len(missing) > 0:
                # If we are missing a significant chunk, it's likely a schema error, not just a few missing values
                if len(missing) > 10:
                    raise ValueError(
                        f"Missing {len(missing)} features (e.g. {missing[:3]}). Check column naming."
                    )
                for col in missing:
                    df[col] = 0.0

            # Extract Features as float64
            X = df[self.feature_columns].astype(np.float64).values
            ids = df["id"].values

            if not is_test:
                y = df["species"].values
                return X, y, ids
            else:
                return X, ids

        # Train
        data["X_train"], data["y_train"], data["train_ids"] = extract_features(
            df_train, is_test=False
        )

        # Val
        data["X_val"], data["y_val"], data["val_ids"] = extract_features(
            df_val, is_test=False
        )

        # Test
        data["X_test"], data["test_ids"] = extract_features(df_test, is_test=True)

        print("Data loading complete.")
        print(f"Train shape: {data['X_train'].shape}")
        print(f"Val shape:   {data['X_val'].shape}")
        print(f"Test shape:  {data['X_test'].shape}")

        return data

    def get_feature_groups(self):
        """
        Returns the column names for each semantic feature group.

        Returns:
            dict: Keys are 'margin', 'shape', 'texture', 'physical'.
                  Values are lists of column names.
        """
        return {
            "margin": self.margin_cols,
            "shape": self.shape_cols,
            "texture": self.texture_cols,
            "physical": self.physical_cols,
        }

    def get_feature_indices(self):
        """
        Returns the column indices for each semantic feature group corresponding
        to the X matrices returned by load_data.

        Returns:
            dict: Keys are 'margin', 'shape', 'texture', 'physical'.
                  Values are lists of integer indices.
        """
        # Create a map of column name -> index
        col_to_idx = {col: i for i, col in enumerate(self.feature_columns)}

        return {
            "margin": [col_to_idx[c] for c in self.margin_cols],
            "shape": [col_to_idx[c] for c in self.shape_cols],
            "texture": [col_to_idx[c] for c in self.texture_cols],
            "physical": [col_to_idx[c] for c in self.physical_cols],
        }
