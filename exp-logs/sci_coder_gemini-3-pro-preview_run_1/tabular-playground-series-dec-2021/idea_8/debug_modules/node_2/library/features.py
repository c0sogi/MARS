import os
import re
import numpy as np
import pandas as pd
from library.config import Config


class FeatureEngineer:
    """
    Handles data loading, feature engineering, and caching for the pipeline.
    Implements Geometric Features and Dual-Representation strategies.
    """

    def __init__(self):
        self.train_path = Config.TRAIN_PATH
        self.val_path = Config.VAL_PATH
        self.test_path = Config.TEST_PATH
        self.cache_dir = Config.WORKING_DIR

        # Cache file paths
        self.train_cache = os.path.join(self.cache_dir, "train_processed.parquet")
        self.val_cache = os.path.join(self.cache_dir, "val_processed.parquet")
        self.test_cache = os.path.join(self.cache_dir, "test_processed.parquet")

    def process_data(self, load_cached_data=True):
        """
        Main method to get processed data.

        Args:
            load_cached_data (bool): If True, attempts to load from Parquet cache.
                                     If False or cache miss, re-processes raw data.

        Returns:
            tuple: (df_train, df_val, df_test)
        """
        # Ensure working directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # 1. Try Loading Cache
        if load_cached_data:
            if (
                os.path.exists(self.train_cache)
                and os.path.exists(self.val_cache)
                and os.path.exists(self.test_cache)
            ):

                print(f"Loading cached data from {self.cache_dir}...")
                df_train = pd.read_parquet(self.train_cache)
                df_val = pd.read_parquet(self.val_cache)
                df_test = pd.read_parquet(self.test_cache)
                return df_train, df_val, df_test
            else:
                print("Cache not found. Processing from scratch...")
        else:
            print("Ignoring cache. Processing from scratch...")

        # 2. Load Raw Data
        df_train, df_val, df_test = self._load_raw_data()

        # 3. Apply Feature Engineering
        print("Applying feature engineering...")
        df_train = self._preprocess(df_train, is_train=True)
        df_val = self._preprocess(df_val, is_train=True)
        df_test = self._preprocess(df_test, is_train=False)

        # 4. Save to Cache
        print(f"Saving processed data to {self.cache_dir}...")
        df_train.to_parquet(self.train_cache, index=False)
        df_val.to_parquet(self.val_cache, index=False)
        df_test.to_parquet(self.test_cache, index=False)

        return df_train, df_val, df_test

    def _load_raw_data(self):
        """Loads raw CSVs from metadata directory."""
        print("Loading raw data from metadata...")
        df_train = pd.read_csv(self.train_path)
        df_val = pd.read_csv(self.val_path)
        df_test = pd.read_csv(self.test_path)

        # Handle Debug Subsampling
        if Config.DEBUG_SAMPLES is not None:
            print(f"DEBUG: Subsampling train/val to {Config.DEBUG_SAMPLES} samples.")

            # Cite debug_lesson_6: Stratified Sampling Fails to Preserve Rare Classes in Small Debug Subsets
            # We implement forced inclusion to ensure all classes are present for XGBoost (contiguous 0..N-1).
            def forced_inclusion_sample(df, n_samples, target_col, seed):
                if target_col not in df.columns or len(df) <= n_samples:
                    return df

                # 1. Force inclusion: Pick 1 sample per class
                unique_classes = df[target_col].unique()
                indices_to_keep = []
                for cls in unique_classes:
                    cls_indices = df[df[target_col] == cls].index
                    if len(cls_indices) > 0:
                        np.random.seed(seed)
                        indices_to_keep.extend(
                            np.random.choice(cls_indices, 1, replace=False)
                        )

                # 2. Fill the rest randomly
                remaining = n_samples - len(indices_to_keep)
                if remaining > 0:
                    all_indices = df.index.values
                    candidates = np.setdiff1d(all_indices, indices_to_keep)
                    if len(candidates) > 0:
                        np.random.seed(seed)
                        indices_to_keep.extend(
                            np.random.choice(
                                candidates,
                                min(len(candidates), remaining),
                                replace=False,
                            )
                        )

                return (
                    df.loc[indices_to_keep]
                    .sample(frac=1, random_state=seed)
                    .reset_index(drop=True)
                )

            df_train = forced_inclusion_sample(
                df_train, Config.DEBUG_SAMPLES, Config.TARGET_COL, Config.SEED
            )
            df_val = forced_inclusion_sample(
                df_val, Config.DEBUG_SAMPLES, Config.TARGET_COL, Config.SEED
            )

            df_test = df_test.sample(
                n=min(len(df_test), Config.DEBUG_SAMPLES), random_state=Config.SEED
            ).reset_index(drop=True)

        return df_train, df_val, df_test

    def _preprocess(self, df, is_train=True):
        """Applies all feature engineering steps to a dataframe."""
        df = df.copy()

        # 1. Geometric Features
        df = self._add_geometric_features(df)

        # 2. Dual Representation (Dense Indices)
        df = self._add_dual_representation(df)

        # 3. Target Mapping (only for train/val)
        if is_train and Config.TARGET_COL in df.columns:
            df = self._map_target(df)

        return df

    def _add_geometric_features(self, df):
        """Adds physics-informed geometric features."""
        # Euclidean Distance to Hydrology
        # sqrt(H_Dist^2 + V_Dist^2)
        h_dist = df["Horizontal_Distance_To_Hydrology"]
        v_dist = df["Vertical_Distance_To_Hydrology"]
        df["Euclidean_Distance_To_Hydrology"] = np.sqrt(h_dist**2 + v_dist**2)

        # Relative Elevation
        # Elevation - Vertical_Distance_To_Hydrology (Elevation of the water source)
        df["Relative_Elevation"] = df["Elevation"] - v_dist

        # Cyclic Aspect Encoding
        # Aspect is in degrees (0-360). Convert to radians for sin/cos.
        # We handle potential values outside 0-360 if any (though usually fixed in data cleaning)
        # by just applying the trig function directly.
        aspect_rad = np.radians(df["Aspect"])
        df["Aspect_Sin"] = np.sin(aspect_rad)
        df["Aspect_Cos"] = np.cos(aspect_rad)

        return df

    def _add_dual_representation(self, df):
        """
        Generates dense integer indices for OHE groups while retaining original columns.
        Explicitly sorts columns numerically to preserve ordinal semantics.
        """
        for prefix in Config.DENSE_PREFIXES:
            # Find all columns belonging to this group
            # e.g. Soil_Type1, Soil_Type2, ...
            cols = [
                c
                for c in df.columns
                if c.startswith(prefix) and c[len(prefix) :].isdigit()
            ]

            if not cols:
                continue

            # Sort columns numerically based on the suffix integer
            # This ensures Soil_Type2 comes before Soil_Type10
            cols.sort(key=lambda x: int(x[len(prefix) :]))

            # Create the dense index
            # np.argmax returns the index of the max value (which is 1 in OHE).
            # We use the sorted order.
            # Result is 0-based index relative to the sorted list.
            # e.g., if Soil_Type1 is active, index is 0.
            # If Soil_Type10 is active, index is 9.
            # This creates a categorical variable representing the type.
            df[f"{prefix}_Index"] = np.argmax(df[cols].values, axis=1)

        return df

    def _map_target(self, df):
        """Maps the target column to 0-N range for XGBoost."""
        # Config.CLASS_MAPPING maps original class (e.g. 1, 2, 7) to (0, 1, 5)
        df[Config.TARGET_COL] = df[Config.TARGET_COL].map(Config.CLASS_MAPPING)
        return df
