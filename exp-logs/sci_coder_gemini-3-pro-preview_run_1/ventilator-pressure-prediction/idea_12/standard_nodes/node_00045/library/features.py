import os
import pandas as pd
import numpy as np
from library.config import Config


class FeatureEngineer:
    """
    Handles feature engineering for the ventilator pressure prediction task.
    Implements physics-based features, time-series lags/diffs, and caching.
    """

    def __init__(self):
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, split_name):
        """Generates a cache file path based on the split name."""
        return os.path.join(self.cache_dir, f"dataset_{split_name}_engineered.parquet")

    def add_time_deltas(self, df):
        """
        Calculates the time difference (dt) between steps within each breath.
        Assumes df is sorted by breath_id and time_step (or id).
        """
        # Calculate global diff
        df["dt"] = df["time_step"].diff()

        # Identify where breath_id changes to reset dt to 0 for the start of new breaths
        # We assume the dataframe is sorted by breath_id, then time_step.
        # The metadata generation ensures this structure.
        breath_change_mask = df["breath_id"] != df["breath_id"].shift(1)

        # Set dt to 0 at the start of each breath and fill initial NaN
        df.loc[breath_change_mask, "dt"] = 0
        df["dt"] = df["dt"].fillna(0)

        # Enforce non-negative dt (handling potential potential data quirks, though unlikely in this dataset)
        df["dt"] = df["dt"].clip(lower=0)

        return df

    def add_physics_features(self, df):
        """
        Adds features derived from physical principles:
        - volume: Integral of flow (u_in) over time.
        - u_in_R: Interaction between input flow and resistance (Pressure ~ Flow * R).
        - vol_C: Interaction between volume and compliance (Pressure ~ Volume / C).
        """
        # 1. Calculate Volume (Cumulative Sum of u_in * dt)
        # We use a temporary column for flow * dt
        df["flow_dt"] = df["u_in"] * df["dt"]

        # Groupby cumsum is necessary to reset volume for each breath
        # Optimization: Since data is sorted, we can use the same mask logic if we implemented a custom cumsum,
        # but groupby().cumsum() is reasonably optimized in modern pandas.
        df["volume"] = df.groupby("breath_id")["flow_dt"].cumsum()

        # Drop temp column
        df.drop(columns=["flow_dt"], inplace=True)

        # 2. Interaction Terms
        df["u_in_R"] = df["u_in"] * df["R"]
        df["vol_C"] = df["volume"] / df["C"]

        return df

    def add_lag_diff_features(self, df):
        """
        Adds lag and difference features for u_in to capture system inertia and dynamics.
        """
        # We use groupby to ensure lags/diffs don't cross breath boundaries.
        # While global shift + mask is faster, groupby is safer and sufficiently fast for this dataset size (~6M rows).

        # Lags 1-4
        for lag in [1, 2, 3, 4]:
            df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0)

        # First Difference (Approximation of derivative of control input)
        df["u_in_diff1"] = df.groupby("breath_id")["u_in"].diff().fillna(0)

        # Second Difference (Approximation of acceleration of control input)
        df["u_in_diff2"] = df.groupby("breath_id")["u_in_diff1"].diff().fillna(0)

        return df

    def process_split(self, split_name, load_cached_data=True):
        """
        Orchestrates the loading, feature engineering, and caching for a specific data split.

        Args:
            split_name (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load from parquet cache first.

        Returns:
            pd.DataFrame: The processed dataframe with all features.
        """
        cache_path = self._get_cache_path(split_name)

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {split_name} data from {cache_path}...")
            return pd.read_parquet(cache_path)

        # 2. Load Raw Data
        print(f"Processing {split_name} data from raw metadata...")
        if split_name == "train":
            path = Config.TRAIN_PATH
        elif split_name == "val":
            path = Config.VAL_PATH
        elif split_name == "test":
            path = Config.TEST_PATH
        else:
            raise ValueError(f"Unknown split_name: {split_name}")

        df = pd.read_csv(path)

        # Debug Mode: Slice data if configured
        if Config.DEBUG:
            print(f"DEBUG mode active. Slicing {split_name} data...")
            unique_breaths = df["breath_id"].unique()[: Config.DEBUG_BREATHS]
            df = df[df["breath_id"].isin(unique_breaths)].copy()

        # 3. Apply Feature Engineering
        print(f"Adding time deltas for {split_name}...")
        df = self.add_time_deltas(df)

        print(f"Adding physics features for {split_name}...")
        df = self.add_physics_features(df)

        print(f"Adding lag/diff features for {split_name}...")
        df = self.add_lag_diff_features(df)

        # 4. Validate Columns
        # Ensure all columns defined in Config.FEATURE_COLS are present
        missing_cols = [col for col in Config.FEATURE_COLS if col not in df.columns]
        if missing_cols:
            raise RuntimeError(
                f"Missing columns after feature engineering: {missing_cols}"
            )

        # 5. Save to Cache
        print(f"Saving processed {split_name} data to {cache_path}...")
        df.to_parquet(cache_path, index=False)

        return df
