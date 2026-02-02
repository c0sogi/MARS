import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler
from library.config import Config


class FeatureEngineer:
    """
    Handles feature engineering, scaling, and caching for the ventilator pressure prediction task.
    Implements physics-informed features (Volume, Interactions) and temporal dynamics (Lags, Diffs).
    """

    def __init__(self):
        self.scaler = RobustScaler()
        # Define continuous columns that require scaling
        self.continuous_cols = [
            "time_step",
            "u_in",
            "R",
            "C",
            "volume",
            "R__u_in",
            "vol__C",
            "u_in_lag1",
            "u_in_lag2",
            "u_in_lag3",
            "u_in_lag4",
            "u_in_diff1",
            "u_in_diff2",
        ]
        # Ensure working directory exists
        Config.setup()

    def _add_physics_features(self, df):
        """
        Computes physics-informed features, lags, and diffs using vectorized operations.
        """
        # Ensure data is sorted by breath and time
        df = df.sort_values([Config.breath_id_col, "time_step"]).reset_index(drop=True)

        # Group object for vectorized operations
        g = df.groupby(Config.breath_id_col)

        # 1. Time Delta (dt)
        # Calculate time difference between steps within each breath
        df["dt"] = g["time_step"].diff().fillna(0)

        # 2. Volume Integration: Cumulative Sum of Flow * dt
        # Vectorized calculation: multiply columns first, then cumsum by group
        df["flow_dt"] = df["u_in"] * df["dt"]
        df["volume"] = g["flow_dt"].cumsum()

        # 3. Physics Interactions
        # R * u_in: Resistance interaction
        df["R__u_in"] = df["R"] * df["u_in"]
        # Volume / C: Compliance interaction
        df["vol__C"] = df["volume"] / df["C"]

        # 4. Lags
        if Config.use_lags:
            for lag in Config.lag_steps:
                df[f"u_in_lag{lag}"] = g["u_in"].shift(lag).fillna(0)

        # 5. Differences (Dynamics)
        if Config.use_diffs:
            # First difference: velocity of valve change
            df["u_in_diff1"] = g["u_in"].diff().fillna(0)
            # Second difference: acceleration of valve change
            df["u_in_diff2"] = g["u_in_diff1"].diff().fillna(0)

        # 6. Control Segregation
        # Keep a raw copy of u_out for loss masking (strictly 0 or 1)
        df["u_out_raw"] = df["u_out"].astype(np.int8)

        # Cleanup temporary columns
        df = df.drop(columns=["dt", "flow_dt"])

        return df

    def _save_scaler(self):
        """Saves scaler parameters to disk."""
        np.save(Config.scaler_center_path, self.scaler.center_)
        np.save(Config.scaler_scale_path, self.scaler.scale_)

    def _load_scaler(self):
        """Loads scaler parameters from disk."""
        if not os.path.exists(Config.scaler_center_path) or not os.path.exists(
            Config.scaler_scale_path
        ):
            raise FileNotFoundError(
                "Scaler files not found in cache. Run process_train_val first."
            )

        # Manually set attributes to reproduce the fitted state
        self.scaler.center_ = np.load(Config.scaler_center_path)
        self.scaler.scale_ = np.load(Config.scaler_scale_path)

    def process_train_val(self, load_cached=True):
        """
        Loads, engineers, scales, and caches training and validation data.

        Args:
            load_cached (bool): If True, attempts to load from parquet cache.

        Returns:
            tuple: (train_df, val_df)
        """
        # 1. Try Loading Cache
        if load_cached:
            if (
                os.path.exists(Config.train_cache)
                and os.path.exists(Config.val_cache)
                and os.path.exists(Config.scaler_center_path)
            ):
                print("Loading train/val data from cache...")
                train_df = pd.read_parquet(Config.train_cache)
                val_df = pd.read_parquet(Config.val_cache)
                self._load_scaler()
                return train_df, val_df

        print("Processing train/val data from scratch...")

        # 2. Load Raw Metadata
        train_df = pd.read_csv(Config.train_file)
        val_df = pd.read_csv(Config.val_file)

        # Debug Mode: Sample data to speed up development
        if Config.debug:
            print("Debug mode: sampling data (100 train breaths, 50 val breaths)...")
            train_breaths = train_df[Config.breath_id_col].unique()[:100]
            val_breaths = val_df[Config.breath_id_col].unique()[:50]
            train_df = train_df[
                train_df[Config.breath_id_col].isin(train_breaths)
            ].copy()
            val_df = val_df[val_df[Config.breath_id_col].isin(val_breaths)].copy()

        # 3. Feature Engineering
        print("Engineering features for Train...")
        train_df = self._add_physics_features(train_df)
        print("Engineering features for Val...")
        val_df = self._add_physics_features(val_df)

        # 4. Scaling
        print("Fitting RobustScaler on Train...")
        self.scaler.fit(train_df[self.continuous_cols])

        print("Transforming Train...")
        train_df[self.continuous_cols] = self.scaler.transform(
            train_df[self.continuous_cols]
        )
        print("Transforming Val...")
        val_df[self.continuous_cols] = self.scaler.transform(
            val_df[self.continuous_cols]
        )

        # 5. Caching
        print(f"Caching data to {Config.working_dir}...")
        train_df.to_parquet(Config.train_cache)
        val_df.to_parquet(Config.val_cache)
        self._save_scaler()

        return train_df, val_df

    def process_test(self, load_cached=True):
        """
        Loads, engineers, scales, and caches test data.
        Requires process_train_val to have been run previously to load the scaler.

        Args:
            load_cached (bool): If True, attempts to load from parquet cache.

        Returns:
            pd.DataFrame: Processed test data
        """
        # 1. Try Loading Cache
        if load_cached:
            if os.path.exists(Config.test_cache):
                print("Loading test data from cache...")
                test_df = pd.read_parquet(Config.test_cache)
                # Load scaler to ensure state consistency
                self._load_scaler()
                return test_df

        print("Processing test data from scratch...")

        # 2. Load Raw Data
        test_df = pd.read_csv(Config.test_file)

        # 3. Feature Engineering
        print("Engineering features for Test...")
        test_df = self._add_physics_features(test_df)

        # 4. Scaling
        # Must load the scaler fitted on training data
        self._load_scaler()

        print("Transforming Test...")
        test_df[self.continuous_cols] = self.scaler.transform(
            test_df[self.continuous_cols]
        )

        # 5. Caching
        print(f"Caching test data to {Config.test_cache}...")
        test_df.to_parquet(Config.test_cache)

        return test_df
