import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import ensure_dir


class FeatureEngineer:
    """
    Handles feature engineering for the Ventilator Pressure Prediction task.
    Implements 'Physics-Fidelity' features including integral volume,
    interaction terms, and temporal dynamics.
    """

    def __init__(self, config: Config):
        """
        Args:
            config (Config): Configuration object containing paths and feature lists.
        """
        self.config = config

    def _add_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds physics-inspired features based on the equation of motion.
        Calculates volume via integration and interaction terms with lung attributes.
        """
        # Calculate time delta (dt) per breath
        # Groupby ensures we don't diff across breath boundaries
        df["dt"] = df.groupby(self.config.BREATH_COL)["time_step"].diff().fillna(0)

        # Calculate Volume: Cumulative sum of flow (u_in) * dt
        # We use a temporary column for vectorization speedup compared to .apply()
        df["flow_dt"] = df["u_in"] * df["dt"]
        df["volume"] = df.groupby(self.config.BREATH_COL)["flow_dt"].cumsum()

        # Interaction Terms (Soft Physics Injection)
        df["u_in_R"] = df["u_in"] * df["R"]
        df["vol_C"] = df["volume"] / df["C"]

        # Cleanup temporary columns
        df = df.drop(columns=["dt", "flow_dt"])
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds temporal dynamics features: Lags and Differences.
        """
        # Create a groupby object for efficiency
        g = df.groupby(self.config.BREATH_COL)["u_in"]

        # Lags 1 to 4
        for i in range(1, 5):
            df[f"u_in_lag{i}"] = g.shift(i).fillna(0)

        # First Difference (Velocity of control input)
        df["u_in_diff1"] = g.diff(1).fillna(0)

        # Second Difference (Acceleration of control input)
        # Note: diff(1) on the diff1 column gives the 2nd order difference
        df["u_in_diff2"] = (
            df.groupby(self.config.BREATH_COL)["u_in_diff1"].diff(1).fillna(0)
        )

        return df

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the full feature engineering pipeline to a dataframe.
        """
        df = df.copy()

        # Apply engineering steps
        df = self._add_physics_features(df)
        df = self._add_temporal_features(df)

        # Filter columns to keep only ID, Breath ID, Target (if present), and Features
        cols_to_keep = [self.config.ID_COL, self.config.BREATH_COL]

        if self.config.TARGET_COL in df.columns:
            cols_to_keep.append(self.config.TARGET_COL)

        # Verify all engineered features are present
        missing_features = [f for f in self.config.FEATURE_LIST if f not in df.columns]
        if missing_features:
            # Check if missing features are just raw features that should be there
            # (RAW_FEATURES are usually in input, ENG_FEATURES are created)
            raise ValueError(f"Missing features after engineering: {missing_features}")

        # Extend with the full feature list (Raw + Engineered)
        # We use a set to avoid duplicates if ID/Breath columns are in FEATURE_LIST (they usually aren't)
        final_cols = []
        for col in cols_to_keep:
            if col not in final_cols:
                final_cols.append(col)

        for col in self.config.FEATURE_LIST:
            if col not in final_cols:
                final_cols.append(col)

        return df[final_cols]

    def _compute_and_save_scaler(self, df: pd.DataFrame):
        """
        Computes RobustScaler statistics (Median and IQR) on the training data
        and saves them to NPY files for consistent scaling.
        """
        print("Computing RobustScaler statistics on training data...")

        # Extract feature matrix
        X = df[self.config.FEATURE_LIST].values

        # Compute Median (Center)
        center = np.nanmedian(X, axis=0)

        # Compute IQR (Scale)
        q75 = np.nanpercentile(X, 75, axis=0)
        q25 = np.nanpercentile(X, 25, axis=0)
        scale = q75 - q25

        # Handle constant features (scale=0) to avoid division by zero
        scale = np.where(scale == 0, 1.0, scale)

        # Save to disk
        ensure_dir(self.config.SCALER_CENTER)
        np.save(self.config.SCALER_CENTER, center)
        np.save(self.config.SCALER_SCALE, scale)

        print(f"Scaler statistics saved to {self.config.WORKING_DIR}")

    def run(self, load_cached_data: bool = True, debug: bool = False):
        """
        Orchestrates the feature engineering process.

        Args:
            load_cached_data (bool): If True, attempts to load processed data from disk.
            debug (bool): If True, processes a small subset of data and skips saving.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        # 1. Check for cached data
        caches_exist = (
            os.path.exists(self.config.TRAIN_CACHE)
            and os.path.exists(self.config.VAL_CACHE)
            and os.path.exists(self.config.TEST_CACHE)
            and os.path.exists(self.config.SCALER_CENTER)
            and os.path.exists(self.config.SCALER_SCALE)
        )

        if load_cached_data and caches_exist and not debug:
            print(
                f"Loading cached engineered datasets from {self.config.WORKING_DIR}..."
            )
            train_df = pd.read_parquet(self.config.TRAIN_CACHE)
            val_df = pd.read_parquet(self.config.VAL_CACHE)
            test_df = pd.read_parquet(self.config.TEST_CACHE)
            return train_df, val_df, test_df

        # 2. Load Raw Data
        print(f"Loading raw data from {self.config.METADATA_DIR}...")
        train_raw = pd.read_csv(self.config.TRAIN_CSV)
        val_raw = pd.read_csv(self.config.VAL_CSV)
        test_raw = pd.read_csv(self.config.TEST_CSV)

        # 3. Debug Sampling
        if debug:
            print("DEBUG MODE: Sampling first 100 breaths for rapid testing.")
            train_breaths = train_raw[self.config.BREATH_COL].unique()[:100]
            val_breaths = val_raw[self.config.BREATH_COL].unique()[:100]
            test_breaths = test_raw[self.config.BREATH_COL].unique()[:100]

            train_raw = train_raw[train_raw[self.config.BREATH_COL].isin(train_breaths)]
            val_raw = val_raw[val_raw[self.config.BREATH_COL].isin(val_breaths)]
            test_raw = test_raw[test_raw[self.config.BREATH_COL].isin(test_breaths)]

        # 4. Process Datasets
        print("Processing training set...")
        train_df = self.process_dataframe(train_raw)

        # Compute scaler only on training data
        self._compute_and_save_scaler(train_df)

        print("Processing validation set...")
        val_df = self.process_dataframe(val_raw)

        print("Processing test set...")
        test_df = self.process_dataframe(test_raw)

        # 5. Save to Cache (Skip in debug mode to avoid corrupting full cache)
        if not debug:
            print(f"Saving datasets to parquet cache in {self.config.WORKING_DIR}...")
            train_df.to_parquet(self.config.TRAIN_CACHE, index=False)
            val_df.to_parquet(self.config.VAL_CACHE, index=False)
            test_df.to_parquet(self.config.TEST_CACHE, index=False)
        else:
            print("DEBUG MODE: Skipping cache save.")

        return train_df, val_df, test_df
