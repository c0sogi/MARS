import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler
from library.config import Config


class FeatureEngineer:
    """
    Handles feature engineering, scaling, and caching for the ventilator dataset.
    Implements physics-informed features and strict control segregation.
    """

    def __init__(self, config: Config):
        self.config = config
        self.scaler_center_path = os.path.join(
            self.config.WORKING_DIR, "scaler_center.npy"
        )
        self.scaler_scale_path = os.path.join(
            self.config.WORKING_DIR, "scaler_scale.npy"
        )

    def _add_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds physics-based features: volume integration and interaction terms.
        """
        # Group by breath_id to ensure calculations don't bleed across breaths
        # We assume data is sorted by breath_id and time_step (standard for this dataset)

        # Calculate dt (time delta)
        # We use groupby().diff() to get the difference within each breath
        df["dt"] = df.groupby(self.config.BREATH_COL)["time_step"].diff().fillna(0)

        # Calculate volume: integral of u_in * dt
        # u_in is 0-100, time is in seconds.
        # We calculate the cumulative sum of flow * dt per breath
        df["volume"] = (
            df.groupby(self.config.BREATH_COL)
            .apply(lambda x: (x["u_in"] * x["dt"]).cumsum())
            .reset_index(level=0, drop=True)
        )

        # Interaction terms
        df["u_in_R"] = df["u_in"] * df["R"]
        df["volume_div_C"] = df["volume"] / df["C"]

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds temporal dynamics: lags and differences for u_in.
        """
        # We can use groupby shift/diff for efficiency
        grouper = df.groupby(self.config.BREATH_COL)["u_in"]

        # Lags 1-4
        for lag in range(1, 5):
            df[f"u_in_lag{lag}"] = grouper.shift(lag).fillna(0)

        # Differences
        df["u_in_diff1"] = grouper.diff(1).fillna(0)
        df["u_in_diff2"] = grouper.diff(2).fillna(0)

        return df

    def _scale_features(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """
        Scales continuous features using RobustScaler.
        Fits on train, transforms on val/test.
        """
        features_to_scale = self.config.CONT_FEATURES

        # Check if all features exist
        missing_cols = [c for c in features_to_scale if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for scaling: {missing_cols}")

        if is_train:
            scaler = RobustScaler()
            # Fit and transform
            df[features_to_scale] = scaler.fit_transform(df[features_to_scale].values)

            # Save scaler parameters manually to avoid pickling the whole object
            np.save(self.scaler_center_path, scaler.center_)
            np.save(self.scaler_scale_path, scaler.scale_)
        else:
            # Load scaler parameters
            if not os.path.exists(self.scaler_center_path) or not os.path.exists(
                self.scaler_scale_path
            ):
                raise FileNotFoundError(
                    "Scaler parameters not found. Process train set first."
                )

            center = np.load(self.scaler_center_path)
            scale = np.load(self.scaler_scale_path)

            # Manual transform: (X - center) / scale
            X = df[features_to_scale].values
            X_scaled = (X - center) / scale
            df[features_to_scale] = X_scaled

        return df

    def process_dataset(
        self, split: str, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Main pipeline to load, engineer, scale, and cache data.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to try loading from cache.

        Returns:
            pd.DataFrame: Processed dataframe ready for the model.
        """
        # Define cache path
        cache_filename = f"{split}_engineered.parquet"
        if self.config.debug:
            cache_filename = f"debug_{cache_filename}"

        cache_path = os.path.join(self.config.WORKING_DIR, cache_filename)

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {split} data from {cache_path}...")
            return pd.read_parquet(cache_path)

        print(f"Processing {split} data from scratch...")

        # 2. Load Raw Data
        if split == "train":
            path = self.config.TRAIN_PATH
        elif split == "val":
            path = self.config.VAL_PATH
        elif split == "test":
            path = self.config.TEST_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        df = pd.read_csv(path)

        # Debug subsampling
        if self.config.debug:
            print(f"Debug mode: Subsampling {split} data...")
            unique_breaths = df[self.config.BREATH_COL].unique()
            # Take first 100 breaths for debug
            sample_breaths = unique_breaths[:100]
            df = df[df[self.config.BREATH_COL].isin(sample_breaths)].copy()

        # 3. Feature Engineering
        df = self._add_physics_features(df)
        df = self._add_temporal_features(df)

        # 4. Scaling
        # Note: u_out is NOT scaled (it is in BINARY_FEATURES, not CONT_FEATURES)
        df = self._scale_features(df, is_train=(split == "train"))

        # 5. Type Casting to save memory
        # Float64 -> Float32
        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].astype("float32")

        # Int64 -> Int32 (except ids if they are large, but here they fit)
        int_cols = df.select_dtypes(include=["int64"]).columns
        df[int_cols] = df[int_cols].astype("int32")

        # 6. Save Cache
        print(f"Saving {split} data to {cache_path}...")
        df.to_parquet(cache_path, index=False)

        return df


def get_processed_data(
    config: Config, split: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Convenience function to initialize the engineer and get data.
    """
    engineer = FeatureEngineer(config)
    return engineer.process_dataset(split, load_cached_data=load_cached_data)
