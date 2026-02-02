import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import get_config_hash


class DataProcessor:
    """
    Handles data loading, feature engineering, scaling, and reshaping for the
    Ventilator Pressure Prediction task. Implements caching based on configuration hashing.
    """

    def __init__(self):
        self.config_hash = get_config_hash()
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def _get_cache_path(self, filename: str) -> str:
        """Constructs a cache file path including the config hash."""
        name, ext = os.path.splitext(filename)
        return os.path.join(self.working_dir, f"{name}_{self.config_hash}{ext}")

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes physics-informed features including derivatives, integrals,
        and interaction terms.
        """
        # Ensure data is sorted by breath_id and time_step
        df = df.sort_values([Config.COL_BREATH_ID, Config.COL_TIME]).reset_index(
            drop=True
        )

        # 1. Temporal Dynamics (dt)
        # Calculate time difference between steps. Fill first value of each breath with 0.
        df["dt"] = df.groupby(Config.COL_BREATH_ID)[Config.COL_TIME].diff().fillna(0)

        # 2. Physics: Cumulative Volume (Integral of flow u_in over time)
        # Volume ~ sum(u_in * dt)
        df["volume_chunk"] = df[Config.COL_U_IN] * df["dt"]
        df["u_in_cumsum"] = df.groupby(Config.COL_BREATH_ID)["volume_chunk"].cumsum()
        df.drop(columns=["volume_chunk", "dt"], inplace=True)

        # 3. Dynamics: Lags
        # Shift u_in by 1 and 2 steps
        df["u_in_lag1"] = (
            df.groupby(Config.COL_BREATH_ID)[Config.COL_U_IN].shift(1).fillna(0)
        )
        df["u_in_lag2"] = (
            df.groupby(Config.COL_BREATH_ID)[Config.COL_U_IN].shift(2).fillna(0)
        )

        # 4. Dynamics: Derivatives (Velocity and Acceleration of valve)
        df["u_in_diff1"] = (
            df.groupby(Config.COL_BREATH_ID)[Config.COL_U_IN].diff(1).fillna(0)
        )
        df["u_in_diff2"] = (
            df.groupby(Config.COL_BREATH_ID)[Config.COL_U_IN].diff(2).fillna(0)
        )

        # 5. Interactions: Control Input * Lung Attributes
        # R and C are constant per breath, but interaction helps linear models/CNNs
        # P ~ Flow * R + Volume / C
        df["u_in_R"] = df[Config.COL_U_IN] * df[Config.COL_R]

        # Cite solution_lesson_node_00005: Explicitly engineering interaction features representing physical equations
        df["volume_div_C"] = df["u_in_cumsum"] / df[Config.COL_C]

        # Additional interactions for high resistance dynamics
        df["u_in_diff1_R"] = df["u_in_diff1"] * df[Config.COL_R]
        df["u_in_diff2_R"] = df["u_in_diff2"] * df[Config.COL_R]

        # Add raw R and C as continuous features for scaling
        df["R_val"] = df[Config.COL_R]
        df["C_val"] = df[Config.COL_C]

        return df

    def _compute_scaler_stats(self, df: pd.DataFrame):
        """
        Computes Median and IQR for continuous features to implement RobustScaling.
        Saves stats to .npy files for consistent application on val/test sets.
        """
        features = df[Config.CONT_FEATURES].values

        # RobustScaler logic: center = median, scale = IQR (75th - 25th quantile)
        center = np.nanmedian(features, axis=0)
        q25 = np.nanpercentile(features, 25, axis=0)
        q75 = np.nanpercentile(features, 75, axis=0)
        scale = q75 - q25

        # Avoid division by zero
        scale[scale == 0.0] = 1.0

        np.save(self._get_cache_path("scaler_center.npy"), center)
        np.save(self._get_cache_path("scaler_scale.npy"), scale)

        return center, scale

    def _load_scaler_stats(self):
        """Loads the saved scaler statistics."""
        center_path = self._get_cache_path("scaler_center.npy")
        scale_path = self._get_cache_path("scaler_scale.npy")

        if not os.path.exists(center_path) or not os.path.exists(scale_path):
            raise FileNotFoundError(
                "Scaler stats not found. Process 'train' split first."
            )

        center = np.load(center_path)
        scale = np.load(scale_path)
        return center, scale

    def _scale_features(
        self, df: pd.DataFrame, center: np.ndarray, scale: np.ndarray
    ) -> pd.DataFrame:
        """Applies RobustScaling to continuous features."""
        df = df.copy()
        features = df[Config.CONT_FEATURES].values
        features = (features - center) / scale
        df[Config.CONT_FEATURES] = features
        return df

    def _reshape_to_sequences(self, df: pd.DataFrame, is_test: bool = False):
        """
        Reshapes DataFrame into 3D tensors (N_breaths, SEQ_LEN, N_features).
        Also extracts targets and mask.
        """
        # Ensure strict ordering
        df = df.sort_values([Config.COL_BREATH_ID, Config.COL_TIME])

        num_breaths = df[Config.COL_BREATH_ID].nunique()
        expected_len = num_breaths * Config.SEQ_LEN

        if len(df) != expected_len:
            # If data length mismatch (e.g. partial breaths), we might need to trim or pad.
            # For this dataset, breaths are usually fixed 80 steps.
            # We will truncate to the nearest full breath count to be safe,
            # though usually the dataset is clean.
            print(
                f"Warning: Data length {len(df)} is not a multiple of {Config.SEQ_LEN}."
            )
            df = df.iloc[:expected_len]

        # 1. Construct Feature Matrix X
        # Continuous features
        x_cont = df[Config.CONT_FEATURES].values

        # One-Hot Encoding for R
        x_r = []
        for val in Config.R_VALUES:
            x_r.append((df[Config.COL_R] == val).values.astype(np.float32))
        x_r = np.stack(x_r, axis=1)

        # One-Hot Encoding for C
        x_c = []
        for val in Config.C_VALUES:
            x_c.append((df[Config.COL_C] == val).values.astype(np.float32))
        x_c = np.stack(x_c, axis=1)

        # u_out feature (as input)
        x_uout = df[[Config.COL_U_OUT]].values.astype(np.float32)

        # Concatenate all features
        # Shape: (Total_Steps, Input_Dim)
        X_flat = np.concatenate([x_cont, x_r, x_c, x_uout], axis=1)

        # Reshape to (N_breaths, SEQ_LEN, Input_Dim)
        X = X_flat.reshape(num_breaths, Config.SEQ_LEN, -1)

        # 2. Construct Mask (u_out)
        # Shape: (N_breaths, SEQ_LEN)
        u_out = df[Config.COL_U_OUT].values.reshape(num_breaths, Config.SEQ_LEN)

        # 3. Construct Target y
        if not is_test:
            y = df[Config.COL_PRESSURE].values.reshape(num_breaths, Config.SEQ_LEN)
            return X, y, u_out
        else:
            return X, u_out

    def load_dataset(self, split: str = "train", load_cached_data: bool = True):
        """
        Main method to load and process data.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load from .npy cache.

        Returns:
            Tuple of numpy arrays.
            If train/val: (X, y, u_out)
            If test: (X, u_out)
        """
        is_test = split == "test"

        # Define cache filenames
        cache_X = self._get_cache_path(f"{split}_x.npy")
        cache_y = self._get_cache_path(f"{split}_y.npy")
        cache_uout = self._get_cache_path(f"{split}_uout.npy")

        # Attempt to load from cache
        if load_cached_data:
            if is_test:
                if os.path.exists(cache_X) and os.path.exists(cache_uout):
                    print(f"Loading cached {split} data...")
                    return np.load(cache_X), np.load(cache_uout)
            else:
                if (
                    os.path.exists(cache_X)
                    and os.path.exists(cache_y)
                    and os.path.exists(cache_uout)
                ):
                    print(f"Loading cached {split} data...")
                    return np.load(cache_X), np.load(cache_y), np.load(cache_uout)

        print(f"Processing {split} data from scratch...")

        # Load raw CSV
        if split == "train":
            path = Config.TRAIN_PATH
        elif split == "val":
            path = Config.VAL_PATH
        else:
            path = Config.TEST_PATH

        df = pd.read_csv(path)

        # Debug Mode: Sample breaths
        if Config.DEBUG:
            print(f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} breaths...")
            unique_breaths = df[Config.COL_BREATH_ID].unique()
            if len(unique_breaths) > Config.DEBUG_SAMPLE_SIZE:
                # Use a fixed seed for sampling consistency within the same debug run
                rng = np.random.default_rng(Config.SEED)
                sample_breaths = rng.choice(
                    unique_breaths, Config.DEBUG_SAMPLE_SIZE, replace=False
                )
                df = df[df[Config.COL_BREATH_ID].isin(sample_breaths)].copy()

        # Feature Engineering
        print("Adding physics features...")
        df = self.add_features(df)

        # Scaling
        print("Scaling features...")
        if split == "train":
            center, scale = self._compute_scaler_stats(df)
        else:
            center, scale = self._load_scaler_stats()

        df = self._scale_features(df, center, scale)

        # Reshaping
        print("Reshaping to tensors...")
        if is_test:
            X, u_out = self._reshape_to_sequences(df, is_test=True)
            # Save to cache
            np.save(cache_X, X)
            np.save(cache_uout, u_out)
            return X, u_out
        else:
            X, y, u_out = self._reshape_to_sequences(df, is_test=False)
            # Save to cache
            np.save(cache_X, X)
            np.save(cache_y, y)
            np.save(cache_uout, u_out)
            return X, y, u_out
