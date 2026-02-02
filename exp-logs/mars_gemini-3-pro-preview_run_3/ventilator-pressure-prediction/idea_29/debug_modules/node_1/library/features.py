import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import seed_everything


class FeatureEngineer:
    """
    Handles feature engineering, preprocessing, and caching for the Ventilator Pressure Prediction task.
    Implements the Dual-Stream strategy:
    - Stream A: Scaled physical features (Model Input)
    - Stream B: Raw binary features (Loss Masking)
    """

    def __init__(self):
        self.scaler = RobustScaler()
        self.is_fitted = False
        self.breath_steps = 80  # Standard breath length for this dataset

    def process_data(self, load_cached_data: bool = True):
        """
        Main pipeline execution method.
        Checks for cached numpy arrays. If found and requested, loads them.
        Otherwise, loads raw metadata, computes features, fits scaler, and saves cache.

        Args:
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: A dictionary containing 'train', 'val', and 'test' datasets.
                  Each dataset is a dictionary with keys like 'x', 'mask', 'y', 'ids'.
        """
        # Define cache file paths
        cache_files = {
            "train_x": os.path.join(Config.CACHE_DIR, "train_x.npy"),
            "train_mask": os.path.join(Config.CACHE_DIR, "train_mask.npy"),
            "train_y": os.path.join(Config.CACHE_DIR, "train_y.npy"),
            "val_x": os.path.join(Config.CACHE_DIR, "val_x.npy"),
            "val_mask": os.path.join(Config.CACHE_DIR, "val_mask.npy"),
            "val_y": os.path.join(Config.CACHE_DIR, "val_y.npy"),
            "test_x": os.path.join(Config.CACHE_DIR, "test_x.npy"),
            "test_mask": os.path.join(Config.CACHE_DIR, "test_mask.npy"),
            "test_ids": os.path.join(Config.CACHE_DIR, "test_ids.npy"),
        }

        # Attempt to load from cache
        if load_cached_data:
            if all(os.path.exists(p) for p in cache_files.values()):
                print("Loading data from cache...")
                return self._load_cache(cache_files)
            else:
                print("Cache not found or incomplete. Recomputing features...")

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Load Raw Data
        print("Loading raw metadata...")
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        if Config.DEBUG:
            print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} breaths...")
            train_df = self._debug_sample(train_df)
            val_df = self._debug_sample(val_df)
            test_df = self._debug_sample(test_df)

        # Feature Engineering
        print("Generating features...")
        train_df = self._engineer_features(train_df)
        val_df = self._engineer_features(val_df)
        test_df = self._engineer_features(test_df)

        # Prepare Stream A (Features to be scaled)
        print("Fitting scaler on training data...")
        train_x_raw = train_df[Config.STREAM_A_COLS].values.astype(np.float32)
        val_x_raw = val_df[Config.STREAM_A_COLS].values.astype(np.float32)
        test_x_raw = test_df[Config.STREAM_A_COLS].values.astype(np.float32)

        # Fit Scaler on Train, Transform All
        self.scaler.fit(train_x_raw)
        train_x = self.scaler.transform(train_x_raw)
        val_x = self.scaler.transform(val_x_raw)
        test_x = self.scaler.transform(test_x_raw)

        # Prepare Stream B (Raw Masks)
        train_mask = train_df[Config.STREAM_B_COLS].values.astype(np.float32)
        val_mask = val_df[Config.STREAM_B_COLS].values.astype(np.float32)
        test_mask = test_df[Config.STREAM_B_COLS].values.astype(np.float32)

        # Prepare Targets and IDs
        train_y = train_df[Config.TARGET_COL].values.astype(np.float32)
        val_y = val_df[Config.TARGET_COL].values.astype(np.float32)
        test_ids = test_df[Config.ID_COL].values.astype(np.int32)

        # Reshape to (N_Breaths, 80, Features)
        print("Reshaping tensors...")
        data = {
            "train": {
                "x": self._reshape_to_sequences(train_x),
                "mask": self._reshape_to_sequences(train_mask),
                "y": self._reshape_to_sequences(train_y),
            },
            "val": {
                "x": self._reshape_to_sequences(val_x),
                "mask": self._reshape_to_sequences(val_mask),
                "y": self._reshape_to_sequences(val_y),
            },
            "test": {
                "x": self._reshape_to_sequences(test_x),
                "mask": self._reshape_to_sequences(test_mask),
                "ids": test_ids,  # Keep IDs flat for submission mapping
            },
        }

        # Save to Cache
        print("Saving to cache...")
        np.save(cache_files["train_x"], data["train"]["x"])
        np.save(cache_files["train_mask"], data["train"]["mask"])
        np.save(cache_files["train_y"], data["train"]["y"])
        np.save(cache_files["val_x"], data["val"]["x"])
        np.save(cache_files["val_mask"], data["val"]["mask"])
        np.save(cache_files["val_y"], data["val"]["y"])
        np.save(cache_files["test_x"], data["test"]["x"])
        np.save(cache_files["test_mask"], data["test"]["mask"])
        np.save(cache_files["test_ids"], data["test"]["ids"])

        return data

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes physical and kinematic features.
        Uses vectorized reshape operations for efficiency instead of groupby.
        """
        # Ensure data is sorted
        df = df.sort_values([Config.BREATH_ID_COL, "time_step"]).reset_index(drop=True)

        # Verify shape
        n_samples = len(df)
        if n_samples % self.breath_steps != 0:
            raise ValueError(
                f"Total samples {n_samples} is not divisible by breath length {self.breath_steps}"
            )

        n_breaths = n_samples // self.breath_steps

        # Extract columns as numpy arrays for reshaping
        u_in = df["u_in"].values.reshape(n_breaths, self.breath_steps)
        time_step = df["time_step"].values.reshape(n_breaths, self.breath_steps)
        C = df["C"].values.reshape(n_breaths, self.breath_steps)

        # 1. Time Delta (dt)
        # dt = time_step[t] - time_step[t-1]. First element is 0.
        dt = np.zeros_like(time_step)
        dt[:, 1:] = time_step[:, 1:] - time_step[:, :-1]

        # 2. Area (Integration of Volume)
        # area = cumsum(u_in * dt)
        vol_change = u_in * dt
        area = np.cumsum(vol_change, axis=1)

        # 3. Explicit Physics Interaction (Area / C)
        area_C = area / C

        # 4. Kinematics: Backward Velocity (u_in_diff1)
        # diff = u_in[t] - u_in[t-1]
        u_in_diff1 = np.zeros_like(u_in)
        u_in_diff1[:, 1:] = u_in[:, 1:] - u_in[:, :-1]

        # 5. Kinematics: Forward Lookahead (u_in_lead1 to 4)
        u_in_lead1 = np.zeros_like(u_in)
        u_in_lead2 = np.zeros_like(u_in)
        u_in_lead3 = np.zeros_like(u_in)
        u_in_lead4 = np.zeros_like(u_in)

        u_in_lead1[:, :-1] = u_in[:, 1:]
        u_in_lead2[:, :-2] = u_in[:, 2:]
        u_in_lead3[:, :-3] = u_in[:, 3:]
        u_in_lead4[:, :-4] = u_in[:, 4:]

        # Assign back to DataFrame (Flattened)
        df["time_delta"] = dt.flatten()
        df["area"] = area.flatten()
        df["area_C"] = area_C.flatten()
        df["u_in_diff1"] = u_in_diff1.flatten()
        df["u_in_lead1"] = u_in_lead1.flatten()
        df["u_in_lead2"] = u_in_lead2.flatten()
        df["u_in_lead3"] = u_in_lead3.flatten()
        df["u_in_lead4"] = u_in_lead4.flatten()

        return df

    def _reshape_to_sequences(self, arr: np.ndarray) -> np.ndarray:
        """
        Reshapes a 2D array (Total_Steps, Features) or 1D array (Total_Steps,)
        into a 3D array (N_Breaths, 80, Features) or (N_Breaths, 80, 1).
        """
        if arr.ndim == 1:
            # Reshape (N,) -> (Breaths, 80, 1)
            return arr.reshape(-1, self.breath_steps, 1)
        else:
            # Reshape (N, Feats) -> (Breaths, 80, Feats)
            return arr.reshape(-1, self.breath_steps, arr.shape[1])

    def _load_cache(self, cache_files: dict) -> dict:
        """
        Loads numpy arrays from the cache.
        """
        return {
            "train": {
                "x": np.load(cache_files["train_x"]),
                "mask": np.load(cache_files["train_mask"]),
                "y": np.load(cache_files["train_y"]),
            },
            "val": {
                "x": np.load(cache_files["val_x"]),
                "mask": np.load(cache_files["val_mask"]),
                "y": np.load(cache_files["val_y"]),
            },
            "test": {
                "x": np.load(cache_files["test_x"]),
                "mask": np.load(cache_files["test_mask"]),
                "ids": np.load(cache_files["test_ids"]),
            },
        }

    def _debug_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Samples a subset of breaths for debugging.
        """
        unique_breaths = df[Config.BREATH_ID_COL].unique()
        if len(unique_breaths) > Config.DEBUG_SAMPLE_SIZE:
            sampled_breaths = unique_breaths[: Config.DEBUG_SAMPLE_SIZE]
            df = df[df[Config.BREATH_ID_COL].isin(sampled_breaths)].copy()
        return df
