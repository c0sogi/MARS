import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from library.config import Config


class FeatureEngineer:
    """
    Handles data loading, feature engineering, scaling, and caching for the PITH-Net pipeline.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.scaler_path = os.path.join(self.working_dir, "scaler_params.npz")
        self.seq_len = 80  # Standard breath length for this dataset

    def get_data(self, split_name, load_cached_data=True):
        """
        Main entry point to get processed data for a specific split (train, val, test).

        Args:
            split_name (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (x, y, u_out, ids)
                x (np.ndarray): Shape (N, 80, n_features) - Input features
                y (np.ndarray): Shape (N, 80) - Target pressure (zeros for test)
                u_out (np.ndarray): Shape (N, 80) - Expiratory valve status
                ids (np.ndarray): Shape (N, 80) - Time step IDs
        """
        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

        # Define cache paths
        cache_prefix = os.path.join(self.working_dir, f"{split_name}")
        path_x = f"{cache_prefix}_x.npy"
        path_y = f"{cache_prefix}_y.npy"
        path_u_out = f"{cache_prefix}_u_out.npy"
        path_ids = f"{cache_prefix}_ids.npy"

        # 1. Try to load from cache
        if load_cached_data:
            if all(os.path.exists(p) for p in [path_x, path_y, path_u_out, path_ids]):
                print(f"Loading {split_name} data from cache...")
                return (
                    np.load(path_x),
                    np.load(path_y),
                    np.load(path_u_out),
                    np.load(path_ids),
                )
            else:
                print(f"Cache miss for {split_name}. Processing from scratch...")

        # 2. Load Raw Data
        df = self._load_raw_data(split_name)

        # 3. Feature Engineering
        print(f"Engineering features for {split_name}...")
        df = self._engineer_features(df)

        # 4. Scaling
        # We scale features before reshaping
        print(f"Scaling features for {split_name}...")
        df = self._scale_features(df, is_train=(split_name == "train"))

        # 5. Reshape and Format
        print(f"Reshaping {split_name} data...")
        x, y, u_out, ids = self._reshape_and_format(df)

        # 6. Save to Cache
        print(f"Saving {split_name} data to cache...")
        np.save(path_x, x)
        np.save(path_y, y)
        np.save(path_u_out, u_out)
        np.save(path_ids, ids)

        return x, y, u_out, ids

    def _load_raw_data(self, split_name):
        """Loads the raw CSV file based on the split name."""
        if split_name == "train":
            path = Config.TRAIN_PATH
        elif split_name == "val":
            path = Config.VAL_PATH
        elif split_name == "test":
            path = Config.TEST_PATH
        else:
            raise ValueError(f"Unknown split: {split_name}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")

        df = pd.read_csv(path)

        # Debugging: Subsample if configured
        if Config.DEBUG:
            print(f"DEBUG MODE: Subsampling {split_name} data...")
            unique_breaths = df[Config.BREATH_ID_COL].unique()
            subset_breaths = unique_breaths[:100]  # Take first 100 breaths
            df = df[df[Config.BREATH_ID_COL].isin(subset_breaths)].copy()

        return df

    def _engineer_features(self, df):
        """Applies physical and lag/lead feature engineering."""

        # --- 1. Pre-computation Setup ---
        # Ensure data is sorted by breath_id and time_step
        df = df.sort_values([Config.BREATH_ID_COL, Config.TIME_COL]).reset_index(
            drop=True
        )

        # Identify breath boundaries
        # mask_start is True at the first time step of each breath
        breath_id = df[Config.BREATH_ID_COL].values
        mask_start = np.concatenate(([True], breath_id[1:] != breath_id[:-1]))
        # mask_end is True at the last time step of each breath
        mask_end = np.concatenate((breath_id[1:] != breath_id[:-1], [True]))

        # --- 2. Dynamic Physics (Derived) ---

        # Calculate dt (time difference)
        # We use numpy for speed.
        time_step = df[Config.TIME_COL].values
        dt = np.zeros_like(time_step)
        dt[1:] = time_step[1:] - time_step[:-1]
        dt[mask_start] = 0  # First step of a breath has dt=0 (or undefined)

        # Physically Accurate Volume (Area = Integral of u_in * dt)
        u_in = df["u_in"].values
        # Cumulative sum resets per breath.
        # We can do this by using groupby, but a faster numpy trick is:
        # cumsum(all) - cumsum(at_start_of_breath)
        # However, groupby().cumsum() in pandas is reasonably optimized.
        df["dt"] = dt
        df["area"] = (
            df.groupby(Config.BREATH_ID_COL)
            .apply(lambda x: (x["u_in"] * x["dt"]).cumsum())
            .reset_index(level=0, drop=True)
        )

        # Acceleration (Derivative of u_in)
        # u_in_diff = u_in(t) - u_in(t-1)
        u_in_diff = np.zeros_like(u_in)
        u_in_diff[1:] = u_in[1:] - u_in[:-1]
        u_in_diff[mask_start] = 0
        df["u_in_diff"] = u_in_diff

        # --- 3. Lookahead Injection (Future Context) ---
        # We shift columns upwards (negative shift).
        # We must handle boundaries so we don't leak data from the next breath.

        for k in range(1, 5):
            col_name = f"u_in_next{k}"
            # Shift raw array
            shifted = np.roll(u_in, -k)
            # Mask out values that wrapped around or crossed breath boundaries
            # If we shift by -k, the last k elements of a breath take values from next breath.
            # We can detect this by comparing breath_ids shifted by -k
            breath_id_shifted = np.roll(breath_id, -k)
            # Valid where breath_id matches
            valid_mask = breath_id == breath_id_shifted
            # Also, the very last k elements of the entire array are invalid (wrap around)
            valid_mask[-k:] = False

            # Apply mask (fill with 0, as valve is usually closed/irrelevant at end)
            shifted[~valid_mask] = 0
            df[col_name] = shifted

        # Derivative at t+1 (u_in_diff_next1)
        # This is u_in(t+1) - u_in(t). Or simply u_in_diff shifted by -1?
        # u_in_diff(t) = u_in(t) - u_in(t-1).
        # u_in_diff(t+1) = u_in(t+1) - u_in(t).
        # So we want u_in_diff shifted by -1.
        shifted_diff = np.roll(u_in_diff, -1)
        breath_id_shifted = np.roll(breath_id, -1)
        valid_mask = breath_id == breath_id_shifted
        valid_mask[-1] = False
        shifted_diff[~valid_mask] = 0
        df["u_in_diff_next1"] = shifted_diff

        # --- 4. Static Physics Interactions ---
        df["R_u_in"] = df["R"] * df["u_in"]
        df["area_C"] = df["area"] / df["C"]

        # Fill any remaining NaNs (just in case)
        df = df.fillna(0)

        return df

    def _scale_features(self, df, is_train):
        """
        Applies RobustScaler to continuous features.
        Fits on train, transforms on val/test.
        Persists scaler parameters to .npz file (no pickle).
        """
        # Identify columns to scale (exclude u_out as it is binary)
        scale_cols = [c for c in Config.FEATURE_COLS if c != "u_out"]

        data_to_scale = df[scale_cols].values.astype(np.float32)

        if is_train:
            # Initialize and fit scaler
            scaler = RobustScaler(quantile_range=(25.0, 75.0))
            data_to_scale = scaler.fit_transform(data_to_scale)

            # Save parameters manually to avoid pickle
            np.savez(self.scaler_path, center=scaler.center_, scale=scaler.scale_)
        else:
            # Load parameters
            if not os.path.exists(self.scaler_path):
                raise RuntimeError(
                    "Scaler parameters not found. Process 'train' split first."
                )

            params = np.load(self.scaler_path)
            center = params["center"]
            scale = params["scale"]

            # Manual transform: (X - center) / scale
            # Handle division by zero if scale is 0 (RobustScaler usually handles this by setting scale=1)
            scale = np.where(scale == 0, 1.0, scale)
            data_to_scale = (data_to_scale - center) / scale

        # Update dataframe
        df[scale_cols] = data_to_scale
        return df

    def _reshape_and_format(self, df):
        """
        Reshapes the dataframe into (N_breaths, 80, N_features).
        Extracts targets and ids.
        """
        # Ensure strict ordering
        df = df.sort_values([Config.BREATH_ID_COL, Config.TIME_COL])

        # Verify sequence length assumption
        # In this dataset, every breath has exactly 80 steps.
        # If DEBUG is on, we might have partial data, but _load_raw_data filters by breath_id.
        n_rows = len(df)
        if n_rows % self.seq_len != 0:
            raise ValueError(
                f"Total rows {n_rows} is not divisible by sequence length {self.seq_len}."
            )

        n_breaths = n_rows // self.seq_len

        # Extract Features
        x_data = df[Config.FEATURE_COLS].values.astype(np.float32)
        x_reshaped = x_data.reshape(n_breaths, self.seq_len, Config.INPUT_DIM)

        # Extract u_out (needed for masking loss)
        u_out_data = df["u_out"].values.astype(np.int8)
        u_out_reshaped = u_out_data.reshape(n_breaths, self.seq_len)

        # Extract IDs (needed for submission)
        ids_data = df[Config.ID_COL].values.astype(np.int32)
        ids_reshaped = ids_data.reshape(n_breaths, self.seq_len)

        # Extract Target (Pressure)
        if Config.TARGET_COL in df.columns:
            y_data = df[Config.TARGET_COL].values.astype(np.float32)
            y_reshaped = y_data.reshape(n_breaths, self.seq_len)
        else:
            # For test set, target might not exist
            y_reshaped = np.zeros((n_breaths, self.seq_len), dtype=np.float32)

        return x_reshaped, y_reshaped, u_out_reshaped, ids_reshaped
