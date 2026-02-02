import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import set_seed


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Reshapes flat data into sequences of (80, Features) for RNN processing.
    """

    def __init__(self, X, u_out, y=None, is_test=False):
        """
        Args:
            X (np.ndarray): Feature matrix [N_steps, N_features].
            u_out (np.ndarray): Binary control input [N_steps,].
            y (np.ndarray, optional): Target pressure [N_steps,].
            is_test (bool): Whether this is the test set (no targets).
        """
        # The dataset is structured as 80 steps per breath.
        # We reshape to (N_breaths, 80, Features)
        self.steps_per_breath = 80

        # Sanity check
        assert (
            len(X) % self.steps_per_breath == 0
        ), "Data length must be divisible by 80."
        self.num_breaths = len(X) // self.steps_per_breath

        # Reshape and convert to tensor
        self.X = torch.tensor(X, dtype=torch.float32).view(
            self.num_breaths, self.steps_per_breath, -1
        )
        self.u_out = torch.tensor(u_out, dtype=torch.float32).view(
            self.num_breaths, self.steps_per_breath
        )

        self.is_test = is_test
        if not self.is_test:
            self.y = torch.tensor(y, dtype=torch.float32).view(
                self.num_breaths, self.steps_per_breath
            )
        else:
            self.y = None

    def __len__(self):
        return self.num_breaths

    def __getitem__(self, idx):
        if self.is_test:
            return self.X[idx], self.u_out[idx]
        else:
            return self.X[idx], self.u_out[idx], self.y[idx]


class Preprocessor:
    """
    Handles feature engineering and segregated scaling.
    """

    def __init__(self):
        self.scaler = RobustScaler()
        self.continuous_cols = Config.CONTINUOUS_FEATURES
        self.binary_cols = Config.BINARY_FEATURES
        self.scaler_fitted = False

    def _add_features(self, df):
        """
        Adds physics-based and dynamic features.
        Vectorized implementation for speed.
        """
        # Ensure sorted
        df = df.sort_values([Config.BREATH_ID_COL, Config.TIME_COL]).reset_index(
            drop=True
        )

        # Group object for breath-wise operations
        # Note: Using shift directly on DataFrame with a mask is faster than groupby().apply()
        # but we must be careful about boundaries.

        # 1. Time Delta
        # We use groupby shift to be safe across breath boundaries
        df["dt"] = df.groupby(Config.BREATH_ID_COL)[Config.TIME_COL].diff().fillna(0)

        # 2. Volume (Time-weighted integration)
        # u_in_cumsum = cumsum(u_in * dt)
        df["volume_fragment"] = df["u_in"] * df["dt"]
        df["u_in_cumsum"] = df.groupby(Config.BREATH_ID_COL)["volume_fragment"].cumsum()
        df.drop(columns=["volume_fragment", "dt"], inplace=True)

        # 3. Multi-step Deltas (Lags)
        # x_t - x_{t-k}
        # Restricted to 1st and 2nd order to avoid noise (Cite solution_lesson_node_00066)
        for k in range(1, 3):
            col_name = f"u_in_diff{k}"
            # Calculate difference
            df[col_name] = df["u_in"] - df.groupby(Config.BREATH_ID_COL)["u_in"].shift(
                k
            )
            # Fill NaNs (start of breath) with 0
            df[col_name] = df[col_name].fillna(0)

        # 4. Explicit Physical Interactions (Cite solution_lesson_node_00076)
        # P ~ R*Flow + Vol/C
        df["R_u_in"] = df["R"] * df["u_in"]
        df["vol_C"] = df["u_in_cumsum"] / df["C"]
        df["R_div_C"] = df["R"] / df["C"]

        return df

    def fit(self, df):
        """
        Fits the RobustScaler on continuous features.
        """
        print("Fitting scaler on continuous features...")
        self.scaler.fit(df[self.continuous_cols].values)
        self.scaler_fitted = True

    def transform(self, df):
        """
        Applies scaling to continuous features and passes binary features raw.
        """
        if not self.scaler_fitted:
            raise RuntimeError("Preprocessor must be fit before transform.")

        print("Transforming data...")
        # 1. Continuous Scaling
        X_cont = self.scaler.transform(df[self.continuous_cols].values)

        # 2. Binary Pass-through
        X_bin = df[self.binary_cols].values

        # 3. Concatenate
        X_combined = np.hstack([X_cont, X_bin])

        return X_combined.astype(np.float32)

    def save_scaler(self, path):
        """Saves scaler statistics to numpy file."""
        if not self.scaler_fitted:
            return
        np.savez(path, center=self.scaler.center_, scale=self.scaler.scale_)
        print(f"Scaler parameters saved to {path}")

    def load_scaler(self, path):
        """Loads scaler statistics."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found: {path}")
        data = np.load(path)
        self.scaler.center_ = data["center"]
        self.scaler.scale_ = data["scale"]
        self.scaler_fitted = True
        print(f"Scaler parameters loaded from {path}")


def load_data(load_cached_data=True):
    """
    Main data loading function.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed parquet files.
                                 If False or cache miss, processes from scratch.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    set_seed(Config.SEED)

    # Check if cache exists
    cache_exists = (
        os.path.exists(Config.TRAIN_CACHE_PATH)
        and os.path.exists(Config.VAL_CACHE_PATH)
        and os.path.exists(Config.TEST_CACHE_PATH)
        and os.path.exists(Config.SCALER_CACHE_PATH)
    )

    if load_cached_data and cache_exists:
        print(f"Loading cached data from {Config.WORKING_DIR}...")

        # Load processed dataframes
        df_train = pd.read_parquet(Config.TRAIN_CACHE_PATH)
        df_val = pd.read_parquet(Config.VAL_CACHE_PATH)
        df_test = pd.read_parquet(Config.TEST_CACHE_PATH)

        # Prepare arrays
        # Note: The parquet files contain the transformed features (X) + targets/meta
        # We need to reconstruct the feature matrix X based on column order
        # However, saving the numpy array directly is often safer for X.
        # But to keep it inspectable, we likely saved the dataframe with feature columns.

        # Let's assume the cache creation below saves the *transformed* features as columns
        # named f0, f1... or simply overrides the original columns.
        # Strategy: To ensure robustness, we will re-transform using the saved scaler
        # OR save the numpy arrays directly.
        # Given the prompt requirement "Use parquet (via pandas) or npy",
        # let's stick to saving the processed DATAFRAMES with engineered features
        # but perform the scaling transform on load to ensure consistency with the `Preprocessor` logic.

        # Actually, for speed, the cache usually stores the *ready-to-train* data.
        # Let's assume the parquet files contain the ENGINEERED features (unscaled)
        # to allow for scaler adjustments, OR contain scaled features.
        # Decision: Cache stores ENGINEERED features (unscaled).
        # Scaling is fast enough to do on load, or we can save scaled.
        # Let's save ENGINEERED features (unscaled) in Parquet.

        # Re-initialize preprocessor and load scaler
        preprocessor = Preprocessor()
        preprocessor.load_scaler(Config.SCALER_CACHE_PATH)

        # Transform
        X_train = preprocessor.transform(df_train)
        y_train = df_train[Config.TARGET_COL].values
        u_out_train = df_train["u_out"].values

        X_val = preprocessor.transform(df_val)
        y_val = df_val[Config.TARGET_COL].values
        u_out_val = df_val["u_out"].values

        X_test = preprocessor.transform(df_test)
        u_out_test = df_test["u_out"].values

    else:
        print("Processing data from scratch...")

        # 1. Load Metadata
        print("Loading metadata...")
        train_meta = pd.read_csv(Config.TRAIN_META_PATH)
        val_meta = pd.read_csv(Config.VAL_META_PATH)
        # test_meta = pd.read_csv(Config.TEST_META_PATH) # Not strictly needed for loading raw

        train_breath_ids = set(train_meta[Config.BREATH_ID_COL].unique())
        val_breath_ids = set(val_meta[Config.BREATH_ID_COL].unique())

        # 2. Load Raw Data
        print(f"Loading raw train data from {Config.TRAIN_DATA_PATH}...")
        df_raw_train = pd.read_csv(Config.TRAIN_DATA_PATH)

        print(f"Loading raw test data from {Config.TEST_DATA_PATH}...")
        df_test = pd.read_csv(Config.TEST_DATA_PATH)

        # 3. Split Train/Val
        print("Splitting train/val...")
        df_train = df_raw_train[
            df_raw_train[Config.BREATH_ID_COL].isin(train_breath_ids)
        ].copy()
        df_val = df_raw_train[
            df_raw_train[Config.BREATH_ID_COL].isin(val_breath_ids)
        ].copy()

        del df_raw_train
        gc.collect()

        # 4. Feature Engineering
        print("Feature engineering (Train)...")
        preprocessor = Preprocessor()
        df_train = preprocessor._add_features(df_train)

        print("Feature engineering (Val)...")
        df_val = preprocessor._add_features(df_val)

        print("Feature engineering (Test)...")
        df_test = preprocessor._add_features(df_test)

        # 5. Fit Scaler (Train only)
        preprocessor.fit(df_train)
        preprocessor.save_scaler(Config.SCALER_CACHE_PATH)

        # 6. Cache Processed Data (Engineered but Unscaled)
        # We save the dataframes with the new features so we don't have to re-compute lags
        print(f"Caching data to {Config.WORKING_DIR}...")
        df_train.to_parquet(Config.TRAIN_CACHE_PATH, index=False)
        df_val.to_parquet(Config.VAL_CACHE_PATH, index=False)
        df_test.to_parquet(Config.TEST_CACHE_PATH, index=False)

        # 7. Transform to Numpy
        X_train = preprocessor.transform(df_train)
        y_train = df_train[Config.TARGET_COL].values
        u_out_train = df_train["u_out"].values

        X_val = preprocessor.transform(df_val)
        y_val = df_val[Config.TARGET_COL].values
        u_out_val = df_val["u_out"].values

        X_test = preprocessor.transform(df_test)
        u_out_test = df_test["u_out"].values

    # Create Datasets
    print("Creating TensorDatasets...")
    train_dataset = VentilatorDataset(X_train, u_out_train, y_train, is_test=False)
    val_dataset = VentilatorDataset(X_val, u_out_val, y_val, is_test=False)
    test_dataset = VentilatorDataset(X_test, u_out_test, is_test=True)

    print(f"Train size: {len(train_dataset)} breaths")
    print(f"Val size: {len(val_dataset)} breaths")
    print(f"Test size: {len(test_dataset)} breaths")

    return train_dataset, val_dataset, test_dataset
