import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import seed_everything


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    """

    def __init__(self, X, y=None, u_out=None):
        """
        Args:
            X (np.ndarray): Input features of shape (N, 80, F).
            y (np.ndarray, optional): Target pressure of shape (N, 80).
            u_out (np.ndarray, optional): Expiratory phase mask of shape (N, 80).
        """
        self.X = X
        self.y = y
        self.u_out = u_out

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = {"x": torch.tensor(self.X[idx], dtype=torch.float32)}

        if self.y is not None:
            data["y"] = torch.tensor(self.y[idx], dtype=torch.float32)

        if self.u_out is not None:
            data["u_out"] = torch.tensor(self.u_out[idx], dtype=torch.float32)

        return data


class DataProcessor:
    """
    Handles data loading, feature engineering, scaling, reshaping, and caching.
    """

    def __init__(self, config=Config):
        self.config = config
        seed_everything(self.config.SEED)

        # Ensure working directory exists
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

    def _add_features(self, df):
        """
        Computes PID and Physics-informed features.
        """
        # Sort to ensure time steps are sequential within breaths
        df = df.sort_values(["breath_id", "time_step"])

        # Vectorized Groupby operations
        # Integral (Volume proxy)
        df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()

        # Derivative (Flow/Acceleration proxy)
        # fillna(0) handles the first time step of each breath
        df["u_in_diff1"] = df.groupby("breath_id")["u_in"].diff().fillna(0)
        df["u_in_diff2"] = df.groupby("breath_id")["u_in_diff1"].diff().fillna(0)

        # Physics Interactions
        # Resistive Pressure ~ R * Flow (approximated by u_in)
        df["R_u_in"] = df["R"] * df["u_in"]
        # Elastic Pressure ~ Volume / C
        df["vol_C"] = df["u_in_cumsum"] / df["C"]

        return df

    def _reshape_sequences(self, df, feature_cols, target_col=None):
        """
        Reshapes long-format dataframe to (N_breaths, 80, N_features).
        """
        # Ensure strict sorting before reshape
        df = df.sort_values(["breath_id", "time_step"])

        # Extract features
        x_data = df[feature_cols].values
        num_breaths = len(df) // self.config.SEQ_LEN

        # Reshape X: (N, 80, F)
        x_reshaped = x_data.reshape(num_breaths, self.config.SEQ_LEN, len(feature_cols))

        # Extract u_out for masking: (N, 80)
        u_out_reshaped = df["u_out"].values.reshape(num_breaths, self.config.SEQ_LEN)

        y_reshaped = None
        if target_col and target_col in df.columns:
            y_data = df[target_col].values
            y_reshaped = y_data.reshape(num_breaths, self.config.SEQ_LEN)

        return x_reshaped, y_reshaped, u_out_reshaped

    def prepare_data(self, load_cached_data=True):
        """
        Main pipeline execution method.
        Checks cache -> Loads or Computes -> Returns arrays.
        """
        # 1. Check Cache
        cache_files = [
            self.config.CACHE_TRAIN_X,
            self.config.CACHE_TRAIN_Y,
            self.config.CACHE_TRAIN_UOUT,
            self.config.CACHE_VAL_X,
            self.config.CACHE_VAL_Y,
            self.config.CACHE_VAL_UOUT,
            self.config.CACHE_TEST_X,
            self.config.CACHE_TEST_UOUT,
        ]

        cache_exists = all(os.path.exists(f) for f in cache_files)

        if load_cached_data and cache_exists:
            print("Loading data from cache...")
            train_x = np.load(self.config.CACHE_TRAIN_X)
            train_y = np.load(self.config.CACHE_TRAIN_Y)
            train_u_out = np.load(self.config.CACHE_TRAIN_UOUT)

            val_x = np.load(self.config.CACHE_VAL_X)
            val_y = np.load(self.config.CACHE_VAL_Y)
            val_u_out = np.load(self.config.CACHE_VAL_UOUT)

            test_x = np.load(self.config.CACHE_TEST_X)
            test_u_out = np.load(self.config.CACHE_TEST_UOUT)

            return (
                (train_x, train_y, train_u_out),
                (val_x, val_y, val_u_out),
                (test_x, None, test_u_out),
            )

        # 2. Process from Scratch
        print("Processing data from scratch...")

        # Load Raw Data
        train_df = pd.read_csv(self.config.TRAIN_PATH)
        val_df = pd.read_csv(self.config.VAL_PATH)
        test_df = pd.read_csv(self.config.TEST_PATH)

        # Debug Mode
        if self.config.DEBUG:
            print(
                f"DEBUG MODE: Truncating data to {self.config.DEBUG_SAMPLES} breaths."
            )
            train_ids = train_df["breath_id"].unique()[: self.config.DEBUG_SAMPLES]
            val_ids = val_df["breath_id"].unique()[: self.config.DEBUG_SAMPLES]
            test_ids = test_df["breath_id"].unique()[: self.config.DEBUG_SAMPLES]

            train_df = train_df[train_df["breath_id"].isin(train_ids)].copy()
            val_df = val_df[val_df["breath_id"].isin(val_ids)].copy()
            test_df = test_df[test_df["breath_id"].isin(test_ids)].copy()

        # Feature Engineering
        print("Generating features...")
        train_df = self._add_features(train_df)
        val_df = self._add_features(val_df)
        test_df = self._add_features(test_df)

        # Scaling
        # We scale continuous features but leave 'u_out' (binary) alone
        scale_cols = [c for c in self.config.FEATURE_COLS if c != "u_out"]
        print(f"Fitting RobustScaler on columns: {scale_cols}")

        scaler = RobustScaler()
        # Fit on Train
        scaler.fit(train_df[scale_cols])

        # Transform all
        train_df[scale_cols] = scaler.transform(train_df[scale_cols])
        val_df[scale_cols] = scaler.transform(val_df[scale_cols])
        test_df[scale_cols] = scaler.transform(test_df[scale_cols])

        # Reshaping
        print("Reshaping to sequences...")
        train_x, train_y, train_u_out = self._reshape_sequences(
            train_df, self.config.FEATURE_COLS, "pressure"
        )
        val_x, val_y, val_u_out = self._reshape_sequences(
            val_df, self.config.FEATURE_COLS, "pressure"
        )
        test_x, _, test_u_out = self._reshape_sequences(
            test_df, self.config.FEATURE_COLS, None
        )

        # Caching
        print(f"Saving processed data to {self.config.WORKING_DIR}...")
        np.save(self.config.CACHE_TRAIN_X, train_x)
        np.save(self.config.CACHE_TRAIN_Y, train_y)
        np.save(self.config.CACHE_TRAIN_UOUT, train_u_out)

        np.save(self.config.CACHE_VAL_X, val_x)
        np.save(self.config.CACHE_VAL_Y, val_y)
        np.save(self.config.CACHE_VAL_UOUT, val_u_out)

        np.save(self.config.CACHE_TEST_X, test_x)
        np.save(self.config.CACHE_TEST_UOUT, test_u_out)

        return (
            (train_x, train_y, train_u_out),
            (val_x, val_y, val_u_out),
            (test_x, None, test_u_out),
        )
