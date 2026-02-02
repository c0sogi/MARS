import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import set_seed

# Ensure reproducibility
set_seed(Config.SEED)


class PreProcessor:
    """
    Handles feature engineering and segregated scaling for the ventilator data.
    Implements the Physics-Fidelity Pipeline.
    """

    def __init__(self):
        self.scaler = RobustScaler()
        self.fitted = False

    def add_physics_features(self, df):
        """
        Adds physics-based features including time-weighted volume integration
        and interaction terms (R*Flow, Vol/C).
        """
        # Ensure data is sorted for correct diff/cumsum
        df = df.sort_values(["breath_id", "time_step"])

        # Calculate dt (time delta)
        # Group by breath_id to avoid diffing across breaths
        df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

        # Time-weighted integration for volume: cumsum(u_in * dt)
        # Note: u_in is 0-100, we treat it as flow rate
        df["volume_chunk"] = df["u_in"] * df["dt"]
        df["u_in_cumsum"] = df.groupby("breath_id")["volume_chunk"].cumsum()

        # Physics Interaction Terms
        # Resistive Pressure ~ R * Flow
        df["R_u_in"] = df["R"] * df["u_in"]

        # Elastic Pressure ~ Volume / Compliance
        # Add epsilon to C to avoid potential division by zero (though C is 10-50)
        df["vol_div_C"] = df["u_in_cumsum"] / df["C"]

        # Cleanup intermediate column
        df = df.drop(columns=["volume_chunk", "dt"])
        return df

    def add_dynamic_features(self, df):
        """
        Adds temporal dynamic features: Lags and Finite Differences.
        Strictly excludes future lags to prevent leakage.
        """
        # Group object for efficiency
        g = df.groupby("breath_id")

        # Past Lags (1 to 2)
        for lag in range(1, 3):
            df[f"u_in_lag{lag}"] = g["u_in"].shift(lag).fillna(0)

        # Finite Differences (1st Derivative)
        df["u_in_diff1"] = g["u_in"].diff().fillna(0)
        df["u_in_diff2"] = g["u_in_diff1"].diff().fillna(0)  # Acceleration

        return df

    def process_features(self, df):
        """
        Orchestrates the feature engineering pipeline.
        """
        df = self.add_physics_features(df)
        df = self.add_dynamic_features(df)
        return df

    def fit_scaler(self, df):
        """
        Fits the RobustScaler on continuous features only.
        """
        if Config.CONT_FEATURES:
            self.scaler.fit(df[Config.CONT_FEATURES])
            self.fitted = True

            # Save scaler parameters manually to NPZ
            os.makedirs(os.path.dirname(Config.SCALER_PATH), exist_ok=True)
            np.savez(
                Config.SCALER_PATH, center=self.scaler.center_, scale=self.scaler.scale_
            )

    def load_scaler(self):
        """
        Loads scaler parameters from cache.
        """
        if os.path.exists(Config.SCALER_PATH):
            data = np.load(Config.SCALER_PATH)
            self.scaler.center_ = data["center"]
            self.scaler.scale_ = data["scale"]
            self.fitted = True
        else:
            raise FileNotFoundError(f"Scaler params not found at {Config.SCALER_PATH}")

    def transform(self, df):
        """
        Applies Segregated Scaling:
        - Continuous features: Scaled via RobustScaler
        - Binary features: Passed through Raw
        """
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted or loaded before transform.")

        df_out = df.copy()

        # Scale continuous features
        if Config.CONT_FEATURES:
            df_out[Config.CONT_FEATURES] = self.scaler.transform(
                df[Config.CONT_FEATURES]
            )

        # Binary features are left as is (already 0/1)

        return df_out


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    """

    def __init__(self, X, u_out, y=None):
        """
        Args:
            X (np.ndarray): Input features of shape (Num_Breaths, Seq_Len, Num_Features)
            u_out (np.ndarray): Binary control input of shape (Num_Breaths, Seq_Len) or (..., 1)
            y (np.ndarray, optional): Targets of shape (Num_Breaths, Seq_Len). Defaults to None.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.u_out = torch.tensor(u_out, dtype=torch.float32)

        # Ensure u_out is shaped correctly for broadcasting/loss calculation if needed
        if self.u_out.ndim == 2:
            self.u_out = self.u_out.unsqueeze(-1)

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.u_out[idx], self.y[idx]
        else:
            return self.X[idx], self.u_out[idx]


def prepare_data(batch_size=Config.BATCH_SIZE, load_cached_data=True, debug=False):
    """
    Main function to load, process, split, and batch the data.

    Args:
        batch_size (int): Batch size for DataLoaders.
        load_cached_data (bool): Whether to try loading from parquet cache.
        debug (bool): If True, uses a small subset of data for quick testing.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
               test_ids is a DataFrame containing IDs for submission mapping.
    """

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Check Cache
    cache_exists = (
        os.path.exists(Config.CACHE_TRAIN_PATH)
        and os.path.exists(Config.CACHE_VAL_PATH)
        and os.path.exists(Config.CACHE_TEST_PATH)
        and os.path.exists(Config.SCALER_PATH)
    )

    if load_cached_data and cache_exists and not debug:
        print("Loading processed data from cache...")
        train_df = pd.read_parquet(Config.CACHE_TRAIN_PATH)
        val_df = pd.read_parquet(Config.CACHE_VAL_PATH)
        test_df = pd.read_parquet(Config.CACHE_TEST_PATH)

        # Load scaler just to ensure state consistency, though data is already scaled
        preprocessor = PreProcessor()
        preprocessor.load_scaler()

    else:
        print("Processing data from scratch...")

        # Load Metadata
        train_meta = pd.read_csv(Config.TRAIN_META_PATH)
        val_meta = pd.read_csv(Config.VAL_META_PATH)

        # Load Raw Data
        # We load the full train.csv and then split based on metadata breath_ids
        raw_train = pd.read_csv(Config.TRAIN_DATA_PATH)
        raw_test = pd.read_csv(Config.TEST_DATA_PATH)

        if debug:
            print("Debug mode: Sampling data...")
            # Sample breaths
            train_breaths = train_meta["breath_id"].unique()[:100]
            val_breaths = val_meta["breath_id"].unique()[:20]
            test_breaths = raw_test["breath_id"].unique()[:20]

            raw_train = raw_train[
                raw_train["breath_id"].isin(
                    np.concatenate([train_breaths, val_breaths])
                )
            ]
            raw_test = raw_test[raw_test["breath_id"].isin(test_breaths)]

            train_meta = train_meta[train_meta["breath_id"].isin(train_breaths)]
            val_meta = val_meta[val_meta["breath_id"].isin(val_breaths)]

        # Identify Split IDs
        train_breath_ids = set(train_meta["breath_id"].unique())
        val_breath_ids = set(val_meta["breath_id"].unique())

        # Split Raw Train into Train/Val
        df_train = raw_train[raw_train["breath_id"].isin(train_breath_ids)].copy()
        df_val = raw_train[raw_train["breath_id"].isin(val_breath_ids)].copy()
        df_test = raw_test.copy()

        del raw_train  # Free memory

        # Feature Engineering
        print("Generating features...")
        preprocessor = PreProcessor()

        df_train = preprocessor.process_features(df_train)
        df_val = preprocessor.process_features(df_val)
        df_test = preprocessor.process_features(df_test)

        # Scaling
        print("Fitting and applying scaler...")
        preprocessor.fit_scaler(df_train)

        train_df = preprocessor.transform(df_train)
        val_df = preprocessor.transform(df_val)
        test_df = preprocessor.transform(df_test)

        # Caching
        if not debug:
            print("Caching processed data...")
            train_df.to_parquet(Config.CACHE_TRAIN_PATH)
            val_df.to_parquet(Config.CACHE_VAL_PATH)
            test_df.to_parquet(Config.CACHE_TEST_PATH)

    # 2. Reshape to Sequences (N, 80, Features)
    print("Reshaping data for LSTM...")

    feature_cols = Config.CONT_FEATURES + Config.BINARY_FEATURES

    def reshape_data(df, is_test=False):
        # Ensure sorting
        df = df.sort_values(["breath_id", "time_step"])

        num_breaths = df["breath_id"].nunique()
        seq_len = Config.SEQ_LEN

        # Extract features
        X = df[feature_cols].values.reshape(num_breaths, seq_len, -1)

        # Extract u_out for loss weighting
        u_out = df["u_out"].values.reshape(num_breaths, seq_len)

        if not is_test:
            y = df["pressure"].values.reshape(num_breaths, seq_len)
            return X, u_out, y
        else:
            # For test, we might need IDs for submission mapping
            ids = df["id"].values  # Flat array
            return X, u_out, None, ids

    X_train, u_out_train, y_train = reshape_data(train_df)
    X_val, u_out_val, y_val = reshape_data(val_df)
    X_test, u_out_test, _, test_ids_flat = reshape_data(test_df, is_test=True)

    # 3. Create Datasets and Loaders
    train_dataset = VentilatorDataset(X_train, u_out_train, y_train)
    val_dataset = VentilatorDataset(X_val, u_out_val, y_val)
    test_dataset = VentilatorDataset(X_test, u_out_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    print(
        f"Data prepared: Train({len(train_dataset)}), Val({len(val_dataset)}), Test({len(test_dataset)})"
    )

    return train_loader, val_loader, test_loader, test_ids_flat
