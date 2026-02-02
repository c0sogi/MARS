import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import seed_everything

# Set random seeds for reproducibility
seed_everything()


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Serves sequences of length 80.
    """

    def __init__(self, X, u_out, y=None):
        """
        Args:
            X (np.ndarray): Input features of shape (num_breaths, 80, num_features)
            u_out (np.ndarray): Control input u_out of shape (num_breaths, 80)
            y (np.ndarray, optional): Target pressure of shape (num_breaths, 80)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.u_out = torch.tensor(u_out, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns:
            X: Input features for the sequence.
            y: Target pressure (or zeros if test).
            u_out: Exploratory valve status (used for metric masking).
        """
        if self.y is not None:
            return self.X[idx], self.y[idx], self.u_out[idx]
        else:
            # Return dummy target for test set
            return self.X[idx], torch.zeros_like(self.u_out[idx]), self.u_out[idx]


def engineer_features(df):
    """
    Computes physics-based features and lookaheads.

    Features computed:
    - dt: Time difference between steps.
    - area: Cumulative integral of u_in * dt (Volume).
    - u_in_diff: Derivative of u_in.
    - R_u_in, area_C: Interaction terms.
    - u_in_next1..4: Future values of u_in.
    """
    # 1. Time Delta (dt)
    # GroupBy is used to ensure diffs don't cross breath boundaries
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # 2. Volume (Area) Integration
    # area = integral(u_in * dt)
    # We calculate the incremental volume first, then cumsum
    df["_vol_chunk"] = df["u_in"] * df["dt"]
    df["area"] = df.groupby("breath_id")["_vol_chunk"].cumsum()
    df.drop(columns=["_vol_chunk"], inplace=True)

    # 3. Derivative (u_in_diff)
    df["u_in_diff"] = df.groupby("breath_id")["u_in"].diff().fillna(0)

    # 4. Interaction Terms
    df["R_u_in"] = df["R"] * df["u_in"]
    df["area_C"] = df["area"] / df["C"]

    # 5. Lookahead Features (u_in_next1 ... u_in_next4)
    # Provides the model with non-causal future context
    grp = df.groupby("breath_id")["u_in"]
    for i in range(1, 5):
        df[f"u_in_next{i}"] = grp.shift(-i).fillna(0)

    return df


def prepare_datasets(load_cached_data=True):
    """
    Loads data, performs feature engineering, scaling, and reshaping.
    Manages caching of processed numpy arrays to speed up experiments.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        train_dataset (VentilatorDataset)
        val_dataset (VentilatorDataset)
        test_dataset (VentilatorDataset)
        scaler (RobustScaler)
    """

    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "train_x": os.path.join(cache_dir, "train_x.npy"),
        "train_y": os.path.join(cache_dir, "train_y.npy"),
        "train_u_out": os.path.join(cache_dir, "train_u_out.npy"),
        "val_x": os.path.join(cache_dir, "val_x.npy"),
        "val_y": os.path.join(cache_dir, "val_y.npy"),
        "val_u_out": os.path.join(cache_dir, "val_u_out.npy"),
        "test_x": os.path.join(cache_dir, "test_x.npy"),
        "test_u_out": os.path.join(cache_dir, "test_u_out.npy"),
        "scaler": os.path.join(cache_dir, "scaler.joblib"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(f) for f in files.values())

    if load_cached_data and cache_exists:
        print(f"Loading cached datasets from {cache_dir}...")
        train_x = np.load(files["train_x"])
        train_y = np.load(files["train_y"])
        train_u_out = np.load(files["train_u_out"])

        val_x = np.load(files["val_x"])
        val_y = np.load(files["val_y"])
        val_u_out = np.load(files["val_u_out"])

        test_x = np.load(files["test_x"])
        test_u_out = np.load(files["test_u_out"])

        scaler = joblib.load(files["scaler"])

    else:
        print("Processing data from scratch...")

        # Load Metadata
        print("Loading CSV files...")
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Feature Engineering
        print("Engineering features...")
        train_df = engineer_features(train_df)
        val_df = engineer_features(val_df)
        test_df = engineer_features(test_df)

        # Select Features
        feature_cols = Config.FEATURE_COLS

        # Scaling
        print("Fitting RobustScaler...")
        scaler = RobustScaler()
        # Fit on train only to prevent leakage
        scaler.fit(train_df[feature_cols])

        # Transform
        print("Transforming data...")
        train_feats = scaler.transform(train_df[feature_cols])
        val_feats = scaler.transform(val_df[feature_cols])
        test_feats = scaler.transform(test_df[feature_cols])

        # Reshaping
        # Data is (N_samples, N_features). Needs to be (N_breaths, 80, N_features)
        # We assume dataset rows are multiples of 80 and contiguous based on dataset specs

        def reshape_data(feats, df, target_col="pressure"):
            # Calculate number of breaths
            n_breaths = len(df) // 80
            n_feats = feats.shape[1]

            # Reshape features
            x_reshaped = feats.reshape(n_breaths, 80, n_feats)

            # Reshape u_out (needed for metric masking)
            u_out_reshaped = df["u_out"].values.reshape(n_breaths, 80)

            # Reshape target if it exists
            if target_col in df.columns:
                y_reshaped = df[target_col].values.reshape(n_breaths, 80)
            else:
                y_reshaped = None

            return x_reshaped, u_out_reshaped, y_reshaped

        print("Reshaping tensors...")
        train_x, train_u_out, train_y = reshape_data(train_feats, train_df)
        val_x, val_u_out, val_y = reshape_data(val_feats, val_df)
        test_x, test_u_out, _ = reshape_data(test_feats, test_df)  # No target for test

        # Save to cache
        print("Saving processed data to cache...")
        np.save(files["train_x"], train_x)
        np.save(files["train_y"], train_y)
        np.save(files["train_u_out"], train_u_out)

        np.save(files["val_x"], val_x)
        np.save(files["val_y"], val_y)
        np.save(files["val_u_out"], val_u_out)

        np.save(files["test_x"], test_x)
        np.save(files["test_u_out"], test_u_out)

        joblib.dump(scaler, files["scaler"])

    # Create Datasets
    print("Creating Dataset objects...")
    train_dataset = VentilatorDataset(train_x, train_u_out, train_y)
    val_dataset = VentilatorDataset(val_x, val_u_out, val_y)
    test_dataset = VentilatorDataset(test_x, test_u_out, None)

    return train_dataset, val_dataset, test_dataset, scaler
