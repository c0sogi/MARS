import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import seed_everything


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Serves sequences of shape (80, Input_Dim).
    """

    def __init__(self, X, y=None, is_test=False):
        self.X = X
        self.y = y
        self.is_test = is_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Load features as float32
        x = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.is_test:
            return x

        # Load targets
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


def add_features(df):
    """
    Implements the Numerical Integration and Feature Engineering pipeline.
    Calculates dt, Area (Volume), and Physics Interactions.
    """
    # Ensure data is sorted by breath and time to guarantee correct diff/cumsum
    df = df.sort_values(["breath_id", "time_step"])

    # 1. Calculate Time Delta (dt)
    # We group by breath_id to avoid diffing across breath boundaries
    # shift(1) gives the previous time_step.
    # For the first step of a breath, we assume dt=0 (or small epsilon, but 0 is safe for integration start)
    df["time_prev"] = df.groupby("breath_id")["time_step"].shift(1)
    df["dt"] = df["time_step"] - df["time_prev"]
    df["dt"] = df["dt"].fillna(0)  # First step has no prev, so dt=0

    # 2. Calculate Accurate Volume (Area) via Numerical Integration
    # Area = Cumulative Sum of (Flow * dt)
    df["area_term"] = df["u_in"] * df["dt"]
    df["Area"] = df.groupby("breath_id")["area_term"].cumsum()

    # 3. Derivative of Control Input (Acceleration)
    df["u_in_diff"] = df.groupby("breath_id")["u_in"].diff().fillna(0)

    # 4. Physics Interaction Terms
    df["R_u_in"] = df["R"] * df["u_in"]  # Resistive Pressure Proxy
    df["Area_C"] = df["Area"] / df["C"]  # Elastic Pressure Proxy

    # Cleanup intermediate columns
    df.drop(columns=["time_prev", "area_term"], inplace=True)

    return df


def prepare_datasets(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Loads data, performs feature engineering, scaling, and creates DataLoaders.
    Manages caching of processed numpy arrays to speed up restart.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # --- Cache Invalidation ---
    # If we force reload, we delete existing cache files to ensure fresh processing
    if not load_cached_data:
        print("Invalidating cache... Deleting stale .npy files.")
        cache_files = [
            Config.TRAIN_CACHE_X,
            Config.TRAIN_CACHE_Y,
            Config.VAL_CACHE_X,
            Config.VAL_CACHE_Y,
            Config.TEST_CACHE_X,
            Config.TEST_IDS,
            Config.SCALER_STATS,
        ]
        for f in cache_files:
            if os.path.exists(f):
                os.remove(f)

    # --- Train & Validation Processing ---
    if not os.path.exists(Config.TRAIN_CACHE_X) or not os.path.exists(
        Config.VAL_CACHE_X
    ):
        print("Processing Train and Validation data from scratch...")

        # Load Metadata CSVs
        train_df = pd.read_csv(Config.TRAIN_CSV)
        val_df = pd.read_csv(Config.VAL_CSV)

        # Apply Feature Engineering
        print("Applying Numerical Integration and Feature Engineering...")
        train_df = add_features(train_df)
        val_df = add_features(val_df)

        # Select Model Features (Excluding time_step as per Lesson 47)
        features = Config.MODEL_FEATURES

        # Prepare for Scaling
        print("Fitting RobustScaler (excluding u_out)...")

        # Identify features to scale
        scale_features = [f for f in features if f != "u_out"]
        u_out_idx = features.index("u_out")

        scaler = RobustScaler()
        train_vals_scale = train_df[scale_features].values.astype(np.float32)

        # Fit on Train only
        scaler.fit(train_vals_scale)

        # Save Scaler Statistics (Center and Scale) for Inference
        np.savez(Config.SCALER_STATS, center=scaler.center_, scale=scaler.scale_)

        def transform_and_reassemble(df):
            # Scale continuous features
            vals = df[scale_features].values.astype(np.float32)
            vals_scaled = scaler.transform(vals)

            # Get u_out
            u_out_vals = df["u_out"].values.astype(np.float32).reshape(-1, 1)

            # Reassemble
            return np.hstack(
                [
                    vals_scaled[:, :u_out_idx],
                    u_out_vals,
                    vals_scaled[:, u_out_idx:],
                ]
            )

        # Transform Train and Val
        train_vals = transform_and_reassemble(train_df)
        val_vals = transform_and_reassemble(val_df)

        # Reshape to Sequences: (N_breaths, 80, N_features)
        # We assume the data is complete with 80 steps per breath
        train_x = train_vals.reshape(-1, 80, len(features))
        val_x = val_vals.reshape(-1, 80, len(features))

        # Process Targets (Pressure)
        train_y = train_df["pressure"].values.astype(np.float32).reshape(-1, 80)
        val_y = val_df["pressure"].values.astype(np.float32).reshape(-1, 80)

        # Save to Cache
        print("Saving processed arrays to cache...")
        np.save(Config.TRAIN_CACHE_X, train_x)
        np.save(Config.TRAIN_CACHE_Y, train_y)
        np.save(Config.VAL_CACHE_X, val_x)
        np.save(Config.VAL_CACHE_Y, val_y)

    else:
        print("Loading Train/Val data from cache...")
        train_x = np.load(Config.TRAIN_CACHE_X)
        train_y = np.load(Config.TRAIN_CACHE_Y)
        val_x = np.load(Config.VAL_CACHE_X)
        val_y = np.load(Config.VAL_CACHE_Y)

    # --- Test Processing ---
    if not os.path.exists(Config.TEST_CACHE_X):
        print("Processing Test data from scratch...")
        test_df = pd.read_csv(Config.TEST_CSV)

        # Apply Feature Engineering
        test_df = add_features(test_df)

        # Load Scaler Stats
        if not os.path.exists(Config.SCALER_STATS):
            raise FileNotFoundError("Scaler stats not found. Process train data first.")

        stats = np.load(Config.SCALER_STATS)
        center = stats["center"]
        scale = stats["scale"]

        # Identify features
        features = Config.MODEL_FEATURES
        scale_features = [f for f in features if f != "u_out"]
        u_out_idx = features.index("u_out")

        # Manual Transform (X - Center) / Scale for continuous features
        test_vals_scale = test_df[scale_features].values.astype(np.float32)
        test_vals_scaled = (test_vals_scale - center) / scale

        # Reassemble with u_out
        u_out_vals = test_df["u_out"].values.astype(np.float32).reshape(-1, 1)
        test_vals = np.hstack(
            [
                test_vals_scaled[:, :u_out_idx],
                u_out_vals,
                test_vals_scaled[:, u_out_idx:],
            ]
        )

        # Reshape
        test_x = test_vals.reshape(-1, 80, len(features))

        # Save IDs for submission alignment
        test_ids = test_df["id"].values.astype(np.int32)

        # Save to Cache
        np.save(Config.TEST_CACHE_X, test_x)
        np.save(Config.TEST_IDS, test_ids)

    else:
        print("Loading Test data from cache...")
        test_x = np.load(Config.TEST_CACHE_X)
        # test_ids are loaded by the submission script, not needed in loader

    # --- Create DataLoaders ---
    print(
        f"Data Shapes - Train: {train_x.shape}, Val: {val_x.shape}, Test: {test_x.shape}"
    )

    train_dataset = VentilatorDataset(train_x, train_y)
    val_dataset = VentilatorDataset(val_x, val_y)
    test_dataset = VentilatorDataset(test_x, None, is_test=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
