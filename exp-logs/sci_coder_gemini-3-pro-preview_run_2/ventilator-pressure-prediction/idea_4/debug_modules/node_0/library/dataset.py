import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.utils import seed_everything


class VentilatorDataset(Dataset):
    def __init__(self, X, y=None, u_out=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.u_out = (
            torch.tensor(u_out, dtype=torch.float32) if u_out is not None else None
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = {"X": self.X[idx]}
        if self.y is not None:
            data["y"] = self.y[idx]
        if self.u_out is not None:
            data["u_out"] = self.u_out[idx]
        return data


def engineer_features(df):
    """
    Applies physics-informed feature engineering to the dataframe.
    """
    # Ensure data is sorted by breath and time
    df = df.sort_values(["breath_id", "time_step"])

    # Create a groupby object for efficiency
    grp = df.groupby("breath_id")

    # Time delta
    df["dt"] = grp["time_step"].diff().fillna(0)

    # Lag features (1-4 steps)
    for i in range(1, 5):
        df[f"u_in_lag{i}"] = grp["u_in"].shift(i).fillna(0)

    # Finite differences (derivatives of u_in)
    df["u_in_diff1"] = grp["u_in"].diff().fillna(0)
    df["u_in_diff2"] = grp["u_in_diff1"].diff().fillna(0)

    # Cumulative Volume (Integral of flow over time)
    # u_in is a proxy for flow, dt is time.
    df["u_in_cumsum"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()

    # Physics-based Interaction Terms
    # Resistive Pressure ~ R * Flow
    df["R_u_in"] = df["R"] * df["u_in"]
    # Elastic Pressure ~ Volume / Compliance
    df["vol_C"] = df["u_in_cumsum"] / df["C"]

    # Define feature columns to include in X
    feature_cols = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",
        "u_in_diff1",
        "u_in_diff2",
        "u_in_cumsum",
        "R_u_in",
        "vol_C",
    ]

    return df, feature_cols


def get_dataloaders(
    data_dir="./input",
    batch_size=128,
    num_workers=4,
    load_cached_data=True,
    cache_dir="./working/idea_4/",
):
    """
    Loads data, performs feature engineering, and returns PyTorch DataLoaders.
    Implements caching to speed up subsequent runs.
    """
    seed_everything(42)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "processed_data.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        data = np.load(cache_path)
        X_train, y_train, u_out_train = (
            data["X_train"],
            data["y_train"],
            data["u_out_train"],
        )
        X_val, y_val, u_out_val = data["X_val"], data["y_val"], data["u_out_val"]
        X_test, u_out_test = data["X_test"], data["u_out_test"]
    else:
        print("Processing data from scratch...")

        # Load Raw Data
        train_path = os.path.join(data_dir, "train.csv")
        test_path = os.path.join(data_dir, "test.csv")

        df_train_raw = pd.read_csv(train_path)
        df_test_raw = pd.read_csv(test_path)

        # Load Metadata for splitting
        train_meta = pd.read_csv("./metadata/train_metadata.csv")
        val_meta = pd.read_csv("./metadata/val_metadata.csv")

        train_breaths = set(train_meta["breath_id"].unique())
        val_breaths = set(val_meta["breath_id"].unique())

        # Feature Engineering
        print("Engineering features for training data...")
        df_train_eng, features = engineer_features(df_train_raw)

        print("Engineering features for test data...")
        df_test_eng, _ = engineer_features(df_test_raw)

        # Split Train and Val based on breath_ids
        df_train = df_train_eng[df_train_eng["breath_id"].isin(train_breaths)].copy()
        df_val = df_train_eng[df_train_eng["breath_id"].isin(val_breaths)].copy()

        # Reshape Data to (N_breaths, 80, N_features)
        SEQ_LEN = 80

        def reshape_to_sequence(df, feats, target_col=None):
            # Ensure strict sorting
            df = df.sort_values(["breath_id", "time_step"])

            num_breaths = len(df) // SEQ_LEN
            # Integrity check
            if len(df) % SEQ_LEN != 0:
                print(
                    f"Warning: Data length {len(df)} not divisible by {SEQ_LEN}. Truncating."
                )
                num_breaths = len(df) // SEQ_LEN
                df = df.iloc[: num_breaths * SEQ_LEN]

            X = df[feats].values.reshape(num_breaths, SEQ_LEN, len(feats))
            u_out = df["u_out"].values.reshape(num_breaths, SEQ_LEN)

            y = None
            if target_col and target_col in df.columns:
                y = df[target_col].values.reshape(num_breaths, SEQ_LEN)

            return X, y, u_out

        print("Reshaping tensors...")
        X_train, y_train, u_out_train = reshape_to_sequence(
            df_train, features, "pressure"
        )
        X_val, y_val, u_out_val = reshape_to_sequence(df_val, features, "pressure")
        X_test, _, u_out_test = reshape_to_sequence(df_test_eng, features)

        # Normalization (RobustScaler)
        print("Fitting Scaler...")
        scaler = RobustScaler()

        # Flatten train X to fit scaler
        N_t, L, F = X_train.shape
        X_train_flat = X_train.reshape(-1, F)
        X_train_flat = scaler.fit_transform(X_train_flat)
        X_train = X_train_flat.reshape(N_t, L, F)

        # Transform Val
        N_v, L, F = X_val.shape
        X_val = scaler.transform(X_val.reshape(-1, F)).reshape(N_v, L, F)

        # Transform Test
        N_test, L, F = X_test.shape
        X_test = scaler.transform(X_test.reshape(-1, F)).reshape(N_test, L, F)

        # Save to cache
        print(f"Saving processed data to {cache_path}...")
        np.savez(
            cache_path,
            X_train=X_train,
            y_train=y_train,
            u_out_train=u_out_train,
            X_val=X_val,
            y_val=y_val,
            u_out_val=u_out_val,
            X_test=X_test,
            u_out_test=u_out_test,
        )

    # Create Datasets
    train_dataset = VentilatorDataset(X_train, y_train, u_out_train)
    val_dataset = VentilatorDataset(X_val, y_val, u_out_val)
    test_dataset = VentilatorDataset(X_test, None, u_out_test)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
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
