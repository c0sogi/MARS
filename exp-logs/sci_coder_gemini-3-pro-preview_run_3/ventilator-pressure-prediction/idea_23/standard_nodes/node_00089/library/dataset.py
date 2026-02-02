import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import joblib
from library.utils import seed_everything


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Returns sequence data for each breath.
    """

    def __init__(self, X, u_out, y=None):
        self.X = X
        self.u_out = u_out
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Inputs are (Seq_Len, Features)
        data = {
            "x": torch.tensor(self.X[idx], dtype=torch.float32),
            "u_out": torch.tensor(self.u_out[idx], dtype=torch.float32),
        }

        if self.y is not None:
            data["y"] = torch.tensor(self.y[idx], dtype=torch.float32)

        return data


def add_features(df, config):
    """
    Applies feature engineering: dt, area, interactions, and lookahead.
    """
    # Ensure data is sorted by breath and time
    df = df.sort_values(["breath_id", "id"]).reset_index(drop=True)

    # Calculate dt (time delta)
    # Groupby ensures we don't diff across different breaths
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # Calculate Area (Numerical Integration of u_in)
    # area = cumsum(u_in * dt)
    df["area"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()

    # Physics Interactions
    df["R__u_in"] = df["R"] * df["u_in"]
    df["area__C"] = df["area"] / df["C"]

    # Lookahead features for u_in (Next 1 to NUM_LAGS steps)
    # shift(-i) brings future values to current row
    grp = df.groupby("breath_id")["u_in"]
    for i in range(1, config.NUM_LAGS + 1):
        df[f"u_in_next{i}"] = grp.shift(-i).fillna(0)

    # Explicit derivatives (Cite solution_lesson_node_00052)
    # diff1 = next1 - current
    df["u_in_diff1"] = df["u_in_next1"] - df["u_in"]
    # diff_i = next_i - next_{i-1}
    for i in range(2, config.NUM_LAGS + 1):
        df[f"u_in_diff{i}"] = df[f"u_in_next{i}"] - df[f"u_in_next{i-1}"]

    return df


def load_data(path, cache_path, config, load_cached_data=True):
    """
    Loads data from CSV or Parquet cache. Applies feature engineering if not cached.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing...")

    # 2. Process from scratch
    print(f"Processing data from {path}")
    df = pd.read_csv(path)
    df = add_features(df, config)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path)

    return df


def get_data_loaders(config):
    """
    Main pipeline: Load -> Scale -> Reshape -> DataLoader.
    """
    seed_everything(config.SEED)

    # 1. Load Data
    train_df = load_data(config.TRAIN_PATH, config.CACHE_TRAIN, config)
    val_df = load_data(config.VAL_PATH, config.CACHE_VAL, config)
    test_df = load_data(config.TEST_PATH, config.CACHE_TEST, config)

    # 2. Scaling (RobustScaler)
    scaler = RobustScaler()
    feature_cols = config.FEATURE_COLS

    print("Fitting scaler on training data...")
    scaler.fit(train_df[feature_cols])

    # Save scaler for inference/reproducibility
    joblib.dump(scaler, config.SCALER_PATH)

    print("Transforming datasets...")
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # 3. Reshape to Sequences (N_breaths, 80, N_features)
    # The dataset guarantees 80 time steps per breath
    SEQ_LEN = 80

    def reshape_data(df, is_test=False):
        # Extract arrays
        X_flat = df[feature_cols].values
        u_out_flat = df["u_out"].values

        # Calculate number of breaths
        num_breaths = len(df) // SEQ_LEN

        # Reshape
        X = X_flat.reshape(num_breaths, SEQ_LEN, -1)
        u_out = u_out_flat.reshape(num_breaths, SEQ_LEN)

        y = None
        if not is_test:
            y_flat = df["pressure"].values
            y = y_flat.reshape(num_breaths, SEQ_LEN)

        return X, u_out, y

    X_train, u_out_train, y_train = reshape_data(train_df)
    X_val, u_out_val, y_val = reshape_data(val_df)
    X_test, u_out_test, _ = reshape_data(test_df, is_test=True)

    # 4. Debug Subsampling
    if config.DEBUG:
        print(f"Debug Mode: Subsampling to {config.DEBUG_BREATH_COUNT} breaths.")
        limit = config.DEBUG_BREATH_COUNT
        X_train, u_out_train, y_train = (
            X_train[:limit],
            u_out_train[:limit],
            y_train[:limit],
        )
        X_val, u_out_val, y_val = X_val[:limit], u_out_val[:limit], y_val[:limit]
        X_test, u_out_test = X_test[:limit], u_out_test[:limit]

    # 5. Create Datasets
    train_dataset = VentilatorDataset(X_train, u_out_train, y_train)
    val_dataset = VentilatorDataset(X_val, u_out_val, y_val)
    test_dataset = VentilatorDataset(X_test, u_out_test)

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Important for BN stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"Data Loaders ready. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
