import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import seed_everything


class VentilatorDataset(Dataset):
    def __init__(self, X, y=None, u_out=None):
        """
        Args:
            X (np.ndarray): Input features of shape (num_breaths, seq_len, num_features)
            y (np.ndarray, optional): Target pressure of shape (num_breaths, seq_len).
            u_out (np.ndarray, optional): Expiratory phase flag of shape (num_breaths, seq_len).
        """
        self.X = torch.tensor(X, dtype=torch.float32)

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

        if u_out is not None:
            self.u_out = torch.tensor(u_out, dtype=torch.float32)
        else:
            self.u_out = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = {"X": self.X[idx]}
        if self.y is not None:
            data["y"] = self.y[idx]
        if self.u_out is not None:
            data["u_out"] = self.u_out[idx]
        return data


def add_physics_features(df):
    """
    Adds physics-based features and time-series aggregations.
    Computes time-weighted volume integration, interaction terms, and multi-step deltas.
    """
    # Ensure data is sorted by breath and time
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    # Calculate dt (time difference between steps)
    # We mask the first step of each breath to 0 since diff() would take the prev breath's time
    df["dt"] = df["time_step"].diff()
    df.loc[df["breath_id"] != df["breath_id"].shift(1), "dt"] = 0
    df["dt"] = df["dt"].fillna(0)

    # Integration: Volume = cumsum(u_in * dt)
    # We calculate the area of the current step and cumsum it per breath
    df["area"] = df["u_in"] * df["dt"]
    df["u_in_cumsum"] = df.groupby("breath_id")["area"].cumsum()

    # Explicit Physics Interactions
    # Resistive Pressure component approximation
    df["R_u_in"] = df["R"] * df["u_in"]
    # Elastic Pressure component approximation
    df["u_in_cumsum_div_C"] = df["u_in_cumsum"] / df["C"]

    # Multi-step Deltas (Dynamics)
    # Calculate lag differences for u_in to capture valve acceleration/velocity
    # Restricted to 1st and 2nd order dynamics (Cite Lesson 00066)
    for i in range(1, 3):
        df[f"u_in_diff{i}"] = df.groupby("breath_id")["u_in"].diff(i).fillna(0)

    return df


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, load_cached_data=True, debug=Config.DEBUG
):
    """
    Main entry point to get PyTorch DataLoaders.
    Handles caching, feature engineering, segregated scaling, and reshaping.
    """
    seed_everything(Config.SEED)

    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_data.npz")
    val_cache = os.path.join(cache_dir, "val_data.npz")
    test_cache = os.path.join(cache_dir, "test_data.npz")

    # Check if we can load from cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        print(f"Loading cached data from {cache_dir}...")
        train_data = np.load(train_cache)
        val_data = np.load(val_cache)
        test_data = np.load(test_cache)

        X_train, y_train, u_out_train = (
            train_data["X"],
            train_data["y"],
            train_data["u_out"],
        )
        X_val, y_val, u_out_val = val_data["X"], val_data["y"], val_data["u_out"]
        X_test, u_out_test = test_data["X"], test_data["u_out"]

    else:
        print("Processing data from scratch...")

        # 1. Load Metadata to identify splits
        train_meta = pd.read_csv(Config.TRAIN_METADATA)
        val_meta = pd.read_csv(Config.VAL_METADATA)

        train_breath_ids = train_meta["breath_id"].unique()
        val_breath_ids = val_meta["breath_id"].unique()

        # 2. Load Raw Data
        # Using specific dtypes to optimize memory usage
        dtype_dict = {
            "id": "int32",
            "breath_id": "int32",
            "R": "float32",
            "C": "float32",
            "time_step": "float32",
            "u_in": "float32",
            "u_out": "int8",
            "pressure": "float32",
        }

        df_train_raw = pd.read_csv(Config.TRAIN_CSV, dtype=dtype_dict)
        # Test set does not have pressure column
        test_dtype = {k: v for k, v in dtype_dict.items() if k != "pressure"}
        df_test_raw = pd.read_csv(Config.TEST_CSV, dtype=test_dtype)

        # Debug Mode: Subsample data
        if debug:
            print("Debug mode: Subsampling data...")
            train_breath_ids = train_breath_ids[:100]
            val_breath_ids = val_breath_ids[:50]
            df_train_raw = df_train_raw[
                df_train_raw["breath_id"].isin(
                    np.concatenate([train_breath_ids, val_breath_ids])
                )
            ]
            test_breath_ids = df_test_raw["breath_id"].unique()[:50]
            df_test_raw = df_test_raw[df_test_raw["breath_id"].isin(test_breath_ids)]

        # 3. Feature Engineering
        # Apply physics features to both raw dataframes
        print("Applying physics feature engineering...")
        df_train_raw = add_physics_features(df_train_raw)
        df_test_raw = add_physics_features(df_test_raw)

        # 4. Split Train/Val based on metadata breath_ids
        df_train = df_train_raw[df_train_raw["breath_id"].isin(train_breath_ids)].copy()
        df_val = df_train_raw[df_train_raw["breath_id"].isin(val_breath_ids)].copy()
        df_test = df_test_raw.copy()

        del df_train_raw  # Free memory

        # 5. Segregated Scaling
        # We fit the scaler ONLY on the continuous features of the training set.
        # Binary features (u_out) are left raw to avoid "Silent Peril".
        print("Applying segregated scaling...")
        scaler = RobustScaler()

        # Extract continuous features
        train_cont = df_train[Config.CONTINUOUS_FEATURES].values.astype(np.float32)
        val_cont = df_val[Config.CONTINUOUS_FEATURES].values.astype(np.float32)
        test_cont = df_test[Config.CONTINUOUS_FEATURES].values.astype(np.float32)

        # Fit on Train, Transform All
        scaler.fit(train_cont)
        train_cont = scaler.transform(train_cont)
        val_cont = scaler.transform(val_cont)
        test_cont = scaler.transform(test_cont)

        # Extract Binary features
        train_bin = df_train[Config.BINARY_FEATURES].values.astype(np.float32)
        val_bin = df_val[Config.BINARY_FEATURES].values.astype(np.float32)
        test_bin = df_test[Config.BINARY_FEATURES].values.astype(np.float32)

        # Concatenate Scaled Continuous + Raw Binary
        X_train_flat = np.hstack([train_cont, train_bin])
        X_val_flat = np.hstack([val_cont, val_bin])
        X_test_flat = np.hstack([test_cont, test_bin])

        # Extract Targets and u_out (for loss weighting)
        y_train_flat = df_train[Config.TARGET_COL].values.astype(np.float32)
        y_val_flat = df_val[Config.TARGET_COL].values.astype(np.float32)

        u_out_train_flat = df_train["u_out"].values.astype(np.float32)
        u_out_val_flat = df_val["u_out"].values.astype(np.float32)
        u_out_test_flat = df_test["u_out"].values.astype(np.float32)

        # 6. Reshape to Sequences (N_breaths, 80, Features)
        # The dataset structure is consistently 80 steps per breath.
        SEQ_LEN = 80

        def reshape_seq(flat_data, seq_len):
            num_samples = len(flat_data) // seq_len
            # Ensure we don't have partial breaths (shouldn't happen with this dataset)
            flat_data = flat_data[: num_samples * seq_len]
            if flat_data.ndim == 1:
                return flat_data.reshape(num_samples, seq_len)
            else:
                return flat_data.reshape(num_samples, seq_len, flat_data.shape[1])

        print("Reshaping to sequences...")
        X_train = reshape_seq(X_train_flat, SEQ_LEN)
        y_train = reshape_seq(y_train_flat, SEQ_LEN)
        u_out_train = reshape_seq(u_out_train_flat, SEQ_LEN)

        X_val = reshape_seq(X_val_flat, SEQ_LEN)
        y_val = reshape_seq(y_val_flat, SEQ_LEN)
        u_out_val = reshape_seq(u_out_val_flat, SEQ_LEN)

        X_test = reshape_seq(X_test_flat, SEQ_LEN)
        u_out_test = reshape_seq(u_out_test_flat, SEQ_LEN)

        # 7. Save to Cache
        print(f"Saving processed data to {cache_dir}...")
        np.savez(train_cache, X=X_train, y=y_train, u_out=u_out_train)
        np.savez(val_cache, X=X_val, y=y_val, u_out=u_out_val)
        np.savez(test_cache, X=X_test, u_out=u_out_test)

    # Create Datasets
    train_dataset = VentilatorDataset(X_train, y_train, u_out_train)
    val_dataset = VentilatorDataset(X_val, y_val, u_out_val)
    test_dataset = VentilatorDataset(X_test, u_out=u_out_test)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
