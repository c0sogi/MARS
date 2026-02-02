import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Holds preprocessed time-series data.
    """

    def __init__(self, X, y=None, is_test=False):
        self.X = X
        self.y = y
        self.is_test = is_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X shape: (Seq_Len, Num_Features)
        # y shape: (Seq_Len,)
        x_item = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.is_test:
            return x_item

        y_item = torch.tensor(self.y[idx], dtype=torch.float32)
        return x_item, y_item


def engineer_features(df):
    """
    Generates physical and temporal features for the ventilator dataset.

    Args:
        df (pd.DataFrame): Raw dataframe containing breath_id, time_step, u_in, u_out, R, C.

    Returns:
        pd.DataFrame: Dataframe with engineered features selected in Config.FEATURE_COLS.
    """
    # Ensure data is sorted
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    # Group by breath_id for window operations
    # Using transform/groupby is safer than assuming strict 80-step blocks for all logic,
    # though the dataset is regular.

    # 1. Temporal derivatives
    # dt: time delta
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # 2. Physical Integration (Volume approximation)
    # area = integral(u_in * dt)
    df["area"] = (
        df.groupby("breath_id")
        .apply(lambda x: (x["u_in"] * x["dt"]).cumsum())
        .reset_index(level=0, drop=True)
    )

    # 3. Derivatives of control input
    df["u_in_diff"] = df.groupby("breath_id")["u_in"].diff().fillna(0)

    # 4. Interaction Terms
    df["R_u_in"] = df["R"] * df["u_in"]
    df["area_C"] = df["area"] / df["C"]

    # 5. Lookahead Features (Future context)
    # We use shift(-k) to look forward. Fill NaNs with 0 (assumption: 0 flow after breath ends).
    grp = df.groupby("breath_id")["u_in"]
    df["u_in_next1"] = grp.shift(-1).fillna(0)
    df["u_in_next2"] = grp.shift(-2).fillna(0)
    df["u_in_next3"] = grp.shift(-3).fillna(0)
    df["u_in_next4"] = grp.shift(-4).fillna(0)

    # Lookahead derivative
    df["u_in_diff_next1"] = df.groupby("breath_id")["u_in_diff"].shift(-1).fillna(0)

    # Select only the features defined in Config
    # We must ensure the order matches Config.FEATURE_COLS
    # Note: Target 'pressure' and IDs are not included in X, handled separately.

    # Check if all required columns exist
    missing_cols = [c for c in Config.FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing engineered features: {missing_cols}")

    return df


def get_data_loaders(load_cached_data=True, debug=False):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.
    Handles caching, feature engineering, scaling, and reshaping.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.
        debug (bool): If True, subsets the data for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "train_x": os.path.join(cache_dir, "train_x.npy"),
        "train_y": os.path.join(cache_dir, "train_y.npy"),
        "val_x": os.path.join(cache_dir, "val_x.npy"),
        "val_y": os.path.join(cache_dir, "val_y.npy"),
        "test_x": os.path.join(cache_dir, "test_x.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(p) for p in files.values())

    if load_cached_data and cache_exists:
        print("Loading cached data from", cache_dir)
        train_x = np.load(files["train_x"])
        train_y = np.load(files["train_y"])
        val_x = np.load(files["val_x"])
        val_y = np.load(files["val_y"])
        test_x = np.load(files["test_x"])
        test_ids = np.load(files["test_ids"])

    else:
        print("Processing data from scratch...")

        # Load Metadata
        train_df = pd.read_csv(Config.TRAIN_CSV)
        val_df = pd.read_csv(Config.VAL_CSV)
        test_df = pd.read_csv(Config.TEST_CSV)

        if debug:
            # Subset by breath_id to maintain sequence integrity
            train_breaths = train_df["breath_id"].unique()[:100]
            val_breaths = val_df["breath_id"].unique()[:50]
            test_breaths = test_df["breath_id"].unique()[:50]

            train_df = train_df[train_df["breath_id"].isin(train_breaths)].copy()
            val_df = val_df[val_df["breath_id"].isin(val_breaths)].copy()
            test_df = test_df[test_df["breath_id"].isin(test_breaths)].copy()

        # Save Test IDs for submission
        # We need one ID per row in the final submission, but here we process by breath.
        # The submission format requires 'id' and 'pressure'.
        # We will store all 'id's to reconstruct the submission later.
        test_ids = test_df[Config.ID_COL].values

        # Extract Targets before feature engineering (as FE might drop non-feature cols)
        train_y_raw = train_df[Config.TARGET_COL].values
        val_y_raw = val_df[Config.TARGET_COL].values

        # Feature Engineering
        print("Engineering features...")
        train_df = engineer_features(train_df)
        val_df = engineer_features(val_df)
        test_df = engineer_features(test_df)

        # Select Features
        X_train = train_df[Config.FEATURE_COLS].values
        X_val = val_df[Config.FEATURE_COLS].values
        X_test = test_df[Config.FEATURE_COLS].values

        # Scaling
        # Fit RobustScaler on Train, transform all
        print("Scaling features...")
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Reshaping to (Num_Breaths, Seq_Len, Num_Features)
        # We assume strict 80 steps per breath as per dataset description
        seq_len = Config.SEQ_LEN

        # Helper to reshape
        def reshape_seq(data, seq_len):
            num_samples = data.shape[0]
            if num_samples % seq_len != 0:
                raise ValueError(
                    f"Data length {num_samples} not divisible by seq_len {seq_len}"
                )
            return data.reshape(-1, seq_len, data.shape[-1])

        def reshape_target(data, seq_len):
            num_samples = data.shape[0]
            return data.reshape(-1, seq_len)

        print("Reshaping tensors...")
        train_x = reshape_seq(X_train, seq_len)
        train_y = reshape_target(train_y_raw, seq_len)

        val_x = reshape_seq(X_val, seq_len)
        val_y = reshape_target(val_y_raw, seq_len)

        test_x = reshape_seq(X_test, seq_len)

        # Save to cache
        print("Saving to cache...")
        np.save(files["train_x"], train_x)
        np.save(files["train_y"], train_y)
        np.save(files["val_x"], val_x)
        np.save(files["val_y"], val_y)
        np.save(files["test_x"], test_x)
        np.save(files["test_ids"], test_ids)

    # Convert to PyTorch Datasets
    train_dataset = VentilatorDataset(train_x, train_y)
    val_dataset = VentilatorDataset(val_x, val_y)
    test_dataset = VentilatorDataset(test_x, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"Data Loaded. Train: {train_x.shape}, Val: {val_x.shape}, Test: {test_x.shape}"
    )

    return train_loader, val_loader, test_loader, test_ids
