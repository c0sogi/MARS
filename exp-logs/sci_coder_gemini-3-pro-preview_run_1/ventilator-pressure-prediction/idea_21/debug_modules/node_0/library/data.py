import os
import gc
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

from library import config
from library.utils import seed_everything

# =============================================================================
# Feature Engineering
# =============================================================================


def engineer_features(df):
    """
    Applies physics-based and temporal feature engineering to the dataframe.
    """
    # Ensure data is sorted
    df = df.sort_values(by=[config.BREATH_ID_COL, config.TIME_COL]).reset_index(
        drop=True
    )

    # Group object for breath-wise operations
    # We use transform where possible to keep shape aligned
    grp = df.groupby(config.BREATH_ID_COL)

    # 1. Temporal Dynamics (Lags and Diffs)
    if config.USE_LAGS:
        for lag in config.LAG_STEPS:
            # Shift u_in
            df[f"u_in_lag{lag}"] = grp[config.U_IN_COL].shift(lag).fillna(0)

            # Diff u_in (current - lag)
            if config.USE_DIFFS:
                df[f"u_in_diff{lag}"] = df[config.U_IN_COL] - df[f"u_in_lag{lag}"]

                # We can also add diff of lags if needed, but sticking to basics first
                # to avoid exploding feature count too much

    # 2. Physics Integration (Volume)
    if config.USE_INTEGRATION:
        # Calculate dt
        # shift(1) gives previous time step.
        # For the first step of each breath, time_step is usually 0, but dt needs care.
        # We assume dt is time_step - prev_time_step. Fillna(0) for first step.
        df["dt"] = grp[config.TIME_COL].diff().fillna(0)

        # Volume = sum(u_in * dt)
        # u_in is 0-100, we treat it as flow rate.
        df["d_vol"] = df[config.U_IN_COL] * df["dt"]
        df["volume"] = grp["d_vol"].cumsum()

        # Cleanup intermediate columns if desired, but dt might be useful
        # Keeping dt and volume

    # 3. Interactions
    if config.USE_INTERACTIONS:
        # R * u_in (Resistance * Flow ~ Pressure drop)
        df["R_u_in"] = df[config.R_COL] * df[config.U_IN_COL]

        # Volume / C (Volume / Compliance ~ Pressure from expansion)
        # Add epsilon to avoid div by zero if C could be 0 (though C is 10, 20, 50)
        df["vol_div_C"] = df["volume"] / (df[config.C_COL] + 1e-6)

        # Interaction of lags with R can also be useful
        if config.USE_LAGS:
            for lag in config.LAG_STEPS:
                df[f"R_u_in_lag{lag}"] = df[config.R_COL] * df[f"u_in_lag{lag}"]

    # 4. Additional useful features derived from previous winning solutions
    # Time since start of breath is already 'time_step'
    # Step index within breath (0 to 79)
    df["step_index"] = grp.cumcount()

    # One-hot encoding R and C is an option, but the "Idea" suggests scaling them.
    # We will keep them as continuous features for the scaler to handle.

    return df


# =============================================================================
# Preprocessing & Caching
# =============================================================================


def preprocess_data(load_cached_data=True):
    """
    Loads data, engineers features, scales, and reshapes.
    Implements caching to ./working/idea_21/
    """
    # Cache file paths
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_x_path = os.path.join(cache_dir, "train_x.npy")
    train_uout_path = os.path.join(cache_dir, "train_uout.npy")
    train_y_path = os.path.join(cache_dir, "train_y.npy")

    val_x_path = os.path.join(cache_dir, "val_x.npy")
    val_uout_path = os.path.join(cache_dir, "val_uout.npy")
    val_y_path = os.path.join(cache_dir, "val_y.npy")

    test_x_path = os.path.join(cache_dir, "test_x.npy")
    test_uout_path = os.path.join(cache_dir, "test_uout.npy")
    test_ids_path = os.path.join(cache_dir, "test_ids.npy")

    scaler_path = config.SCALER_PATH

    # Check if all cache files exist
    files_exist = all(
        os.path.exists(p)
        for p in [
            train_x_path,
            train_uout_path,
            train_y_path,
            val_x_path,
            val_uout_path,
            val_y_path,
            test_x_path,
            test_uout_path,
            test_ids_path,
            scaler_path,
        ]
    )

    if load_cached_data and files_exist:
        print("Loading cached data from", cache_dir)
        train_x = np.load(train_x_path)
        train_uout = np.load(train_uout_path)
        train_y = np.load(train_y_path)

        val_x = np.load(val_x_path)
        val_uout = np.load(val_uout_path)
        val_y = np.load(val_y_path)

        test_x = np.load(test_x_path)
        test_uout = np.load(test_uout_path)
        test_ids = np.load(test_ids_path)

        return (
            (train_x, train_uout, train_y),
            (val_x, val_uout, val_y),
            (test_x, test_uout, test_ids),
        )

    print("Cache not found or disabled. Processing data from scratch...")

    # Load Raw Data
    print("Loading CSVs...")
    train_df = pd.read_csv(config.TRAIN_PATH)
    val_df = pd.read_csv(config.VAL_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    # Apply Feature Engineering
    print("Engineering features...")
    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    # Identify feature columns
    # Exclude IDs, Targets, and u_out (handled separately)
    exclude_cols = [
        config.ID_COL,
        config.BREATH_ID_COL,
        config.TARGET_COL,
        config.U_OUT_COL,
    ]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    print(f"Selected {len(feature_cols)} features: {feature_cols}")

    # Scaling
    # Fit on Train only
    print("Fitting Scaler...")
    scaler = RobustScaler()
    scaler.fit(train_df[feature_cols])

    # Transform
    print("Transforming data...")
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # Save Scaler
    joblib.dump(scaler, scaler_path)

    # Reshaping to (N_breaths, 80, N_features)
    # We assume 80 steps per breath based on dataset standards
    steps_per_breath = 80

    def reshape_dataset(df, is_test=False):
        # Ensure sorting is correct before reshape
        df = df.sort_values(by=[config.BREATH_ID_COL, config.TIME_COL])

        n_breaths = len(df) // steps_per_breath

        # Features
        x = df[feature_cols].values.reshape(
            n_breaths, steps_per_breath, len(feature_cols)
        )

        # u_out (keep separate for masking)
        u_out = df[config.U_OUT_COL].values.reshape(n_breaths, steps_per_breath)

        if not is_test:
            y = df[config.TARGET_COL].values.reshape(n_breaths, steps_per_breath)
            return x.astype(np.float32), u_out.astype(np.float32), y.astype(np.float32)
        else:
            ids = df[config.ID_COL].values.reshape(n_breaths, steps_per_breath)
            return x.astype(np.float32), u_out.astype(np.float32), ids.astype(np.int32)

    print("Reshaping datasets...")
    train_x, train_uout, train_y = reshape_dataset(train_df)
    val_x, val_uout, val_y = reshape_dataset(val_df)
    test_x, test_uout, test_ids = reshape_dataset(test_df, is_test=True)

    # Save to Cache
    print("Saving to cache...")
    np.save(train_x_path, train_x)
    np.save(train_uout_path, train_uout)
    np.save(train_y_path, train_y)

    np.save(val_x_path, val_x)
    np.save(val_uout_path, val_uout)
    np.save(val_y_path, val_y)

    np.save(test_x_path, test_x)
    np.save(test_uout_path, test_uout)
    np.save(test_ids_path, test_ids)

    # Clean up memory
    del train_df, val_df, test_df
    gc.collect()

    return (
        (train_x, train_uout, train_y),
        (val_x, val_uout, val_y),
        (test_x, test_uout, test_ids),
    )


# =============================================================================
# Dataset Class
# =============================================================================


class VentilatorDataset(Dataset):
    def __init__(self, X, u_out, y=None, is_test=False):
        """
        Args:
            X (np.ndarray): Feature tensor (N, 80, F)
            u_out (np.ndarray): Control input u_out (N, 80)
            y (np.ndarray): Target pressure or IDs (N, 80)
            is_test (bool): If True, y contains IDs.
        """
        self.X = X
        self.u_out = u_out
        self.y = y
        self.is_test = is_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert to tensor
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        u_out_tensor = torch.tensor(self.u_out[idx], dtype=torch.float32)

        if self.y is not None:
            if self.is_test:
                # y contains IDs (int)
                y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
            else:
                # y contains pressure (float)
                y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
            return x_tensor, u_out_tensor, y_tensor

        return x_tensor, u_out_tensor


# =============================================================================
# DataLoader Factory
# =============================================================================


def get_dataloaders(
    batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, load_cached_data=True
):
    """
    Orchestrates the data pipeline and returns DataLoaders.
    """
    seed_everything()

    # Get processed data
    train_data, val_data, test_data = preprocess_data(load_cached_data=load_cached_data)

    train_x, train_uout, train_y = train_data
    val_x, val_uout, val_y = val_data
    test_x, test_uout, test_ids = test_data

    # Create Datasets
    train_dataset = VentilatorDataset(train_x, train_uout, train_y, is_test=False)
    val_dataset = VentilatorDataset(val_x, val_uout, val_y, is_test=False)
    test_dataset = VentilatorDataset(test_x, test_uout, test_ids, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
