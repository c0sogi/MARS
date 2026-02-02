import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import get_config_hash


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Returns:
        - x: Scaled feature tensor (Time, Features)
        - y: Target pressure tensor (Time,) [Optional]
        - u_out: Unscaled binary control input for masking (Time,) [Optional]
    """

    def __init__(self, X, y=None, u_out=None):
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


def engineer_features(df):
    """
    Generates features as defined in Config.features.
    Includes correct volume integration, lags, diffs, and physics interactions.
    """
    # Ensure data is sorted by breath_id and time_step
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    # Calculate time delta (dt)
    # Use groupby to ensure we don't diff across breaths.
    # fillna(0) handles the first time step of each breath.
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # Correct Volume Integration: sum(u_in * dt)
    # u_in is flow rate, so volume is the integral of flow over time.
    # Groupby ensures cumulative sum resets for each breath.
    df["volume"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()

    # Lags for u_in (capturing system delay/inertia)
    grp_u_in = df.groupby("breath_id")["u_in"]
    for lag in range(1, 5):
        df[f"u_in_lag{lag}"] = grp_u_in.shift(lag).fillna(0)

    # Differences for u_in (capturing rate of change in control)
    df["u_in_diff1"] = grp_u_in.diff(1).fillna(0)
    df["u_in_diff2"] = grp_u_in.diff(2).fillna(0)
    df["u_in_diff3"] = grp_u_in.diff(3).fillna(0)
    df["u_in_diff4"] = grp_u_in.diff(4).fillna(0)

    # Physics Interaction Terms (Equation of Motion proxies)
    # Pressure drop across airway ~ Flow * Resistance
    df["u_in_R"] = df["u_in"] * df["R"]
    # Pressure due to elasticity ~ Volume / Compliance
    df["volume_C"] = df["volume"] / df["C"]

    # Derivative Interactions (Resistance * Acceleration/Velocity changes)
    df["u_in_diff1_R"] = df["u_in_diff1"] * df["R"]
    df["u_in_diff2_R"] = df["u_in_diff2"] * df["R"]

    return df


def get_dataloaders(
    debug=Config.debug,
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
    load_cached_data=True,
):
    """
    Prepares DataLoaders for train, val, and test sets.
    Handles caching of processed numpy arrays to avoid re-computing features.

    Args:
        debug (bool): If True, uses a small subset of data.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of worker processes for DataLoaders.
        load_cached_data (bool): If True, attempts to load from disk cache.

    Returns:
        train_loader, val_loader, test_loader
    """

    # Ensure cache directory exists
    os.makedirs(Config.cache_dir, exist_ok=True)

    # Generate a unique hash based on the feature configuration
    feature_hash = get_config_hash(Config.features)

    # Define cache file paths
    # Include 'debug' in filename to separate full vs debug caches
    suffix = f"{feature_hash}_debug" if debug else feature_hash

    cache_files = {
        "train_x": os.path.join(Config.cache_dir, f"train_x_{suffix}.npy"),
        "train_y": os.path.join(Config.cache_dir, f"train_y_{suffix}.npy"),
        "train_uout": os.path.join(Config.cache_dir, f"train_uout_{suffix}.npy"),
        "val_x": os.path.join(Config.cache_dir, f"val_x_{suffix}.npy"),
        "val_y": os.path.join(Config.cache_dir, f"val_y_{suffix}.npy"),
        "val_uout": os.path.join(Config.cache_dir, f"val_uout_{suffix}.npy"),
        "test_x": os.path.join(Config.cache_dir, f"test_x_{suffix}.npy"),
        "test_uout": os.path.join(Config.cache_dir, f"test_uout_{suffix}.npy"),
        "test_ids": os.path.join(Config.cache_dir, f"test_ids_{suffix}.npy"),
        "scaler_center": os.path.join(Config.cache_dir, f"scaler_center_{suffix}.npy"),
        "scaler_scale": os.path.join(Config.cache_dir, f"scaler_scale_{suffix}.npy"),
    }

    # Check if all required cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print(f"Loading cached data from {Config.cache_dir}...")
        train_x = np.load(cache_files["train_x"])
        train_y = np.load(cache_files["train_y"])
        train_uout = np.load(cache_files["train_uout"])

        val_x = np.load(cache_files["val_x"])
        val_y = np.load(cache_files["val_y"])
        val_uout = np.load(cache_files["val_uout"])

        test_x = np.load(cache_files["test_x"])
        test_uout = np.load(cache_files["test_uout"])
        # test_ids are cached for inference script usage

    else:
        print("Cache not found or reload requested. Processing data from scratch...")

        # Load raw CSV data
        print(f"Loading raw data from {Config.train_path}...")
        train_df = pd.read_csv(Config.train_path)
        val_df = pd.read_csv(Config.val_path)
        test_df = pd.read_csv(Config.test_path)

        if debug:
            print("Debug mode: Sampling subset of breaths...")
            # Sample by breath_id to keep time series intact
            train_breaths = train_df["breath_id"].unique()[:200]
            val_breaths = val_df["breath_id"].unique()[:100]
            test_breaths = test_df["breath_id"].unique()[:100]

            train_df = train_df[train_df["breath_id"].isin(train_breaths)].copy()
            val_df = val_df[val_df["breath_id"].isin(val_breaths)].copy()
            test_df = test_df[test_df["breath_id"].isin(test_breaths)].copy()

        print("Engineering features...")
        train_df = engineer_features(train_df)
        val_df = engineer_features(val_df)
        test_df = engineer_features(test_df)

        # Fit Scaler
        print("Fitting RobustScaler on training data...")
        scaler = RobustScaler()
        scaler.fit(train_df[Config.features])

        # Save scaler parameters
        np.save(cache_files["scaler_center"], scaler.center_)
        np.save(cache_files["scaler_scale"], scaler.scale_)

        def process_split(df, is_test=False):
            # Extract unscaled u_out for masking
            u_out = df["u_out"].values.astype(np.float32)

            # Extract target
            y = None
            if not is_test:
                y = df[Config.target_col].values.astype(np.float32)

            # Extract IDs for test submission
            ids = None
            if is_test:
                ids = df[Config.id_col].values

            # Scale features
            x_scaled = scaler.transform(df[Config.features]).astype(np.float32)

            # Reshape to (N_breaths, 80, N_features)
            n_steps = 80
            n_features = len(Config.features)

            # Handle potential length mismatch (though dataset is clean)
            num_rows = len(df)
            if num_rows % n_steps != 0:
                trim = num_rows % n_steps
                print(
                    f"Warning: Data length {num_rows} not divisible by {n_steps}. Truncating last {trim} rows."
                )
                x_scaled = x_scaled[:-trim]
                u_out = u_out[:-trim]
                if y is not None:
                    y = y[:-trim]
                if ids is not None:
                    ids = ids[:-trim]

            x_reshaped = x_scaled.reshape(-1, n_steps, n_features)
            u_out_reshaped = u_out.reshape(-1, n_steps)

            y_reshaped = None
            if y is not None:
                y_reshaped = y.reshape(-1, n_steps)

            return x_reshaped, y_reshaped, u_out_reshaped, ids

        print("Processing Train Split...")
        train_x, train_y, train_uout, _ = process_split(train_df, is_test=False)

        print("Processing Val Split...")
        val_x, val_y, val_uout, _ = process_split(val_df, is_test=False)

        print("Processing Test Split...")
        test_x, _, test_uout, test_ids = process_split(test_df, is_test=True)

        # Save to cache
        print(f"Saving processed data to {Config.cache_dir}...")
        np.save(cache_files["train_x"], train_x)
        np.save(cache_files["train_y"], train_y)
        np.save(cache_files["train_uout"], train_uout)

        np.save(cache_files["val_x"], val_x)
        np.save(cache_files["val_y"], val_y)
        np.save(cache_files["val_uout"], val_uout)

        np.save(cache_files["test_x"], test_x)
        np.save(cache_files["test_uout"], test_uout)
        np.save(cache_files["test_ids"], test_ids)

        # Memory cleanup
        del train_df, val_df, test_df
        gc.collect()

    # Create Datasets
    train_dataset = VentilatorDataset(train_x, train_y, train_uout)
    val_dataset = VentilatorDataset(val_x, val_y, val_uout)
    test_dataset = VentilatorDataset(test_x, None, test_uout)

    # Create DataLoaders
    # Train: Shuffle, Drop Last (for batch norm stability if used, though we use LayerNorm/None)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Val/Test: No Shuffle
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

    print(
        f"DataLoaders ready. Train Batches: {len(train_loader)}, Val Batches: {len(val_loader)}"
    )

    return train_loader, val_loader, test_loader
