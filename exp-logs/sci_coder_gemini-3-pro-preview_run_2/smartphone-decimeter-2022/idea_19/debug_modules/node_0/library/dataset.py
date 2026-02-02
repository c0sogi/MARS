import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
from library.config import config
from library.preprocessing import preprocess_dataset
from library.utils import degrees_to_meters


class GNSSWindowDataset(Dataset):
    def __init__(self, X, y=None, window_size=15, scaler=None, mode="train"):
        """
        Args:
            X (pd.DataFrame): Feature dataframe containing GNSS/IMU/WLS data.
            y (np.array, optional): Target array (N, 2) for train/val.
            window_size (int): Size of the sliding window (must be odd).
            scaler (StandardScaler, optional): Fitted scaler.
            mode (str): 'train', 'val', or 'test'.
        """
        self.X = X.reset_index(drop=True)
        self.y = y
        self.window_size = window_size
        self.half_window = window_size // 2
        self.mode = mode

        # Pre-compute trip boundaries for fast indexing
        self.trip_ids = self.X["tripId"].values

        # Find indices where tripId changes
        # 0 and indices where trip_ids[i] != trip_ids[i-1]
        change_points = np.where(self.trip_ids[:-1] != self.trip_ids[1:])[0] + 1
        trip_starts = np.concatenate(([0], change_points))
        trip_ends = np.concatenate((change_points, [len(self.X)])) - 1

        # Create a lookup array: index -> (start, end) of the trip it belongs to
        self.sample_trip_bounds = np.zeros((len(self.X), 2), dtype=int)
        for start, end in zip(trip_starts, trip_ends):
            self.sample_trip_bounds[start : end + 1, 0] = start
            self.sample_trip_bounds[start : end + 1, 1] = end

        # Determine valid indices
        if self.mode == "test":
            # In test mode, we must predict for EVERY row.
            # We will handle edge cases by clamping indices in __getitem__
            self.valid_indices = np.arange(len(self.X))
        else:
            # In train/val, we only use windows that fully fit within a trip
            # to avoid edge artifacts and padding noise.
            # Check if window [i-half, i+half] is within [trip_start, trip_end]
            indices = np.arange(len(self.X))
            starts = self.sample_trip_bounds[:, 0]
            ends = self.sample_trip_bounds[:, 1]

            mask = (indices - self.half_window >= starts) & (
                indices + self.half_window <= ends
            )
            self.valid_indices = indices[mask]

        # Define feature columns
        self.wls_cols = ["wls_lat", "wls_lon", "wls_alt"]
        self.vel_cols = ["vel_lat_m", "vel_lon_m", "vel_alt_m"]
        self.signal_cols = ["cn0", "unc_m"]
        self.env_cols = ["mean_elev", "std_elev", "mean_azim", "std_azim"]
        self.imu_cols = ["mean_acc_mag", "std_acc_mag", "mean_gyro_mag", "std_gyro_mag"]

        # Scaler handling
        self.scaler = scaler
        if self.scaler is None and mode == "train":
            self._fit_scaler()

    def _fit_scaler(self):
        print("Fitting scaler on training data...")
        # Fit on a random sample of windows to capture distribution of relative coords
        sample_size = min(10000, len(self.valid_indices))
        if sample_size > 0:
            sample_indices = np.random.choice(
                self.valid_indices, sample_size, replace=False
            )
            sample_features = []
            for idx in sample_indices:
                feat, _ = self._get_window_features(idx)
                sample_features.append(feat)

            sample_features = np.stack(sample_features)
            self.scaler = StandardScaler()
            self.scaler.fit(sample_features)
            print("Scaler fitted.")
        else:
            print("Warning: No valid windows to fit scaler.")
            self.scaler = StandardScaler()

    def _get_window_features(self, idx):
        # Get trip bounds for this sample
        t_start, t_end = self.sample_trip_bounds[idx]

        # Calculate raw window indices
        raw_start = idx - self.half_window
        raw_end = idx + self.half_window

        # Generate indices, clamping to trip boundaries (Edge Padding logic)
        # This repeats the first/last elements if the window extends beyond the trip
        window_indices = np.clip(np.arange(raw_start, raw_end + 1), t_start, t_end)

        # Extract data for the window
        # Note: iloc with integer array is efficient enough here
        window_df = self.X.iloc[window_indices]

        # Center WLS (Ego-Centric Origin)
        center_lat = self.X.at[idx, "wls_lat"]
        center_lon = self.X.at[idx, "wls_lon"]
        center_alt = self.X.at[idx, "wls_alt"]

        # 1. Trajectory Block: Relative Positions
        lats = window_df["wls_lat"].values
        lons = window_df["wls_lon"].values
        alts = window_df["wls_alt"].values

        # Convert degrees diff to meters
        rel_n, rel_e = degrees_to_meters(
            lats - center_lat, lons - center_lon, center_lat
        )
        rel_u = alts - center_alt

        # Other trajectory features
        vels = window_df[self.vel_cols].values
        sigs = window_df[self.signal_cols].values

        # Stack trajectory features: (WindowSize, 8)
        # Features: [rel_n, rel_e, rel_u, vel_n, vel_e, vel_u, cn0, unc]
        traj_feats = np.column_stack([rel_n, rel_e, rel_u, vels, sigs])

        # Flatten trajectory: (WindowSize * 8,)
        traj_flat = traj_feats.flatten()

        # 2. Context Blocks (Aggregated over window)
        # Environmental Context
        env_data = window_df[self.env_cols].values
        env_agg = np.mean(env_data, axis=0)  # Mean

        # Inertial Context
        imu_data = window_df[self.imu_cols].values
        imu_agg = np.mean(imu_data, axis=0)  # Mean

        # Concatenate all features
        full_features = np.concatenate([traj_flat, env_agg, imu_agg])

        # Target
        target = None
        if self.y is not None:
            target = self.y[idx]

        return full_features, target

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        idx = self.valid_indices[i]
        features, target = self._get_window_features(idx)

        # Apply scaler
        if self.scaler:
            # Reshape for scaler (1, -1) then flatten back
            features = self.scaler.transform(features.reshape(1, -1)).flatten()

        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)

        if target is not None:
            target_tensor = torch.tensor(target, dtype=torch.float32)
            return features_tensor, target_tensor
        else:
            return features_tensor


def get_dataloaders(batch_size=config.BATCH_SIZE, num_workers=4):
    """
    Prepares Train and Validation DataLoaders.
    """
    # Load Data
    print("Loading Train Data...")
    X_train, y_train = preprocess_dataset(
        config.TRAIN_METADATA_PATH, mode="train", load_cached_data=True
    )

    print("Loading Validation Data...")
    X_val, y_val = preprocess_dataset(
        config.VAL_METADATA_PATH, mode="val", load_cached_data=True
    )

    # Create Datasets
    # Train dataset fits scaler
    train_dataset = GNSSWindowDataset(
        X_train, y_train, window_size=config.WINDOW_SIZE, mode="train"
    )

    # Save scaler for inference
    scaler_path = os.path.join(config.WORKING_DIR, "scaler.pkl")
    joblib.dump(train_dataset.scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Val dataset uses fitted scaler
    val_dataset = GNSSWindowDataset(
        X_val,
        y_val,
        window_size=config.WINDOW_SIZE,
        scaler=train_dataset.scaler,
        mode="val",
    )

    # DataLoaders
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

    return train_loader, val_loader


def get_test_dataloader(batch_size=config.BATCH_SIZE, num_workers=4):
    """
    Prepares Test DataLoader and returns raw X_test for reconstruction.
    """
    print("Loading Test Data...")
    X_test = preprocess_dataset(
        config.TEST_METADATA_PATH, mode="test", load_cached_data=True
    )

    # Load Scaler
    scaler_path = os.path.join(config.WORKING_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler from {scaler_path}")
    else:
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. Train model first."
        )

    test_dataset = GNSSWindowDataset(
        X_test, y=None, window_size=config.WINDOW_SIZE, scaler=scaler, mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader, X_test
