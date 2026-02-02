import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import (
    WORKING_DIR,
    TRAJECTORY_FEATURES,
    CONTEXT_FEATURES,
    TARGET_FEATURES,
    WINDOW_SIZE,
    DEG_TO_M_LAT,
    DEG_TO_M_LON,
)


class GNSSScaler:
    """
    StandardScaler that handles both static column-wise features and
    dynamically computed window-relative features.
    """

    def __init__(self):
        self.means = {}
        self.stds = {}

    def fit(self, df: pd.DataFrame, sample_size: int = 100000):
        """
        Compute mean and std for features.
        For window-relative features, we simulate windows on a sample.
        """
        # 1. Static features (Velocity, Cn0, Uncertainty, Context)
        # These exist directly in the dataframe (except relative pos)
        static_feats = [
            f for f in TRAJECTORY_FEATURES if not f.startswith("rel_")
        ] + CONTEXT_FEATURES

        for feat in static_feats:
            if feat in df.columns:
                self.means[feat] = float(df[feat].mean())
                self.stds[feat] = float(df[feat].std())
                # Avoid divide by zero
                if self.stds[feat] == 0:
                    self.stds[feat] = 1.0
            else:
                # Fallback if column missing (should not happen with correct loader)
                self.means[feat] = 0.0
                self.stds[feat] = 1.0

        # 2. Relative Position Features
        # We need to estimate the distribution of (pos - center_pos) within windows.
        rel_feats = [f for f in TRAJECTORY_FEATURES if f.startswith("rel_")]
        if rel_feats:
            # Group by trip to ensure validity
            groups = df.groupby("tripId")
            all_rel_lats = []
            all_rel_lons = []

            # Sample a few trips to estimate distribution
            trip_ids = list(groups.groups.keys())
            selected_trips = np.random.choice(
                trip_ids, size=min(len(trip_ids), 50), replace=False
            )

            for tid in selected_trips:
                grp = groups.get_group(tid)
                lats = grp["wls_lat"].values
                lons = grp["wls_lon"].values
                n = len(grp)
                if n < WINDOW_SIZE:
                    continue

                # Sample indices within this trip
                # We take a random subset of valid center points
                valid_indices = np.arange(WINDOW_SIZE // 2, n - WINDOW_SIZE // 2)
                if len(valid_indices) > 200:
                    sample_idxs = np.random.choice(
                        valid_indices, size=200, replace=False
                    )
                else:
                    sample_idxs = valid_indices

                for idx in sample_idxs:
                    # Define window bounds
                    start = idx - WINDOW_SIZE // 2
                    end = idx + WINDOW_SIZE // 2 + 1

                    center_lat = lats[idx]
                    center_lon = lons[idx]

                    # Compute relative
                    win_lats = (lats[start:end] - center_lat) * DEG_TO_M_LAT
                    # Approximate lon scaling using center lat
                    win_lons = (
                        (lons[start:end] - center_lon)
                        * DEG_TO_M_LON
                        * np.cos(np.radians(center_lat))
                    )

                    all_rel_lats.extend(win_lats)
                    all_rel_lons.extend(win_lons)

            if all_rel_lats:
                self.means["rel_wls_lat_m"] = float(np.mean(all_rel_lats))
                self.stds["rel_wls_lat_m"] = float(np.std(all_rel_lats)) + 1e-6
                self.means["rel_wls_lon_m"] = float(np.mean(all_rel_lons))
                self.stds["rel_wls_lon_m"] = float(np.std(all_rel_lons)) + 1e-6
            else:
                self.means["rel_wls_lat_m"] = 0.0
                self.stds["rel_wls_lat_m"] = 10.0
                self.means["rel_wls_lon_m"] = 0.0
                self.stds["rel_wls_lon_m"] = 10.0

    def transform(self, feature_name, value):
        mean = self.means.get(feature_name, 0.0)
        std = self.stds.get(feature_name, 1.0)
        return (value - mean) / std

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"means": self.means, "stds": self.stds}, f, indent=4)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
            self.means = data["means"]
            self.stds = data["stds"]


class GNSSSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        scaler: GNSSScaler,
        window_size: int = WINDOW_SIZE,
        is_test: bool = False,
    ):
        self.scaler = scaler
        self.window_size = window_size
        self.is_test = is_test
        self.half_window = window_size // 2

        # Group data by tripId for efficient access
        self.trips = {}
        self.trip_ids = df["tripId"].unique()

        # Define columns to extract
        self.traj_static_cols = [
            "vel_lat_m",
            "vel_lon_m",
            "vel_alt_m",
            "mean_cn0",
            "mean_uncertainty",
        ]
        self.context_cols = CONTEXT_FEATURES

        # Ensure columns exist (fill with 0 if missing to prevent crash)
        for c in (
            self.traj_static_cols
            + self.context_cols
            + ["wls_lat", "wls_lon", "wls_alt"]
        ):
            if c not in df.columns:
                df[c] = 0.0

        if not is_test:
            for c in TARGET_FEATURES:
                if c not in df.columns:
                    df[c] = 0.0

        # Build index of valid samples
        self.samples = []

        grouped = df.groupby("tripId")

        for tid in self.trip_ids:
            grp = grouped.get_group(tid)

            # Store arrays for fast access
            trip_data = {
                "wls_lat": grp["wls_lat"].values.astype(np.float32),
                "wls_lon": grp["wls_lon"].values.astype(np.float32),
                "traj_static": grp[self.traj_static_cols].values.astype(np.float32),
                "context": grp[self.context_cols].values.astype(np.float32),
                "length": len(grp),
            }

            if not is_test:
                trip_data["targets"] = grp[TARGET_FEATURES].values.astype(np.float32)

            self.trips[tid] = trip_data

            n = len(grp)
            if is_test:
                # Test mode: predict for every timestamp, padding if necessary
                indices = range(n)
            else:
                # Train mode: only valid windows
                start = self.half_window
                end = n - self.half_window
                if start < end:
                    indices = range(start, end)
                else:
                    indices = []

            for idx in indices:
                self.samples.append((tid, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tid, center_idx = self.samples[idx]
        trip = self.trips[tid]
        n = trip["length"]

        # Determine window bounds
        start = center_idx - self.half_window
        end = center_idx + self.half_window + 1

        # Handle padding for test set if out of bounds
        pad_left = 0
        pad_right = 0

        if start < 0:
            pad_left = -start
            start = 0
        if end > n:
            pad_right = end - n
            end = n

        # Extract data
        # 1. WLS coords for relative calculation
        wls_lat_win = trip["wls_lat"][start:end]
        wls_lon_win = trip["wls_lon"][start:end]

        center_lat = trip["wls_lat"][center_idx]
        center_lon = trip["wls_lon"][center_idx]

        # Calculate relative coords in meters
        rel_lat = (wls_lat_win - center_lat) * DEG_TO_M_LAT
        rel_lon = (
            (wls_lon_win - center_lon) * DEG_TO_M_LON * np.cos(np.radians(center_lat))
        )

        # 2. Static Trajectory Features
        traj_static_win = trip["traj_static"][start:end]

        # Combine Trajectory Features
        # Order must match TRAJECTORY_FEATURES:
        # ["rel_wls_lat_m", "rel_wls_lon_m", "vel_lat_m", "vel_lon_m", "vel_alt_m", "mean_cn0", "mean_uncertainty"]
        traj_feats = np.concatenate(
            [rel_lat[:, None], rel_lon[:, None], traj_static_win], axis=1
        )

        # 3. Context Features (Center epoch only)
        context_feats = trip["context"][center_idx].copy()

        # Apply Padding if needed (Edge padding)
        if pad_left > 0 or pad_right > 0:
            traj_feats = np.pad(
                traj_feats, ((pad_left, pad_right), (0, 0)), mode="edge"
            )

        # Normalize
        # Trajectory Stream
        for i, name in enumerate(TRAJECTORY_FEATURES):
            traj_feats[:, i] = self.scaler.transform(name, traj_feats[:, i])

        # Context Stream
        for i, name in enumerate(CONTEXT_FEATURES):
            context_feats[i] = self.scaler.transform(name, context_feats[i])

        # Convert to tensors
        # CNN expects [Channels, Length]
        traj_tensor = torch.tensor(traj_feats, dtype=torch.float32).transpose(0, 1)
        context_tensor = torch.tensor(context_feats, dtype=torch.float32)

        if self.is_test:
            return traj_tensor, context_tensor
        else:
            target = trip["targets"][center_idx]
            target_tensor = torch.tensor(target, dtype=torch.float32)
            return traj_tensor, context_tensor, target_tensor
