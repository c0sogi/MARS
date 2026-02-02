import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import deg_to_meters
from library.preprocessing import process_dataset


class StandardScaler:
    def __init__(self):
        self.mean = {}
        self.std = {}

    def fit(self, df, features, relative_features=None):
        # Fit on static features present in DataFrame
        for col in features:
            if col in df.columns:
                self.mean[col] = float(df[col].mean())
                self.std[col] = float(df[col].std())
                if self.std[col] == 0:
                    self.std[col] = 1.0

        # Handle relative features (computed on the fly)
        # We estimate their stats based on velocity and window size.
        # Relative positions are centered at 0, so mean is 0.
        # Std is approximated as: std(velocity) * (window_size / 4)
        # This is a heuristic assuming linear motion over half the window radius.
        if relative_features:
            w_factor = Config.WINDOW_SIZE / 4.0
            if "vel_lat_m" in self.std:
                self.mean["rel_lat_m"] = 0.0
                self.std["rel_lat_m"] = self.std["vel_lat_m"] * w_factor
            if "vel_lon_m" in self.std:
                self.mean["rel_lon_m"] = 0.0
                self.std["rel_lon_m"] = self.std["vel_lon_m"] * w_factor

    def transform(self, data_dict):
        # data_dict: {feature_name: value_array}
        normalized = {}
        for k, v in data_dict.items():
            if k in self.mean and k in self.std:
                normalized[k] = (v - self.mean[k]) / self.std[k]
            else:
                normalized[k] = v  # No scaling if not found
        return normalized

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"mean": self.mean, "std": self.std}, f, indent=4)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
            self.mean = data["mean"]
            self.std = data["std"]


class GNSSWindowDataset(Dataset):
    def __init__(self, df, scaler, window_size=Config.WINDOW_SIZE, mode="train"):
        self.mode = mode
        self.window_size = window_size
        self.scaler = scaler
        self.half_window = window_size // 2

        # Group by tripId to handle boundaries
        self.trip_ids = df["tripId"].unique()
        self.trip_data = {}
        self.trip_indices = []  # List of (trip_id, local_index)

        # Columns required for features
        # Static features from DF
        self.static_feature_cols = [
            f for f in Config.INPUT_FEATURES if f not in ["rel_lat_m", "rel_lon_m"]
        ]
        # Columns needed for relative calc
        self.pos_cols = ["wls_lat", "wls_lon"]

        # Pre-convert trips to numpy dictionaries for speed
        for tid in self.trip_ids:
            trip_df = df[df["tripId"] == tid].reset_index(drop=True)

            # Data dict
            data = {}
            for col in self.static_feature_cols:
                data[col] = trip_df[col].values.astype(np.float32)

            for col in self.pos_cols:
                data[col] = trip_df[col].values.astype(
                    np.float64
                )  # Keep precision for coords

            if mode in ["train", "val"]:
                data["target_lat_m"] = trip_df["target_lat_m"].values.astype(np.float32)
                data["target_lon_m"] = trip_df["target_lon_m"].values.astype(np.float32)

            # Metadata for reconstruction
            data["UnixTimeMillis"] = trip_df["UnixTimeMillis"].values

            # Store WLS ECEF if available for reconstruction, otherwise we rely on wls_lat/lon
            if "WlsPositionXEcefMeters" in trip_df.columns:
                data["WlsPositionXEcefMeters"] = trip_df[
                    "WlsPositionXEcefMeters"
                ].values
                data["WlsPositionYEcefMeters"] = trip_df[
                    "WlsPositionYEcefMeters"
                ].values
                data["WlsPositionZEcefMeters"] = trip_df[
                    "WlsPositionZEcefMeters"
                ].values

            self.trip_data[tid] = data

            # Create indices
            # We predict for every point in the trip
            num_points = len(trip_df)
            self.trip_indices.extend([(tid, i) for i in range(num_points)])

    def __len__(self):
        return len(self.trip_indices)

    def __getitem__(self, idx):
        tid, center_idx = self.trip_indices[idx]
        data = self.trip_data[tid]
        num_points = len(data["wls_lat"])

        # Calculate window indices with padding
        start_idx = center_idx - self.half_window
        end_idx = center_idx + self.half_window + 1

        # Handle boundaries by clamping and padding
        # Strategy: Clamp indices to [0, num_points-1]
        # This effectively repeats the first/last element
        indices = np.arange(start_idx, end_idx)
        indices = np.clip(indices, 0, num_points - 1)

        # Extract Static Features
        features = {}
        for col in self.static_feature_cols:
            features[col] = data[col][indices]

        # Calculate Relative Positions
        center_lat = data["wls_lat"][center_idx]
        center_lon = data["wls_lon"][center_idx]

        window_lat = data["wls_lat"][indices]
        window_lon = data["wls_lon"][indices]

        d_lat = window_lat - center_lat
        d_lon = window_lon - center_lon

        rel_lat_m, rel_lon_m = deg_to_meters(d_lat, d_lon, center_lat)

        features["rel_lat_m"] = rel_lat_m.astype(np.float32)
        features["rel_lon_m"] = rel_lon_m.astype(np.float32)

        # Normalize
        features = self.scaler.transform(features)

        # Stack into tensor (Sequence Length, Input Dim)
        # Order must match Config.INPUT_FEATURES
        feature_list = [features[f] for f in Config.INPUT_FEATURES]
        x = np.stack(feature_list, axis=1)  # Shape: (Window, Features)
        x = torch.tensor(x, dtype=torch.float32)

        # Metadata
        meta = {
            "tripId": tid,
            "UnixTimeMillis": data["UnixTimeMillis"][center_idx],
            "wls_lat": center_lat,
            "wls_lon": center_lon,
        }

        if self.mode in ["train", "val"]:
            y = np.array(
                [data["target_lat_m"][center_idx], data["target_lon_m"][center_idx]],
                dtype=np.float32,
            )
            y = torch.tensor(y, dtype=torch.float32)
            return x, y, meta
        else:
            return x, meta


def get_dataset(mode, load_cached_data=True):
    """
    Factory function to create the dataset.
    Handles loading/processing data and fitting/loading the scaler.
    """
    # 1. Process/Load Data
    df = process_dataset(mode, load_cached_data=load_cached_data)

    # 2. Handle Scaler
    scaler = StandardScaler()
    if mode == "train":
        print("Fitting scaler on training data...")
        # Identify static features available in DF
        static_feats = [
            f for f in Config.INPUT_FEATURES if f not in ["rel_lat_m", "rel_lon_m"]
        ]
        scaler.fit(df, static_feats, relative_features=True)
        print("Saving scaler...")
        scaler.save(Config.SCALER_PATH)
    else:
        if os.path.exists(Config.SCALER_PATH):
            print("Loading scaler...")
            scaler.load(Config.SCALER_PATH)
        else:
            raise FileNotFoundError(
                f"Scaler not found at {Config.SCALER_PATH}. Run training first."
            )

    # 3. Create Dataset
    dataset = GNSSWindowDataset(df, scaler, window_size=Config.WINDOW_SIZE, mode=mode)

    return dataset
