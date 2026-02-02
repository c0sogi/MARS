import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


class GnssScaler:
    """
    Standard Scaler for GNSS features.
    Fits on training data and saves parameters to disk for consistent scaling across splits.
    """

    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.mean = None
        self.scale = None
        self.save_path = os.path.join(Config.WORKING_DIR, "scaler_params.json")

    def fit(self, df):
        """Compute mean and std to be used for later scaling."""
        print("Fitting scaler on training data...")
        stats = {}
        # We compute stats on the whole dataframe at once for efficiency
        # Handling potential infinite values or NaNs by dropping them for stat calculation
        for col in self.feature_cols:
            series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if series.empty:
                stats[col] = {"mean": 0.0, "scale": 1.0}
            else:
                mean_val = float(series.mean())
                std_val = float(series.std())
                # Avoid division by zero
                scale_val = std_val if std_val > 1e-9 else 1.0
                stats[col] = {"mean": mean_val, "scale": scale_val}

        self.mean = np.array(
            [stats[c]["mean"] for c in self.feature_cols], dtype=np.float32
        )
        self.scale = np.array(
            [stats[c]["scale"] for c in self.feature_cols], dtype=np.float32
        )

        # Save to disk
        with open(self.save_path, "w") as f:
            json.dump(stats, f)
        print(f"Scaler parameters saved to {self.save_path}")

    def load(self):
        """Load scaler parameters from disk."""
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(
                f"Scaler params not found at {self.save_path}. Fit scaler on train data first."
            )

        with open(self.save_path, "r") as f:
            stats = json.load(f)

        # Ensure order matches feature_cols
        self.mean = np.array(
            [stats[c]["mean"] for c in self.feature_cols], dtype=np.float32
        )
        self.scale = np.array(
            [stats[c]["scale"] for c in self.feature_cols], dtype=np.float32
        )
        print(f"Scaler parameters loaded from {self.save_path}")

    def transform(self, features):
        """
        Standardize features by removing the mean and scaling to unit variance.
        Args:
            features: Numpy array of shape (Seq_Len, Num_Features) or (Num_Features,)
        Returns:
            Transformed numpy array (float32)
        """
        if self.mean is None or self.scale is None:
            raise RuntimeError("Scaler has not been fitted or loaded.")

        # Broadcasting handles both (Seq, Feat) and (Feat,) if last dim matches
        return ((features - self.mean) / self.scale).astype(np.float32)


class DualStreamGnssDataset(Dataset):
    """
    PyTorch Dataset for Dual-Stream 1D ResUNet.
    Yields (stream_a, stream_b, targets, metadata).
    """

    def __init__(self, df, mode="train", scaler=None):
        """
        Args:
            df (pd.DataFrame): Preprocessed data containing features and targets.
            mode (str): 'train', 'val', or 'test'.
            scaler (GnssScaler): Fitted scaler instance. If None and mode='train', a new one is fitted.
        """
        self.df = df
        self.mode = mode
        self.config = Config

        # Identify Feature Columns
        # We construct the column names based on the logic in preprocessing.py
        self.feat_cols_a = []
        self.feat_cols_b = []

        # Stat features
        for feat in self.config.STAT_FEATURES:
            for stat in self.config.STATS_LIST:
                self.feat_cols_a.append(f"A_{feat}_{stat}")
                self.feat_cols_b.append(f"B_{feat}_{stat}")

        # Mean features
        for feat in self.config.MEAN_FEATURES:
            self.feat_cols_a.append(f"A_{feat}_mean")
            self.feat_cols_b.append(f"B_{feat}_mean")

        # Count features
        self.feat_cols_a.append("A_sat_count")
        self.feat_cols_b.append("B_sat_count")

        self.all_feature_cols = self.feat_cols_a + self.feat_cols_b

        # Verify columns exist
        missing_cols = [c for c in self.all_feature_cols if c not in df.columns]
        if missing_cols:
            # In case of empty streams (e.g. no L5 signals ever), columns might be missing.
            # We fill them with 0 in the dataframe to be safe.
            for c in missing_cols:
                self.df[c] = 0.0

        # Handle Scaling
        if scaler is None:
            self.scaler = GnssScaler(self.all_feature_cols)
            if mode == "train":
                self.scaler.fit(self.df)
            else:
                self.scaler.load()
        else:
            self.scaler = scaler

        # Pre-calculate sequence indices
        self.indices = self._prepare_indices()

    def _prepare_indices(self):
        """
        Generates a list of (start_idx, end_idx) tuples for valid sequences.
        Ensures sequences do not cross drive/phone boundaries.
        """
        indices = []
        # Group by drive and phone to isolate trips
        groups = self.df.groupby(["drive_id", "phone_name"])

        seq_len = self.config.SEQ_LEN
        stride = (
            self.config.TRAIN_STRIDE
            if self.mode == "train"
            else self.config.TEST_STRIDE
        )

        for _, group in groups:
            # Get the global integer indices of this group
            # Note: The df passed to __init__ must have a RangeIndex or we reset it here
            # We assume df is RangeIndex'ed 0..N
            group_indices = group.index.values
            n_samples = len(group_indices)

            if n_samples == 0:
                continue

            # If drive is shorter than seq_len
            if n_samples < seq_len:
                if self.mode == "test":
                    # For test, we must predict for these points.
                    # We will pad in __getitem__. We register one sequence starting at 0 of group.
                    indices.append((group_indices[0], n_samples))
                # For train/val, we skip extremely short drives to avoid padding noise
                continue

            # Generate sliding windows
            # range(start, stop, step)
            # We want the last index i such that i + seq_len <= n_samples
            max_start = n_samples - seq_len

            for start_local in range(0, max_start + 1, stride):
                global_start = group_indices[start_local]
                indices.append((global_start, seq_len))

            # Handle the tail for Testing
            # If the last window didn't exactly cover the end of the drive, add one more window
            # aligned to the end of the drive.
            if self.mode == "test":
                last_covered_local = (range(0, max_start + 1, stride)[-1]) + seq_len
                if last_covered_local < n_samples:
                    # Add a window ending exactly at the last sample
                    # start = n_samples - seq_len
                    tail_start_local = n_samples - seq_len
                    global_start = group_indices[tail_start_local]
                    # Check if we haven't already added this exact window (possible if stride=1 or coincidental)
                    if not indices or indices[-1][0] != global_start:
                        indices.append((global_start, seq_len))

        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx, length = self.indices[idx]

        # Extract the slice
        # If length < seq_len (only possible for short test drives), we handle padding
        end_idx = start_idx + length

        # Get raw features
        # Shape: (Length, Num_Features)
        data_slice = self.df.iloc[start_idx:end_idx]

        # Extract features
        features_a = data_slice[self.feat_cols_a].values.astype(np.float32)
        features_b = data_slice[self.feat_cols_b].values.astype(np.float32)

        # Concatenate for scaling (since scaler expects all features)
        combined_features = np.concatenate([features_a, features_b], axis=1)

        # Apply scaling
        combined_scaled = self.scaler.transform(combined_features)

        # Split back
        n_feat_a = len(self.feat_cols_a)
        features_a_scaled = combined_scaled[:, :n_feat_a]
        features_b_scaled = combined_scaled[:, n_feat_a:]

        # Handle Padding if necessary (for short test drives)
        target_len = self.config.SEQ_LEN
        if length < target_len:
            pad_size = target_len - length
            # Pad with zeros
            features_a_scaled = np.pad(
                features_a_scaled, ((0, pad_size), (0, 0)), "constant"
            )
            features_b_scaled = np.pad(
                features_b_scaled, ((0, pad_size), (0, 0)), "constant"
            )

        # Transpose to (Channels, Length) for PyTorch Conv1d
        # Input: (L, C) -> Output: (C, L)
        tensor_a = torch.from_numpy(features_a_scaled.transpose(1, 0))
        tensor_b = torch.from_numpy(features_b_scaled.transpose(1, 0))

        result = {
            "stream_a": tensor_a,
            "stream_b": tensor_b,
        }

        # Metadata for reconstruction
        # We need this for both train (debugging) and test (submission)
        # If padded, we pad metadata with the last value or dummy?
        # For inference, we track valid length.

        # Extract WLS and Time
        wls = data_slice[["wls_lat", "wls_lon", "wls_alt"]].values.astype(np.float64)
        times = data_slice["UnixTimeMillis"].values.astype(np.int64)

        if length < target_len:
            # Pad metadata arrays to keep batch collation simple
            pad_width = ((0, target_len - length), (0, 0))
            wls = np.pad(wls, pad_width, "edge")  # Pad with edge values
            times = np.pad(times, (0, target_len - length), "edge")

        result["wls"] = torch.from_numpy(
            wls
        ).float()  # Keep as float for collation, convert to double later if needed
        result["time"] = torch.from_numpy(times)
        result["drive_id"] = data_slice.iloc[0]["drive_id"]
        result["phone_name"] = data_slice.iloc[0]["phone_name"]
        result["valid_len"] = length

        # Targets
        if self.mode in ["train", "val"]:
            targets = data_slice[["target_e", "target_n"]].values.astype(np.float32)
            if length < target_len:
                targets = np.pad(
                    targets, ((0, target_len - length), (0, 0)), "constant"
                )

            # (L, 2) -> (2, L)
            tensor_target = torch.from_numpy(targets.transpose(1, 0))
            result["target"] = tensor_target

        return result
