import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


class GnssSequenceDataset(Dataset):
    def __init__(
        self, df, mode="train", window_size=Config.TRAIN_WINDOW_SIZE, stride=None
    ):
        """
        Args:
            df (pd.DataFrame): Preprocessed data containing features and targets.
            mode (str): 'train', 'val', or 'test'.
            window_size (int): Length of the sequence window.
            stride (int): Stride for sliding window. If None, defaults to window_size.
        """
        self.df = df.copy()
        self.mode = mode
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size

        # Identify Feature Columns based on Config
        self.feature_cols = []

        # Global Features
        self.feature_cols.extend(
            [
                "global_cn0_mean",
                "global_cn0_std",
                "global_cn0_min",
                "global_cn0_max",
                "global_elev_mean",
                "global_elev_std",
                "global_elev_min",
                "global_elev_max",
                "global_sat_count",
                "global_pr_unc_mean",
            ]
        )

        # Panorama Features
        for i in range(Config.NUM_AZIMUTH_BINS):
            self.feature_cols.extend(
                [f"bin_{i}_cn0_max", f"bin_{i}_elev_mean", f"bin_{i}_sat_count"]
            )

        # Ensure columns exist
        missing_cols = [c for c in self.feature_cols if c not in self.df.columns]
        if missing_cols:
            # In case of empty dataframe or missing cols, fill with 0
            for c in missing_cols:
                self.df[c] = 0.0

        # Prepare samples (windows)
        self.samples = []
        self._prepare_windows()

    def _prepare_windows(self):
        # Group by drive to ensure continuity
        # Note: In test mode, drive_id and phone_name identify the sequence
        grouped = self.df.groupby(["drive_id", "phone_name"])

        for (drive, phone), group in grouped:
            # Sort by time just in case
            group = group.sort_values("UnixTimeMillis")
            indices = group.index.to_numpy()
            seq_len = len(indices)

            if self.mode == "train":
                # Create sliding windows
                for start in range(0, seq_len, self.stride):
                    end = min(start + self.window_size, seq_len)
                    # If the remaining chunk is too small (e.g., < 10% of window), skip it unless it's the only one
                    if (end - start) < (self.window_size // 10) and start > 0:
                        continue

                    # For training, we usually want fixed size.
                    # If chunk < window_size, we can pad.
                    self.samples.append(
                        (indices[start:end], True)
                    )  # True = needs padding if short
            else:
                # Validation/Test: Return full sequence
                # We handle variable length in the model or collate_fn, or batch_size=1
                self.samples.append((indices, False))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        indices, pad_flag = self.samples[idx]

        # Extract data
        subset = self.df.loc[indices]

        # 1. Features
        features = subset[self.feature_cols].values.astype(np.float32)

        # Normalize Features (Simple robust scaling)
        # Cn0: roughly 10-50 -> (x-30)/10
        # Elev: 0-90 -> x/90
        # Counts: 0-30 -> x/30
        # PrUnc: Log1p

        # Indices for specific feature types
        # Global: 0-9
        # Bins: 10-33 (8 bins * 3 feats) -> 0,1,2, 3,4,5...

        # Global Normalization
        features[:, 0:4] = (features[:, 0:4] - 30.0) / 10.0  # Cn0
        features[:, 4:8] = features[:, 4:8] / 90.0  # Elev
        features[:, 8] = features[:, 8] / 30.0  # Sat Count
        features[:, 9] = np.log1p(features[:, 9])  # Pr Unc

        # Panorama Normalization
        # Structure: [Cn0, Elev, Count] repeated
        for i in range(Config.NUM_AZIMUTH_BINS):
            base = 10 + i * 3
            features[:, base] = (features[:, base] - 30.0) / 10.0  # Cn0
            features[:, base + 1] = features[:, base + 1] / 90.0  # Elev
            features[:, base + 2] = (
                features[:, base + 2] / 10.0
            )  # Count (per bin is smaller)

        # 2. Targets & Baseline
        # Baseline WLS
        baseline_lat = subset["wls_lat"].values.astype(np.float64)
        baseline_lon = subset["wls_lon"].values.astype(np.float64)

        if self.mode != "test":
            # Targets: ENU offsets
            target_e = subset["target_east"].values.astype(np.float32)
            target_n = subset["target_north"].values.astype(np.float32)
            targets = np.stack([target_e, target_n], axis=1)  # (T, 2)
        else:
            # Dummy targets for test
            targets = np.zeros((len(indices), 2), dtype=np.float32)

        # 3. Padding (if required for training fixed size batches)
        seq_len = features.shape[0]
        if pad_flag and seq_len < self.window_size:
            pad_len = self.window_size - seq_len
            # Pad features with 0
            features = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
            # Pad targets with 0
            targets = np.pad(targets, ((0, pad_len), (0, 0)), mode="constant")
            # Pad baseline (just repeat last or 0, doesn't matter as long as we track length)
            baseline_lat = np.pad(baseline_lat, (0, pad_len), mode="edge")
            baseline_lon = np.pad(baseline_lon, (0, pad_len), mode="edge")

            # Create a mask for valid data
            mask = np.concatenate([np.ones(seq_len), np.zeros(pad_len)]).astype(
                np.float32
            )
        else:
            mask = np.ones(seq_len, dtype=np.float32)

        # 4. Transpose for PyTorch (Channels, Time)
        # Features: (T, C) -> (C, T)
        features = torch.tensor(features).permute(1, 0)
        targets = torch.tensor(targets).permute(1, 0)

        # Metadata
        meta = {
            "drive_id": subset["drive_id"].iloc[0],
            "phone_name": subset["phone_name"].iloc[0],
            "timestamp": subset["UnixTimeMillis"].values,  # Int64 array
            "baseline_lat": baseline_lat,
            "baseline_lon": baseline_lon,
            "mask": torch.tensor(mask),
        }

        return features, targets, meta
