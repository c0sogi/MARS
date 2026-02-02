import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.data_processing import process_dataset


class GnssDataset(Dataset):
    def __init__(self, metadata_path, mode="train", load_cached_data=True, debug=False):
        """
        PyTorch Dataset for GNSS sequences.

        Args:
            metadata_path (str): Path to metadata CSV.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Use cached parquet files if available.
            debug (bool): If True, use a small subset of data.
        """
        self.mode = mode
        self.debug = debug

        # Load processed data
        self.df = process_dataset(
            metadata_path, mode=mode, load_cached_data=load_cached_data
        )

        if self.df.empty:
            print(f"Warning: Dataset for {mode} is empty.")
            self.trips = []
            return

        # Define Feature Columns
        # L1 Features
        self.l1_cols = []
        for feat in Config.STAT_FEATURES:
            self.l1_cols.extend(
                [f"L1_{feat}_{stat}" for stat in ["mean", "std", "min", "max"]]
            )

        # L5 Features
        self.l5_cols = []
        for feat in Config.STAT_FEATURES:
            self.l5_cols.extend(
                [f"L5_{feat}_{stat}" for stat in ["mean", "std", "min", "max"]]
            )

        # Global Features
        # Note: Az_X and Az_Y are added by data_processing.py even if not in Config.GLOBAL_FEATURES
        self.global_cols = [
            "SatCount",
            "RawPseudorangeUncertaintyMeters",
            "Az_X",
            "Az_Y",
        ]

        self.feature_cols = self.l1_cols + self.l5_cols + self.global_cols
        self.input_dim = len(self.feature_cols)

        # Group by Trip
        # For test data, tripId is unique per submission requirement, but physically it maps to drive+phone
        # We group by drive_id and phone_name to ensure temporal continuity
        self.grouped = self.df.sort_values("UnixTimeMillis").groupby(
            ["drive_id", "phone_name"]
        )
        self.trips = list(self.grouped.groups.keys())

        if self.debug:
            self.trips = self.trips[:10]  # Limit to 10 trips for debugging

    def __len__(self):
        return len(self.trips)

    def __getitem__(self, idx):
        drive_id, phone_name = self.trips[idx]
        group = self.grouped.get_group((drive_id, phone_name))

        # Extract Features
        features = group[self.feature_cols].values.astype(np.float32)

        # Cite debug_lesson_11: Filter for Finiteness
        # Ensure no NaNs or Infs enter the model (final safety net)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply Scaling (Heuristic)
        # Cn0: 0-60 -> 0-1
        # Elev: 0-90 -> 0-1
        # SatCount: 0-50 -> 0-1
        # Uncertainty: log1p -> 0-1 approx

        # Indices for scaling
        cn0_indices = [i for i, c in enumerate(self.feature_cols) if "Cn0" in c]
        elev_indices = [i for i, c in enumerate(self.feature_cols) if "Elevation" in c]
        sat_idx = (
            self.feature_cols.index("SatCount")
            if "SatCount" in self.feature_cols
            else -1
        )
        unc_idx = (
            self.feature_cols.index("RawPseudorangeUncertaintyMeters")
            if "RawPseudorangeUncertaintyMeters" in self.feature_cols
            else -1
        )

        if len(cn0_indices) > 0:
            features[:, cn0_indices] /= 60.0
        if len(elev_indices) > 0:
            features[:, elev_indices] /= 90.0
        if sat_idx != -1:
            features[:, sat_idx] /= 50.0
        if unc_idx != -1:
            features[:, unc_idx] = np.log1p(features[:, unc_idx]) / 5.0

        # Extract Targets
        if self.mode in ["train", "val"]:
            targets = group[["Target_E", "Target_N"]].values.astype(np.float32)
        else:
            # Dummy targets for test
            targets = np.zeros((len(group), 2), dtype=np.float32)

        # Meta info for reconstruction
        meta = {
            "drive_id": drive_id,
            "phone_name": phone_name,
            "UnixTimeMillis": group["UnixTimeMillis"].values,
            "Wls_Lat": group["Wls_Lat"].values,
            "Wls_Lon": group["Wls_Lon"].values,
            "Wls_Alt": group["Wls_Alt"].values,
        }

        # If test mode, we might need tripId if it exists in the dataframe
        if "tripId" in group.columns:
            meta["tripId"] = group["tripId"].values

        return torch.tensor(features), torch.tensor(targets), meta


def collate_fn(batch):
    """
    Collate function to pad variable length sequences.
    """
    features, targets, metas = zip(*batch)

    # Pad sequences (Batch, Len, Dim)
    # batch_first=True
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0.0)

    # Create Mask (Batch, Len) - True for real data, False for padding
    lengths = torch.tensor([len(f) for f in features])
    max_len = padded_features.shape[1]
    mask = torch.arange(max_len)[None, :] < lengths[:, None]

    return padded_features, padded_targets, mask, metas


def get_dataloaders(debug=False):
    """
    Factory function to create dataloaders for train, val, and test.
    """
    # Train Loader
    train_ds = GnssDataset(
        Config.TRAIN_METADATA_PATH, mode="train", load_cached_data=True, debug=debug
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Val Loader
    val_ds = GnssDataset(
        Config.VAL_METADATA_PATH, mode="val", load_cached_data=True, debug=debug
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Test Loader
    test_ds = GnssDataset(
        Config.TEST_METADATA_PATH, mode="test", load_cached_data=True, debug=debug
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
