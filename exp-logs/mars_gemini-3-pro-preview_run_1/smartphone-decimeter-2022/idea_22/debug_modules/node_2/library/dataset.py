import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from library.config import Config
from library.feature_engineering import process_dataset


class GNSSSequenceDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True, debug=False):
        """
        PyTorch Dataset for GNSS sequences.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load cached processed data.
            debug (bool): If True, limits the dataset size for debugging.
        """
        self.split = split
        self.debug = debug

        # Determine metadata path based on split
        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            meta_path = Config.VAL_METADATA_PATH
        elif split == "test":
            meta_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load processed data using feature_engineering module
        # This handles the caching logic required
        self.df = process_dataset(
            meta_path, load_cached_data=load_cached_data, split_name=split
        )

        # Debugging: Limit dataset size
        if self.debug:
            drives = self.df["drive_id"].unique()
            if len(drives) > Config.DEBUG_DRIVE_COUNT:
                selected_drives = drives[: Config.DEBUG_DRIVE_COUNT]
                self.df = self.df[self.df["drive_id"].isin(selected_drives)].copy()
                print(f"Debug mode: Limited dataset to {len(selected_drives)} drives.")

        # Define Feature Columns matching feature_engineering.aggregate_features
        self.feature_cols = []

        # Strata Features
        variables = ["Cn0DbHz", "SvElevationDegrees"]
        stats = Config.STRATA_STATS

        for i in [1, 2, 3]:
            for var in variables:
                for stat in stats:
                    self.feature_cols.append(f"S{i}_{var}_{stat}")

        # Global Features
        self.feature_cols.extend(Config.GLOBAL_FEATURES)

        # Validation: Ensure all expected columns exist
        missing_cols = [c for c in self.feature_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing feature columns in processed data: {missing_cols}"
            )

        # Group by drive and phone to form sequences
        # Sort by UnixTimeMillis to ensure correct temporal order
        self.groups = [
            g.sort_values("UnixTimeMillis")
            for _, g in self.df.groupby(["drive_id", "phone_name"])
        ]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]

        # Extract Features
        # Dataframe shape: (L, C)
        # Target tensor shape: (C, L) for 1D Conv
        features_np = group[self.feature_cols].values.astype(np.float32)
        features = torch.tensor(features_np).transpose(0, 1)

        # Extract Metadata
        # WLS positions and timestamps are needed for reconstructing the final path
        wls_pos = group[["WlsLat", "WlsLon"]].values.astype(np.float64)
        timestamps = group["UnixTimeMillis"].values

        metadata = {
            "drive_id": group["drive_id"].iloc[0],
            "phone_name": group["phone_name"].iloc[0],
            "wls_pos": torch.tensor(wls_pos),  # (L, 2)
            "timestamps": torch.tensor(timestamps),  # (L,)
        }

        # Extract Targets (if available)
        # dEast and dNorth are computed in process_drive if ground truth is present
        if "dEast" in group.columns and "dNorth" in group.columns:
            targets_np = group[["dEast", "dNorth"]].values.astype(np.float32)
            targets = torch.tensor(targets_np).transpose(0, 1)  # (2, L)
            return features, targets, metadata
        else:
            return features, metadata


def gnss_collate_fn(batch):
    """
    Collate function to pad sequences in a batch.

    Args:
        batch: List of tuples returned by __getitem__

    Returns:
        features_padded: (B, C, L_max)
        targets_padded: (B, 2, L_max) or None
        mask: (B, L_max) Boolean mask (True where data is valid)
        metadata_list: List of metadata dicts
    """
    # Check if batch contains targets (Train/Val vs Test)
    has_targets = len(batch[0]) == 3

    features_list = []
    metadata_list = []
    targets_list = []

    for item in batch:
        if has_targets:
            feat, targ, meta = item
            # Transpose back to (L, 2) for pad_sequence
            targets_list.append(targ.transpose(0, 1))
        else:
            feat, meta = item

        # Transpose back to (L, C) for pad_sequence
        features_list.append(feat.transpose(0, 1))
        metadata_list.append(meta)

    # Pad features
    # batch_first=True -> Output (B, L_max, C)
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)

    # Create Mask (True for valid data, False for padding)
    # Calculate lengths based on the processed list
    lengths = torch.tensor([f.shape[0] for f in features_list])
    max_len = features_padded.shape[1]

    # Create mask: (B, L_max)
    # arange(max_len) creates [0, 1, ..., max_len-1]
    # Compare with lengths[:, None] which is (B, 1)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]

    # Permute features to (B, C, L_max) for PyTorch Conv1d
    features_padded = features_padded.permute(0, 2, 1)

    if has_targets:
        # Pad targets
        targets_padded = pad_sequence(targets_list, batch_first=True, padding_value=0.0)
        # Permute targets to (B, 2, L_max)
        targets_padded = targets_padded.permute(0, 2, 1)
        return features_padded, targets_padded, mask, metadata_list
    else:
        return features_padded, mask, metadata_list
