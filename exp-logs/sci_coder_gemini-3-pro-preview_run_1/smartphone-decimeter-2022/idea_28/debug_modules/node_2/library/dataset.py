import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import pandas as pd
import library.config as config
from library.features import generate_dataset


def get_feature_columns():
    """
    Reconstructs the list of feature columns to ensure consistent ordering
    matching the logic in library/features.py.
    """
    # Global Context Features
    global_cols = [
        "global_SatCount",
        "global_PrUncMean",
        "global_SinAzCentroid",
        "global_CosAzCentroid",
    ]

    # Binned Features
    # Logic matches features.py:
    # for s in range(sectors): for q in range(strata): for feat in base: for stat in stats:
    bin_cols = []
    stats = ["mean", "std", "min", "max"]
    base_feats = ["Cn0DbHz", "SvElevationDegrees"]

    for s in range(config.AZIMUTH_SECTORS):
        for q in range(config.QUALITY_STRATA):
            bin_id = f"s{s}_q{q}"
            for feat in base_feats:
                for stat in stats:
                    col_name = f"{stat}_{feat}_{bin_id}"
                    bin_cols.append(col_name)

    return global_cols + bin_cols


class GnssSequenceDataset(Dataset):
    def __init__(self, split, load_cached_data=True):
        """
        PyTorch Dataset for GNSS sequences.
        Groups the flat processed dataframe into sequences by (drive_id, phone_name).

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from cache or re-process.
        """
        self.split = split

        # Load the processed dataframe (flat 1Hz records)
        # generate_dataset handles caching and processing logic
        self.df = generate_dataset(split, load_cached_data=load_cached_data)

        # Identify feature columns
        self.feature_cols = get_feature_columns()

        # Verify feature columns exist
        missing = [c for c in self.feature_cols if c not in self.df.columns]
        if missing:
            raise RuntimeError(
                f"Missing feature columns in dataset: {missing}. "
                f"Cache might be stale. Try setting load_cached_data=False."
            )

        # Group by drive and phone to create sequences
        # We sort by UnixTimeMillis inside generate_dataset, so groups are temporally ordered.
        self.groups = [
            group
            for _, group in self.df.groupby(["drive_id", "phone_name"], sort=False)
        ]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group_df = self.groups[idx]

        # Extract Features
        # Shape: (Length, Channels)
        x = group_df[self.feature_cols].values.astype(np.float32)

        # Extract Targets (East, North)
        # Shape: (Length, 2)
        y = group_df[["target_East", "target_North"]].values.astype(np.float32)

        # Extract Metadata for reconstruction/submission
        # We need WLS positions to add the predicted offsets back
        wls = group_df[["WlsLat", "WlsLon"]].values.astype(np.float64)
        timestamps = group_df["UnixTimeMillis"].values.astype(np.int64)

        # Metadata strings (take the first one as they are constant per group)
        drive_id = group_df.iloc[0]["drive_id"]
        phone_name = group_df.iloc[0]["phone_name"]

        # Transpose for PyTorch Conv1d: Input must be (Channels, Length)
        x = torch.tensor(x).transpose(0, 1)  # -> (C, L)
        y = torch.tensor(y).transpose(0, 1)  # -> (2, L)

        return {
            "features": x,
            "targets": y,
            "wls": wls,
            "timestamps": timestamps,
            "drive_id": drive_id,
            "phone_name": phone_name,
        }


def collate_fn(batch):
    """
    Custom collate function to pad sequences in a batch.
    Ensures sequence length is divisible by 16 for U-Net compatibility.
    """
    # Separate elements
    # pad_sequence expects (L, C), so we transpose back temporarily
    features = [item["features"].transpose(0, 1) for item in batch]
    targets = [item["targets"].transpose(0, 1) for item in batch]

    # Pad sequences to max length in batch
    # batch_first=True -> (Batch, Length, Channels)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0.0)

    # Get original lengths for masking
    lengths = torch.tensor([f.shape[0] for f in features])
    current_max_len = features_padded.shape[1]

    # Ensure max_len is divisible by 16 (2^4) for U-Net pooling operations
    # This avoids shape mismatch errors in the decoder concatenation
    pad_alignment = 16
    remainder = current_max_len % pad_alignment

    if remainder != 0:
        pad_amt = pad_alignment - remainder
        # Pad features: (B, L, C) -> Pad last dim 0, 2nd last dim pad_amt
        # F.pad format for 3D input (B, L, C) is (pad_left, pad_right, pad_top, pad_bottom)
        # covering the last two dimensions (C, L).
        # We want to pad L (dimension 1), which corresponds to pad_top/pad_bottom in F.pad semantics for 4D,
        # but for 3D tensor it pads starting from last dim.
        # Format: (pad_last_dim_left, pad_last_dim_right, pad_2nd_last_left, pad_2nd_last_right)
        # We want to pad dim 1 (Length). Dim 2 is Channels.
        # So we pad (0, 0, 0, pad_amt)
        features_padded = F.pad(features_padded, (0, 0, 0, pad_amt), "constant", 0.0)
        targets_padded = F.pad(targets_padded, (0, 0, 0, pad_amt), "constant", 0.0)
        current_max_len += pad_amt

    # Generate mask (1 for valid data, 0 for padding)
    # Shape: (Batch, Length)
    mask = torch.arange(current_max_len).expand(
        len(lengths), current_max_len
    ) < lengths.unsqueeze(1)

    # Transpose back to (Batch, Channels, Length) for Conv1d
    features_padded = features_padded.transpose(1, 2)
    targets_padded = targets_padded.transpose(1, 2)

    # Collect metadata
    wls_list = [item["wls"] for item in batch]
    timestamps_list = [item["timestamps"] for item in batch]
    drive_ids = [item["drive_id"] for item in batch]
    phone_names = [item["phone_name"] for item in batch]

    return {
        "features": features_padded,  # (B, C, L)
        "targets": targets_padded,  # (B, 2, L)
        "mask": mask,  # (B, L)
        "wls": wls_list,  # List of np.arrays
        "timestamps": timestamps_list,  # List of np.arrays
        "drive_ids": drive_ids,
        "phone_names": phone_names,
    }
