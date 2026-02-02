import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from library.config import Config


class GnssSequenceDataset(Dataset):
    """
    PyTorch Dataset for GNSS sequences.
    Loads processed data grouped by drive and phone, handling feature extraction and normalization.
    """

    def __init__(self, df, is_test=False):
        """
        Args:
            df (pd.DataFrame): Processed dataframe containing features and targets.
            is_test (bool): If True, targets are not expected/loaded.
        """
        self.is_test = is_test

        # Group data by drive_id and phone_name to form continuous sequences
        # Data is assumed to be sorted by time within the dataframe
        self.groups = list(df.groupby(["drive_id", "phone_name"]))

        # Define feature columns matching the data_processing.py logic
        # 3 Strata * 2 Base Features * 4 Stats + 2 Context = 26 Features
        self.feature_cols = []
        stats = ["mean", "std", "min", "max"]
        base_feats = ["Cn0DbHz", "SvElevationDegrees"]
        strata = ["S1", "S2", "S3"]

        for s in strata:
            for feat in base_feats:
                for stat in stats:
                    self.feature_cols.append(f"{s}_{feat}_{stat}")

        # Add Context Features
        self.feature_cols.extend(["Azimuth_Sin", "Azimuth_Cos"])

        # Validate feature count
        if len(self.feature_cols) != Config.IN_CHANNELS:
            raise ValueError(
                f"Feature count mismatch. Expected {Config.IN_CHANNELS}, got {len(self.feature_cols)}"
            )

        # Pre-calculate indices for normalization to avoid doing it in __getitem__ loop
        # Cn0 columns contain 'Cn0DbHz', Elev columns contain 'SvElevationDegrees'
        self.cn0_indices = [
            i for i, c in enumerate(self.feature_cols) if "Cn0DbHz" in c
        ]
        self.elev_indices = [
            i for i, c in enumerate(self.feature_cols) if "SvElevationDegrees" in c
        ]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        (drive_id, phone_name), group_df = self.groups[idx]

        # 1. Extract Features
        # Shape: (Sequence_Length, In_Channels)
        features = group_df[self.feature_cols].values.astype(np.float32)

        # 2. Normalize Features
        # In-place modification for efficiency
        features[:, self.cn0_indices] /= Config.CN0_SCALE
        features[:, self.elev_indices] /= Config.ELEV_SCALE

        features_tensor = torch.from_numpy(features)

        # 3. Prepare Result Dictionary
        result = {
            "features": features_tensor,
            "drive_id": drive_id,
            "phone_name": phone_name,
            "t_millis": torch.from_numpy(
                group_df["UnixTimeMillis"].values.astype(np.int64)
            ),
        }

        # 4. Extract Targets and WLS Baseline
        # WLS is needed for both training (metric calc) and inference (submission)
        if "WLS_Lat" in group_df.columns and "WLS_Lon" in group_df.columns:
            result["wls_lat"] = torch.from_numpy(
                group_df["WLS_Lat"].values.astype(np.float64)
            )
            result["wls_lon"] = torch.from_numpy(
                group_df["WLS_Lon"].values.astype(np.float64)
            )

        if not self.is_test:
            if "Target_North" in group_df.columns and "Target_East" in group_df.columns:
                targets = group_df[["Target_North", "Target_East"]].values.astype(
                    np.float32
                )
                result["targets"] = torch.from_numpy(targets)
            else:
                raise ValueError("Targets missing in training data.")

        return result


def gnss_collate_fn(batch):
    """
    Collate function to pad variable-length sequences into a batch.
    Returns a dictionary with padded tensors and masks.
    """
    # Extract lists from batch
    features_list = [item["features"] for item in batch]
    drive_ids = [item["drive_id"] for item in batch]
    phone_names = [item["phone_name"] for item in batch]
    t_millis_list = [item["t_millis"] for item in batch]

    # Pad Features: (Batch, Max_Seq_Len, Features)
    # batch_first=True
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)

    # Create Mask: (Batch, Max_Seq_Len)
    # True indicates valid data, False indicates padding
    lengths = torch.tensor([len(f) for f in features_list], dtype=torch.long)
    max_len = features_padded.size(1)
    # arange: [0, 1, ..., max_len-1]
    # mask: [[0 < len0, 1 < len0, ...], [0 < len1, ...]]
    mask = torch.arange(max_len)[None, :] < lengths[:, None]

    batch_dict = {
        "features": features_padded,
        "mask": mask,
        "drive_id": drive_ids,
        "phone_name": phone_names,
        "t_millis": t_millis_list,  # List of 1D tensors (variable length)
        "lengths": lengths,
    }

    # Handle Targets if present
    if "targets" in batch[0]:
        targets_list = [item["targets"] for item in batch]
        # Pad targets with 0.0 (masked out by loss function anyway)
        targets_padded = pad_sequence(targets_list, batch_first=True, padding_value=0.0)
        batch_dict["targets"] = targets_padded

    # Handle WLS Baseline (Pad similarly for aligned indexing during metrics/inference)
    if "wls_lat" in batch[0]:
        wls_lat_list = [item["wls_lat"] for item in batch]
        wls_lon_list = [item["wls_lon"] for item in batch]
        batch_dict["wls_lat"] = pad_sequence(
            wls_lat_list, batch_first=True, padding_value=0.0
        )
        batch_dict["wls_lon"] = pad_sequence(
            wls_lon_list, batch_first=True, padding_value=0.0
        )

    return batch_dict
