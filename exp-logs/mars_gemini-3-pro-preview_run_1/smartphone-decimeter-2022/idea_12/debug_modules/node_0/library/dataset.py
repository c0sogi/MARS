import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from library.config import Config


class GNSSSequenceDataset(Dataset):
    def __init__(self, df, mode="train", scaler=None):
        """
        PyTorch Dataset for GNSS sequences.

        Args:
            df (pd.DataFrame): The preprocessed dataframe containing features and metadata.
            mode (str): 'train' or 'test'.
            scaler (dict, optional): Dictionary containing 'mean' and 'std' for normalization.
                                     Computed from data if None and mode='train'.
        """
        self.mode = mode
        self.feature_cols = Config.INPUT_FEATURES
        self.target_cols = Config.TARGET_COLS

        # Handle Normalization
        if scaler is None and mode == "train":
            # Compute mean and std for features
            print("Computing feature statistics for normalization...")
            features = df[self.feature_cols].values
            # Handle potential NaNs by filling with column means temporarily
            features_df = pd.DataFrame(features, columns=self.feature_cols)
            features_df = features_df.fillna(features_df.mean())

            mean = features_df.mean().values
            std = features_df.std().values

            # Avoid division by zero
            std[std == 0] = 1.0

            self.scaler = {
                "mean": mean.astype(np.float32),
                "std": std.astype(np.float32),
            }
        else:
            self.scaler = scaler

        # Group by trip (drive_id + phone_name)
        # We need to ensure the sequence is sorted by time
        print(f"Grouping data by trip for {mode} dataset...")
        self.sequences = []

        # Create a unique trip identifier if not present
        if "trip_id" not in df.columns:
            df["trip_id"] = df["drive_id"] + "-" + df["phone_name"]

        # Grouping
        for trip_id, group in df.groupby("trip_id"):
            group = group.sort_values("UnixTimeMillis").reset_index(drop=True)

            # Extract features
            # Fill NaNs with 0 before normalization as a robust fallback
            feat_data = group[self.feature_cols].fillna(0).values.astype(np.float32)

            # Normalize
            if self.scaler:
                feat_data = (feat_data - self.scaler["mean"]) / self.scaler["std"]

            seq_data = {
                "features": feat_data,
                "trip_id": trip_id,
                "wls_pos": group[["wls_lat", "wls_lon"]].values.astype(
                    np.float64
                ),  # Keep high precision for coords
                "timestamps": group["UnixTimeMillis"].values,
            }

            if self.mode == "train":
                target_data = (
                    group[self.target_cols].fillna(0).values.astype(np.float32)
                )
                seq_data["targets"] = target_data

            self.sequences.append(seq_data)

        print(f"Created {len(self.sequences)} sequences.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_data = self.sequences[idx]

        # Prepare features: (L, C) -> (C, L) for 1D Conv
        features = torch.tensor(seq_data["features"]).permute(1, 0)

        item = {
            "features": features,
            "trip_id": seq_data["trip_id"],
            "wls_pos": seq_data["wls_pos"],
            "timestamps": seq_data["timestamps"],
        }

        if self.mode == "train":
            # Targets: (L, C) -> (C, L)
            targets = torch.tensor(seq_data["targets"]).permute(1, 0)
            item["targets"] = targets

        return item


def gnss_collate_fn(batch):
    """
    Collate function to handle variable length sequences.
    Pads sequences to the maximum length in the batch.
    """
    # Find max length in this batch
    max_len = max([item["features"].shape[1] for item in batch])

    # Initialize batch tensors
    batch_size = len(batch)
    n_features = batch[0]["features"].shape[0]

    # Features: (B, C, L)
    padded_features = torch.zeros(
        (batch_size, n_features, max_len), dtype=torch.float32
    )

    # Masks: (B, L) - 1 for valid data, 0 for padding
    masks = torch.zeros((batch_size, max_len), dtype=torch.float32)

    trip_ids = []
    wls_pos_list = []
    timestamps_list = []

    padded_targets = None
    if "targets" in batch[0]:
        n_targets = batch[0]["targets"].shape[0]
        padded_targets = torch.zeros(
            (batch_size, n_targets, max_len), dtype=torch.float32
        )

    for i, item in enumerate(batch):
        seq_len = item["features"].shape[1]

        # Copy features
        padded_features[i, :, :seq_len] = item["features"]

        # Set mask
        masks[i, :seq_len] = 1.0

        # Copy targets if exist
        if padded_targets is not None:
            padded_targets[i, :, :seq_len] = item["targets"]

        trip_ids.append(item["trip_id"])
        wls_pos_list.append(item["wls_pos"])
        timestamps_list.append(item["timestamps"])

    output = {
        "features": padded_features,
        "masks": masks,
        "trip_ids": trip_ids,
        "wls_pos": wls_pos_list,  # List of numpy arrays (variable length)
        "timestamps": timestamps_list,  # List of numpy arrays
    }

    if padded_targets is not None:
        output["targets"] = padded_targets

    return output
