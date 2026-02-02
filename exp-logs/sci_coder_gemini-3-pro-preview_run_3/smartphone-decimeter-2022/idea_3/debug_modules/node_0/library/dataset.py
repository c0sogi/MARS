import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from library import config
from library import data_factory


class SmartphoneSequenceDataset(Dataset):
    def __init__(self, df, scaler=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Processed dataframe containing features and targets.
            scaler (StandardScaler, optional): Scaler for normalizing features.
            mode (str): 'train', 'val', or 'test'.
        """
        self.mode = mode
        self.trip_ids = df["tripId"].unique()
        self.feature_cols = config.FEATURE_NAMES

        # Group data by tripId to form sequences
        # Assuming df is already sorted by tripId and utcTimeMillis from data_factory
        self.grouped = df.groupby("tripId")

        # Pre-process sequences into list of numpy arrays to avoid overhead in __getitem__
        self.sequences = []
        self.targets = []
        self.meta = []  # Stores (tripId, timestamps, wls_pos)

        # Collect all features for scaling if training
        all_features = []

        for trip_id in self.trip_ids:
            group = self.grouped.get_group(trip_id)

            # Extract features
            feats = group[self.feature_cols].values.astype(np.float32)

            # Extract targets: dLat, dLon
            targs = group[["dLat", "dLon"]].values.astype(np.float32)

            # Meta info for reconstruction/submission
            # lat_wls, lon_wls are needed to add predictions back
            timestamps = group["utcTimeMillis"].values
            wls_pos = group[["lat_wls", "lon_wls"]].values.astype(np.float32)

            self.sequences.append(feats)
            self.targets.append(targs)
            self.meta.append(
                {"tripId": trip_id, "timestamps": timestamps, "wls_pos": wls_pos}
            )

            if mode == "train" and scaler is None:
                all_features.append(feats)

        # Handle Scaling
        if mode == "train":
            if scaler is None:
                self.scaler = StandardScaler()
                # Concatenate all features to fit
                if all_features:
                    self.scaler.fit(np.concatenate(all_features, axis=0))
            else:
                self.scaler = scaler
        else:
            self.scaler = scaler

    def __len__(self):
        return len(self.trip_ids)

    def __getitem__(self, idx):
        features = self.sequences[idx]
        targets = self.targets[idx]
        meta = self.meta[idx]

        # Normalize features
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)

        return {
            "features": features_tensor,
            "targets": targets_tensor,
            "tripId": meta["tripId"],
            "timestamps": meta["timestamps"],
            "wls_pos": meta["wls_pos"],
        }


def collate_fn(batch):
    """
    Collate function to pad sequences and create masks.
    """
    # Extract items
    features_list = [item["features"] for item in batch]
    targets_list = [item["targets"] for item in batch]
    trip_ids = [item["tripId"] for item in batch]
    timestamps_list = [item["timestamps"] for item in batch]
    wls_pos_list = [item["wls_pos"] for item in batch]

    # Get lengths
    lengths = torch.tensor([len(f) for f in features_list], dtype=torch.long)

    # Pad sequences
    # batch_first=True results in (batch, max_seq_len, feature_dim)
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    targets_padded = pad_sequence(targets_list, batch_first=True, padding_value=0.0)

    # Create mask (True for valid tokens, False for padding)
    # Shape: (batch, max_seq_len)
    max_len = features_padded.size(1)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]

    return {
        "features": features_padded,
        "targets": targets_padded,
        "mask": mask,
        "lengths": lengths,
        "tripIds": trip_ids,
        "timestamps": timestamps_list,  # List of numpy arrays (variable length)
        "wls_pos": wls_pos_list,  # List of numpy arrays
    }


def get_dataloaders(batch_size=32, num_workers=4):
    """
    Loads data, creates datasets and dataloaders.

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    # 1. Load Data
    train_df = data_factory.process_dataset(
        config.TRAIN_METADATA_PATH, config.TRAIN_CACHE_PATH
    )
    val_df = data_factory.process_dataset(
        config.VAL_METADATA_PATH, config.VAL_CACHE_PATH
    )
    test_df = data_factory.process_dataset(
        config.TEST_METADATA_PATH, config.TEST_CACHE_PATH
    )

    # 2. Create Datasets
    # Train dataset fits the scaler
    train_dataset = SmartphoneSequenceDataset(train_df, mode="train")
    scaler = train_dataset.scaler

    val_dataset = SmartphoneSequenceDataset(val_df, scaler=scaler, mode="val")
    test_dataset = SmartphoneSequenceDataset(test_df, scaler=scaler, mode="test")

    # 3. Create Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, scaler
