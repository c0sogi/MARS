import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import library.config as C
import library.data_processing as DP


def get_feature_columns():
    """
    Generates the list of feature column names based on the aggregation map
    and derived columns defined in the configuration.
    """
    cols = []
    for col, stats in C.AGGREGATION_MAP.items():
        for stat in stats:
            cols.append(f"{col}_{stat}")
    cols.extend(C.DERIVED_COLS)
    return cols


class GNSSSequenceDataset(Dataset):
    def __init__(self, sequences, split="train", mean=None, std=None):
        """
        Args:
            sequences (list[pd.DataFrame]): List of DataFrames, where each DF is a drive sequence.
            split (str): 'train', 'val', or 'test'.
            mean (np.array): Mean for normalization. If None, computed from sequences (only for train).
            std (np.array): Std for normalization. If None, computed from sequences (only for train).
        """
        self.sequences = sequences
        self.split = split
        self.feature_cols = get_feature_columns()

        # Compute or set normalization stats
        if mean is None or std is None:
            if split == "train" and len(sequences) > 0:
                all_features = np.concatenate(
                    [seq[self.feature_cols].values for seq in sequences], axis=0
                )
                self.mean = np.mean(all_features, axis=0).astype(np.float32)
                self.std = np.std(all_features, axis=0).astype(np.float32)
                # Avoid division by zero
                self.std[self.std < 1e-6] = 1.0
            else:
                # Default to identity if not provided and not training
                self.mean = np.zeros(len(self.feature_cols), dtype=np.float32)
                self.std = np.ones(len(self.feature_cols), dtype=np.float32)
        else:
            self.mean = mean.astype(np.float32)
            self.std = std.astype(np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        df = self.sequences[idx]

        # 1. Extract Features
        # Fill NaNs with 0 before normalization (though processing should have handled this)
        features = df[self.feature_cols].fillna(0).values.astype(np.float32)

        # Normalize
        features = (features - self.mean) / self.std

        # 2. Extract Metadata
        # Phone index is constant for the drive
        phone_idx = df["phone_idx"].iloc[0]

        # Timestamps
        timestamps = df["UnixTimeMillis"].values

        # 3. Extract Baseline WLS
        # These are needed to reconstruct absolute position from predicted residuals
        wls_pos = df[["wls_lat", "wls_lon", "wls_alt"]].values.astype(np.float64)

        # 4. Extract Targets (if available)
        targets = np.zeros((len(df), C.OUTPUT_DIM), dtype=np.float32)
        if self.split != "test" and all(c in df.columns for c in C.TARGET_COLS):
            targets = df[C.TARGET_COLS].values.astype(np.float32)

        return {
            "features": torch.tensor(features),
            "targets": torch.tensor(targets),
            "wls_pos": torch.tensor(wls_pos),
            "phone_idx": torch.tensor(phone_idx, dtype=torch.long),
            "timestamps": torch.tensor(timestamps, dtype=torch.long),
        }


def gnss_collate_fn(batch):
    """
    Collate function to pad variable length sequences in a batch.
    """
    features_list = [item["features"] for item in batch]
    targets_list = [item["targets"] for item in batch]
    wls_pos_list = [item["wls_pos"] for item in batch]
    phone_idx_list = [item["phone_idx"] for item in batch]
    timestamps_list = [item["timestamps"] for item in batch]

    # Pad sequences
    # batch_first=True -> (Batch, Time, Feat)
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    targets_padded = pad_sequence(targets_list, batch_first=True, padding_value=0.0)
    wls_pos_padded = pad_sequence(wls_pos_list, batch_first=True, padding_value=0.0)
    timestamps_padded = pad_sequence(timestamps_list, batch_first=True, padding_value=0)

    # Stack scalar metadata
    phone_idxs = torch.stack(phone_idx_list)

    # Create Padding Mask
    # True indicates padding (ignored position), False indicates real data
    # Shape: (Batch, Time)
    batch_size = len(batch)
    max_len = features_padded.size(1)
    lengths = torch.tensor([len(x) for x in features_list], dtype=torch.long)

    padding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, length in enumerate(lengths):
        padding_mask[i, length:] = True

    return {
        "features": features_padded,
        "targets": targets_padded,
        "wls_pos": wls_pos_padded,
        "phone_idx": phone_idxs,
        "padding_mask": padding_mask,
        "timestamps": timestamps_padded,
        "lengths": lengths,
    }


def get_datasets(load_cached_data=True, max_drives=None):
    """
    Factory function to create train, val, and test datasets.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed parquet files.
        max_drives (int): If set, limits the number of drives per split for debugging.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Load sequences using data_processing module
    train_seqs = DP.prepare_sequences(
        os.path.join(C.METADATA_DIR, "train_metadata.csv"),
        load_cached_data=load_cached_data,
        split_name="train",
    )

    val_seqs = DP.prepare_sequences(
        os.path.join(C.METADATA_DIR, "val_metadata.csv"),
        load_cached_data=load_cached_data,
        split_name="val",
    )

    test_seqs = DP.prepare_sequences(
        os.path.join(C.METADATA_DIR, "test_metadata.csv"),
        load_cached_data=load_cached_data,
        split_name="test",
    )

    # Debugging: Limit size
    if max_drives is not None:
        train_seqs = train_seqs[:max_drives]
        val_seqs = val_seqs[:max_drives]
        test_seqs = test_seqs[:max_drives]

    # Create Train Dataset first to compute normalization stats
    train_dataset = GNSSSequenceDataset(train_seqs, split="train")

    # Use training stats for validation and test
    val_dataset = GNSSSequenceDataset(
        val_seqs, split="val", mean=train_dataset.mean, std=train_dataset.std
    )

    test_dataset = GNSSSequenceDataset(
        test_seqs, split="test", mean=train_dataset.mean, std=train_dataset.std
    )

    return train_dataset, val_dataset, test_dataset
