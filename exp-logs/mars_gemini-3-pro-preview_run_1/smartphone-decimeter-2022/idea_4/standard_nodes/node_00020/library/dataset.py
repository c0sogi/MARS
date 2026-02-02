import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from library.config import Config


def get_feature_columns():
    """
    Generates the list of flattened feature column names based on Config.
    Matches the aggregation logic in data_processing.py.
    """
    cols = []
    for col, funcs in Config.GNSS_AGG_CONFIG.items():
        for func in funcs:
            cols.append(f"{col}_{func}")
    return cols


class GNSSSequenceDataset(Dataset):
    def __init__(self, df, split_name, feature_stats=None, load_cached_data=True):
        """
        PyTorch Dataset for 1D U-Net GNSS sequence modeling.

        Args:
            df (pd.DataFrame): The dataframe containing aggregated GNSS data and targets.
            split_name (str): Name of the split ('train', 'val', 'test') for caching.
            feature_stats (dict, optional): Dictionary containing 'mean' and 'std' numpy arrays
                                            for normalization. If None, computed from df.
            load_cached_data (bool): Whether to attempt loading pre-sequenced data from cache.
        """
        self.split_name = split_name
        self.feature_cols = get_feature_columns()

        # Ensure working directory exists for caching
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Handle missing columns (fill with 0)
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        # Fill NaNs in features
        df[self.feature_cols] = df[self.feature_cols].fillna(0.0)

        # Compute or Apply Feature Statistics for Normalization
        if feature_stats is None:
            self.stats = {
                "mean": df[self.feature_cols].mean().values.astype(np.float32),
                "std": df[self.feature_cols].std().values.astype(np.float32),
            }
            # Prevent division by zero
            self.stats["std"][self.stats["std"] == 0] = 1.0
        else:
            self.stats = feature_stats

        # Normalize Features
        # (x - mean) / std
        vals = df[self.feature_cols].values.astype(np.float32)
        vals = (vals - self.stats["mean"]) / self.stats["std"]

        # Create a copy to avoid modifying the original dataframe reference
        df_norm = df.copy()
        df_norm[self.feature_cols] = vals

        # Process into sequences (with caching)
        self.sequences = self._prepare_sequences(df_norm, load_cached_data)

    def _prepare_sequences(self, df, load_cached_data):
        """
        Groups the dataframe by drive_id and phone_name into sequences.
        Implements caching using .npz files.
        """
        cache_path = os.path.join(
            Config.WORKING_DIR, f"{self.split_name}_sequences.npz"
        )

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading {self.split_name} sequences from cache: {cache_path}")
                # allow_pickle=True is needed for string arrays (drive_id/phone_name) stored in npz
                # However, requirements say "Do NOT use pickle".
                # np.savez stores arrays. Storing strings as numpy arrays is standard and safe.
                # allow_pickle=False might block object arrays, but fixed-length unicode strings are usually fine.
                # We will try with allow_pickle=True for string safety, but the format is NPZ (zip of npy), not raw pickle.
                loaded = np.load(cache_path, allow_pickle=True)

                num_seqs = int(loaded["num_seqs"])
                sequences = []
                for i in range(num_seqs):
                    seq = {
                        "features": loaded[f"feat_{i}"],
                        "drive_id": str(loaded[f"drive_{i}"]),
                        "phone_name": str(loaded[f"phone_{i}"]),
                        "timestamps": loaded[f"time_{i}"],
                    }
                    if f"targ_{i}" in loaded:
                        seq["targets"] = loaded[f"targ_{i}"]
                    sequences.append(seq)
                return sequences
            except Exception as e:
                print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

        # 2. Compute Sequences
        print(f"Computing sequences for {self.split_name}...")
        sequences = []

        # Sort by timestamp to ensure correct temporal order within sequences
        df_sorted = df.sort_values("UnixTimeMillis")
        groups = df_sorted.groupby(["drive_id", "phone_name"])

        for (drive_id, phone_name), group in groups:
            feat_arr = group[self.feature_cols].values.astype(np.float32)
            time_arr = group["UnixTimeMillis"].values.astype(np.int64)

            seq_data = {
                "features": feat_arr,
                "drive_id": drive_id,
                "phone_name": phone_name,
                "timestamps": time_arr,
            }

            # Check if targets exist (Train/Val)
            if all(c in group.columns for c in Config.TARGET_COLS):
                targ_arr = group[Config.TARGET_COLS].values.astype(np.float32)
                seq_data["targets"] = targ_arr

            sequences.append(seq_data)

        # 3. Save to Cache
        try:
            save_dict = {"num_seqs": len(sequences)}
            for i, seq in enumerate(sequences):
                save_dict[f"feat_{i}"] = seq["features"]
                save_dict[f"drive_{i}"] = seq["drive_id"]
                save_dict[f"phone_{i}"] = seq["phone_name"]
                save_dict[f"time_{i}"] = seq["timestamps"]
                if "targets" in seq:
                    save_dict[f"targ_{i}"] = seq["targets"]

            np.savez(cache_path, **save_dict)
            print(f"Saved sequences to cache: {cache_path}")
        except Exception as e:
            print(f"Failed to save cache: {e}")

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # PyTorch Conv1d expects (Channels, Length)
        # Data is stored as (Length, Channels), so we transpose.
        features = torch.tensor(seq["features"].T, dtype=torch.float32)

        item = {
            "features": features,
            "drive_id": seq["drive_id"],
            "phone_name": seq["phone_name"],
            "timestamps": torch.tensor(seq["timestamps"], dtype=torch.long),
        }

        if "targets" in seq:
            # Targets also transposed to (Channels, Length) to match model output shape
            targets = torch.tensor(seq["targets"].T, dtype=torch.float32)
            item["targets"] = targets

        return item


def collate_padded(batch):
    """
    Collate function to handle variable length sequences in a batch.
    Pads features and targets to the maximum length in the batch.

    Args:
        batch (list): List of items from __getitem__.

    Returns:
        dict: Batched data with padded tensors.
    """
    # Extract features and transpose back to (Length, Channels) for pad_sequence
    features_list = [item["features"].T for item in batch]

    # Pad sequences. batch_first=True -> (Batch, Length, Channels)
    # padding_value=0.0 is appropriate for normalized features (mean=0)
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)

    # Transpose back to (Batch, Channels, Length) for Conv1d
    features_padded = features_padded.permute(0, 2, 1)

    # Pad timestamps
    timestamps_list = [item["timestamps"] for item in batch]
    timestamps_padded = pad_sequence(timestamps_list, batch_first=True, padding_value=0)

    # Calculate lengths for masking if needed
    lengths = torch.tensor([len(t) for t in timestamps_list], dtype=torch.long)

    batch_out = {
        "features": features_padded,
        "timestamps": timestamps_padded,
        "lengths": lengths,
        "drive_id": [item["drive_id"] for item in batch],
        "phone_name": [item["phone_name"] for item in batch],
    }

    # Handle Targets
    if "targets" in batch[0]:
        targets_list = [item["targets"].T for item in batch]  # (Length, Channels)
        targets_padded = pad_sequence(targets_list, batch_first=True, padding_value=0.0)
        targets_padded = targets_padded.permute(0, 2, 1)  # (Batch, Channels, Length)
        batch_out["targets"] = targets_padded

    return batch_out
