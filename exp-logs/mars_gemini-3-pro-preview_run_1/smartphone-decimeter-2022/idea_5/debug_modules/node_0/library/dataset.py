import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config
from library.data_processing import load_dataset


class GnssSequenceDataset(Dataset):
    def __init__(
        self, metadata_path, split="train", max_seq_len=None, load_cached_data=True
    ):
        """
        PyTorch Dataset for GNSS sequences.
        Loads preprocessed data, groups it by drive, and prepares padded tensors.

        Args:
            metadata_path (str): Path to the metadata CSV.
            split (str): 'train', 'val', or 'test'. Determines if targets are loaded.
            max_seq_len (int, optional): Fixed sequence length for padding/truncation.
                                         If None, uses the maximum length found in the data (capped at 4096).
            load_cached_data (bool): Whether to use cached parquet files via load_dataset.
        """
        self.split = split
        self.max_seq_len = max_seq_len

        # Load the concatenated dataframe using the provided library function
        # This handles caching of the heavy preprocessing steps
        self.df = load_dataset(metadata_path, load_cached_data=load_cached_data)

        if self.df.empty:
            print(f"Warning: Dataset at {metadata_path} is empty.")
            self.grouped = []
            return

        # Construct the list of feature column names based on Config
        self.feature_cols = ["SatCount"]
        for col, stats in Config.AGGREGATION_SPECS.items():
            for stat in stats:
                self.feature_cols.append(f"{col}_{stat}")

        # Ensure all feature columns exist; fill missing with 0.0
        # This handles potential mismatches if cache was generated with different specs
        for c in self.feature_cols:
            if c not in self.df.columns:
                self.df[c] = 0.0

        # Fill NaNs in features (e.g., std of single value is NaN)
        self.df[self.feature_cols] = self.df[self.feature_cols].fillna(0.0)

        # Sort by timestamp to ensure correct temporal order within sequences
        self.df = self.df.sort_values(by=["drive_id", "phone_name", "UnixTimeMillis"])

        # Group by drive and phone to form sequences
        # self.grouped becomes a list of ((drive_id, phone_name), dataframe_subset)
        self.grouped = list(self.df.groupby(["drive_id", "phone_name"]))

        # Determine max sequence length
        if self.max_seq_len is None:
            if len(self.grouped) > 0:
                max_len_in_data = max(len(g[1]) for g in self.grouped)
                # Cap at 4096 to prevent OOM on exceptionally long outlier drives
                self.max_seq_len = min(max_len_in_data, 4096)
            else:
                self.max_seq_len = 1024  # Fallback

        print(
            f"Dataset {split}: Loaded {len(self.grouped)} sequences. Max sequence length set to {self.max_seq_len}."
        )

    def __len__(self):
        return len(self.grouped)

    def __getitem__(self, idx):
        keys, df_seq = self.grouped[idx]
        drive_id, phone_name = keys

        # --- 1. Extract Features ---
        # Shape: (Seq_Len, Num_Features)
        features = df_seq[self.feature_cols].values.astype(np.float32)

        # Log-transform Uncertainty features to compress dynamic range
        # This is a critical preprocessing step for NN stability
        unc_col = "RawPseudorangeUncertaintyMeters_mean"
        if unc_col in self.feature_cols:
            unc_idx = self.feature_cols.index(unc_col)
            features[:, unc_idx] = np.log1p(features[:, unc_idx])

        current_len = features.shape[0]

        # --- 2. Extract Targets ---
        # We predict residuals: DeltaEast, DeltaNorth
        if self.split in ["train", "val"] and "DeltaEast" in df_seq.columns:
            targets = df_seq[["DeltaEast", "DeltaNorth"]].values.astype(np.float32)
        else:
            # Dummy targets for test set or if columns missing
            targets = np.zeros((current_len, Config.OUT_CHANNELS), dtype=np.float32)

        # --- 3. Extract Metadata ---
        # Necessary for reconstructing Lat/Lon from predicted ENU residuals
        meta_cols = ["UnixTimeMillis", "WlsLat", "WlsLon", "WlsAlt"]
        # Ensure columns exist (Test set preprocessing ensures this, but safety check)
        for c in meta_cols:
            if c not in df_seq.columns:
                df_seq[c] = 0.0

        metadata = df_seq[meta_cols].values.astype(np.float64)

        # --- 4. Padding / Truncation ---
        # We need uniform length for batching.
        # Output format for Conv1d: (Channels, Length)

        # Truncate if longer than max_seq_len
        if current_len > self.max_seq_len:
            features = features[: self.max_seq_len]
            targets = targets[: self.max_seq_len]
            metadata = metadata[: self.max_seq_len]
            current_len = self.max_seq_len

        # Pad if shorter than max_seq_len
        pad_len = self.max_seq_len - current_len

        if pad_len > 0:
            # Pad with zeros
            features = np.pad(
                features, ((0, pad_len), (0, 0)), mode="constant", constant_values=0
            )
            targets = np.pad(
                targets, ((0, pad_len), (0, 0)), mode="constant", constant_values=0
            )
            metadata = np.pad(
                metadata, ((0, pad_len), (0, 0)), mode="constant", constant_values=0
            )

        # Create Mask (1 for valid data, 0 for padding)
        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:current_len] = 1.0

        # Transpose to (Channels, Length) for PyTorch Conv1d
        features_tensor = torch.tensor(features).permute(1, 0)  # (C, L)
        targets_tensor = torch.tensor(targets).permute(1, 0)  # (2, L)
        mask_tensor = torch.tensor(mask)  # (L,)

        # Info dictionary for evaluation/submission
        info = {
            "drive_id": drive_id,
            "phone_name": phone_name,
            "metadata": metadata,  # Numpy array (Batch, L, 4)
            "original_length": current_len,
        }

        return features_tensor, targets_tensor, mask_tensor, info
