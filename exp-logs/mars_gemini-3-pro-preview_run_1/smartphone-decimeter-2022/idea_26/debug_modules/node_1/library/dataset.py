import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
import os
from library.config import Config


class GnssSequenceDataset(Dataset):
    """
    PyTorch Dataset for loading stratified GNSS feature sequences.

    Handles:
    - Sequence chunking (sliding window for train, full drive for test)
    - Feature normalization (Z-score)
    - Padding for U-Net architecture compatibility
    - Target alignment
    """

    def __init__(self, df, split="train", config=None, max_samples=None):
        """
        Args:
            df (pd.DataFrame): Preprocessed DataFrame containing features and metadata.
            split (str): 'train', 'val', or 'test'.
            config (Config): Configuration object.
            max_samples (int, optional): Limit dataset size for debugging.
        """
        self.config = config if config else Config()
        self.split = split
        self.feature_names = self.config.get_feature_names()

        # Ensure dataframe is sorted by timestamp within each drive for valid sequencing
        df = df.sort_values(
            by=["drive_id", "phone_name", "UnixTimeMillis"]
        ).reset_index(drop=True)

        # --- Normalization ---
        self.scaler_path = os.path.join(self.config.WORKING_DIR, "scaler_params.json")

        # If training split, compute and save stats. Otherwise load them.
        if self.split == "train":
            self._fit_scaler(df)

        self.scaler_stats = self._load_scaler()

        # Pre-normalize features to speed up __getitem__
        # Convert to float32 for PyTorch
        features_raw = df[self.feature_names].values.astype(np.float32)
        mean_vec = np.array(self.scaler_stats["mean"], dtype=np.float32)
        std_vec = np.array(self.scaler_stats["std"], dtype=np.float32)

        # Avoid division by zero
        std_vec[std_vec == 0] = 1.0

        self.features = (features_raw - mean_vec) / std_vec

        # --- Metadata & Targets ---
        self.metadata = df[["drive_id", "phone_name", "UnixTimeMillis"]].copy()

        # Load WLS Baseline for reconstruction (if available)
        if "WlsLatitudeDegrees" in df.columns and "WlsLongitudeDegrees" in df.columns:
            self.wls_coords = df[
                ["WlsLatitudeDegrees", "WlsLongitudeDegrees"]
            ].values.astype(np.float64)
        else:
            # Fallback for test set if not explicitly merged (though preprocessing should handle it)
            self.wls_coords = np.zeros((len(df), 2), dtype=np.float64)

        # Load Targets (Regression offsets)
        if self.split in ["train", "val"]:
            # Ensure targets exist
            if all(c in df.columns for c in self.config.TARGET_COLUMNS):
                self.targets = df[self.config.TARGET_COLUMNS].values.astype(np.float32)
            else:
                raise ValueError(
                    f"Target columns {self.config.TARGET_COLUMNS} missing in {split} set."
                )
        else:
            self.targets = None

        # --- Sequence Indexing ---
        self.samples = []
        self._create_samples(df)

        # Debugging limit
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _fit_scaler(self, df):
        """Computes mean and std of features and saves to JSON."""
        stats = {
            "mean": df[self.feature_names].mean().tolist(),
            "std": df[self.feature_names].std().tolist(),
        }
        with open(self.scaler_path, "w") as f:
            json.dump(stats, f)

    def _load_scaler(self):
        """Loads scaling parameters from JSON."""
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, "r") as f:
                return json.load(f)
        else:
            # Fallback (should be avoided by running train first)
            print("Warning: Scaler params not found. Using identity scaling.")
            return {
                "mean": [0.0] * len(self.feature_names),
                "std": [1.0] * len(self.feature_names),
            }

    def _create_samples(self, df):
        """
        Generates a list of (start_row, length) tuples defining valid sequences.
        """
        # Create a unique identifier for each continuous drive
        group_ids = df["drive_id"].astype(str) + "_" + df["phone_name"].astype(str)

        # Find indices where the drive changes
        # np.where returns indices where condition is true.
        # We look for transitions between rows.
        group_id_values = group_ids.values
        change_indices = np.where(group_id_values[:-1] != group_id_values[1:])[0] + 1

        # Start indices of every drive
        drive_starts = np.r_[0, change_indices]
        # End indices of every drive
        drive_ends = np.r_[change_indices, len(df)]

        seq_len = self.config.TRAIN_SEQUENCE_LENGTH

        for start, end in zip(drive_starts, drive_ends):
            drive_len = end - start

            if self.split == "train":
                # Training: Sliding window augmentation
                # Stride = half sequence length for overlap
                stride = seq_len // 2

                if drive_len <= seq_len:
                    # If drive is shorter than window, take it as is (will be padded)
                    self.samples.append((start, drive_len))
                else:
                    # Generate sliding windows
                    for i in range(0, drive_len - seq_len + 1, stride):
                        self.samples.append((start + i, seq_len))

                    # Ensure the tail of the drive is used if not perfectly divisible
                    if (drive_len - seq_len) % stride != 0:
                        self.samples.append((start + drive_len - seq_len, seq_len))
            else:
                # Validation/Test: Use full drive as one sequence
                # Batch size must be 1 for these splits due to variable lengths
                self.samples.append((start, drive_len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_row, length = self.samples[idx]

        # 1. Extract Features
        # Shape: (Length, Channels)
        x = self.features[start_row : start_row + length]

        # 2. Determine Padding
        # U-Net with 4 downsamples requires length divisible by 2^4 = 16
        pad_len = 0
        if self.split == "train":
            # Fixed length for training batches
            target_len = self.config.TRAIN_SEQUENCE_LENGTH
            if length < target_len:
                pad_len = target_len - length
        else:
            # Variable length for inference, but must be divisible by 16
            divisor = 16
            remainder = length % divisor
            if remainder != 0:
                pad_len = divisor - remainder

        # 3. Apply Padding
        # Pad end of sequence with zeros
        if pad_len > 0:
            x = np.pad(x, ((0, pad_len), (0, 0)), mode="constant", constant_values=0)

        # 4. Transpose for PyTorch Conv1d
        # Input expected: (Channels, Length)
        x = x.transpose(1, 0)

        # 5. Create Mask (True for valid data, False for padding)
        mask = torch.ones(length + pad_len, dtype=torch.bool)
        if pad_len > 0:
            mask[length:] = False

        sample = {
            "features": torch.from_numpy(x),
            "mask": mask,
            "original_length": length,
            "drive_id": self.metadata.iloc[start_row]["drive_id"],
            "phone_name": self.metadata.iloc[start_row]["phone_name"],
        }

        # 6. Handle Targets
        if self.targets is not None:
            # Shape: (Length, OutputChannels)
            y = self.targets[start_row : start_row + length]
            if pad_len > 0:
                y = np.pad(
                    y, ((0, pad_len), (0, 0)), mode="constant", constant_values=0
                )

            # Transpose to (OutputChannels, Length) for consistency with model output
            y = y.transpose(1, 0)
            sample["targets"] = torch.from_numpy(y)

        # 7. Metadata for reconstruction (Not padded, numpy arrays)
        # Used in inference loop to map predictions back to global coordinates
        sample["wls_coords"] = self.wls_coords[start_row : start_row + length]
        sample["timestamps"] = self.metadata.iloc[start_row : start_row + length][
            "UnixTimeMillis"
        ].values

        return sample
