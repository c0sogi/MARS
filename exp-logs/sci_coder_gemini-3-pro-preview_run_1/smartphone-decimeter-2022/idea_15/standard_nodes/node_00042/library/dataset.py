import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from library.config import Config

# Explicit feature list matching data_processing.py output
FEATURE_COLUMNS = [
    "Cn0DbHz_mean",
    "Cn0DbHz_std",
    "Cn0DbHz_min",
    "Cn0DbHz_max",
    "SvElevationDegrees_mean",
    "SvElevationDegrees_std",
    "SvElevationDegrees_min",
    "SvElevationDegrees_max",
    "sin_az_mean",
    "cos_az_mean",
    "weighted_sin_az",
    "weighted_cos_az",
    "SatCount",
    "RawPseudorangeUncertaintyMeters_mean",
]


class GnssSequenceDataset(Dataset):
    def __init__(
        self, df, mode="train", mean=None, std=None, window_size=256, stride=128
    ):
        """
        Args:
            df (pd.DataFrame): Processed dataframe containing features and targets.
            mode (str): 'train', 'val', or 'test'.
            mean (np.array): Mean for feature normalization.
            std (np.array): Std for feature normalization.
            window_size (int): Length of sequence chunks for training.
            stride (int): Stride for sliding window in training.
        """
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
        self.feature_cols = FEATURE_COLUMNS

        # Group data by drive and phone to form sequences
        # Sort just in case, though processing should have sorted it
        df = df.sort_values(["drive_id", "phone_name", "UnixTimeMillis"]).reset_index(
            drop=True
        )

        self.sequences = []
        self.metadata = []

        # Group by unique drive-phone pairs
        groups = df.groupby(["drive_id", "phone_name"])

        for (drive_id, phone_name), group in groups:
            # Extract features
            feats = group[self.feature_cols].values.astype(np.float32)

            # Extract targets if available
            targets = None
            if "target_east" in group.columns and "target_north" in group.columns:
                targets = group[["target_east", "target_north"]].values.astype(
                    np.float32
                )

            # Extract metadata needed for reconstruction/submission
            meta = {
                "drive_id": drive_id,
                "phone_name": phone_name,
                "UnixTimeMillis": group["UnixTimeMillis"].values,
                "wls_lat": (
                    group["wls_lat"].values
                    if "wls_lat" in group.columns
                    else np.zeros(len(group))
                ),
                "wls_lon": (
                    group["wls_lon"].values
                    if "wls_lon" in group.columns
                    else np.zeros(len(group))
                ),
            }

            # Slicing logic
            seq_len = len(feats)

            if mode == "train":
                # Create sliding windows for training
                # If sequence is shorter than window_size, take the whole thing (will be padded later)
                if seq_len <= window_size:
                    self.sequences.append((feats, targets))
                    self.metadata.append(meta)
                else:
                    # Slide window
                    for start in range(0, seq_len - window_size + 1, stride):
                        end = start + window_size
                        self.sequences.append(
                            (
                                feats[start:end],
                                targets[start:end] if targets is not None else None,
                            )
                        )
                        # Slice metadata
                        sliced_meta = {
                            k: v[start:end] if isinstance(v, np.ndarray) else v
                            for k, v in meta.items()
                        }
                        self.metadata.append(sliced_meta)

                    # Handle remainder if needed (optional, often skipped or handled by overlap)
                    # For strict coverage, one could add the last window_size segment
                    if (seq_len - window_size) % stride != 0:
                        self.sequences.append(
                            (
                                feats[-window_size:],
                                targets[-window_size:] if targets is not None else None,
                            )
                        )
                        sliced_meta = {
                            k: v[-window_size:] if isinstance(v, np.ndarray) else v
                            for k, v in meta.items()
                        }
                        self.metadata.append(sliced_meta)
            else:
                # For val/test, keep full sequences
                # We will pad them to nearest multiple of 16 in __getitem__
                self.sequences.append((feats, targets))
                self.metadata.append(meta)

        # Calculate or set normalization stats
        if mean is None or std is None:
            # Compute from current data (usually passed from train set)
            all_feats = np.concatenate([s[0] for s in self.sequences], axis=0)
            self.mean = np.mean(all_feats, axis=0)
            self.std = np.std(all_feats, axis=0)
            # Avoid division by zero
            self.std[self.std < 1e-6] = 1.0
        else:
            self.mean = mean
            self.std = std

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        feats, targets = self.sequences[idx]
        meta = self.metadata[idx]

        # 1. Normalize Features
        feats = (feats - self.mean) / self.std

        # 2. Transpose to (C, L) for PyTorch Conv1d
        feats = feats.transpose(1, 0)  # Shape: (Num_Features, Length)

        original_length = feats.shape[1]

        # 3. Pad to multiple of 16 (2^4 for U-Net depth)
        pad_mod = 16
        rem = original_length % pad_mod
        pad_len = 0
        if rem != 0:
            pad_len = pad_mod - rem

        # Pad features
        # PyTorch padding is (left, right, top, bottom...)
        # Here we pad the last dimension (Length) on the right
        if pad_len > 0:
            feats = np.pad(
                feats, ((0, 0), (0, pad_len)), mode="constant", constant_values=0
            )

        # Convert to Tensor
        feats_tensor = torch.from_numpy(feats)

        # Create Mask (1 for real data, 0 for padding)
        mask = torch.ones(original_length, dtype=torch.float32)
        if pad_len > 0:
            mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)

        result = {
            "features": feats_tensor,
            "mask": mask,
            "original_length": original_length,
            "meta": meta,
        }

        # 4. Handle Targets (Multi-Scale Deep Supervision)
        if targets is not None:
            # Transpose targets to (C_out, L) -> (2, L)
            targets = targets.transpose(1, 0)

            # Pad targets
            if pad_len > 0:
                targets = np.pad(
                    targets, ((0, 0), (0, pad_len)), mode="constant", constant_values=0
                )

            targets_tensor = torch.from_numpy(targets)

            # Generate Multi-Scale Targets via Average Pooling
            # Scales: 1 (Original), 1/2, 1/4, 1/8
            # Input to avg_pool1d must be (N, C, L), so we unsqueeze and squeeze
            ms_targets = []

            # Scale 1 (Original)
            ms_targets.append(targets_tensor)

            # Auxiliary Scales
            curr_target = targets_tensor.unsqueeze(0)  # (1, 2, L)

            for _ in range(3):  # 3 downsampling steps for depth 4 (outputs at decoders)
                # Kernel size 2, stride 2 performs 2x downsampling
                curr_target = torch.nn.functional.avg_pool1d(
                    curr_target, kernel_size=2, stride=2
                )
                ms_targets.append(curr_target.squeeze(0))

            result["targets"] = ms_targets

        return result


def get_datasets(train_df, val_df, test_df):
    """
    Factory function to create datasets.
    Calculates stats on train and applies to val/test.
    """
    # Create Training Dataset
    train_dataset = GnssSequenceDataset(
        train_df,
        mode="train",
        window_size=(
            Config.BATCH_SIZE * 16 if Config.BATCH_SIZE * 16 < 512 else 512
        ),  # Dynamic window size or fixed
        stride=256,
    )

    # Extract stats
    mean = train_dataset.mean
    std = train_dataset.std

    # Create Validation Dataset
    val_dataset = GnssSequenceDataset(val_df, mode="val", mean=mean, std=std)

    # Create Test Dataset
    test_dataset = GnssSequenceDataset(test_df, mode="test", mean=mean, std=std)

    return train_dataset, val_dataset, test_dataset
