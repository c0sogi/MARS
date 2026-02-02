import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import List, Dict, Tuple

from library.config import Config
from library.data_processing import load_data


class GnssDriveDataset(Dataset):
    """
    PyTorch Dataset for GNSS Drive Data.
    Handles variable length sequences of GNSS features and targets.
    """

    def __init__(self, data_list: List[Dict]):
        """
        Args:
            data_list: List of dictionaries, where each dictionary contains data for one drive
                       (features, targets, timestamps, baseline, etc.) as returned by load_data.
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        drive_data = self.data_list[idx]

        # Features: (Time, Channels) -> (Channels, Time)
        # Ensure float32 for PyTorch
        features = torch.tensor(drive_data["features"], dtype=torch.float32).transpose(
            0, 1
        )

        # Targets: (Time, 2) -> (2, Time)
        # If targets are None (test set), create dummy targets of same length
        if drive_data["targets"] is not None:
            targets_np = drive_data["targets"]
            # Create mask: Valid if target is NOT NaN
            # We assume if one coordinate is NaN, the other is too, or the point is invalid.
            # Check if any coordinate is NaN for the timestamp
            mask_np = ~np.isnan(targets_np).any(axis=1)

            # Fill NaNs with 0.0 to avoid errors in padding/tensor creation,
            # but mask will prevent them from being used in loss.
            targets_np = np.nan_to_num(targets_np, nan=0.0)

            targets = torch.tensor(targets_np, dtype=torch.float32).transpose(0, 1)
            mask = torch.tensor(mask_np, dtype=torch.bool)
        else:
            time_steps = features.shape[1]
            targets = torch.zeros((2, time_steps), dtype=torch.float32)
            mask = torch.zeros(time_steps, dtype=torch.bool)  # All invalid for test

        return {
            "features": features,
            "targets": targets,
            "mask": mask,
            "baseline": drive_data["baseline"],
            "timestamps": drive_data["timestamps"],
            "drive_id": drive_data.get("drive_id", ""),
            "phone_name": drive_data.get("phone_name", ""),
        }


def gnss_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function to handle variable length sequences.
    Pads features, targets, and masks to the maximum length in the batch.
    """
    # Extract items
    features_list = [
        item["features"].transpose(0, 1) for item in batch
    ]  # Back to (Time, Channels) for pad_sequence
    targets_list = [
        item["targets"].transpose(0, 1) for item in batch
    ]  # Back to (Time, 2)
    masks_list = [item["mask"] for item in batch]  # (Time,)

    # Pad sequences (batch_first=True -> Batch, Time, Channels)
    # Padding value 0.0
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    targets_padded = pad_sequence(targets_list, batch_first=True, padding_value=0.0)
    masks_padded = pad_sequence(masks_list, batch_first=True, padding_value=False)

    # Transpose back to (Batch, Channels, Time) for Conv1d compatibility
    features_padded = features_padded.transpose(1, 2)
    targets_padded = targets_padded.transpose(1, 2)

    # Collect metadata
    baselines = [item["baseline"] for item in batch]
    timestamps = [item["timestamps"] for item in batch]
    drive_ids = [item["drive_id"] for item in batch]
    phone_names = [item["phone_name"] for item in batch]

    return {
        "features": features_padded,
        "targets": targets_padded,
        "mask": masks_padded,
        "baseline": baselines,  # List of numpy arrays (variable length)
        "timestamps": timestamps,  # List of numpy arrays (variable length)
        "drive_id": drive_ids,
        "phone_name": phone_names,
    }


def get_dataloaders(
    load_cached_data: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads data for train, val, and test splits and returns DataLoaders.

    Args:
        load_cached_data: Whether to use cached .npz files in prepare_drive_data.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load raw data lists
    train_data = load_data(split="train", load_cached_data=load_cached_data)
    val_data = load_data(split="val", load_cached_data=load_cached_data)
    test_data = load_data(split="test", load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = GnssDriveDataset(train_data)
    val_dataset = GnssDriveDataset(val_data)
    test_dataset = GnssDriveDataset(test_data)

    # Create DataLoaders
    # Shuffle train, but not val/test
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=gnss_collate_fn,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=gnss_collate_fn,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=gnss_collate_fn,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
