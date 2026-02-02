import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import List, Tuple, Dict, Optional

from library.config import Config
from library.features import get_train_data, get_val_data, get_test_data


class GestureDataset(Dataset):
    """
    PyTorch Dataset for the MS-TCN Gesture Recognition task.
    Wraps pre-computed features and dense frame-wise labels.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        ids: np.ndarray,
    ):
        """
        Args:
            features: Numpy object array where each element is a (T, D) float32 array.
            labels: Numpy object array where each element is a (T,) int64 array.
            ids: Numpy array of sample strings.
        """
        self.features = features
        self.labels = labels
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            feature_tensor: (T, Input_Dim) FloatTensor
            label_tensor: (T,) LongTensor
            sample_id: str
        """
        # Retrieve numpy arrays from the object array
        feat_np = self.features[idx]
        label_np = self.labels[idx]
        sample_id = self.ids[idx]

        # Convert to Tensor
        feature_tensor = torch.from_numpy(feat_np).float()
        label_tensor = torch.from_numpy(label_np).long()

        return feature_tensor, label_tensor, sample_id


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, str]]):
    """
    Custom collate function to handle variable sequence lengths.

    Args:
        batch: List of tuples (features, labels, sample_id)

    Returns:
        padded_features: (B, T_max, D)
        padded_labels: (B, T_max) - padded with -100 (ignore_index)
        mask: (B, T_max) - Boolean mask (True for valid frames, False for padding)
        lengths: (B,) - Original lengths of sequences
        ids: List of sample IDs
    """
    features, labels, ids = zip(*batch)

    # Calculate original lengths
    lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)

    # Pad features with 0.0
    # Shape: (Batch, Max_Len, Dim)
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)

    # Pad labels with -100 (standard ignore_index for CrossEntropyLoss)
    # Shape: (Batch, Max_Len)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    # Create Mask (1 for valid data, 0 for padding)
    # Shape: (Batch, Max_Len)
    max_len = padded_features.size(1)
    batch_size = padded_features.size(0)

    # Create a range tensor (0, 1, ..., max_len-1) and expand to batch
    # Compare with lengths to generate mask
    range_tensor = torch.arange(max_len, device=padded_features.device).expand(
        batch_size, max_len
    )
    lengths_expanded = lengths.unsqueeze(1).expand(batch_size, max_len)
    mask = range_tensor < lengths_expanded

    return padded_features, padded_labels, mask, lengths, ids


def create_dataloaders(
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    debug_subset_size: Optional[int] = Config.DEBUG_SUBSET_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for train, validation, and test sets.
    Uses the library.features module to load or compute data (cached).

    Args:
        batch_size: Batch size for training/inference.
        num_workers: Number of subprocesses for data loading.
        debug_subset_size: If provided, loads only a subset of data for debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load data (cached via library.features)
    # These return dicts with keys: 'features', 'labels', 'ids'
    train_data = get_train_data(subset_size=debug_subset_size)
    val_data = get_val_data(subset_size=debug_subset_size)
    test_data = get_test_data(subset_size=debug_subset_size)

    # Initialize Datasets
    train_dataset = GestureDataset(
        train_data["features"], train_data["labels"], train_data["ids"]
    )
    val_dataset = GestureDataset(
        val_data["features"], val_data["labels"], val_data["ids"]
    )
    test_dataset = GestureDataset(
        test_data["features"], test_data["labels"], test_data["ids"]
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
