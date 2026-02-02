import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import HierarchyMapper


class CachedFeatureDataset(Dataset):
    """
    Dataset for loading pre-computed features and hierarchical labels from disk.
    Loads entire dataset into RAM for high-speed training.
    """

    def __init__(
        self, feature_path, label_or_id_path, hierarchy_mapper=None, mode="train"
    ):
        """
        Args:
            feature_path (str): Path to the .npy file containing features (N, D).
            label_or_id_path (str): Path to the .npy file containing labels (train/val) or IDs (test).
            hierarchy_mapper (HierarchyMapper): Instance of HierarchyMapper to convert raw IDs to hierarchical targets.
            mode (str): 'train', 'val', or 'test'.
        """
        self.mode = mode

        # Load features into RAM
        # Expected shape: (N, 3328).
        # 7M samples * 3328 floats * 4 bytes ~= 93 GB. Fits comfortably in 220 GB RAM.
        try:
            self.features = np.load(feature_path)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load features from {feature_path}: {e}")

        if self.mode in ["train", "val"]:
            if hierarchy_mapper is None:
                raise ValueError("HierarchyMapper must be provided for train/val mode.")

            # Load raw category IDs
            try:
                raw_labels = np.load(label_or_id_path)
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to load labels from {label_or_id_path}: {e}"
                )

            # Map raw IDs to hierarchical indices (L1, L2, L3)
            # We use the pre-computed maps in HierarchyMapper.
            # This is deterministic processing, but fast enough to run in __init__ (~5-10s for 7M rows)
            # so disk caching of processed labels is not strictly necessary.

            # 1. Map Raw ID -> L3 Index
            # raw_to_l3_map is a dict. We use pandas for efficient vectorized mapping.
            l3_map = hierarchy_mapper.raw_to_l3_map
            l3_indices = pd.Series(raw_labels).map(l3_map)

            if l3_indices.isnull().any():
                raise ValueError(
                    "Some category IDs in the dataset were not found in the hierarchy map."
                )

            l3_indices = l3_indices.values.astype(np.int64)

            # 2. Map L3 Index -> L1, L2 Indices using fast array lookups
            l1_indices = hierarchy_mapper.l3_to_l1_map[l3_indices]
            l2_indices = hierarchy_mapper.l3_to_l2_map[l3_indices]

            # Store as tensors
            self.targets_l1 = torch.from_numpy(l1_indices)
            self.targets_l2 = torch.from_numpy(l2_indices)
            self.targets_l3 = torch.from_numpy(l3_indices)

        elif self.mode == "test":
            # Load product IDs for submission
            try:
                self.ids = np.load(label_or_id_path)
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to load IDs from {label_or_id_path}: {e}"
                )
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # Retrieve feature vector
        # torch.from_numpy creates a tensor from the numpy array (zero-copy if possible)
        feature = torch.from_numpy(self.features[idx]).float()

        if self.mode in ["train", "val"]:
            target_l1 = self.targets_l1[idx]
            target_l2 = self.targets_l2[idx]
            target_l3 = self.targets_l3[idx]
            return feature, (target_l1, target_l2, target_l3)
        else:
            _id = self.ids[idx]
            return feature, _id


def create_dataloader(
    feature_path,
    label_or_id_path,
    hierarchy_mapper=None,
    mode="train",
    batch_size=Config.TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=Config.NUM_WORKERS,
):
    """
    Factory function to create a DataLoader for the CachedFeatureDataset.

    Args:
        feature_path (str): Path to features .npy
        label_or_id_path (str): Path to labels/ids .npy
        hierarchy_mapper (HierarchyMapper): Processed mapper instance
        mode (str): 'train', 'val', or 'test'
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of worker threads

    Returns:
        DataLoader: Configured PyTorch DataLoader
    """
    dataset = CachedFeatureDataset(
        feature_path=feature_path,
        label_or_id_path=label_or_id_path,
        hierarchy_mapper=hierarchy_mapper,
        mode=mode,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
