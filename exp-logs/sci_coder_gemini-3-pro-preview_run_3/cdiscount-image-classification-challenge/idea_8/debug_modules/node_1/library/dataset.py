import torch
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import Dataset
from library.config import Config


class FeatureDataset(Dataset):
    """
    Dataset for loading pre-computed features and hierarchical labels from .npy files.
    Designed to load the entire dataset into RAM for maximum training speed.
    """

    def __init__(
        self, features_path, labels_path=None, limit=None, load_in_memory=True
    ):
        """
        Args:
            features_path (str): Path to the .npy file containing feature vectors (N, 2048).
            labels_path (str, optional): Path to the .npy file containing labels (N, 3). None for test set.
            limit (int, optional): If provided, limits the dataset to the first `limit` samples (for debugging).
            load_in_memory (bool): If True, loads data into RAM. If False, uses mmap_mode='r'.
        """
        super().__init__()

        self.features_path = features_path
        self.labels_path = labels_path

        # Load Features
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found at {features_path}")

        if load_in_memory:
            self.features = np.load(features_path)
        else:
            self.features = np.load(features_path, mmap_mode="r")

        # Load Labels if present
        self.labels = None
        if labels_path and os.path.exists(labels_path):
            if load_in_memory:
                self.labels = np.load(labels_path)
            else:
                self.labels = np.load(labels_path, mmap_mode="r")

        # Apply limit for debugging
        if limit is not None:
            self.features = self.features[:limit]
            if self.labels is not None:
                self.labels = self.labels[:limit]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Load feature as float32 tensor
        # If mmap_mode is used, this reads from disk. If in memory, it's a fast array slice.
        feature = torch.tensor(self.features[idx], dtype=torch.float32)

        if self.labels is not None:
            # Load labels as long tensor (indices for L1, L2, L3)
            # Shape: (3,) -> [l1_idx, l2_idx, l3_idx]
            labels = torch.tensor(self.labels[idx], dtype=torch.long)
            return feature, labels

        # Test set case
        return feature


class MixupCollate:
    """
    Custom collate function that applies Feature-Space MixUp.
    Interpolates feature vectors and generates soft one-hot targets for all hierarchy levels.
    """

    def __init__(self, alpha=0.2):
        self.alpha = alpha
        # Retrieve class counts from Config to ensure correct one-hot encoding dimensions
        self.n_l1 = Config.NUM_CLASSES_L1
        self.n_l2 = Config.NUM_CLASSES_L2
        self.n_l3 = Config.NUM_CLASSES_L3

    def __call__(self, batch):
        """
        Args:
            batch: List of tuples (feature, labels)

        Returns:
            mixed_features: (B, 2048) Tensor
            y1_soft: (B, n_l1) Tensor - Soft targets for Level 1
            y2_soft: (B, n_l2) Tensor - Soft targets for Level 2
            y3_soft: (B, n_l3) Tensor - Soft targets for Level 3
        """
        # Unzip batch
        features, labels = zip(*batch)

        # Stack into tensors
        features = torch.stack(features)  # (B, 2048)
        labels = torch.stack(labels)  # (B, 3)

        batch_size = features.size(0)

        # Sample MixUp lambda
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Generate shuffle indices
        index = torch.randperm(batch_size)

        # Mix Features
        # x' = lambda * x + (1 - lambda) * x[index]
        mixed_features = lam * features + (1 - lam) * features[index]

        # Helper to create mixed soft targets
        def mix_targets(target_indices, num_classes):
            # Convert indices to one-hot
            # target_indices: (B,)
            one_hot = F.one_hot(target_indices, num_classes=num_classes).float()

            # Mix targets
            # y' = lambda * y + (1 - lambda) * y[index]
            mixed_targets = lam * one_hot + (1 - lam) * one_hot[index]
            return mixed_targets

        # Generate soft targets for each hierarchical level
        # labels[:, 0] -> Level 1
        # labels[:, 1] -> Level 2
        # labels[:, 2] -> Level 3
        y1_soft = mix_targets(labels[:, 0], self.n_l1)
        y2_soft = mix_targets(labels[:, 1], self.n_l2)
        y3_soft = mix_targets(labels[:, 2], self.n_l3)

        return mixed_features, y1_soft, y2_soft, y3_soft
