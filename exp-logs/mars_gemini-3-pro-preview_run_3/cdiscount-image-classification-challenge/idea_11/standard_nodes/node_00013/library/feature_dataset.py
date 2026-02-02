import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import SEED


class CachedFeatureDataset(Dataset):
    """
    Dataset for loading pre-computed features and labels from disk.
    Maps raw category IDs to hierarchical indices (L1, L2, L3) for training.
    """

    def __init__(
        self,
        features_path,
        labels_path=None,
        ids_path=None,
        hierarchy_manager=None,
        load_in_memory=True,
    ):
        """
        Args:
            features_path (str): Path to the .npy file containing feature vectors.
            labels_path (str, optional): Path to the .npy file containing raw category IDs.
            ids_path (str, optional): Path to the .npy file containing product IDs (for test set).
            hierarchy_manager (HierarchyManager, optional): Instance for mapping labels.
            load_in_memory (bool): If True, loads data into RAM. If False, uses mmap.
        """
        self.features_path = features_path
        self.labels_path = labels_path
        self.ids_path = ids_path
        self.hierarchy_manager = hierarchy_manager

        # Load Features
        if load_in_memory:
            self.features = np.load(features_path)
        else:
            self.features = np.load(features_path, mmap_mode="r")

        self.has_labels = labels_path is not None

        # Process Labels if available
        if self.has_labels:
            if self.hierarchy_manager is None:
                raise ValueError(
                    "HierarchyManager is required when labels_path is provided."
                )

            self.raw_labels = np.load(labels_path)

            # Create mapping dictionaries from HierarchyManager
            cat_map = self.hierarchy_manager.cat_to_indices
            l1_map = {k: v["l1_idx"] for k, v in cat_map.items()}
            l2_map = {k: v["l2_idx"] for k, v in cat_map.items()}
            l3_map = {k: v["l3_idx"] for k, v in cat_map.items()}

            # Use Pandas for fast vectorized mapping of millions of labels
            s_labels = pd.Series(self.raw_labels)

            self.l1_labels = torch.tensor(s_labels.map(l1_map).values, dtype=torch.long)
            self.l2_labels = torch.tensor(s_labels.map(l2_map).values, dtype=torch.long)
            self.l3_labels = torch.tensor(s_labels.map(l3_map).values, dtype=torch.long)

        # Process IDs if available (Test set)
        elif ids_path is not None:
            self.ids = np.load(ids_path)
        else:
            self.ids = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Retrieve feature vector
        # Copy to ensure it's a writable tensor if using mmap, and float32
        feat = torch.from_numpy(self.features[idx]).float()

        if self.has_labels:
            return feat, self.l1_labels[idx], self.l2_labels[idx], self.l3_labels[idx]
        else:
            pid = self.ids[idx] if self.ids is not None else -1
            return feat, pid


class HierarchicalMixupCollate:
    """
    Collate function that applies Feature-Space MixUp.
    Returns mixed features and dual targets for loss calculation.
    """

    def __init__(self, alpha=0.2, use_mixup=True):
        self.alpha = alpha
        self.use_mixup = use_mixup

    def __call__(self, batch):
        """
        Args:
            batch: List of tuples (feature, l1, l2, l3)
        Returns:
            mixed_features: Tensor (B, D)
            targets_a: Tuple (l1, l2, l3)
            targets_b: Tuple (l1, l2, l3)
            lam: float (mixing coefficient)
        """
        # Transpose batch: [(f1, l1..), (f2, l1..)] -> ([f1, f2], [l1..], ...)
        batch_data = list(zip(*batch))

        features = torch.stack(batch_data[0])
        l1 = torch.stack(batch_data[1])
        l2 = torch.stack(batch_data[2])
        l3 = torch.stack(batch_data[3])

        batch_size = features.size(0)

        if self.use_mixup and self.alpha > 0:
            # Sample lambda
            lam = np.random.beta(self.alpha, self.alpha)

            # Shuffle indices
            index = torch.randperm(batch_size)

            # Mix features
            mixed_features = lam * features + (1 - lam) * features[index, :]

            # Prepare dual targets
            targets_a = (l1, l2, l3)
            targets_b = (l1[index], l2[index], l3[index])

            return mixed_features, targets_a, targets_b, lam
        else:
            # No MixUp: Return original features and identical targets
            targets = (l1, l2, l3)
            return features, targets, targets, 1.0


def get_dataloaders(
    train_features_path,
    train_labels_path,
    val_features_path,
    val_labels_path,
    test_features_path,
    test_ids_path,
    hierarchy_manager,
    batch_size=2048,
    mixup_alpha=0.2,
    num_workers=12,
):
    """
    Helper to construct DataLoaders for all splits.
    """
    # --- Train Loader ---
    train_ds = CachedFeatureDataset(
        train_features_path,
        labels_path=train_labels_path,
        hierarchy_manager=hierarchy_manager,
    )

    train_collate = HierarchicalMixupCollate(alpha=mixup_alpha, use_mixup=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate,
        pin_memory=True,
    )

    # --- Val Loader ---
    val_ds = CachedFeatureDataset(
        val_features_path,
        labels_path=val_labels_path,
        hierarchy_manager=hierarchy_manager,
    )

    # Disable MixUp for validation (use_mixup=False)
    val_collate = HierarchicalMixupCollate(alpha=0.0, use_mixup=False)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate,
        pin_memory=True,
    )

    # --- Test Loader ---
    test_ds = CachedFeatureDataset(
        test_features_path, ids_path=test_ids_path, hierarchy_manager=None
    )

    # Use default collate for test (returns features, ids)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
