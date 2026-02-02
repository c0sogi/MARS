import torch
import numpy as np
from torch.utils.data import Dataset
from library.utils import HierarchyMapper


class CachedFeatureDataset(Dataset):
    def __init__(
        self,
        features_path,
        labels_path,
        is_train=False,
        is_test=False,
        mixup_alpha=0.0,
        limit=None,
    ):
        """
        Dataset for loading pre-computed features and generating hierarchical labels.

        Args:
            features_path (str): Path to the .npy file containing the feature matrix.
            labels_path (str): Path to the .npy file containing labels (category_id) or IDs (_id).
            is_train (bool): If True, enables Feature-Space MixUp augmentation.
            is_test (bool): If True, the dataset returns (feature, _id) and skips label processing.
            mixup_alpha (float): The alpha parameter for the Beta distribution used in MixUp.
            limit (int, optional): If provided, limits the dataset size (useful for debugging).
        """
        self.features_path = features_path
        self.labels_path = labels_path
        self.is_train = is_train
        self.is_test = is_test
        self.mixup_alpha = mixup_alpha

        # Load Features
        # We load into memory for speed as we have plenty of RAM (220GB)
        # The features are float32 arrays of shape (N, 3328)
        print(f"Loading features from {features_path}...")
        self.features = np.load(features_path)

        # Load Labels/IDs
        # The labels are int64 arrays of shape (N,)
        print(f"Loading labels/IDs from {labels_path}...")
        self.labels = np.load(labels_path)

        # Apply limit for debugging if requested
        if limit is not None and limit > 0:
            print(f"Limiting dataset to first {limit} samples.")
            self.features = self.features[:limit]
            self.labels = self.labels[:limit]

        # Pre-process Hierarchical Targets (Train/Val only)
        if not self.is_test:
            self._prepare_targets()

    def _prepare_targets(self):
        """
        Maps raw category_ids to L1, L2, and L3 model indices using HierarchyMapper.
        Stores the result as PyTorch tensors for efficient access.
        """
        print("Initializing HierarchyMapper...")
        self.mapper = HierarchyMapper(load_cached_data=True)

        print("Mapping raw category IDs to hierarchical indices...")
        # 1. Map raw category_id to L3 index (Target Class Index: 0..5269)
        # transform_targets returns a numpy array of integers
        l3_indices = self.mapper.transform_targets(self.labels)

        # 2. Map L3 index to L1 and L2 indices using lookup arrays
        # l3_to_l1 and l3_to_l2 are numpy arrays where array[l3_idx] = l1_idx
        l3_to_l1 = self.mapper.l3_to_l1
        l3_to_l2 = self.mapper.l3_to_l2

        l1_indices = l3_to_l1[l3_indices]
        l2_indices = l3_to_l2[l3_indices]

        # 3. Store as PyTorch Tensors for efficient access in __getitem__
        self.l1_targets = torch.from_numpy(l1_indices).long()
        self.l2_targets = torch.from_numpy(l2_indices).long()
        self.l3_targets = torch.from_numpy(l3_indices).long()

        print("Target mapping complete.")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Retrieve feature vector
        # features are float32, convert to tensor
        feature = torch.from_numpy(self.features[idx])

        # Case 1: Test Set (Return Feature + ID)
        if self.is_test:
            product_id = self.labels[idx]
            return feature, product_id

        # Case 2: Train/Val (Return Feature + Hierarchical Labels)
        l1 = self.l1_targets[idx]
        l2 = self.l2_targets[idx]
        l3 = self.l3_targets[idx]

        # Feature-Space MixUp (Training Only)
        if self.is_train and self.mixup_alpha > 0.0:
            return self._apply_mixup(feature, l1, l2, l3)

        # Standard return for Validation or Training without MixUp
        return feature, l1, l2, l3

    def _apply_mixup(self, feature, l1, l2, l3):
        """
        Applies MixUp augmentation by mixing with a random sample from the dataset.
        Returns the mixed feature, both sets of labels, and the lambda coefficient.
        """
        # Select a random index from the dataset
        idx2 = np.random.randint(0, len(self))

        feature2 = torch.from_numpy(self.features[idx2])
        l1_b = self.l1_targets[idx2]
        l2_b = self.l2_targets[idx2]
        l3_b = self.l3_targets[idx2]

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        # Mix features: x' = lambda * x1 + (1 - lambda) * x2
        mixed_feature = lam * feature + (1 - lam) * feature2

        # Return mixed feature, both sets of labels, and lambda
        return (
            mixed_feature,
            l1,
            l2,
            l3,
            l1_b,
            l2_b,
            l3_b,
            torch.tensor(lam, dtype=torch.float32),
        )
