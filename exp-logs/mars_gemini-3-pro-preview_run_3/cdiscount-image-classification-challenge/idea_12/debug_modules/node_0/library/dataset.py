import torch
from torch.utils.data import Dataset
import numpy as np
import os
import library.config as config
from library.hierarchy_utils import HierarchyMapper


class CachedFeatureDataset(Dataset):
    """
    A custom PyTorch Dataset for loading pre-computed features and hierarchical labels.

    Features:
        - Uses memory mapping (mmap) for feature arrays to handle datasets larger than RAM.
        - Automatically expands Level 3 (fine-grained) labels into Level 1 and Level 2
          parent labels using the HierarchyMapper for multi-task learning.
        - Supports subsetting for debugging via the sample_size argument.
    """

    def __init__(
        self,
        features_path,
        labels_path=None,
        ids_path=None,
        hierarchy_mapper=None,
        sample_size=None,
    ):
        """
        Args:
            features_path (str): Path to the .npy file containing feature vectors (N, D).
            labels_path (str, optional): Path to the .npy file containing L3 labels (N,).
            ids_path (str, optional): Path to the .npy file containing product IDs (N,).
            hierarchy_mapper (HierarchyMapper, optional): Instance for mapping labels.
                                                          If None, a new one is created.
            sample_size (int, optional): If provided, limits the dataset to the first N samples.
        """
        super().__init__()

        # 1. Load Features (Memory Mapped)
        # We use mmap_mode='r' to avoid loading the entire 60GB+ feature set into RAM.
        # The OS will handle paging efficiently.
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found at {features_path}")

        self.features = np.load(features_path, mmap_mode="r")

        # 2. Setup Metadata Flags
        self.has_labels = labels_path is not None and os.path.exists(labels_path)
        self.has_ids = ids_path is not None and os.path.exists(ids_path)

        # 3. Process Labels (Hierarchical Expansion)
        if self.has_labels:
            # Load L3 labels into memory.
            # Unlike features, labels are small (~20MB for 5M samples), so we load them
            # fully into RAM for fast random access during mapping.
            self.labels_l3 = np.load(labels_path)

            # Validate alignment
            if self.features.shape[0] != self.labels_l3.shape[0]:
                raise ValueError(
                    f"Size mismatch: Features ({self.features.shape[0]}) vs "
                    f"Labels ({self.labels_l3.shape[0]})."
                )

            # Initialize Mapper if not provided
            if hierarchy_mapper is None:
                hierarchy_mapper = HierarchyMapper(load_cached_data=True)

            # Get lookup arrays: index -> parent_index
            # This returns numpy arrays where array[l3_idx] = l1_idx (or l2_idx)
            l3_to_l1, l3_to_l2 = hierarchy_mapper.get_all_hierarchy_targets()

            # Vectorized mapping of all labels to their parents
            # This is much faster than doing it per-item in __getitem__
            self.labels_l1 = l3_to_l1[self.labels_l3]
            self.labels_l2 = l3_to_l2[self.labels_l3]

        else:
            self.labels_l3 = None
            self.labels_l1 = None
            self.labels_l2 = None

        # 4. Load IDs
        if self.has_ids:
            # IDs are also small enough to load into RAM
            self.ids = np.load(ids_path)
            if self.features.shape[0] != self.ids.shape[0]:
                raise ValueError("Size mismatch: Features vs IDs.")
        else:
            self.ids = None

        # 5. Handle Subsetting (Debugging)
        self.length = self.features.shape[0]
        if sample_size is not None:
            if sample_size < self.length:
                self.length = sample_size
                # Note: We do not slice the mmap array physically to avoid creating a copy.
                # We simply limit the __len__ and access indices 0..sample_size-1.

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            If labels exist (Train/Val):
                (feature_tensor, (l1_target, l2_target, l3_target))
            If labels do not exist (Test):
                (feature_tensor, product_id)
        """
        # Read feature from mmap
        # Accessing self.features[idx] reads from disk.
        # wrapping in np.array() ensures we get a copy, which is necessary for
        # converting to a PyTorch tensor (mmap arrays are not directly writable/tensor-compatible).
        feature_np = np.array(self.features[idx])
        feature_t = torch.from_numpy(feature_np).float()

        if self.has_labels:
            # Retrieve hierarchical targets
            l1 = torch.tensor(self.labels_l1[idx], dtype=torch.long)
            l2 = torch.tensor(self.labels_l2[idx], dtype=torch.long)
            l3 = torch.tensor(self.labels_l3[idx], dtype=torch.long)

            # Return tuple for Multi-Task Learning
            return feature_t, (l1, l2, l3)

        else:
            # Inference Mode
            if self.has_ids:
                # Return ID for submission mapping
                _id = self.ids[idx]
                return feature_t, _id
            else:
                return feature_t
