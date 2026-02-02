import torch
from torch.utils.data import Dataset
import numpy as np
import os
from library.config import Config
from library.utils import get_hierarchy_mappings


class CachedFeatureDataset(Dataset):
    """
    A PyTorch Dataset for loading pre-computed features and generating hierarchical labels.
    Uses memory mapping for large feature arrays to optimize memory usage.
    """

    def __init__(self, feature_path, label_path=None, id_path=None, is_test=False):
        """
        Args:
            feature_path (str): Path to the .npy file containing features (N, D).
            label_path (str, optional): Path to the .npy file containing raw category_ids (N,).
            id_path (str, optional): Path to the .npy file containing product _ids (N,).
            is_test (bool): If True, returns (feature, product_id). If False, returns (feature, targets).
        """
        self.feature_path = feature_path
        self.label_path = label_path
        self.id_path = id_path
        self.is_test = is_test

        # Load Hierarchy Mappings
        # mapping_dict: category_id -> {'l1': idx, 'l2': idx, 'l3': idx}
        self.mapping_dict, _ = get_hierarchy_mappings(load_cached_data=True)

        # Load Features
        # Use mmap_mode='r' to keep large arrays on disk and page in as needed.
        # This prevents OOM errors given the large dataset size (~90GB for features).
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found at {feature_path}")

        self.features = np.load(feature_path, mmap_mode="r")

        # Load Labels (if available)
        if not self.is_test:
            if label_path is None or not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found at {label_path}")
            # Labels are small enough to load into memory
            self.labels = np.load(label_path)
        else:
            self.labels = None

        # Load IDs (if available/needed for test)
        if self.id_path and os.path.exists(self.id_path):
            self.ids = np.load(self.id_path)
        else:
            self.ids = None

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # Load feature vector
        # Copying creates a writable tensor in memory, detaching from the mmap backing
        feature = torch.from_numpy(self.features[idx].copy()).float()

        if self.is_test:
            # For test set, return feature and product ID
            product_id = int(self.ids[idx]) if self.ids is not None else -1
            return feature, product_id
        else:
            # For train/val, generate hierarchical targets
            raw_category_id = int(self.labels[idx])

            # Retrieve hierarchical indices
            if raw_category_id in self.mapping_dict:
                hierarchy = self.mapping_dict[raw_category_id]
                l1_target = hierarchy["l1"]
                l2_target = hierarchy["l2"]
                l3_target = hierarchy["l3"]
            else:
                # Fallback for unknown categories (should not happen in clean data)
                # Assigning 0 or a specific 'unknown' index if defined.
                # Given strict dataset, we assume validity.
                l1_target, l2_target, l3_target = 0, 0, 0

            # Return feature and tuple of targets
            return feature, l1_target, l2_target, l3_target
