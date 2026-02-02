import os
import torch
import numpy as np
from torch.utils.data import Dataset
from library.config import WORKING_DIR
from library.data_utils import HierarchyEncoder


class DecoupledFeatureDataset(Dataset):
    """
    PyTorch Dataset for loading pre-computed features and hierarchical labels.

    This dataset assumes features have been extracted using `feature_extractor.py`
    and stored as numpy arrays. It handles the conversion of raw category_ids
    into hierarchical targets (l1, l2, l3) and supports caching of these
    processed labels.
    """

    def __init__(
        self,
        feature_path,
        label_path,
        hierarchy_encoder=None,
        is_test=False,
        load_cached_data=True,
    ):
        """
        Args:
            feature_path (str): Path to the .npy file containing extracted features.
            label_path (str): Path to the .npy file containing raw category_ids (or _ids for test).
            hierarchy_encoder (HierarchyEncoder, optional): Instance for mapping IDs. Required if is_test=False.
            is_test (bool): Whether this is the test set (returns _id instead of labels).
            load_cached_data (bool): Whether to attempt loading processed labels from cache.
        """
        self.is_test = is_test
        self.feature_path = feature_path
        self.label_path = label_path

        # Load Features
        # We load into memory (RAM) because 220GB is available and it speeds up training significantly
        # compared to mmap_mode='r'.
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        print(f"Loading features from {feature_path}...")
        self.features = np.load(feature_path)

        # Load Raw Labels / IDs
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        print(f"Loading raw labels/ids from {label_path}...")
        self.raw_labels = np.load(label_path)

        # Ensure consistency
        if self.features.shape[0] != self.raw_labels.shape[0]:
            raise ValueError(
                f"Mismatch: Features {self.features.shape[0]} vs Labels {self.raw_labels.shape[0]}"
            )

        # Process Labels if Training/Validation
        if not self.is_test:
            if hierarchy_encoder is None:
                raise ValueError(
                    "hierarchy_encoder must be provided for training/validation sets."
                )

            self.hierarchy_encoder = hierarchy_encoder

            # Define cache path for hierarchical labels
            # e.g., ./working/idea_7/train_labels_hierarchical.npy
            base_name = os.path.basename(label_path).replace(".npy", "")
            self.cache_path = os.path.join(WORKING_DIR, f"{base_name}_hierarchical.npy")

            self.targets = self._load_or_process_labels(load_cached_data)
        else:
            self.targets = self.raw_labels  # For test, targets are just the _ids

    def _load_or_process_labels(self, load_cached_data):
        """
        Loads hierarchical labels from cache or computes them from raw category_ids.
        Returns a numpy array of shape (N, 3) containing (l1, l2, l3) indices.
        """
        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading cached hierarchical labels from {self.cache_path}...")
            try:
                targets = np.load(self.cache_path)
                if targets.shape[0] == self.raw_labels.shape[0]:
                    return targets
                else:
                    print("Cached labels size mismatch. Recomputing...")
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")

        # 2. Compute from Scratch
        print("Computing hierarchical labels...")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        l1_list, l2_list, l3_list = [], [], []

        # Iterate and map
        # Optimization: Accessing the lookup dict is fast.
        # We assume raw_labels are category_ids (int)
        for cat_id in self.raw_labels:
            l1, l2, l3 = self.hierarchy_encoder.get_labels(cat_id)
            l1_list.append(l1)
            l2_list.append(l2)
            l3_list.append(l3)

        # Stack into (N, 3) array
        targets = np.column_stack(
            [
                np.array(l1_list, dtype=np.int64),
                np.array(l2_list, dtype=np.int64),
                np.array(l3_list, dtype=np.int64),
            ]
        )

        # 3. Save to Cache
        print(f"Saving hierarchical labels to {self.cache_path}...")
        np.save(self.cache_path, targets)

        return targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Features are already float32, but ensure torch tensor
        # Shape: (1280,)
        feature = torch.from_numpy(self.features[idx])

        if self.is_test:
            # Return feature and product_id
            product_id = self.targets[idx]
            return feature, product_id
        else:
            # Return feature and (l1, l2, l3) targets
            # targets[idx] is [l1, l2, l3]
            l1 = self.targets[idx, 0]
            l2 = self.targets[idx, 1]
            l3 = self.targets[idx, 2]

            return feature, (l1, l2, l3)
