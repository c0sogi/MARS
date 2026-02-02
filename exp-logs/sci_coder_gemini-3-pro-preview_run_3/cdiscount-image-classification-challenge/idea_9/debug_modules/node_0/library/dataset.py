import torch
from torch.utils.data import Dataset
import numpy as np
import os
from library.config import Config
from library.utils import get_hierarchy_mappings


class CachedFeatureDataset(Dataset):
    def __init__(self, features_path, labels_path=None, is_test=False):
        """
        Dataset for loading pre-computed ResNet50 features and hierarchical labels from RAM.

        This dataset loads the entire feature matrix into memory (approx 57GB for the full train set)
        to maximize training throughput for the MLP ensemble. It also handles the conversion
        of raw category IDs into hierarchical targets (Level 1, 2, and 3) required for the
        multi-task loss function.

        Args:
            features_path (str): Path to the .npy file containing features (N, 2048).
            labels_path (str): Path to the .npy file containing raw category_ids (N,) or _ids for test.
            is_test (bool): If True, labels_path contains _ids, and no hierarchy mapping is done.
        """
        self.is_test = is_test

        # 1. Load Features
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found at {features_path}")

        print(f"Loading features from {features_path}...")
        # Load fully into RAM. The machine has 220GB RAM, dataset is ~58GB.
        # This provides the fastest access speeds for training.
        self.features = np.load(features_path)

        # 2. Load Labels or IDs
        if labels_path is not None:
            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Labels file not found at {labels_path}")

            print(f"Loading labels/ids from {labels_path}...")
            self.labels_or_ids = np.load(labels_path)
        else:
            self.labels_or_ids = None

        # 3. Process Hierarchical Labels (Train/Val only)
        if not self.is_test and self.labels_or_ids is not None:
            print("Processing hierarchical labels...")
            # Retrieve mappings from the utility function
            # raw_to_l3: dict(category_id -> l3_index)
            # l3_to_l1: array where index is l3_index, value is l1_index
            # l3_to_l2: array where index is l3_index, value is l2_index
            raw_to_l3, _, l3_to_l1, l3_to_l2 = get_hierarchy_mappings(
                load_cached_data=True
            )

            # Map raw category_ids to Level 3 indices
            # Using list comprehension for mapping is efficient enough for 7M integers
            try:
                l3_targets = np.array(
                    [raw_to_l3[rid] for rid in self.labels_or_ids], dtype=np.int64
                )
            except KeyError as e:
                raise KeyError(
                    f"Found a category_id in labels that is not in category_names.csv: {e}"
                )

            # Map Level 3 indices to Level 1 and Level 2 indices using the lookup arrays
            l1_targets = l3_to_l1[l3_targets]
            l2_targets = l3_to_l2[l3_targets]

            # Store targets as Tensors to avoid overhead in __getitem__
            self.l1_targets = torch.from_numpy(l1_targets)
            self.l2_targets = torch.from_numpy(l2_targets)
            self.l3_targets = torch.from_numpy(l3_targets)

            # Free up the raw labels memory as they are no longer needed
            del self.labels_or_ids
            self.labels_or_ids = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Returns:
            If is_test=False: (feature_tensor, (l1_target, l2_target, l3_target))
            If is_test=True:  (feature_tensor, product_id)
        """
        # Convert numpy feature to tensor
        # Features are float32, no need for casting if saved correctly
        x = torch.from_numpy(self.features[idx])

        if self.is_test:
            # Return feature and ID for submission generation
            _id = self.labels_or_ids[idx]
            return x, _id
        else:
            # Return feature and tuple of hierarchical targets for multi-task loss
            y1 = self.l1_targets[idx]
            y2 = self.l2_targets[idx]
            y3 = self.l3_targets[idx]
            return x, (y1, y2, y3)
