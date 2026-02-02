import torch
import numpy as np
from torch.utils.data import Dataset
from library.config import Config
from library.utils import HierarchyMapper


class RamFeatureDataset(Dataset):
    """
    A PyTorch Dataset that loads pre-computed features entirely into RAM.
    It handles the mapping of raw category IDs to hierarchical targets (L1, L2, L3)
    using the HierarchyMapper.
    """

    def __init__(self, feature_path, label_path=None, mode="train"):
        """
        Args:
            feature_path (str): Path to the .npy file containing feature vectors.
            label_path (str, optional): Path to the .npy file containing raw category_ids.
            mode (str): 'train', 'val', or 'test'. Used for logging.
        """
        self.mode = mode
        self.feature_path = feature_path
        self.label_path = label_path

        print(f"[{self.mode.upper()}] Loading features from {self.feature_path}...")
        # Load features into RAM.
        # Given 220GB RAM and ~93GB dataset, we can load fully without mmap_mode.
        self.features = np.load(self.feature_path)

        self.has_labels = self.label_path is not None and list(self.label_path)

        if self.has_labels:
            print(f"[{self.mode.upper()}] Loading labels from {self.label_path}...")
            raw_labels = np.load(self.label_path)

            print(f"[{self.mode.upper()}] Mapping labels to hierarchy indices...")
            # Initialize mapper (loads cached mapping df)
            mapper = HierarchyMapper(load_cached_data=True)

            # Convert raw category_ids to L1, L2, L3 indices
            targets = mapper.get_training_targets(raw_labels)

            self.l1_targets = targets["l1"]
            self.l2_targets = targets["l2"]
            self.l3_targets = targets["l3"]
        else:
            self.l1_targets = None
            self.l2_targets = None
            self.l3_targets = None

        # Debugging: Slice dataset if configured
        if Config.DEBUG:
            print(
                f"[{self.mode.upper()}] DEBUG MODE: Slicing dataset to {Config.DEBUG_SAMPLES} samples."
            )
            self.features = self.features[: Config.DEBUG_SAMPLES]
            if self.has_labels:
                self.l1_targets = self.l1_targets[: Config.DEBUG_SAMPLES]
                self.l2_targets = self.l2_targets[: Config.DEBUG_SAMPLES]
                self.l3_targets = self.l3_targets[: Config.DEBUG_SAMPLES]

        print(f"[{self.mode.upper()}] Dataset ready. Size: {len(self.features)}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Returns:
            If labels exist: (feature_tensor, l1_target, l2_target, l3_target)
            If no labels:    (feature_tensor)
        """
        # Convert numpy feature to torch tensor
        # Features are expected to be float32
        feature = torch.from_numpy(self.features[idx])

        if self.has_labels:
            # Retrieve hierarchical targets
            l1 = torch.tensor(self.l1_targets[idx], dtype=torch.long)
            l2 = torch.tensor(self.l2_targets[idx], dtype=torch.long)
            l3 = torch.tensor(self.l3_targets[idx], dtype=torch.long)

            return feature, l1, l2, l3
        else:
            return feature
