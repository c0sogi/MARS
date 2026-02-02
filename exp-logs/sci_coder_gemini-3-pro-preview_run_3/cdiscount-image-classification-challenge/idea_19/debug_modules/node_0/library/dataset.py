import torch
import numpy as np
import os
from torch.utils.data import Dataset
from library.config import Config


class FeatureDataset(Dataset):
    """
    Dataset for loading pre-computed features and hierarchical labels for training/validation.

    Expected Input Shapes:
    - resnet_feats: (N, 2048)
    - effnet_feats: (N, 1280)
    - labels: (N, 3) where columns are [Level1_Idx, Level2_Idx, Level3_Idx]
    """

    def __init__(self, resnet_path, effnet_path, labels_path, in_memory=True):
        super().__init__()
        self.resnet_path = resnet_path
        self.effnet_path = effnet_path
        self.labels_path = labels_path
        self.in_memory = in_memory

        # Check if files exist
        if not (
            os.path.exists(resnet_path)
            and os.path.exists(effnet_path)
            and os.path.exists(labels_path)
        ):
            raise FileNotFoundError(
                f"One or more feature files missing: {resnet_path}, {effnet_path}, {labels_path}"
            )

        # Load data
        if self.in_memory:
            # Load fully into RAM for fastest access
            self.resnet_feats = np.load(self.resnet_path)
            self.effnet_feats = np.load(self.effnet_path)
            self.labels = np.load(self.labels_path)
        else:
            # Memory-map for lower RAM usage (slower random access)
            self.resnet_feats = np.load(self.resnet_path, mmap_mode="r")
            self.effnet_feats = np.load(self.effnet_path, mmap_mode="r")
            self.labels = np.load(self.labels_path, mmap_mode="r")

        # validation
        assert (
            len(self.resnet_feats) == len(self.effnet_feats) == len(self.labels)
        ), f"Mismatch in dataset lengths: ResNet={len(self.resnet_feats)}, EffNet={len(self.effnet_feats)}, Labels={len(self.labels)}"

    def __len__(self):
        return len(self.resnet_feats)

    def __getitem__(self, idx):
        # Retrieve features (numpy views)
        r_feat = self.resnet_feats[idx]
        e_feat = self.effnet_feats[idx]

        # Retrieve labels [L1, L2, L3]
        # Labels are stored as int64 in the NPY file
        labels = self.labels[idx]

        # Convert to Tensor
        # Note: torch.from_numpy creates a tensor sharing memory if possible.
        # We cast features to float32 (standard for NN inputs) and labels to long.
        return {
            "resnet_feat": torch.from_numpy(r_feat).float(),
            "effnet_feat": torch.from_numpy(e_feat).float(),
            "label_l1": torch.tensor(labels[0], dtype=torch.long),
            "label_l2": torch.tensor(labels[1], dtype=torch.long),
            "label_l3": torch.tensor(labels[2], dtype=torch.long),
        }


class TestFeatureDataset(Dataset):
    """
    Dataset for loading pre-computed features and IDs for inference.

    Expected Input Shapes:
    - resnet_feats: (N, 2048)
    - effnet_feats: (N, 1280)
    - ids: (N,)
    """

    def __init__(self, resnet_path, effnet_path, ids_path, in_memory=True):
        super().__init__()
        self.resnet_path = resnet_path
        self.effnet_path = effnet_path
        self.ids_path = ids_path
        self.in_memory = in_memory

        if not (
            os.path.exists(resnet_path)
            and os.path.exists(effnet_path)
            and os.path.exists(ids_path)
        ):
            raise FileNotFoundError(
                f"One or more feature files missing: {resnet_path}, {effnet_path}, {ids_path}"
            )

        if self.in_memory:
            self.resnet_feats = np.load(self.resnet_path)
            self.effnet_feats = np.load(self.effnet_path)
            self.ids = np.load(self.ids_path)
        else:
            self.resnet_feats = np.load(self.resnet_path, mmap_mode="r")
            self.effnet_feats = np.load(self.effnet_path, mmap_mode="r")
            self.ids = np.load(self.ids_path, mmap_mode="r")

        assert (
            len(self.resnet_feats) == len(self.effnet_feats) == len(self.ids)
        ), f"Mismatch in dataset lengths: ResNet={len(self.resnet_feats)}, EffNet={len(self.effnet_feats)}, IDs={len(self.ids)}"

    def __len__(self):
        return len(self.resnet_feats)

    def __getitem__(self, idx):
        r_feat = self.resnet_feats[idx]
        e_feat = self.effnet_feats[idx]
        _id = self.ids[idx]

        return {
            "resnet_feat": torch.from_numpy(r_feat).float(),
            "effnet_feat": torch.from_numpy(e_feat).float(),
            "_id": torch.tensor(_id, dtype=torch.long),
        }
