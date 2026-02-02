import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import HierarchyMap


class FeatureDataset(Dataset):
    """
    Dataset class for loading pre-computed feature embeddings and hierarchical labels.
    Supports Train, Validation, and Test modes.
    """

    def __init__(
        self,
        feature_path,
        label_path=None,
        id_path=None,
        hierarchy_map=None,
        mode="train",
    ):
        """
        Args:
            feature_path (str): Path to the .npy file containing feature vectors.
            label_path (str, optional): Path to the .npy file containing raw category_ids.
            id_path (str, optional): Path to the .npy file containing product IDs (for test).
            hierarchy_map (HierarchyMap, optional): Instance to map raw IDs to hierarchical targets.
            mode (str): One of 'train', 'val', 'test'.
        """
        self.mode = mode
        self.feature_path = feature_path

        # Load features
        # We load into memory (mmap_mode=None) since we have 220GB RAM and features are ~100GB total.
        # This significantly speeds up training compared to reading from disk.
        print(f"Loading features from {feature_path}...")
        self.features = np.load(feature_path, mmap_mode=None)

        # Convert to torch tensor (float32) to avoid overhead in __getitem__
        # Doing this conversion on-the-fly in __getitem__ is often faster for very large datasets
        # to keep numpy's memory layout, but converting to tensor once can save CPU->GPU copy time overhead
        # if memory permits. We'll keep as numpy in RAM and convert in __getitem__ to be safe with memory fragmentation.

        if self.mode != "test":
            if label_path is None or hierarchy_map is None:
                raise ValueError(
                    "label_path and hierarchy_map are required for train/val modes"
                )

            print(f"Loading labels from {label_path}...")
            raw_labels = np.load(label_path)

            # Pre-compute hierarchical targets
            print("Mapping raw category IDs to hierarchical targets...")
            l1_targets = []
            l2_targets = []
            l3_targets = []

            # Vectorizing this lookup would be ideal, but dictionary lookup in a loop is acceptable
            # for 5-7M items given it runs once during init.
            for cat_id in raw_labels:
                l1, l2, l3 = hierarchy_map.get_targets(cat_id)
                l1_targets.append(l1)
                l2_targets.append(l2)
                l3_targets.append(l3)

            self.y_l1 = torch.tensor(l1_targets, dtype=torch.long)
            self.y_l2 = torch.tensor(l2_targets, dtype=torch.long)
            self.y_l3 = torch.tensor(l3_targets, dtype=torch.long)

        else:
            if id_path is None:
                raise ValueError("id_path is required for test mode")
            print(f"Loading IDs from {id_path}...")
            self.ids = np.load(id_path)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # Return feature as tensor
        feature = torch.from_numpy(self.features[idx])

        if self.mode == "test":
            product_id = self.ids[idx]
            return feature, product_id
        else:
            return feature, self.y_l1[idx], self.y_l2[idx], self.y_l3[idx]


def mixup_data(x, y_l1, y_l2, y_l3, alpha=0.2, device="cuda"):
    """
    Applies MixUp regularization to a batch of data.
    Returns mixed inputs, pairs of targets, and lambda.

    Args:
        x (torch.Tensor): Input features.
        y_l1, y_l2, y_l3 (torch.Tensor): Hierarchical targets.
        alpha (float): MixUp hyperparameter.
        device (str): Device to perform mixing on.

    Returns:
        mixed_x: Mixed features.
        y_l1_a, y_l1_b: Pair of L1 targets.
        y_l2_a, y_l2_b: Pair of L2 targets.
        y_l3_a, y_l3_b: Pair of L3 targets.
        lam: Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_l1_a, y_l1_b = y_l1, y_l1[index]
    y_l2_a, y_l2_b = y_l2, y_l2[index]
    y_l3_a, y_l3_b = y_l3, y_l3[index]

    return mixed_x, y_l1_a, y_l1_b, y_l2_a, y_l2_b, y_l3_a, y_l3_b, lam
