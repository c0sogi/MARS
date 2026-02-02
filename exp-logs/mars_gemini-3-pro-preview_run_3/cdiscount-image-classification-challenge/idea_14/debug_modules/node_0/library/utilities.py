import os
import random
import numpy as np
import pandas as pd
import torch
from library.configuration import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_accuracy(preds, targets):
    """
    Calculates simple accuracy.
    Args:
        preds: Tensor of predictions (indices).
        targets: Tensor of ground truth labels (indices).
    Returns:
        float: Accuracy value.
    """
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    if total == 0:
        return 0.0
    return correct / total


class HierarchyManager:
    """
    Manages the mapping between raw category_ids and hierarchical indices (L1, L2, L3).
    Handles loading from source CSV or cached Parquet file.
    """

    def __init__(self, load_cached_data=True):
        self.map_df = None

        # Lookups
        self.cat_id_to_l3_idx = {}
        self.l3_idx_to_cat_id = {}
        self.l3_idx_to_l2_idx = {}
        self.l3_idx_to_l1_idx = {}

        # Initialize
        if load_cached_data and os.path.exists(Config.HIERARCHY_MAPPING_PATH):
            self._load_cache()
        else:
            self._build_hierarchy()

    def _build_hierarchy(self):
        """
        Reads category_names.csv, generates indices for L1, L2, and L3,
        and saves the mapping to cache.
        """
        # Load raw category names
        df = pd.read_csv(Config.CATEGORY_NAMES)

        # Ensure category_id is int
        df["category_id"] = df["category_id"].astype(int)

        # 1. Encode Level 1
        l1_uniques = sorted(df["category_level1"].unique())
        l1_map = {name: idx for idx, name in enumerate(l1_uniques)}
        df["l1_idx"] = df["category_level1"].map(l1_map)

        # 2. Encode Level 2
        l2_uniques = sorted(df["category_level2"].unique())
        l2_map = {name: idx for idx, name in enumerate(l2_uniques)}
        df["l2_idx"] = df["category_level2"].map(l2_map)

        # 3. Encode Level 3 (This corresponds to the specific category_id)
        # We sort by category_id to ensure deterministic mapping
        l3_uniques = sorted(df["category_id"].unique())
        l3_map = {cat_id: idx for idx, cat_id in enumerate(l3_uniques)}
        df["l3_idx"] = df["category_id"].map(l3_map)

        # Verify counts match Config
        assert (
            len(l1_uniques) == Config.NUM_CLASSES_L1
        ), f"L1 count mismatch: {len(l1_uniques)} vs {Config.NUM_CLASSES_L1}"
        assert (
            len(l2_uniques) == Config.NUM_CLASSES_L2
        ), f"L2 count mismatch: {len(l2_uniques)} vs {Config.NUM_CLASSES_L2}"
        assert (
            len(l3_uniques) == Config.NUM_CLASSES_L3
        ), f"L3 count mismatch: {len(l3_uniques)} vs {Config.NUM_CLASSES_L3}"

        self.map_df = df

        # Save to cache
        os.makedirs(os.path.dirname(Config.HIERARCHY_MAPPING_PATH), exist_ok=True)
        df.to_parquet(Config.HIERARCHY_MAPPING_PATH, index=False)

        self._populate_lookups()

    def _load_cache(self):
        """Loads the hierarchy mapping from the cached Parquet file."""
        self.map_df = pd.read_parquet(Config.HIERARCHY_MAPPING_PATH)
        self._populate_lookups()

    def _populate_lookups(self):
        """Populates dictionary lookups for fast access."""
        # Create dictionaries from the dataframe
        # cat_id -> l3_idx
        self.cat_id_to_l3_idx = dict(
            zip(self.map_df["category_id"], self.map_df["l3_idx"])
        )

        # l3_idx -> cat_id (for submission)
        self.l3_idx_to_cat_id = dict(
            zip(self.map_df["l3_idx"], self.map_df["category_id"])
        )

        # l3_idx -> l2_idx (for training targets)
        self.l3_idx_to_l2_idx = dict(zip(self.map_df["l3_idx"], self.map_df["l2_idx"]))

        # l3_idx -> l1_idx (for training targets)
        self.l3_idx_to_l1_idx = dict(zip(self.map_df["l3_idx"], self.map_df["l1_idx"]))

    def get_training_targets(self, category_ids):
        """
        Converts a list/array of raw category_ids into L1, L2, and L3 target indices.

        Args:
            category_ids: List or numpy array of raw category_ids.

        Returns:
            Tuple of numpy arrays (l1_targets, l2_targets, l3_targets)
        """
        l3_targets = []
        l2_targets = []
        l1_targets = []

        for cid in category_ids:
            # Handle potential unknown IDs if any (though shouldn't happen with correct metadata)
            if cid not in self.cat_id_to_l3_idx:
                # Fallback to 0 or raise error. Raising error is safer for debugging.
                raise KeyError(f"Category ID {cid} not found in hierarchy mapping.")

            l3 = self.cat_id_to_l3_idx[cid]
            l2 = self.l3_idx_to_l2_idx[l3]
            l1 = self.l3_idx_to_l1_idx[l3]

            l3_targets.append(l3)
            l2_targets.append(l2)
            l1_targets.append(l1)

        return (
            np.array(l1_targets, dtype=np.int64),
            np.array(l2_targets, dtype=np.int64),
            np.array(l3_targets, dtype=np.int64),
        )

    def decode_predictions(self, l3_indices):
        """
        Converts L3 predicted indices back to raw category_ids.

        Args:
            l3_indices: List, numpy array, or Tensor of predicted L3 indices.

        Returns:
            Numpy array of raw category_ids.
        """
        if isinstance(l3_indices, torch.Tensor):
            l3_indices = l3_indices.cpu().numpy()

        result = []
        for idx in l3_indices:
            result.append(
                self.l3_idx_to_cat_id.get(int(idx), 0)
            )  # Default to 0 if not found

        return np.array(result, dtype=np.int64)
