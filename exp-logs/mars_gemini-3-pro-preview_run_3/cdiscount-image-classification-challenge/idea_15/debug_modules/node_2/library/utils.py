import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HierarchyMapper:
    """
    Handles the mapping between raw category_ids and hierarchical indices (L1, L2, L3)
    required for the Deep Feature Cascading model.
    """

    def __init__(self, load_cached_data=True):
        self.mapping_df = self._load_or_build_mapping(load_cached_data)

        # Create fast lookup maps
        # Map raw category_id -> l3_idx (0 to 5269)
        self.cat_to_l3 = dict(
            zip(self.mapping_df["category_id"], self.mapping_df["l3_idx"])
        )

        # Map l3_idx -> raw category_id (for submission)
        self.l3_to_cat = dict(
            zip(self.mapping_df["l3_idx"], self.mapping_df["category_id"])
        )

        # Map l3_idx -> l2_idx (0 to 482)
        self.l3_to_l2 = dict(zip(self.mapping_df["l3_idx"], self.mapping_df["l2_idx"]))

        # Map l3_idx -> l1_idx (0 to 48)
        self.l3_to_l1 = dict(zip(self.mapping_df["l3_idx"], self.mapping_df["l1_idx"]))

    def _load_or_build_mapping(self, load_cached_data):
        """
        Loads the hierarchy mapping from cache or builds it from source.
        """
        # Ensure working directory exists
        os.makedirs(os.path.dirname(Config.HIERARCHY_MAPPING_PATH), exist_ok=True)

        if load_cached_data and os.path.exists(Config.HIERARCHY_MAPPING_PATH):
            try:
                df = pd.read_parquet(Config.HIERARCHY_MAPPING_PATH)
                return df
            except Exception:
                pass  # Fallback to rebuild if load fails

        # Build from scratch
        df_cats = pd.read_csv(Config.CATEGORY_NAMES)

        # 1. Process Level 1
        # Sort alphabetically to ensure deterministic encoding
        l1_names = sorted(df_cats["category_level1"].unique())
        l1_map = {name: i for i, name in enumerate(l1_names)}

        # 2. Process Level 2
        l2_names = sorted(df_cats["category_level2"].unique())
        l2_map = {name: i for i, name in enumerate(l2_names)}

        # 3. Process Level 3 (Target)
        # Sort by ID numerically to ensure deterministic encoding
        l3_ids = sorted(df_cats["category_id"].unique())
        l3_map = {cat_id: i for i, cat_id in enumerate(l3_ids)}

        # Apply mappings
        df_cats["l1_idx"] = df_cats["category_level1"].map(l1_map)
        df_cats["l2_idx"] = df_cats["category_level2"].map(l2_map)
        df_cats["l3_idx"] = df_cats["category_id"].map(l3_map)

        # Select relevant columns for the cache
        mapping_df = df_cats[["category_id", "l1_idx", "l2_idx", "l3_idx"]].copy()

        # Save to cache
        mapping_df.to_parquet(Config.HIERARCHY_MAPPING_PATH, index=False)

        return mapping_df

    def get_training_targets(self, category_ids):
        """
        Converts a list/array of raw category_ids into hierarchical target indices.

        Args:
            category_ids: List or numpy array of raw category_ids.

        Returns:
            dict: Contains 'l1', 'l2', 'l3' tensors/arrays of indices.
        """
        l3_indices = [self.cat_to_l3[cid] for cid in category_ids]
        l2_indices = [self.l3_to_l2[idx] for idx in l3_indices]
        l1_indices = [self.l3_to_l1[idx] for idx in l3_indices]

        return {
            "l1": np.array(l1_indices, dtype=np.int64),
            "l2": np.array(l2_indices, dtype=np.int64),
            "l3": np.array(l3_indices, dtype=np.int64),
        }

    def get_submission_ids(self, l3_indices):
        """
        Converts model predictions (L3 indices) back to raw category_ids.

        Args:
            l3_indices: List or numpy array of predicted indices (0-5269).

        Returns:
            np.array: Array of raw category_ids.
        """
        return np.array([self.l3_to_cat[idx] for idx in l3_indices], dtype=np.int64)
