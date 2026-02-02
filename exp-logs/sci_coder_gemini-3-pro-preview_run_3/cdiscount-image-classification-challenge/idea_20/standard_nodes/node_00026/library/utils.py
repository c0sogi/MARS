import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class HierarchyMapper:
    """
    Handles mapping between raw category IDs and hierarchical model indices (L1, L2, L3).
    Manages the translation of string categories to integer indices and provides
    lookup tables for parent-child relationships in the category tree.
    """

    def __init__(self, load_cached_data=True):
        self.mapping_df = self._load_or_create_mapping(load_cached_data)

        # Create fast lookups for L3 (Target)
        # Map raw category_id -> model output index (0..5269)
        self.cat_id_to_idx = dict(
            zip(self.mapping_df["category_id"], self.mapping_df["l3_idx"])
        )
        # Map model output index -> raw category_id
        self.idx_to_cat_id = dict(
            zip(self.mapping_df["l3_idx"], self.mapping_df["category_id"])
        )

        # Prepare hierarchical mapping arrays (Index -> Index)
        # Sort by l3_idx to ensure array index i corresponds to class i
        sorted_df = self.mapping_df.sort_values("l3_idx").reset_index(drop=True)

        self.l3_to_l1 = sorted_df["l1_idx"].values
        self.l3_to_l2 = sorted_df["l2_idx"].values

        self.num_l1 = self.mapping_df["l1_idx"].max() + 1
        self.num_l2 = self.mapping_df["l2_idx"].max() + 1
        self.num_l3 = self.mapping_df["l3_idx"].max() + 1

        # Validation print
        print(
            f"HierarchyMapper initialized: L1={self.num_l1}, L2={self.num_l2}, L3={self.num_l3} classes."
        )

    def _load_or_create_mapping(self, load_cached):
        cache_path = Config.HIERARCHY_MAPPING

        # Try loading cache
        if load_cached and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                # Minimal validation to ensure cache is valid
                required_cols = {"category_id", "l1_idx", "l2_idx", "l3_idx"}
                if required_cols.issubset(df.columns):
                    print(f"Loaded hierarchy mapping from {cache_path}")
                    return df
                else:
                    print("Cache missing required columns. Recomputing...")
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        print("Computing hierarchy mapping from source...")
        df = pd.read_csv(Config.CATEGORY_NAMES)

        # 1. Encode Level 1
        l1_cats = sorted(df["category_level1"].unique())
        l1_map = {c: i for i, c in enumerate(l1_cats)}
        df["l1_idx"] = df["category_level1"].map(l1_map)

        # 2. Encode Level 2
        l2_cats = sorted(df["category_level2"].unique())
        l2_map = {c: i for i, c in enumerate(l2_cats)}
        df["l2_idx"] = df["category_level2"].map(l2_map)

        # 3. Encode Level 3 (Target)
        # Sort by category_id to ensure deterministic mapping
        unique_ids = sorted(df["category_id"].unique())
        l3_map = {c: i for i, c in enumerate(unique_ids)}
        df["l3_idx"] = df["category_id"].map(l3_map)

        # Save cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path)
        print(f"Saved hierarchy mapping to {cache_path}")

        return df

    def get_l3_to_l1_map(self):
        """Returns a torch tensor mapping L3 indices to L1 indices."""
        return torch.tensor(self.l3_to_l1, dtype=torch.long)

    def get_l3_to_l2_map(self):
        """Returns a torch tensor mapping L3 indices to L2 indices."""
        return torch.tensor(self.l3_to_l2, dtype=torch.long)

    def transform_targets(self, category_ids):
        """Converts raw category_ids (int) to model class indices (int)."""
        # Handle scalar
        if isinstance(category_ids, (int, np.integer)):
            return self.cat_id_to_idx.get(category_ids, -1)

        # Handle iterable (list, numpy array, pandas series)
        return np.array([self.cat_id_to_idx.get(c, -1) for c in category_ids])

    def inverse_transform_targets(self, l3_indices):
        """Converts model class indices (int) back to raw category_ids (int)."""
        # Handle scalar
        if isinstance(l3_indices, (int, np.integer)):
            return self.idx_to_cat_id.get(l3_indices, -1)

        # Handle iterable
        return np.array([self.idx_to_cat_id.get(i, -1) for i in l3_indices])
