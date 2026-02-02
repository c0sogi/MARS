import os
import pandas as pd
import numpy as np
import library.config as config


class HierarchyMapper:
    """
    Handles the mapping between raw category_ids and hierarchical integer indices
    (Level 1, Level 2, Level 3) for multi-task learning.
    """

    def __init__(self, load_cached_data=True):
        """
        Initialize the mapper.

        Args:
            load_cached_data (bool): If True, attempts to load pre-computed mappings
                                     from disk. If False or file missing, recomputes.
        """
        self.mapping_path = config.HIERARCHY_MAPPING
        self._ensure_directories()

        # Set seeds for reproducibility where applicable (though pandas ops are deterministic here)
        np.random.seed(config.SEED)

        if load_cached_data and os.path.exists(self.mapping_path):
            print(f"Loading hierarchy mapping from {self.mapping_path}")
            self.df_mapping = pd.read_parquet(self.mapping_path)
        else:
            print("Computing hierarchy mapping from scratch...")
            self.df_mapping = self._create_mapping()
            print(f"Saving hierarchy mapping to {self.mapping_path}")
            self.df_mapping.to_parquet(self.mapping_path, index=False)

        # Build fast lookup structures
        self._build_lookups()

    def _ensure_directories(self):
        os.makedirs(os.path.dirname(self.mapping_path), exist_ok=True)

    def _create_mapping(self):
        """
        Reads the raw category names CSV and generates integer encodings for all levels.
        """
        # Load raw category names
        if not os.path.exists(config.CATEGORY_NAMES):
            raise FileNotFoundError(
                f"Category names file not found at {config.CATEGORY_NAMES}"
            )

        df = pd.read_csv(config.CATEGORY_NAMES)

        # Ensure deterministic order by sorting by category_id
        df = df.sort_values("category_id").reset_index(drop=True)

        # Level 3 (Target) Mapping
        # We map the raw category_id to a continuous index 0..N-1 based on the sorted order.
        # This index will be the primary target for the classifier.
        df["l3_idx"] = df.index.astype(np.int32)

        # Level 2 Mapping
        # Factorize creates a unique integer for each unique string.
        # sort=True ensures the mapping is deterministic (alphabetical order).
        l2_codes, l2_uniques = pd.factorize(df["category_level2"], sort=True)
        df["l2_idx"] = l2_codes.astype(np.int32)

        # Level 1 Mapping
        l1_codes, l1_uniques = pd.factorize(df["category_level1"], sort=True)
        df["l1_idx"] = l1_codes.astype(np.int32)

        # Keep only the necessary mapping columns
        mapping_df = df[["category_id", "l3_idx", "l2_idx", "l1_idx"]].copy()

        # Validation print
        print(f"Hierarchy Mapping Created:")
        print(f"  - Total L3 Categories (Classes): {len(df)}")
        print(f"  - Total L2 Categories: {len(l2_uniques)}")
        print(f"  - Total L1 Categories: {len(l1_uniques)}")

        return mapping_df

    def _build_lookups(self):
        """
        Creates dictionary and array-based lookups for O(1) access during training/inference.
        """
        # Map: raw category_id -> model class index (0..5269)
        self.cat_to_l3 = dict(
            zip(self.df_mapping["category_id"], self.df_mapping["l3_idx"])
        )

        # Map: model class index -> raw category_id (for submission file)
        # We sort by l3_idx to ensure the array index corresponds to l3_idx
        self.l3_to_cat = self.df_mapping.sort_values("l3_idx")["category_id"].values

        # Map: model class index -> L2 parent index
        self.l3_to_l2 = self.df_mapping.sort_values("l3_idx")["l2_idx"].values

        # Map: model class index -> L1 parent index
        self.l3_to_l1 = self.df_mapping.sort_values("l3_idx")["l1_idx"].values

    def get_l3_index(self, category_id):
        """
        Returns the model target index (0-5269) for a given raw category_id.
        Returns -1 if category_id is not found.
        """
        return self.cat_to_l3.get(category_id, -1)

    def get_category_id(self, l3_idx):
        """
        Returns the raw category_id for a given model prediction index.
        """
        if 0 <= l3_idx < len(self.l3_to_cat):
            return self.l3_to_cat[l3_idx]
        raise ValueError(f"Index {l3_idx} out of bounds for category list.")

    def get_hierarchy_targets(self, l3_idx):
        """
        Returns the parent indices (l1_idx, l2_idx) for a given l3_idx.
        Useful for calculating auxiliary hierarchical losses for a single sample.
        """
        return self.l3_to_l1[l3_idx], self.l3_to_l2[l3_idx]

    def get_all_hierarchy_targets(self):
        """
        Returns the full lookup arrays for L1 and L2 targets.

        Returns:
            tuple: (l1_targets_array, l2_targets_array)
            Where array[i] is the parent index for class i.
        """
        return self.l3_to_l1, self.l3_to_l2
