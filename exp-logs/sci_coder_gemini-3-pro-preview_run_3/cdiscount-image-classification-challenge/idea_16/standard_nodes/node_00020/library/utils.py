import os
import random
import logging
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: str = None, level=logging.INFO):
    """
    Configures the logging system to print to console and optionally to a file.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


class HierarchyMapper:
    """
    Manages the mapping between raw category_ids and hierarchical integer labels (L1, L2, L3).
    Handles caching of the mapping table to parquet for efficiency.
    """

    def __init__(self, hierarchy_csv_path: str, cache_path: str):
        self.hierarchy_csv_path = hierarchy_csv_path
        self.cache_path = cache_path
        self.mapping_df = None
        self.l3_to_cat_id = None

    def process_hierarchy(self, load_cached_data: bool = True):
        """
        Loads or creates the hierarchical mapping table.

        Logic:
        1. If load_cached_data is True and cache exists, load from Parquet.
        2. Otherwise, read CSV, encode L1/L2/L3 to integers, and save to Parquet.
        """
        # Ensure working directory exists
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading hierarchy mapping from {self.cache_path}")
            self.mapping_df = pd.read_parquet(self.cache_path)
        else:
            print(f"Processing hierarchy from {self.hierarchy_csv_path}")
            df = pd.read_csv(self.hierarchy_csv_path)

            # Ensure category_id is int
            df["category_id"] = df["category_id"].astype(int)

            # Sort to ensure deterministic encoding
            df = df.sort_values("category_id").reset_index(drop=True)

            # Encode Levels
            # Level 1
            df["label_l1"] = (
                df["category_level1"].astype("category").cat.codes.astype(int)
            )

            # Level 2
            df["label_l2"] = (
                df["category_level2"].astype("category").cat.codes.astype(int)
            )

            # Level 3 (Target) - We map the category_id directly to a 0..N index
            # Since we sorted by category_id, this is effectively a rank encoding,
            # but using cat.codes is safer for consistency.
            # However, for L3, the class index MUST correspond to the model's output layer.
            # We treat each unique category_id as a distinct class.
            df["label_l3"] = np.arange(len(df))

            # Select relevant columns
            self.mapping_df = df[["category_id", "label_l1", "label_l2", "label_l3"]]

            # Cache the result
            print(f"Saving hierarchy mapping to {self.cache_path}")
            self.mapping_df.to_parquet(self.cache_path, index=False)

        # Create reverse mapping for submission (L3 Index -> Category ID)
        # We convert to a numpy array where index is label_l3 and value is category_id
        # This assumes label_l3 are continuous integers 0..N-1, which our construction ensures.
        self.l3_to_cat_id = (
            self.mapping_df.set_index("label_l3")["category_id"].sort_index().values
        )

        # Validation
        assert (
            len(self.mapping_df) == Config.NUM_CLASSES_L3
        ), f"Mismatch in L3 classes: {len(self.mapping_df)} vs config {Config.NUM_CLASSES_L3}"

        return self.mapping_df

    def get_labels(self, category_ids):
        """
        Given a list/array of raw category_ids, returns the corresponding L1, L2, L3 integer labels.

        Args:
            category_ids: List or numpy array of raw category_ids.

        Returns:
            dict: {'l1': np.array, 'l2': np.array, 'l3': np.array}
        """
        if self.mapping_df is None:
            raise ValueError("Hierarchy not processed. Call process_hierarchy() first.")

        # Convert input to dataframe for merging
        input_df = pd.DataFrame({"category_id": category_ids})
        input_df["category_id"] = input_df["category_id"].astype(int)

        # Merge to get labels (preserve order)
        merged = input_df.merge(self.mapping_df, on="category_id", how="left")

        # Check for missing values (unknown categories)
        if merged.isnull().any().any():
            missing_count = merged.isnull().any(axis=1).sum()
            print(
                f"Warning: {missing_count} category_ids were not found in the hierarchy mapping."
            )
            # Fill with -1 or handle appropriately. For now, we assume data integrity.
            merged = merged.fillna(-1)

        return {
            "l1": merged["label_l1"].values.astype(np.int64),
            "l2": merged["label_l2"].values.astype(np.int64),
            "l3": merged["label_l3"].values.astype(np.int64),
        }

    def get_category_id_from_label(self, l3_labels):
        """
        Converts model predicted L3 indices back to raw category_ids.

        Args:
            l3_labels: Numpy array or tensor of predicted indices (0..N-1).

        Returns:
            np.array: Array of raw category_ids.
        """
        if self.l3_to_cat_id is None:
            raise ValueError("Hierarchy not processed. Call process_hierarchy() first.")

        if isinstance(l3_labels, torch.Tensor):
            l3_labels = l3_labels.detach().cpu().numpy()

        return self.l3_to_cat_id[l3_labels]
