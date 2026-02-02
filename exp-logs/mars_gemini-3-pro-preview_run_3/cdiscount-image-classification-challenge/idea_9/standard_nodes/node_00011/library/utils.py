import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_hierarchy_mappings(load_cached_data=True):
    """
    Generates or loads mappings between raw category IDs and hierarchical indices (Level 1, 2, 3).
    This function processes the category_names.csv file to create integer indices for
    each level of the hierarchy, which are required for the multi-task loss function.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed mappings from the cache directory.

    Returns:
        tuple: (raw_to_l3, l3_to_raw, l3_to_l1, l3_to_l2)
            - raw_to_l3 (dict): Maps raw category_id (int) to Level 3 class index (0-5269).
            - l3_to_raw (dict): Maps Level 3 class index back to raw category_id.
            - l3_to_l1 (np.ndarray): Array of shape (5270,) where index is L3 idx and value is L1 idx.
            - l3_to_l2 (np.ndarray): Array of shape (5270,) where index is L3 idx and value is L2 idx.
    """
    cache_path = Config.HIERARCHY_MAP_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
    else:
        # 2. Compute from scratch
        if not os.path.exists(Config.CATEGORY_NAMES):
            raise FileNotFoundError(
                f"Category names file not found at {Config.CATEGORY_NAMES}"
            )

        cats = pd.read_csv(Config.CATEGORY_NAMES)

        # Sort unique values to ensure deterministic indexing across runs
        l1_unique = sorted(cats["category_level1"].unique())
        l2_unique = sorted(cats["category_level2"].unique())
        l3_unique = sorted(
            cats["category_id"].unique()
        )  # category_id is the unique identifier for L3

        # Create mappings from Name/ID to Integer Index
        l1_map = {name: i for i, name in enumerate(l1_unique)}
        l2_map = {name: i for i, name in enumerate(l2_unique)}
        l3_map = {cid: i for i, cid in enumerate(l3_unique)}

        # Apply mappings to the dataframe
        df = cats.copy()
        df["l1_idx"] = df["category_level1"].map(l1_map)
        df["l2_idx"] = df["category_level2"].map(l2_map)
        df["l3_idx"] = df["category_id"].map(l3_map)

        # Select relevant columns for the map
        df = df[["category_id", "l1_idx", "l2_idx", "l3_idx"]]

        # Save to cache for future use
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path)

    # 3. Construct return objects
    # Dictionary for encoding targets during training
    raw_to_l3 = df.set_index("category_id")["l3_idx"].to_dict()

    # Dictionary for decoding predictions for submission
    l3_to_raw = df.set_index("l3_idx")["category_id"].to_dict()

    # Create numpy lookup arrays for hierarchy
    # These allow fast lookup of parent categories given the child category index
    num_l3 = df["l3_idx"].max() + 1
    l3_to_l1 = np.zeros(num_l3, dtype=np.int64)
    l3_to_l2 = np.zeros(num_l3, dtype=np.int64)

    # Fill arrays by sorting the dataframe by l3_idx to align with array indices
    df_sorted = df.sort_values("l3_idx")
    l3_to_l1[:] = df_sorted["l1_idx"].values
    l3_to_l2[:] = df_sorted["l2_idx"].values

    return raw_to_l3, l3_to_raw, l3_to_l1, l3_to_l2


def save_submission(
    test_ids, predicted_l3_indices, l3_to_raw_map, file_path=Config.SUBMISSION_PATH
):
    """
    Formats the predictions into the required CSV format and saves them.

    Args:
        test_ids (np.ndarray or list): The _id values for the test set.
        predicted_l3_indices (np.ndarray or list): The predicted class indices (Level 3).
        l3_to_raw_map (dict): Mapping from L3 index to raw category_id.
        file_path (str): Path to save the CSV.
    """
    # Handle case where inputs might be tensors
    if isinstance(predicted_l3_indices, torch.Tensor):
        predicted_l3_indices = predicted_l3_indices.detach().cpu().numpy()
    if isinstance(test_ids, torch.Tensor):
        test_ids = test_ids.detach().cpu().numpy()

    # Map internal indices back to the original category_ids
    predicted_raw_ids = [l3_to_raw_map[idx] for idx in predicted_l3_indices]

    submission = pd.DataFrame({"_id": test_ids, "category_id": predicted_raw_ids})

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save
    submission.to_csv(file_path, index=False)
    print(f"Submission saved to {file_path}")
